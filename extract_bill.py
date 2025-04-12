import google.generativeai as genai
import easyocr
import cv2
import os
import re
import json
from datetime import datetime 
import numpy as np
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from typing import Optional
import pandas as pd
from tempfile import NamedTemporaryFile

# Initialize FastAPI
app = FastAPI(
    title="AI Invoice Extractor API",
    description="Extract structured data from invoices using OCR and AI enhancement",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600
)

# Configuration
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global variables for models
use_gemini = False
use_phi2 = False
reader = None
phi2_model = None
phi2_tokenizer = None

def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        return [gray, binary, eroded]
    except Exception:
        return [image]

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image file")
        all_text = ""
        for img in preprocess_image(image):
            try:
                results = reader.readtext(img, detail=0)
                all_text += " ".join(results) + "\n"
            except Exception:
                continue
        return re.sub(r'\s+', ' ', all_text.strip())
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return ""

def is_valid_invoice_number(candidate):
    invalid_invoice_terms = ['Date', 'Time', 'Page', 'GST']
    if not re.search(r'\d{3,}', candidate): return False
    if re.fullmatch(r'20\d{2}', candidate): return False
    if any(word.lower() in candidate.lower() for word in invalid_invoice_terms): return False
    return True

def extract_invoice_number(text):
    invoice_patterns = [
        r'(?:invoice|bill|inv|receipt)\s*(?:no\.?|number|#|id)?\s*[:.]*\s*(\d{3,10})',
        r'(?:invoice|bill|inv)\s*[.:#]*\s*(\d{3,10})',
        r'(?:no|number|#|ref)\s*[.:]*\s*(\d{3,10})',
        r'(?:order|reference)\s*[.:]*\s*(\d{3,10})',
        r'\b\d{6,10}\b',
        r'(?:invoice|bill).{0,20}?(\d{3,8}-\d{3,8})',
    ]
    
    candidates = []
    for pattern in invoice_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = next((m for m in match if m), None)
            if match and is_valid_invoice_number(match):
                candidates.append(match)
    candidates += [num for num in re.findall(r'\b\d{4,8}\b', text) if is_valid_invoice_number(num)]
    return max(candidates, key=len) if candidates else 'N/A'

def extract_fields(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    data = {
        'invoice_info': {
            'invoice_number': extract_invoice_number(text),
            'invoice_date': 'N/A',
            'invoice_time': 'N/A'
        },
        'company_info': {
            'name': lines[0] if lines else 'N/A',
            'gst_number': 'N/A'
        },
        'financial_info': {
            'subtotal': 'N/A',
            'tax_amount': 'N/A',
            'total_amount': 'N/A',
            'payment_method': 'N/A'
        },
        'products': []
    }

    date_patterns = [
        r'(?:Date|Dated)[:.\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s]?\d{1,2},?[-\s]?\d{4}\b',
        r'\b\d{1,2}[-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s]?\d{4}\b'
    ]
    
    for pattern in date_patterns:
        try:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                data['invoice_info']['invoice_date'] = date_match.group(1) if date_match.group(1) else date_match.group(0)
                break
        except IndexError:
            continue

    try:
        time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)', text, re.IGNORECASE)
        if time_match:
            data['invoice_info']['invoice_time'] = time_match.group(0)
    except Exception:
        pass

    try:
        gst_match = re.search(r'([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})', text)
        if gst_match:
            data['company_info']['gst_number'] = gst_match.group(0)
    except Exception:
        pass

    amount_patterns = {
        'total_amount': r'(?:Total|Grand Total|Amount Due)[:\s]*(?:Rs\.?|₹|INR)?[.\s]*(\d[\d,]*(?:\.\d+)?)',
        'subtotal': r'(?:Sub ?total|Base Amount)[:\s]*(?:Rs\.?|₹|INR)?[.\s]*(\d[\d,]*(?:\.\d+)?)',
        'tax_amount': r'(?:Tax|GST|VAT)[:\s]*(?:Rs\.?|₹|INR)?[.\s]*(\d[\d,]*(?:\.\d+)?)'
    }
    
    for field, pattern in amount_patterns.items():
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['financial_info'][field] = match.group(1).replace(',', '') if match.group(1) else match.group(0).replace(',', '')
        except Exception:
            continue

    try:
        payment_match = re.search(r'(?:Payment Method|Paid by|Mode of Payment)[:\s]*([A-Za-z\s]+(?:,\s*[A-Za-z\s]+)*)', text, re.IGNORECASE)
        if payment_match:
            data['financial_info']['payment_method'] = payment_match.group(1).strip() if payment_match.group(1) else payment_match.group(0).strip()
    except Exception:
        pass

    try:
        product_matches = re.finditer(
            r'(\d+)\s+([^\n]{3,50}?)\s+(\d+\.\d{2})\s+(\d+\.\d{2})',
            text
        )
        
        for match in product_matches:
            try:
                qty, name, price, total = match.groups()
                data['products'].append({
                    'product_name': name.strip(),
                    'quantity': qty,
                    'unit_price': price,
                    'total_price': total
                })
            except Exception:
                continue
    except Exception:
        pass

    return data

def enhance_with_gemini(ocr_text, data):
    if not use_gemini:
        return data
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = f"""
        Analyze this OCR text from an invoice and improve the extracted data:
        
        OCR TEXT:
        {ocr_text}
        
        CURRENT EXTRACTION:
        {json.dumps(data, indent=2)}
        
        Instructions:
        1. Correct obvious OCR errors
        2. Preserve original values unless clearly wrong
        3. Don't invent new data
        4. Return only improved JSON in same structure
        5. Format dates as DD/MM/YYYY
        6. Format amounts as numbers without symbols
        """
        response = model.generate_content(prompt)
        output = response.text.strip()

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception as e:
                    print("⚠️ JSON parsing failed:", e)
            print("⚠️ Gemini response was not valid JSON:\n", output)
            raise
    except Exception as e:
        print("⚠️ Gemini enhancement failed:", e)
        return data

def enhance_with_phi2(ocr_text, data):
    if not use_phi2:
        return data
    
    try:
        prompt = f"""
        Analyze this OCR text from an invoice and improve the extracted data:
        
        OCR TEXT:
        {ocr_text}
        
        CURRENT EXTRACTION:
        {json.dumps(data, indent=2)}
        
        Instructions:
        1. Correct obvious OCR errors
        2. Preserve original values unless clearly wrong
        3. Return only the improved JSON data
        4. Format dates as DD/MM/YYYY
        5. Format amounts as numbers without symbols
        
        Respond with ONLY the JSON output:
        """
        
        inputs = phi2_tokenizer(prompt, return_tensors="pt").to(phi2_model.device)
        outputs = phi2_model.generate(
            **inputs,
            max_length=1024,
            temperature=0.7,
            do_sample=True
        )
        response = phi2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"⚠️ Phi-2 response parsing failed: {e}")
        
        return data
    except Exception as e:
        print(f"⚠️ Phi-2 enhancement failed: {e}")
        return data

def enhance_data(ocr_text, data):
    if use_gemini:
        return enhance_with_gemini(ocr_text, data)
    elif use_phi2:
        return enhance_with_phi2(ocr_text, data)
    return data

@app.get("/api/health")
async def health_check():
    status = {
        "api": "running",
        "gemini": "connected" if use_gemini else "disconnected",
        "phi2": "connected" if use_phi2 else "disconnected",
        "ocr": "ready" if reader else "not ready",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_directory": os.path.abspath(OUTPUT_DIR),
        "version": "1.1.0",
        "timestamp": datetime.now().isoformat()
    }
    return status

@app.get("/api/models")
async def list_models():
    return {
        "available_models": [
            {"name": "gemini-1.5-pro", "status": "active" if use_gemini else "inactive"},
            {"name": "phi-2", "status": "active" if use_phi2 else "inactive"}
        ],
        "active_model": "gemini-1.5-pro" if use_gemini else "phi-2" if use_phi2 else "none",
        "recommendation": "gemini-1.5-pro" if use_gemini else "phi-2" if use_phi2 else "none"
    }

@app.post("/api/extract")
async def extract_invoice(
    file: UploadFile = File(...),
    use_ai: bool = True,
    output_format: str = "json",
    save_to_file: bool = True
):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        temp_file = f"temp_{uuid.uuid4().hex}{file_ext}"
        
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        
        ocr_text = process_image(temp_file)
        if not ocr_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from image")
        
        data = extract_fields(ocr_text)
        
        if use_ai and (use_gemini or use_phi2):
            data = enhance_data(ocr_text, data)
        
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_DIR, f"invoice_output_{timestamp}.json")
            try:
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)
                data['_output_file'] = output_file
            except Exception as e:
                print(f"❌ Failed to save output: {e}")
        
        os.remove(temp_file)
        
        if output_format == "pretty":
            return JSONResponse(content={
                "status": "success",
                "data": data,
                "formatted": format_structured_data(data),
                "ocr_text": ocr_text
            })
        return JSONResponse(content={
            "status": "success",
            "data": data,
            "ocr_text": ocr_text
        })
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-excel")
async def export_to_excel(request: Request):
    try:
        data = await request.json()
        if not data or 'invoice_info' not in data:
            raise HTTPException(status_code=400, detail="Invalid invoice data")
        
        # Create DataFrames for each section with proper formatting
        invoice_df = pd.DataFrame({
            "Field": ["Invoice Number", "Invoice Date", "Invoice Time"],
            "Value": [
                data['invoice_info'].get('invoice_number', 'N/A'),
                data['invoice_info'].get('invoice_date', 'N/A'),
                data['invoice_info'].get('invoice_time', 'N/A')
            ]
        })
        
        company_df = pd.DataFrame({
            "Field": ["Company Name", "GST Number"],
            "Value": [
                data['company_info'].get('name', 'N/A'),
                data['company_info'].get('gst_number', 'N/A')
            ]
        })
        
        financial_df = pd.DataFrame({
            "Field": ["Subtotal", "Tax Amount", "Total Amount", "Payment Method"],
            "Value": [
                data['financial_info'].get('subtotal', 'N/A'),
                data['financial_info'].get('tax_amount', 'N/A'),
                data['financial_info'].get('total_amount', 'N/A'),
                data['financial_info'].get('payment_method', 'N/A')
            ]
        })
        
        # Clean up products data
        products_data = []
        for idx, product in enumerate(data.get('products', []), 1):
            products_data.append({
                "ID": idx,
                "Product Name": product.get('product_name', 'N/A'),
                "Quantity": product.get('quantity', 'N/A'),
                "Unit Price": product.get('unit_price', 'N/A'),
                "Total Price": product.get('total_price', 'N/A')
            })
        
        products_df = pd.DataFrame(products_data)
        
        # Create a temporary file
        with NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            with pd.ExcelWriter(tmp.name, engine='openpyxl') as writer:
                # Invoice Info sheet
                invoice_df.to_excel(
                    writer,
                    sheet_name='Invoice Info',
                    index=False
                )
                
                # Company Info sheet
                company_df.to_excel(
                    writer,
                    sheet_name='Company Info', 
                    index=False
                )
                
                # Financial Info sheet
                financial_df.to_excel(
                    writer,
                    sheet_name='Financial Info',
                    index=False
                )
                
                # Products sheet (only if there are products)
                if not products_df.empty:
                    products_df.to_excel(
                        writer,
                        sheet_name='Products',
                        index=False
                    )
            
            return FileResponse(
                tmp.name,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"invoice_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_structured_data(data):
    result = ""
    
    # Invoice Info
    result += "=== INVOICE INFORMATION ===\n"
    result += f"Number: {data.get('invoice_info', {}).get('invoice_number', 'N/A')}\n"
    result += f"Date: {data.get('invoice_info', {}).get('invoice_date', 'N/A')}\n"
    result += f"Time: {data.get('invoice_info', {}).get('invoice_time', 'N/A')}\n\n"
    
    # Company Info
    result += "=== COMPANY INFORMATION ===\n"
    result += f"Name: {data.get('company_info', {}).get('name', 'N/A')}\n"
    result += f"GST Number: {data.get('company_info', {}).get('gst_number', 'N/A')}\n\n"
    
    # Financial Info
    result += "=== FINANCIAL INFORMATION ===\n"
    result += f"Subtotal: {data.get('financial_info', {}).get('subtotal', 'N/A')}\n"
    result += f"Tax Amount: {data.get('financial_info', {}).get('tax_amount', 'N/A')}\n"
    result += f"Total Amount: {data.get('financial_info', {}).get('total_amount', 'N/A')}\n"
    result += f"Payment Method: {data.get('financial_info', {}).get('payment_method', 'N/A')}\n\n"
    
    # Products
    products = data.get('products', [])
    if products:
        result += f"=== PRODUCTS ({len(products)}) ===\n"
        for idx, product in enumerate(products, 1):
            result += f"Product {idx}:\n"
            result += f"  Name: {product.get('product_name', 'N/A')}\n"
            result += f"  Quantity: {product.get('quantity', 'N/A')}\n"
            result += f"  Unit Price: {product.get('unit_price', 'N/A')}\n"
            result += f"  Total Price: {product.get('total_price', 'N/A')}\n\n"
    
    return result

def initialize_models():
    global use_gemini, use_phi2, reader, phi2_model, phi2_tokenizer
    
    # Configure Gemini
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content("Test connection")
            print("✅ Gemini API is working!")
            use_gemini = True
        except Exception as e:
            print("❌ Gemini API failed, disabling Gemini features:")
            print(e)
            use_gemini = False
    else:
        print("❌ Gemini API key not found, disabling Gemini features")
        use_gemini = False

    # Configure Phi-2
    try:
        phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        phi2_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Phi-2 model loaded successfully")
        use_phi2 = True
    except Exception as e:
        print(f"❌ Phi-2 initialization failed: {e}")
        use_phi2 = False

    # Initialize EasyOCR
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✅ EasyOCR initialized successfully")
    except Exception as e:
        print(f"❌ EasyOCR initialization failed: {e}")
        raise

if __name__ == "__main__":
    initialize_models()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)