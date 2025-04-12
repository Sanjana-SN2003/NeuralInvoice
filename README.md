#NeuralInvoice
NeuralInvoice is an AI-driven invoice parser that combines Optical Character Recognition (OCR) and Generative AI (GenAI) to intelligently extract, analyze, and structure data from invoices and bills. It provides a user-friendly interface and optional chatbot support to streamline business billing workflows with reduced manual effort and increased accuracy.

🚀 Features
🔍 OCR-Based Data Extraction using advanced techniques to pull key fields (invoice number, date, GSTIN, total, etc.)

🤖 GenAI-Powered Template Population for accurate and structured invoice interpretation

🧾 Invoice Preview Interface using a simple web front-end

💬 Optional Chatbot to guide users in navigating billing workflows

📦 Export Capability to integrate structured data with external systems

🛠️ Tech Stack
Python 3.x

OCR Engine (EasyOCR or similar)

GenAI Model Integration (LLM-powered text interpretation)

HTML/CSS for front-end

Optional: FastAPI for backend services (extendable)

📂 Project Structure
graphql
Copy
Edit
NeuralInvoice/
├── extract_bill.py      # Main invoice OCR and parsing logic
├── genai.py             # Handles integration with generative models
├── index.html           # Front-end UI for uploading/viewing invoice data
├── invoice.jpg          # Sample invoice image
├── 70ByteBuilders.pptx  # Presentation overview of the project
└── README.md            # Project documentation
📥 Installation & Setup
Clone the Repository

bash
Copy
Edit
git clone https://github.com/Sanjana-SN2003/NeuralInvoice.git
cd NeuralInvoice
Install Required Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Application

bash
Copy
Edit
python extract_bill.py
Access the UI Open index.html in your browser to upload an invoice and see the output.

📸 Sample Workflow
Upload an invoice (.jpg, .png, etc.).

The system extracts key fields using OCR.

GenAI refines and fills structured templates.

View and download structured data for further use.

Collaborators
1.Archana A L : Building ML Model,Backend Integration 
2.Sanjana S N: Building ML Model,Backend Integration.
3.Ananya A S: UI and Frontend Integration
4.Chaithra M:UI and Frontend Integration

🤝 Contributing
Contributions are welcome! If you have ideas to improve the extraction pipeline, enhance the UI, or expand template support, feel free to fork the repo and create a pull request.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

📧 Contact
For any queries or collaborations, feel free to connect with the project maintainer via GitHub.

