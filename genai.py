from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Phi2Generator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "microsoft/phi-2"
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            print("✅ Phi-2 model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Phi-2 model: {e}")
            return False

    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text response from Phi-2 model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return ""