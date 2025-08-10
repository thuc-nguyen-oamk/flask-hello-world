import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)

class Translator:
    def __init__(self, hf_model_name="chi-vi/hirashiba-mt-tiny-zh-vi"):
        self.HF_MODEL_NAME = hf_model_name
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.LOCAL_MODEL_DIR = os.path.join(self.script_dir, "local_translator_models", self.HF_MODEL_NAME.replace('/', '_'))
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f'LOCAL_MODEL_DIR = {self.LOCAL_MODEL_DIR}')
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)

        self.ensure_model_local()
        self.tokenizer, self.model = self.load_model(self.HF_MODEL_NAME, local_dir=self.LOCAL_MODEL_DIR)

    def ensure_model_local(self):
        """Download model from HF if not present locally."""
        if os.path.exists(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            try:
                print("📂 Local model directory found. Attempting to load...")
                AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
                print("✅ Local model loaded successfully — using offline mode.")
                return
            except Exception as e:
                print(f"⚠️ Local model load failed: {e}")
                import shutil
                shutil.rmtree(self.LOCAL_MODEL_DIR, ignore_errors=True)

        print("⬇ Downloading model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.HF_MODEL_NAME)
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
        model.save_pretrained(self.LOCAL_MODEL_DIR)
        print("✅ Model downloaded and saved locally.")

    def load_model(self, model_name, local_dir=None):
        model_path = local_dir or model_name
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.DEVICE)
        return tokenizer, model

    def translate(self, text, max_length=128):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Initialize translator once on startup
translator = Translator()

@app.route('/')
def hello():
    # Test translation
    # test_text = "你好，世界"
    # translated = translator.translate(test_text)
    
    welcome_msg = "🌍 Hello, World! Chinese-Vietnamese AI Translator is running here.\n"
    # test_result = f"🔍 Test translation: '{test_text}' → '{translated}'"
    
    return welcome_msg  # + test_result

@app.route('/translate', methods=['GET'])
def translate_text():
    chinese_text = request.args.get("text", "").strip()

    if not chinese_text:
        return jsonify({"error": "No text provided. Use ?text=..."}), 400

    try:
        translated = translator.translate(chinese_text)
        return jsonify({
            "input": chinese_text,
            "translation": translated
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) # Use Render's port or default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False) # debug=False for production
