# app.py

import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from threading import Thread
from flask import Flask, request, jsonify

# -----------------------------
# Translator Class (same as before)
# -----------------------------
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

# -----------------------------
# Load Translator Once
# -----------------------------
@st.cache_resource
def load_translator():
    return Translator()

translator = load_translator()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 Chinese to Vietnamese Translator")
chinese_text = st.text_input("Enter Chinese text to translate:", "你好，世界")

if st.button("Translate"):
    if chinese_text.strip():
        with st.spinner("Translating..."):
            translated = translator.translate(chinese_text)
        st.success(f"**Translation:** {translated}")
    else:
        st.warning("Please enter some text to translate.")

# -----------------------------
# Flask API in Background Thread
# -----------------------------
flask_app = Flask(__name__)

@flask_app.route('/translate', methods=['GET'])
def api_translate():
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = translator.translate(text)
        return jsonify({"input": text, "translation": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    port = int(os.environ.get("FLASK_PORT", 8000))
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# Start Flask in background
Thread(target=run_flask, daemon=True).start()
