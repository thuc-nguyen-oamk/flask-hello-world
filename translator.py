import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, hf_model_name):
        self.HF_MODEL_NAME = hf_model_name
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.LOCAL_MODEL_DIR = os.path.join(
            self.script_dir,
            "local_translator_models",
            self.HF_MODEL_NAME.replace('/', '_')
        )
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        self.ensure_model_local()
        self.tokenizer, self.model = self.load_model(self.LOCAL_MODEL_DIR)

    def ensure_model_local(self):
        if os.path.exists(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            try:
                AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
                return
            except Exception:
                import shutil
                shutil.rmtree(self.LOCAL_MODEL_DIR, ignore_errors=True)

        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.HF_MODEL_NAME)
        tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
        model.save_pretrained(self.LOCAL_MODEL_DIR)

    def load_model(self, model_path):
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


class TranslatorManager:
    def __init__(self, model_names):
        self.translators = {name: Translator(name) for name in model_names}

    def translate_all(self, text):
        results = {}
        for name, translator in self.translators.items():
            try:
                results[name] = translator.translate(text)
            except Exception as e:
                results[name] = f"Error: {e}"
        return results
