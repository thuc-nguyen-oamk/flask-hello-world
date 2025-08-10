import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig
)

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

        # Detect model type from config
        self.config = AutoConfig.from_pretrained(self.LOCAL_MODEL_DIR)
        if self.config.model_type in [
            "t5", "mt5", "mbart", "marian", "m2m_100", "fsmt", "pegasus"
        ]:
            self.model_type = "seq2seq"
            self.tokenizer, self.model = self.load_seq2seq()
        else:
            self.model_type = "causal"
            self.tokenizer, self.model = self.load_causal()

    def ensure_model_local(self):
        if os.path.exists(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            try:
                AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
                return
            except Exception:
                import shutil
                shutil.rmtree(self.LOCAL_MODEL_DIR, ignore_errors=True)

        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_NAME)
        if self.is_seq2seq_model():
            model = AutoModelForSeq2SeqLM.from_pretrained(self.HF_MODEL_NAME)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.HF_MODEL_NAME)
        tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
        model.save_pretrained(self.LOCAL_MODEL_DIR)

    def is_seq2seq_model(self):
        try:
            cfg = AutoConfig.from_pretrained(self.HF_MODEL_NAME)
            return cfg.model_type in [
                "t5", "mt5", "mbart", "marian", "m2m_100", "fsmt", "pegasus"
            ]
        except Exception:
            return False

    def load_seq2seq(self):
        tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.LOCAL_MODEL_DIR).to(self.DEVICE)
        return tokenizer, model

    def load_causal(self):
        tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(self.LOCAL_MODEL_DIR).to(self.DEVICE)
        return tokenizer, model

    def translate(self, text, max_length=128):
        if self.model_type == "seq2seq":
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.DEVICE)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:  # causal LM translation
            # Build a simple translation prompt
            prompt = f"Translate this from Chinese to Vietnamese:\n{text}\nTranslation:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.DEVICE)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=False
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Translation:")[-1].strip()


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
