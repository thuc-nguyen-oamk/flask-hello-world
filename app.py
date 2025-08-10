import re
import jieba
import os
import json
from collections import Counter
import shutil

os.environ['HF_HUB_CACHE'] = r'E:\huggingface_cache'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Local translator with AI
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    MBart50TokenizerFast,
    MBartForConditionalGeneration
)
import torch


class Translator:
    def __init__(self, hf_model_name=None):
        # Constants for dictionary/rule file paths
        self.BASE_DIR = r"E:\z\vietphrase_tu_dien_cua_no"
        self.PHIENAM_FILE = os.path.join(self.BASE_DIR, "PhienAm.txt")
        self.VIETPHRASE_FILE = os.path.join(self.BASE_DIR, "VietPhrase.txt")
        self.NAMES_FILE = os.path.join(self.BASE_DIR, "Names.txt")
        self.LUATNHAN_FILE = os.path.join(self.BASE_DIR, "LuatNhan.txt")

        # File to log unknown tokens
        self.UNKNOWN_LOG_FILE = os.path.join(self.BASE_DIR, "unknown_tokens_log.json")

        # Settings for the AI model
        if hf_model_name:
            self.HF_MODEL_NAME = hf_model_name
        else:
            # self.HF_MODEL_NAME = "sail/Sailor-7B"  # Hugging Face model name
            # self.HF_MODEL_NAME = "sail/Sailor-4B"  # Hugging Face model name
            # self.HF_MODEL_NAME = "sail/Sailor-1.8B"
            # sluggish: # self.HF_MODEL_NAME = "sail/Sailor-0.5B"
            # too large: # self.HF_MODEL_NAME = "arcee-ai/Arcee-VyLinh"  # Hugging Face model name
            # cui: # self.HF_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"  # Hugging Face model name
            # self.HF_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-vi"  # Hugging Face model name
            # self.HF_MODEL_NAME = "chi-vi/gemma-3-1b-novels"  # Hugging Face model name
            self.HF_MODEL_NAME = "chi-vi/hirashiba-mt-tiny-zh-vi"  # Hugging Face model name
        self.script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
        self.LOCAL_MODEL_DIR = os.path.join(self.script_dir, "local_translator_models", self.HF_MODEL_NAME.replace('/', '_'))
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the directory structure
        print(f'LOCAL_MODEL_DIR = {self.LOCAL_MODEL_DIR}')
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        
        self.load_all_resources()
        self.unknown_counter = Counter()
        self.load_unknown_log()
        self.ensure_model_local()
        self.tokenizer, self.model, self.cfg = self.load_model(self.HF_MODEL_NAME, local_dir=self.LOCAL_MODEL_DIR)

    def get_model_name(self):
        return self.HF_MODEL_NAME

    def load_all_resources(self):
        self.phienam_dict = self.load_dictionary(self.PHIENAM_FILE)
        self.vietphrase_dict = self.load_dictionary(self.VIETPHRASE_FILE)
        self.names_dict = self.load_dictionary(self.NAMES_FILE)
        self.rules = self.load_rules(self.LUATNHAN_FILE)
        jieba.initialize()  # preload jieba dict to avoid delay on first use

    def reload_resources(self):
        """Reload dictionaries and rules from disk, e.g. after updating files."""
        self.load_all_resources()
        print("Dictionaries and rules reloaded.")

    def load_dictionary(self, path):
        d = {}
        if os.path.exists(path):
            with open(path, encoding='utf-8-sig') as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        d[k] = v
                    # testing...
                    # if '第' in line:
                    #     print(line)
        else:
            print(f"Warning: Dictionary file {path} not found.")
        return d

    def load_rules(self, path):
        rules = []
        if os.path.exists(path):
            with open(path, encoding='utf-8-sig') as f:
                for line in f:
                    if '=' in line:
                        pattern, replacement = line.strip().split('=', 1)
                        try:
                            rules.append((re.compile(pattern), replacement))
                        except re.error as e:
                            print(f"Invalid regex pattern: {pattern} in {path}, error: {e}")
        else:
            print(f"Warning: Rules file {path} not found.")
        return rules

    def load_unknown_log(self):
        self.unknown_counter = Counter()
        if os.path.exists(self.UNKNOWN_LOG_FILE):
            try:
                with open(self.UNKNOWN_LOG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.unknown_counter.update(data)
            except Exception as e:
                print(f"Error loading unknown tokens log: {e}")

    def save_unknown_log(self):
        with open(self.UNKNOWN_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.unknown_counter, f, ensure_ascii=False, indent=2)

    # --- Named Entity Detection (simple heuristic) ---
    def is_named_entity(self, token):
        # Detect hashtags, mentions, emojis, latin words, numbers, punctuation, and some symbols to keep as is
        if re.match(r'^[@#][\w]+$', token):  # hashtags or mentions
            return True
        if re.match(r'^[A-Za-z0-9_]+$', token):  # Latin letters and numbers (e.g. 'Chinadra')
            return True
        if re.match(r'^[\u2600-\u27BF]+$', token):  # common emojis & symbols Unicode range
            return True
        if re.match(r'^[\u4e00-\u9fff]$', token) is None and len(token) > 1:
            # If token contains no Chinese char and longer than 1, treat as named entity
            return True
        # Extend here with more sophisticated NER if needed
        return False

    def segment_mixed_text(self, text):
        """
        Splits text into tokens: 
        - Each Chinese character is a separate token
        - Non-Chinese sequences (Vietnamese/Latin words) stay as one token
        """
        tokens = re.findall(r'[\u4e00-\u9fff]|[^\u4e00-\u9fff]+', text)
        return [tok.strip() for tok in tokens if tok.strip()]

    # --- Han-Viet transliteration ---
    def translate_locally_hanviet(self, text):
        result = []
        buffer = ""
        for char in text:
            # If Chinese char, transliterate and add space-separated
            if '\u4e00' <= char <= '\u9fff':
                if buffer:
                    # Flush previous non-Chinese buffer as a single word
                    result.append(buffer)
                    buffer = ""
                result.append(self.phienam_dict.get(char, char))
            else:
                # Buffer non-Chinese chars (Latin, Vietnamese, punctuation)
                buffer += char
        if buffer:
            result.append(buffer)
        return ' '.join(result)

    # --- Thuần Việt dictionary + rule-based translation ---
    def translate_locally_thuanviet(self, text):
        tokens = self.segment_mixed_text(text)
        translated_tokens = []

        for token in tokens:
            if self.is_named_entity(token):
                translated_tokens.append(token)
                continue

            if token in self.names_dict:
                translated_tokens.append(self.names_dict[token])
            elif token in self.vietphrase_dict:
                translated_tokens.append(self.vietphrase_dict[token].split('/')[0])
            elif all('\u4e00' <= ch <= '\u9fff' for ch in token):
                # Multi-character Chinese word
                hanviet = ''.join(self.phienam_dict.get(ch, ch) for ch in token)
                translated_tokens.append(hanviet)
            else:
                self.unknown_counter[token] += 1
                translated_tokens.append(token)

        translated_text = ' '.join(translated_tokens)
        for pattern, replacement in self.rules:
            translated_text = pattern.sub(replacement, translated_text)

        self.save_unknown_log()
        return translated_text

    # --- Wrapper ---
    def translate_locally(self, text, mode='hanviet'):
        if mode == 'hanviet':
            return self.translate_locally_hanviet(text)
        elif mode == 'thuanviet':
            return self.translate_locally_thuanviet(text)
        else:
            raise ValueError("Invalid mode. Use 'hanviet' or 'thuanviet'.")

    def ensure_model_local(self):
        """Download from HF if local folder missing or incomplete."""
        print(f"Checking local model directory: {self.LOCAL_MODEL_DIR}")

        # Check if the directory exists and is not empty
        if os.path.exists(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            # Basic check: see if key files that indicate a successful save exist
            # Checking for config.json is usually reliable for both models and tokenizers
            config_path = os.path.join(self.LOCAL_MODEL_DIR, "config.json")
            # For Seq2Seq models, checking for model index or pytorch_model bin files is common
            pytorch_bin_present = any(fname.startswith("pytorch_model") and fname.endswith(".bin") for fname in os.listdir(self.LOCAL_MODEL_DIR))
            # Tokenizers often have specific files, but checking load is better. Let's check for common tokenizer files or rely on loading attempt.
            # A more robust check is simply to try loading it.

            # Attempt to load the model and tokenizer to verify integrity
            try:
                print("📂 Local model directory found. Attempting to load...")
                # Test loading tokenizer
                test_tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
                # Test loading model config (lighter than full model)
                # from transformers import AutoConfig
                # config = AutoConfig.from_pretrained(self.LOCAL_MODEL_DIR) # Optional extra check
                print("✅ Local model loaded successfully — using offline mode.")
                return # Success, no need to download
            except Exception as e:
                print(f"⚠️  Local model load failed: {e}")
                print("   Local model appears corrupted or incomplete. Re-downloading...")
                # Remove the corrupted directory
                shutil.rmtree(self.LOCAL_MODEL_DIR, ignore_errors=True)

        # If directory doesn't exist, is empty, or loading failed, download
        print("⬇ Downloading model from Hugging Face...")
        # Ensure trust_remote_code=True is used during download for tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_NAME, trust_remote_code=True)
        # Use AutoModelForCausalLM for Arcee-VyLinh (as per load_model)
        model = AutoModelForCausalLM.from_pretrained(self.HF_MODEL_NAME, trust_remote_code=True)
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
        model.save_pretrained(self.LOCAL_MODEL_DIR)
        print("✅ Model downloaded and saved locally.")

    def load_model(self, model_name, local_dir=None, device=None):
        """
        Loads different types of Chinese→Vietnamese models
        into a unified (tokenizer, model, config) tuple.
        Supports Seq2Seq (OPUS, Hirashiba), mBART, and Causal LLMs (Sailor, Arcee).
        """

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Prefer local directory if given, else load from HF
        model_path = local_dir or model_name

        # Identify model type based on name
        if "mbart" in model_name.lower():
            print(f"[INFO] Loading mBART model: {model_name}")
            tokenizer = MBart50TokenizerFast.from_pretrained(model_path, src_lang="zh_CN")
            model = MBartForConditionalGeneration.from_pretrained(model_path)
            forced_bos_token_id = tokenizer.lang_code_to_id["vi_VN"]
            config = {"type": "mbart", "forced_bos_token_id": forced_bos_token_id}

        elif "opus-mt" in model_name.lower() or "hirashiba" in model_name.lower():
            print(f"[INFO] Loading Seq2Seq translation model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            config = {"type": "seq2seq", "forced_bos_token_id": None}

        elif "sailor" in model_name.lower() or "arcee" in model_name.lower() or "qwen" in model_name.lower(): # Add Qwen check
            print(f"[INFO] Loading Causal LLM model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # Add trust_remote_code if needed for Qwen/Arcee
            # Use AutoModelForCausalLM, not AutoModelForSeq2SeqLM
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True) # Add trust_remote_code
            config = {"type": "causal", "forced_bos_token_id": None}

        else:
            raise ValueError(f"Unknown model type for: {model_name}")

        model = model.to(device)
        return tokenizer, model, config


    def translate_locally_thuanviet_with_AI(self, texts, max_len=128, mode='thuanviet'):
        # mode is not just, just for convinience (to align with other translation functions)
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.DEVICE)
        
        # Prepare generation arguments
        generate_kwargs = {
            "max_length": max_len,
            "num_beams": 4,
            "early_stopping": True,
        }
        
        # Explicitly set the target language for mBART models
        if self.cfg.get("type") == "mbart" and self.cfg.get("forced_bos_token_id") is not None:
            generate_kwargs["forced_bos_token_id"] = self.cfg["forced_bos_token_id"]
            # print(f"[DEBUG] Using forced_bos_token_id: {generate_kwargs['forced_bos_token_id']} for target language.")
        # else:
            # print(f"[DEBUG] Model type: {self.cfg.get('type')}, forced_bos_token_id: {self.cfg.get('forced_bos_token_id')}")

        outputs = self.model.generate(**inputs, **generate_kwargs) # Pass the kwargs
        
        results = [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        return results if len(results) > 1 else results[0]


translator = Translator("chi-vi/hirashiba-mt-tiny-zh-vi")
sample = r'''[Multi SUB] (全集)《似此星辰非昨夜》六千条消息她只回三十八条，五年替身婚姻到期，我带着呼吸机去了西北科研基地 #都市 #逆袭 #虐恋 #爱情 #完整版'''
print(translator.translate_locally(sample, mode='thuanviet'))
print(translator.translate_locally_thuanviet_with_AI(sample))


if __name__ == "__main__":
    translator = Translator("chi-vi/hirashiba-mt-tiny-zh-vi")
    sample = r'''[Multi SUB] (全集)《似此星辰非昨夜》六千条消息她只回三十八条，五年替身婚姻到期，我带着呼吸机去了西北科研基地 #都市 #逆袭 #虐恋 #爱情 #完整版'''
    print(translator.translate_locally(sample, mode='thuanviet'))
    print(translator.translate_locally_thuanviet_with_AI(sample))
    import sys
    sys.exit()

    # translator_helsinki = Translator("Helsinki-NLP/opus-mt-zh-vi")
    # translator_3 = Translator("chi-vi/WN-VN-14B-v0.2-GPTQ-Int4")  # not work, quan4 need GPU

    sample_text = "天下第一剑"
    sample_text_2 = '✨MULTISUB《雲深不知處》第1-75集丨蘇袀禾&賈青 #最新短劇#短劇合集#短劇#都市#甜寵#虐戀#愛情#倫理#復仇#大陸劇#熱門短劇#泡芙'
    sample_text_3 = '《雲深不知處》'

    print("Han-Viet transliteration:")
    print(translator.translate_locally(sample_text, mode='hanviet'))
    
    print("\nThuần Việt translation:")
    # print(translator.translate_locally(sample_text, mode='thuanviet'))

    print("\nThuần Việt translation with AI:")
    while True:
        text = input('Enter text to translate: ')
        # print(f'-- Translation by {translator.get_model_name()}: ', end='')
        # print(translator.translate_locally_thuanviet_with_AI(text))
        # print(f'-- Translation by {translator_helsinki.get_model_name()}: ', end='')
        # print(translator_helsinki.translate_locally_thuanviet_with_AI(text))
        # print(f'-- Translation by {translator_3.get_model_name()}: ', end='')
        # print(translator_3.translate_locally_thuanviet_with_AI(text))

    # Example: dynamically reload dictionaries if updated externally
    # translator.reload_resources()
