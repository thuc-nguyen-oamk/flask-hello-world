# app.py
import streamlit as st
from translator import TranslatorManager
# Import the new Han-Viet translator
from dich_han_viet import Translator as HanVietTranslator

# List of Hugging Face models you want to use
MODEL_NAMES = [
    "chi-vi/hirashiba-mt-tiny-zh-vi",
    "Helsinki-NLP/opus-mt-zh-vi",
]

@st.cache_resource
def load_translators():
    return TranslatorManager(MODEL_NAMES)

# Load AI translators
translators = load_translators()

# Initialize Han-Viet translator
@st.cache_resource
def load_hanviet_translator():
    return HanVietTranslator()

hanviet_translator = load_hanviet_translator()

st.title("🌍 Chinese to Vietnamese Multi-Model Translator")
chinese_text = st.text_input("Enter Chinese text to translate:", "你好，世界")

if st.button("Translate"):
    if chinese_text.strip():
        with st.spinner("Translating with all models..."):
            # Get AI model translations
            results = translators.translate_all(chinese_text)
            
            # Get Han-Viet transliteration
            hanviet_result = hanviet_translator.translate_locally_hanviet(chinese_text)
        
        # Display AI model translations
        for model_name, translation in results.items():
            st.markdown(f"**{model_name}:** {translation}")
        
        # Display Han-Viet transliteration
        st.markdown(f"**Han-Viet Transliteration:** {hanviet_result}")
    else:
        st.warning("Please enter some text to translate.")
