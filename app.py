# app.py
import streamlit as st
from translator import TranslatorManager
# Import the new Han-Viet translator
from dich_han_viet import Translator as HanVietTranslator
# Import the Chinese-English translator
from chinese_english_translator import translate_text

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

# Wrap the input and button in a form to enable Enter key submission
with st.form(key="translation_form"):
    chinese_text = st.text_input("Enter Chinese text to translate:", "")
    submit_button = st.form_submit_button(label="Translate")

if submit_button:
    if chinese_text.strip():
        with st.spinner("Translating with all models..."):
            # Get AI model translations
            results = translators.translate_all(chinese_text)
            
            # Get Han-Viet transliteration
            hanviet_result = hanviet_translator.translate_locally_hanviet(chinese_text)
            
            # Get English translation
            english_result = translate_text(chinese_text, 'en')
            
            # Capitalize first letter of Han-Viet result
            hanviet_capitalized = hanviet_result.capitalize() if hanviet_result else ""
        
        # Display AI model translations
        for model_name, translation in results.items():
            st.markdown(f"**{model_name}:** {translation}")
        
        # Display Han-Viet transliteration
        st.markdown(f"**Han-Viet Transliteration:** {hanviet_result}")
        
        # Create combined translation text
        vietnamese_translations = list(results.values())
        newline = "\n"  # Workaround for f-string backslash limitation
        combined_text = f"""{chinese_text}
{english_result}
{hanviet_capitalized}
{newline.join(vietnamese_translations[:1])}"""  # Helsinki translation is temporarily ignored in the combination
        
        # Display combined translation in a markdown code block
        st.markdown("### Combined Translation:")
        st.code(combined_text, language=None)
    else:
        st.warning("Please enter some text to translate.")
