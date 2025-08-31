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
chinese_text = st.text_input("Enter Chinese text to translate:", "你好，世界")

if st.button("Translate"):
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
        
        # Create combined translation text (strip any leading/trailing whitespace)
        chinese_text_clean = chinese_text.strip()
        vietnamese_translations = list(results.values())
        newline = "\n"  # Workaround for f-string backslash limitation
        combined_text = f"""{chinese_text_clean}
{english_result}
{hanviet_capitalized}
{newline.join(vietnamese_translations[:2])}"""
        
        # Display combined translation with copy button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Combined Translation:")
        with col2:
            st.code(combined_text)  # Hidden code block for copying
            # Create escaped version for JavaScript
            js_escaped_text = combined_text.replace('`', '\\`').replace('$', '\\$')
            button_html = f"""
            <div style="position: relative; top: -40px; text-align: right;">
                <button title="Copy to clipboard" onclick="navigator.clipboard.writeText(`{js_escaped_text}`)" style="
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 18px;
                ">📋</button>
            </div>
            """
            st.markdown(button_html, unsafe_allow_html=True)
        
        # Display combined translation in a styled div without scrollbars
        st.markdown(f"""
        <div style="
            background-color: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 16px;
            font-family: monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-width: 100%;
            margin-top: -30px;
        ">
        {combined_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to translate.")
