import streamlit as st
from translator import TranslatorManager
from dich_han_viet import Translator as HanVietTranslator
from chinese_english_translator import translate_text

MODEL_NAMES = [
    "chi-vi/hirashiba-mt-tiny-zh-vi",
    "Helsinki-NLP/opus-mt-zh-vi",
]

@st.cache_resource
def load_translators():
    return TranslatorManager(MODEL_NAMES)

@st.cache_resource
def load_hanviet_translator():
    return HanVietTranslator()

translators = load_translators()
hanviet_translator = load_hanviet_translator()

st.title("🌍 Chinese to Vietnamese Multi-Model Translator")

def do_translation():
    chinese_text = st.session_state.chinese_text
    if chinese_text.strip():
        with st.spinner("Translating with all models..."):
            results = translators.translate_all(chinese_text)
            hanviet_result = hanviet_translator.translate_locally_hanviet(chinese_text)
            english_result = translate_text(chinese_text, 'en')
            hanviet_capitalized = hanviet_result.capitalize() if hanviet_result else ""

        for model_name, translation in results.items():
            st.markdown(f"**{model_name}:** {translation}")

        st.markdown(f"**Han-Viet Transliteration:** {hanviet_result}")

        vietnamese_translations = list(results.values())
        newline = "\n"
        combined_text = f"""{chinese_text}
{english_result}
{hanviet_capitalized}
{newline.join(vietnamese_translations[:2])}"""

        st.markdown("### Combined Translation:")
        st.code(combined_text, language=None)
    else:
        st.warning("Please enter some text to translate.")

st.text_input(
    "Enter Chinese text to translate:",
    key="chinese_text",
    on_change=do_translation
)
