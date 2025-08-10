import streamlit as st
from translator import TranslatorManager

# List of Hugging Face models you want to use
MODEL_NAMES = [
    "chi-vi/hirashiba-mt-tiny-zh-vi",
    "Helsinki-NLP/opus-mt-zh-vi",
]

@st.cache_resource
def load_translators():
    return TranslatorManager(MODEL_NAMES)

translators = load_translators()

st.title("🌍 Chinese to Vietnamese Multi-Model Translator")
chinese_text = st.text_input("Enter Chinese text to translate:", "你好，世界")

if st.button("Translate"):
    if chinese_text.strip():
        with st.spinner("Translating with all models..."):
            results = translators.translate_all(chinese_text)
        for model_name, translation in results.items():
            st.markdown(f"**{model_name}:** {translation}")
    else:
        st.warning("Please enter some text to translate.")
