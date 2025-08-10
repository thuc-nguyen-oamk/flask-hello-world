# streamlit_app.py
import streamlit as st
from translator import Translator

@st.cache_resource
def load_translator():
    return Translator()

translator = load_translator()

st.title("🌍 Chinese to Vietnamese Translator")
chinese_text = st.text_input("Enter Chinese text to translate:", "你好，世界")

if st.button("Translate"):
    if chinese_text.strip():
        with st.spinner("Translating..."):
            translated = translator.translate(chinese_text)
        st.success(f"**Translation:** {translated}")
    else:
        st.warning("Please enter some text to translate.")
