import streamlit as st
from translator import TranslatorManager

# List of Hugging Face models you want to use
MODEL_NAMES = [
    "chi-vi/hirashiba-mt-tiny-zh-vi",
    "Helsinki-NLP/opus-mt-zh-vi",
    "arcee-ai/Arcee-VyLinh"  # Newly added causal LM
]

@st.cache_resource
def load_translators():
    return TranslatorManager(MODEL_NAMES)

translators = load_translators()

st.title("🌍 Dịch tiếng Trung sang tiếng Việt (Nhiều mô hình)")
chinese_text = st.text_input("Nhập câu tiếng Trung cần dịch:", "你好，世界")

if st.button("Dịch"):
    if chinese_text.strip():
        with st.spinner("Đang dịch với tất cả mô hình..."):
            results = translators.translate_all(chinese_text)
        for idx, (model_name, translation) in enumerate(results.items(), start=1):
            # Show "Bản dịch 1", "Bản dịch 2", etc.
            st.markdown(f"**Bản dịch {idx}:** {translation}  \n<hr/>", unsafe_allow_html=True)
    else:
        st.warning("Vui lòng nhập nội dung cần dịch.")
