# 🌍 Chinese to Vietnamese Multi-Model Translator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flask-hello-world.streamlit.app/)

This Streamlit application provides side-by-side translations of Chinese text into Vietnamese using multiple Hugging Face transformer models.  
It is designed to be **easily extendable** — adding a new translation model requires only adding its name to a list.

## 🚀 Demo
🔗 **Live App:** [https://flask-hello-world.streamlit.app/](https://flask-hello-world.streamlit.app/)

---

## ✨ Features
- **Multiple Models**: Compare translations from different Hugging Face models in one click.
- **Offline Caching**: Models are downloaded once and stored locally for faster future use.
- **Easy Model Extension**: Add new models by simply updating a list in the config section.
- **GPU Support**: Automatically uses GPU if available for faster inference.
- **Streamlit UI**: Simple, responsive web interface with instant results.

---

## 📦 Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/your-username/chinese-vietnamese-translator.git
cd chinese-vietnamese-translator
