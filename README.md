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
git clone https://github.com/thuc-nguyen-oamk/flask-hello-world.git
cd flask-hello-world
````

### 2️⃣ Install dependencies

We recommend using Python **3.9+** with `venv` or `conda`.

```bash
pip install -r requirements.txt
```

---

## ⚙️ Usage

### Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app will be available at:

```
http://localhost:8501
```

---

## 🛠 Configuration

All models are defined in `streamlit_app.py`:

```python
MODEL_NAMES = [
    "chi-vi/hirashiba-mt-tiny-zh-vi",
    "Helsinki-NLP/opus-mt-zh-vi"
]
```

To add another model, simply append its Hugging Face model name to the list.

---

## 📂 Project Structure

```
.
├── app.py      # Main Streamlit UI
├── translator.py         # Translator classes (single and multi-model)
├── requirements.txt      # Python dependencies
└── local_translator_models/
    └── ...               # Cached Hugging Face models (auto-created)
```

---

## 📋 Requirements

* Python 3.9+
* [Streamlit](https://streamlit.io/)
* [Transformers](https://huggingface.co/docs/transformers)
* PyTorch (with GPU support recommended)

---

## 💡 Example

Input:

```
你好，世界
```

Output:

```
chi-vi/hirashiba-mt-tiny-zh-vi: Chào ngươi, thế giới
Helsinki-NLP/opus-mt-zh-vi: Chào thế giới
```

---

## 📜 License

MIT License — feel free to use and modify.

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome!
If you add a new model that works well, consider submitting it so others can benefit.

---

**Author:** Your Name
🔗 **Live App:** [https://flask-hello-world.streamlit.app/](https://flask-hello-world.streamlit.app/)

---

Do you want me to also **add screenshots and a "How it works" diagram** so the README looks even more professional for GitHub? That would make it stand out.
```
