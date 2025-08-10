# api.py
from fastapi import FastAPI
from translator import Translator

app = FastAPI()
translator = Translator()

@app.get("/translate")
def translate(text: str):
    try:
        result = translator.translate(text)
        return {"input": text, "translation": result}
    except Exception as e:
        return {"error": str(e)}
