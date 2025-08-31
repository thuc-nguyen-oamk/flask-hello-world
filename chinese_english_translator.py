# chinese_english_translator.py
import requests
from deep_translator import GoogleTranslator

# Translation helper
def translate_text(text, target_lang):
    # print('⚠️ translate_text is temporarily OFF.')
    # return ''

    TRANSLATORS = [
        translate_text_with_google,
        translate_text_with_ftapi,
    ]

    if not text or not text.strip():
        return ''

    _, text, _ = unwrap_text(text)
    text = re.sub(r'[《》「」【】\[\]\(\)#]', '', text)

    last_error = ""
    for translator in TRANSLATORS:
        try:
            result = translator(text, target_lang)
            if not result:
                continue  # Try next translator
            else:
                return result
        except Exception as e:
            last_error = f"[{translator.__name__} Error: {e}]"
            print('⚠️ [translate_text] Falling back to the next translator...')
            continue  # Try next translator

    return f"Translation failed. Errors: {last_error}"


def translate_text_with_google(text, target_lang):
    return GoogleTranslator(source='zh-CN', target=target_lang).translate(text)


def translate_text_with_ftapi(text, target_lang):
    # Still use google translate at the core
    url = f"https://ftapi.pythonanywhere.com/translate?dl={target_lang}&text={requests.utils.quote(text)}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("destination-text", "").strip('"')  # Strip quotes if needed
    else:
        return f"[Fallback Translation Error] HTTP {response.status_code}"
 
