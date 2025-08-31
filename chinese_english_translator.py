# chinese_english_translator.py
import requests
import re
from deep_translator import GoogleTranslator

def unwrap_text(text):
    # Define all possible brackets treated as standalone prefix or suffix
    brackets = ['(', ')', '[', ']', '{', '}', '<', '>', 
                '《', '》', '“', '”', '‘', '’', '"', '"', 
                "'", "'", "「", "」", "【", "】", "『", "』"]

    # Normalize and trim input
    text = text.strip()

    prefix_bracket = ''
    suffix_bracket = ''

    # Check for prefix bracket
    for b in brackets:
        if text.startswith(b):
            prefix_bracket = b
            break

    # Check for suffix bracket
    for b in brackets:
        if text.endswith(b):
            suffix_bracket = b
            break

    # Extract inner text based on detected brackets
    inner_start = len(prefix_bracket)
    inner_end = -len(suffix_bracket) if suffix_bracket else None

    if inner_end is not None:
        inner_text = text[inner_start:inner_end]
    else:
        inner_text = text[inner_start:]

    # Strip inner text before return
    return prefix_bracket, inner_text.strip(), suffix_bracket

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
 
