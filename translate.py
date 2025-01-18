from functools import lru_cache
import requests
from flask import current_app

class TranslationError(Exception):
    """Custom exception for translation errors"""
    pass

@lru_cache(maxsize=100)
def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text using Together.xyz API with caching
    
    Args:
        text: Text to translate
        target_lang: Target language ('en' for English, 'fa' for Persian)
    
    Returns:
        Translated text
    
    Raises:
        TranslationError: If translation fails
    """
    if not text:
        return text
        
    # Skip translation if text is already in target language (basic check)
    if target_lang == 'fa' and any('\u0600' <= c <= '\u06FF' for c in text):
        return text
    elif target_lang == 'en' and all(ord(c) < 128 for c in text):
        return text

    API_KEY = current_app.config.get('FLUX_API_KEY')
    CHAT_API_URL = current_app.config.get('CHAT_API_URL')

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {API_KEY}"
    }
    
    system_prompt = {
        'fa': "You are a professional Persian (Farsi) translator. Translate the following text to natural, fluent Persian. Only respond with the translation, no explanations.",
        'en': "You are a professional English translator. Translate the following text to natural, fluent English. Only respond with the translation, no explanations."
    }
    
    data = {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "messages": [
            {
                "role": "system",
                "content": system_prompt[target_lang]
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.3,  # Lower temperature for more consistent translations
        "max_tokens": 200
    }
    
    try:
        response = requests.post(CHAT_API_URL, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            translated_text = result['choices'][0]['message']['content'].strip()
            current_app.logger.debug(f"Translation: {text} -> {translated_text}")
            return translated_text
        else:
            raise TranslationError(f"Unexpected API response format: {result}")
            
    except requests.exceptions.Timeout:
        current_app.logger.error("Translation API timeout")
        raise TranslationError("Translation service timeout")
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Translation API error: {str(e)}")
        raise TranslationError(f"Translation service error: {str(e)}")
    except Exception as e:
        current_app.logger.error(f"Unexpected translation error: {str(e)}")
        raise TranslationError(f"Unexpected error: {str(e)}")

def translate_to_persian(text: str) -> str:
    """Translate text to Persian with fallback"""
    try:
        return translate_text(text, 'fa')
    except TranslationError as e:
        current_app.logger.error(f"Persian translation failed: {str(e)}")
        # Return a hardcoded Persian message for common errors
        error_messages = {
            'You don\'t have enough credits': 'اعتبار شما کافی نیست',
            'Please enter an image description': 'لطفاً توضیحات تصویر را وارد کنید',
            'Error generating image. Please try again': 'خطا در تولید تصویر. لطفاً دوباره تلاش کنید',
            'Image generated successfully': 'تصویر با موفقیت ایجاد شد'
        }
        return error_messages.get(text, text)

def translate_to_english(text: str) -> str:
    """Translate text to English with fallback"""
    try:
        return translate_text(text, 'en')
    except TranslationError as e:
        current_app.logger.error(f"English translation failed: {str(e)}")
        return text 