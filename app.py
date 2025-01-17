import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from oauthlib.oauth2 import WebApplicationClient
import requests
from datetime import datetime
import json
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")

# Handle Render PostgreSQL URL
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url or "sqlite:///app.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    app.logger.error("Google OAuth credentials not configured!")

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# OAuth 2 client setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# API Configuration
API_KEY = os.environ.get('FLUX_API_KEY')
if not API_KEY:
    app.logger.error("API key not configured!")

API_URL = "https://api.together.xyz/v1/images/generations"
CHAT_API_URL = "https://api.together.xyz/v1/chat/completions"

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    name = db.Column(db.String(100))
    profile_pic = db.Column(db.String(100))
    credits = db.Column(db.Integer, default=10)
    images = db.relationship('Image', backref='user', lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.String(500), nullable=False)
    image_url = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Test login route - only available in development
@app.route('/test-login')
def test_login():
    if not app.debug:
        return "Not available in production", 403
    
    test_user = User.query.filter_by(email='test@example.com').first()
    if not test_user:
        test_user = User(
            email='test@example.com',
            name='کاربر تست',
            profile_pic='https://ui-avatars.com/api/?name=Test+User&background=random',
            credits=999
        )
        db.session.add(test_user)
        db.session.commit()
    
    login_user(test_user)
    return redirect(url_for('dashboard'))

def get_google_provider_cfg():
    try:
        return requests.get(GOOGLE_DISCOVERY_URL).json()
    except:
        return None

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
            app.logger.debug(f"Translation: {text} -> {translated_text}")
            return translated_text
        else:
            raise TranslationError(f"Unexpected API response format: {result}")
            
    except requests.exceptions.Timeout:
        app.logger.error("Translation API timeout")
        raise TranslationError("Translation service timeout")
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Translation API error: {str(e)}")
        raise TranslationError(f"Translation service error: {str(e)}")
    except Exception as e:
        app.logger.error(f"Unexpected translation error: {str(e)}")
        raise TranslationError(f"Unexpected error: {str(e)}")

def translate_to_persian(text: str) -> str:
    """Translate text to Persian with fallback"""
    try:
        return translate_text(text, 'fa')
    except TranslationError as e:
        app.logger.error(f"Persian translation failed: {str(e)}")
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
        app.logger.error(f"English translation failed: {str(e)}")
        return text

def generate_image(prompt: str) -> str:
    """Generate image from prompt"""
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {API_KEY}"
    }
    
    try:
        # Translate prompt to English if needed
        english_prompt = translate_to_english(prompt)
        app.logger.info(f"Translated prompt: {english_prompt}")
        
        # Enhanced prompt with better guidance
        enhanced_prompt = f"professional high-quality detailed {english_prompt}, cinematic lighting, 8k uhd, highly detailed"
        
        data = {
            "model": "black-forest-labs/FLUX.1-schnell-Free",
            "prompt": enhanced_prompt,
            "steps": 4,
            "n": 1,
            "height": 1024,
            "width": 1024
        }
        
        app.logger.info(f"Sending request to API with prompt: {enhanced_prompt}")
        response = requests.post(API_URL, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        app.logger.debug(f"API Response: {result}")
        
        if 'data' in result and len(result['data']) > 0 and 'url' in result['data'][0]:
            return result['data'][0]['url']
        else:
            app.logger.error(f"Unexpected API response format: {result}")
            return None
            
    except requests.exceptions.Timeout:
        app.logger.error("Image generation API timeout")
        return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Image generation API error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            app.logger.error(f"Error Response: {e.response.text}")
        return None
    except Exception as e:
        app.logger.error(f"Unexpected error in image generation: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    # Find out what URL to hit for Google login
    google_provider_cfg = get_google_provider_cfg()
    if not google_provider_cfg:
        return "Error loading Google configuration", 500
    
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    
    # Use library to construct the request for Google login
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route('/login/callback')
def callback():
    # Get authorization code Google sent back
    code = request.args.get("code")
    
    # Find out what URL to hit to get tokens
    google_provider_cfg = get_google_provider_cfg()
    if not google_provider_cfg:
        return "Error loading Google configuration", 500
    
    token_endpoint = google_provider_cfg["token_endpoint"]
    
    # Prepare and send request to get tokens
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code,
    )
    
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )
    
    # Parse the tokens
    client.parse_request_body_response(json.dumps(token_response.json()))
    
    # Get user info from Google
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)
    
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        users_name = userinfo_response.json()["name"]
        picture = userinfo_response.json()["picture"]
        
        # Create or update user
        user = User.query.filter_by(email=users_email).first()
        if not user:
            user = User(
                email=users_email,
                name=users_name,
                profile_pic=picture,
                credits=10
            )
            db.session.add(user)
            db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    else:
        return "User email not verified by Google.", 400

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/generate', methods=['POST'])
@login_required
def generate():
    if current_user.credits <= 0:
        return jsonify({'error': translate_to_persian("You don't have enough credits")}), 400
    
    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': translate_to_persian("Please enter an image description")}), 400
    
    image_url = generate_image(prompt)
    if not image_url:
        return jsonify({'error': translate_to_persian("Error generating image. Please try again")}), 500
    
    try:
        # Create new image record
        image = Image(prompt=prompt, image_url=image_url, user_id=current_user.id)
        current_user.credits -= 1
        
        db.session.add(image)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'image_url': image_url,
            'credits_remaining': current_user.credits,
            'message': translate_to_persian("Image generated successfully")
        })
    except Exception as e:
        app.logger.error(f"Database error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': translate_to_persian("Error saving image. Please try again")}), 500

@app.route('/add_credits', methods=['POST'])
@login_required
def add_credits():
    # In a real application, verify ad was watched
    current_user.credits += 10
    db.session.commit()
    return jsonify({'credits': current_user.credits})

@app.route('/images')
@login_required
def images():
    user_images = Image.query.filter_by(user_id=current_user.id).order_by(Image.created_at.desc()).all()
    return render_template('images.html', images=user_images)

@app.route('/download-image/<int:image_id>')
@login_required
def download_image(image_id):
    image = Image.query.get_or_404(image_id)
    
    # Verify the image belongs to the current user
    if image.user_id != current_user.id:
        return "Unauthorized", 403
    
    try:
        # Download the image from the URL
        response = requests.get(image.image_url)
        response.raise_for_status()
        
        # Create a BytesIO object from the image data
        image_data = BytesIO(response.content)
        
        # Send the file with a proper filename
        return send_file(
            image_data,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'generated-image-{image_id}.png'
        )
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return "Error downloading image", 500

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    if app.debug:
        app.run(debug=True, ssl_context="adhoc")
    else:
        app.run(ssl_context='adhoc')  # Enable HTTPS for OAuth 