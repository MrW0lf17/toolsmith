import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file, Response, abort
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
import sys
from urllib.parse import urljoin

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-key-for-local-only")

# Handle database configuration
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.logger.info("Using PostgreSQL database")
else:
    # For local development, use SQLite
    sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{sqlite_path}'
    app.logger.info("Using SQLite database for local development")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# Configure SSL based on environment
if os.environ.get('FLASK_ENV') == 'production':
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['REMEMBER_COOKIE_SECURE'] = True

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

# Function to get base URL
def get_base_url():
    if os.environ.get('FLASK_ENV') == 'production':
        return "https://toolsmith.onrender.com"
    return "http://localhost:5000" if os.environ.get('FLASK_DEBUG') == '1' else "https://localhost:5000"

# Initialize SQLAlchemy after all configurations
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize OAuth properly
client = None

def init_oauth():
    global client
    if GOOGLE_CLIENT_ID:
        client = WebApplicationClient(GOOGLE_CLIENT_ID)
    else:
        app.logger.error("Google OAuth client ID not configured!")

# Call initialization after app creation
init_oauth()

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
        response = requests.get(
            GOOGLE_DISCOVERY_URL,
            timeout=5  # Add timeout
        )
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except Exception as e:
        app.logger.error(f"Error fetching Google configuration: {str(e)}")
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
        
        # Get model from request
        model = request.form.get('model', 'realistic')
        
        # Model-specific prompt enhancements
        model_prompts = {
            'realistic': f"professional high-quality detailed photograph of {english_prompt}, cinematic lighting, 8k uhd, highly detailed, photorealistic",
            'anime': f"high-quality anime illustration of {english_prompt}, anime style, vibrant colors, detailed anime art, studio ghibli inspired",
            'painting': f"artistic digital painting of {english_prompt}, oil painting style, detailed brushstrokes, artistic composition, vibrant colors",
            'pixel': f"pixel art style {english_prompt}, retro game art, 16-bit style, clear pixel definition, nostalgic gaming aesthetic",
            'minimal': f"minimalist design of {english_prompt}, clean lines, simple shapes, minimal color palette, elegant composition",
            '3d': f"3D rendered scene of {english_prompt}, octane render, volumetric lighting, subsurface scattering, high-end 3D visualization"
        }
        
        # Enhanced prompt with model-specific guidance
        enhanced_prompt = model_prompts.get(model, model_prompts['realistic'])
        
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
    if not client:
        app.logger.error("OAuth client not initialized")
        flash("Login service not properly configured", "error")
        return redirect(url_for('index'))

    try:
        google_provider_cfg = get_google_provider_cfg()
        if not google_provider_cfg:
            app.logger.error("Error loading Google configuration")
            flash("Login service temporarily unavailable", "error")
            return redirect(url_for('index'))
        
        authorization_endpoint = google_provider_cfg["authorization_endpoint"]
        
        # Use the base URL function to dynamically determine the callback URL
        callback_url = urljoin(get_base_url(), "login/callback")
        app.logger.info(f"Using callback URL: {callback_url}")
        
        # Use library to construct the request for Google login
        request_uri = client.prepare_request_uri(
            authorization_endpoint,
            redirect_uri=callback_url,
            scope=["openid", "email", "profile"],
        )
        return redirect(request_uri)
    except Exception as e:
        app.logger.error(f"Error in login route: {str(e)}")
        flash("Unable to initiate login process", "error")
        return redirect(url_for('index'))

@app.route('/login/callback')
def callback():
    try:
        # Get authorization code Google sent back
        code = request.args.get("code")
        if not code:
            app.logger.error("Authorization code not received")
            flash("Login failed: No authorization code received", "error")
            return redirect(url_for('index'))
        
        google_provider_cfg = get_google_provider_cfg()
        if not google_provider_cfg:
            app.logger.error("Error loading Google configuration")
            flash("Login failed: Could not load Google configuration", "error")
            return redirect(url_for('index'))
        
        token_endpoint = google_provider_cfg["token_endpoint"]
        callback_url = urljoin(get_base_url(), "/login/callback")
        
        # Prepare and send request to get tokens
        token_url, headers, body = client.prepare_token_request(
            token_endpoint,
            authorization_response=request.url,
            redirect_url=callback_url,
            code=code,
        )
        
        token_response = requests.post(
            token_url,
            headers=headers,
            data=body,
            auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
            timeout=10
        )
        token_response.raise_for_status()

        # Parse the tokens
        client.parse_request_body_response(json.dumps(token_response.json()))
        
        # Get user info from Google
        userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
        uri, headers, body = client.add_token(userinfo_endpoint)
        userinfo_response = requests.get(uri, headers=headers, data=body, timeout=10)
        userinfo_response.raise_for_status()
        
        if userinfo_response.json().get("email_verified"):
            users_email = userinfo_response.json()["email"]
            users_name = userinfo_response.json()["name"]
            picture = userinfo_response.json()["picture"]
            
            try:
                # Create or update user
                user = User.query.filter_by(email=users_email).first()
                if not user:
                    user = User(
                        email=users_email,
                        name=users_name,
                        profile_pic=picture,
                        credits=5
                    )
                    db.session.add(user)
                    db.session.commit()
                
                login_user(user)
                return redirect(url_for('dashboard'))
                
            except Exception as db_error:
                app.logger.error(f"Database error during user creation: {str(db_error)}")
                db.session.rollback()
                flash("Error creating user account", "error")
                return redirect(url_for('index'))
        else:
            app.logger.error("User email not verified by Google")
            flash("Login failed: Email not verified", "error")
            return redirect(url_for('index'))
            
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error in token exchange: {str(e)}")
        flash("Login failed: Authentication error", "error")
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"Unexpected error in callback: {str(e)}")
        flash("Login failed: Unexpected error", "error")
        return redirect(url_for('index'))

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
            'image_id': image.id,
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
    current_user.credits += 3
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
        response = requests.get(image.image_url, timeout=10)
        response.raise_for_status()
        
        # Create a BytesIO object from the image data
        image_data = BytesIO(response.content)
        
        # Use 'application/octet-stream' to enforce download
        return send_file(
            image_data,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'generated-image-{image.id}.jpeg'
        )
    except Exception as e:
        app.logger.error(f"Error downloading image: {str(e)}")
        return "Error downloading image", 500

@app.route('/serve-image/<int:image_id>')
@login_required
def serve_image(image_id):
    image = Image.query.get_or_404(image_id)
    
    # Verify the image belongs to the current user
    if image.user_id != current_user.id:
        return "Unauthorized", 403
    
    try:
        # Download the image from the URL
        response = requests.get(image.image_url, timeout=10)
        response.raise_for_status()
        
        # Serve the image with caching headers
        return Response(
            response.content,
            mimetype='image/jpeg',
            headers={
                'Content-Disposition': f'inline; filename=generated-image-{image.id}.jpeg',
                'Cache-Control': 'public, max-age=3600'
            }
        )
    except Exception as e:
        app.logger.error(f"Error serving image: {str(e)}")
        return "Error serving image", 500

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database tables created successfully")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {e}")
            sys.exit(1)
    
    debug_mode = os.environ.get('FLASK_DEBUG') == '1'
    if debug_mode:
        app.run(debug=True, port=5000)
    else:
        app.run(ssl_context='adhoc', port=5000)  # Enable HTTPS for OAuth 