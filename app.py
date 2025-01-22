import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file, Response, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from oauthlib.oauth2 import WebApplicationClient
import requests
from datetime import datetime, timedelta
import json
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from functools import lru_cache
import sys
from urllib.parse import urljoin
from translations import translations
from sqlalchemy.exc import SQLAlchemyError
import stripe
from sqlalchemy import inspect

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-key-for-local-only")

# Supported languages
LANGUAGES = {
    'fa': 'Persian',
    'en': 'English',
    'zh': 'Chinese',
    'it': 'Italian',
    'de': 'German',
    'ja': 'Japanese',
    'fr': 'French',
    'hi': 'Hindi',
    'ru': 'Russian',
    'tr': 'Turkish',
    'az': 'Azerbaijani',
    'id': 'Indonesian'
}

def _(text):
    """Translation function"""
    lang = session.get('lang', 'fa')
    return translations.get(lang, {}).get(text, text)

@app.context_processor
def inject_translations():
    """Make translation function available in templates"""
    return dict(_=_)

@app.route('/set-language', methods=['POST'])
def set_language():
    data = request.get_json()
    if data and 'lang' in data and data['lang'] in LANGUAGES:
        session['lang'] = data['lang']
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 400

# Handle database configuration
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # Handle Render.com's PostgreSQL URL format
    if database_url.startswith("postgres://"):
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
    'pool_pre_ping': True,  # Verify database connection before using it
    'pool_recycle': 300,    # Recycle connections every 5 minutes
    'pool_timeout': 30,     # Connection timeout of 30 seconds
    'max_overflow': 15      # Allow up to 15 extra connections
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

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    name = db.Column(db.String(100))
    profile_pic = db.Column(db.String(100))
    credits = db.Column(db.Integer, default=10)
    images = db.relationship('Image', backref='user', lazy=True)
    subscription_id = db.Column(db.Integer, db.ForeignKey('subscription.id'), nullable=True)
    subscription_end_date = db.Column(db.DateTime, nullable=True)
    stripe_customer_id = db.Column(db.String(100), unique=True, nullable=True)  # Stripe Customer ID
    stripe_subscription_id = db.Column(db.String(100), unique=True, nullable=True)  # Stripe Subscription ID
    payment_method_id = db.Column(db.String(100), nullable=True)  # Stripe Payment Method ID
    subscription_status = db.Column(db.String(20), default='none')  # active, trialing, past_due, canceled, none

class Subscription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    monthly_credits = db.Column(db.Integer, default=0)
    no_ads = db.Column(db.Boolean, default=False)
    stripe_price_id = db.Column(db.String(100), unique=True)  # Stripe Price ID
    stripe_product_id = db.Column(db.String(100), unique=True)  # Stripe Product ID
    features = db.Column(db.JSON)  # Store features as JSON
    users = db.relationship('User', backref='subscription', lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.String(500), nullable=False)
    image_url = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class PaymentHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stripe_payment_intent_id = db.Column(db.String(100), unique=True)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='usd')
    status = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize Stripe configuration
STRIPE_PUBLIC_KEY = os.environ.get('STRIPE_PUBLIC_KEY', 'pk_test_51QjSomG1Y6czweQ3rB4iSwWqYRsGKxBHUKlyB2PV1LrPoqyY2qCCMoOxiJeQuQt0RLrWEYfvcDgjrgYr0E2csGdQ00ckQHLilZ')
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_51QjSomG1Y6czweQ3SDlIpDp3UHzM6fglBEYRU08kh7JpoEo8XgF3ifvWcVis4uYL3pvhBISNgtwoEt79Hs6wgXTV00IP25Flaf')

# Make Stripe keys available in templates
@app.context_processor
def inject_stripe_key():
    return dict(stripe_public_key=STRIPE_PUBLIC_KEY)

# Configure Stripe
stripe.api_key = STRIPE_SECRET_KEY
stripe.api_version = '2023-10-16'  # Use latest API version

def create_stripe_customer(user):
    """Create a Stripe customer for a user"""
    try:
        customer = stripe.Customer.create(
            email=user.email,
            name=user.name,
            metadata={'user_id': user.id}
        )
        user.stripe_customer_id = customer.id
        db.session.commit()
        return customer
    except stripe.error.StripeError as e:
        app.logger.error(f"Error creating Stripe customer: {str(e)}")
        return None

def get_or_create_stripe_customer(user):
    """Get existing Stripe customer or create new one"""
    if user.stripe_customer_id:
        try:
            return stripe.Customer.retrieve(user.stripe_customer_id)
        except stripe.error.StripeError:
            pass
    return create_stripe_customer(user)

def create_subscription_plans():
    """Create initial subscription plans if they don't exist"""
    try:
        # First verify the table exists with correct schema
        inspector = inspect(db.engine)
        
        # Check if subscription table exists
        if not inspector.has_table('subscription'):
            app.logger.info("Creating subscription table...")
            db.create_all()
        else:
            # Check if all required columns exist
            columns = {c['name'] for c in inspector.get_columns('subscription')}
            required_columns = {'stripe_price_id', 'stripe_product_id', 'features'}
            
            # If any required column is missing, drop and recreate tables
            if not required_columns.issubset(columns):
                app.logger.info("Subscription table schema outdated, recreating tables...")
                db.drop_all()
                db.create_all()
            
        # Check if any subscriptions exist
        existing_subs = Subscription.query.first()
        if existing_subs:
            app.logger.info("Subscription plans already exist")
            return
            
        app.logger.info("Creating initial subscription plans...")
        
        # Create Basic VIP Plan in Stripe
        if os.environ.get('STRIPE_SECRET_KEY'):
            try:
                basic_product = stripe.Product.create(
                    name='Basic VIP',
                    description='Basic VIP subscription with 100 monthly credits'
                )
                basic_price = stripe.Price.create(
                    product=basic_product.id,
                    unit_amount=500,  # $5.00
                    currency='usd',
                    recurring={'interval': 'month'}
                )
                
                # Create Premium VIP Plan in Stripe
                premium_product = stripe.Product.create(
                    name='Premium VIP',
                    description='Premium VIP subscription with 2000 monthly credits'
                )
                premium_price = stripe.Price.create(
                    product=premium_product.id,
                    unit_amount=1000,  # $10.00
                    currency='usd',
                    recurring={'interval': 'month'}
                )
                
                stripe_basic_product_id = basic_product.id
                stripe_basic_price_id = basic_price.id
                stripe_premium_product_id = premium_product.id
                stripe_premium_price_id = premium_price.id
            except stripe.error.StripeError as e:
                app.logger.error(f"Stripe API error: {str(e)}")
                stripe_basic_product_id = None
                stripe_basic_price_id = None
                stripe_premium_product_id = None
                stripe_premium_price_id = None
        else:
            app.logger.warning("STRIPE_SECRET_KEY not set, skipping Stripe product creation")
            stripe_basic_product_id = None
            stripe_basic_price_id = None
            stripe_premium_product_id = None
            stripe_premium_price_id = None
        
        # Create Free Plan (no Stripe product needed)
        free = Subscription(
            name='Free',
            price=0.0,
            monthly_credits=5,
            no_ads=False,
            features={
                'credits': 5,
                'ads': True,
                'support': 'Basic',
                'generation_speed': 'Normal'
            }
        )
        
        # Create Basic VIP Plan
        basic = Subscription(
            name='Basic VIP',
            price=5.0,
            monthly_credits=100,
            no_ads=True,
            stripe_product_id=stripe_basic_product_id,
            stripe_price_id=stripe_basic_price_id,
            features={
                'credits': 100,
                'ads': False,
                'support': 'Priority',
                'generation_speed': 'Fast',
                'video_ad_credits': True
            }
        )
        
        # Create Premium VIP Plan
        premium = Subscription(
            name='Premium VIP',
            price=10.0,
            monthly_credits=2000,
            no_ads=True,
            stripe_product_id=stripe_premium_product_id,
            stripe_price_id=stripe_premium_price_id,
            features={
                'credits': 2000,
                'ads': False,
                'support': '24/7 Priority',
                'generation_speed': 'Ultra-Fast',
                'storage': 'Unlimited',
                'early_access': True
            }
        )
        
        # Add all plans to session
        db.session.add(free)
        db.session.add(basic)
        db.session.add(premium)
        
        # Commit transaction
        db.session.commit()
        app.logger.info("Subscription plans created successfully")
        
    except Exception as e:
        app.logger.error(f"Error creating subscription plans: {str(e)}")
        db.session.rollback()
        # Re-raise the exception to be handled by the caller
        raise

# Create all tables first, then initialize data
with app.app_context():
    try:
        # Create all tables
        db.create_all()
        app.logger.info("Database tables created successfully")
        # Initialize subscription plans
        create_subscription_plans()
    except Exception as e:
        app.logger.error(f"Error initializing database: {str(e)}")
        raise

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
        target_lang: Target language code ('en', 'fa', 'zh', 'it', 'de', 'ja', 'fr', 'hi', 'ru')
    
    Returns:
        Translated text
    
    Raises:
        TranslationError: If translation fails
    """
    if not text:
        return text
        
    # Skip translation if text appears to be in target language (basic check)
    if target_lang == 'fa' and any('\u0600' <= c <= '\u06FF' for c in text):
        return text
    elif target_lang == 'zh' and any('\u4e00' <= c <= '\u9fff' for c in text):
        return text
    elif target_lang == 'ja' and any('\u3040' <= c <= '\u30ff' for c in text):
        return text
    elif target_lang == 'ru' and any('\u0400' <= c <= '\u04FF' for c in text):
        return text
    elif target_lang == 'hi' and any('\u0900' <= c <= '\u097F' for c in text):
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
        'en': "You are a professional English translator. Translate the following text to natural, fluent English. Only respond with the translation, no explanations.",
        'zh': "You are a professional Chinese (Simplified) translator. Translate the following text to natural, fluent Chinese. Only respond with the translation, no explanations.",
        'it': "You are a professional Italian translator. Translate the following text to natural, fluent Italian. Only respond with the translation, no explanations.",
        'de': "You are a professional German translator. Translate the following text to natural, fluent German. Only respond with the translation, no explanations.",
        'ja': "You are a professional Japanese translator. Translate the following text to natural, fluent Japanese. Only respond with the translation, no explanations.",
        'fr': "You are a professional French translator. Translate the following text to natural, fluent French. Only respond with the translation, no explanations.",
        'hi': "You are a professional Hindi translator. Translate the following text to natural, fluent Hindi. Only respond with the translation, no explanations.",
        'ru': "You are a professional Russian translator. Translate the following text to natural, fluent Russian. Only respond with the translation, no explanations.",
        'tr': "You are a professional Turkish translator. Translate the following text to natural, fluent Turkish. Only respond with the translation, no explanations.",
        'az': "You are a professional Azerbaijani translator. Translate the following text to natural, fluent Azerbaijani. Only respond with the translation, no explanations.",
        'id': "You are a professional Indonesian translator. Translate the following text to natural, fluent Indonesian. Only respond with the translation, no explanations."
    }
    
    if target_lang not in system_prompt:
        raise TranslationError(f"Unsupported target language: {target_lang}")

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
        "temperature": 0.3,
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

# Add translation functions for each language
def translate_to_chinese(text: str) -> str:
    try:
        return translate_text(text, 'zh')
    except TranslationError as e:
        app.logger.error(f"Chinese translation failed: {str(e)}")
        return text

def translate_to_italian(text: str) -> str:
    try:
        return translate_text(text, 'it')
    except TranslationError as e:
        app.logger.error(f"Italian translation failed: {str(e)}")
        return text

def translate_to_german(text: str) -> str:
    try:
        return translate_text(text, 'de')
    except TranslationError as e:
        app.logger.error(f"German translation failed: {str(e)}")
        return text

def translate_to_japanese(text: str) -> str:
    try:
        return translate_text(text, 'ja')
    except TranslationError as e:
        app.logger.error(f"Japanese translation failed: {str(e)}")
        return text

def translate_to_french(text: str) -> str:
    try:
        return translate_text(text, 'fr')
    except TranslationError as e:
        app.logger.error(f"French translation failed: {str(e)}")
        return text

def translate_to_hindi(text: str) -> str:
    try:
        return translate_text(text, 'hi')
    except TranslationError as e:
        app.logger.error(f"Hindi translation failed: {str(e)}")
        return text

def translate_to_russian(text: str) -> str:
    try:
        return translate_text(text, 'ru')
    except TranslationError as e:
        app.logger.error(f"Russian translation failed: {str(e)}")
        return text

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

def upload_to_imgbb(image_url: str) -> str:
    """Upload an image to ImgBB and return the permanent URL"""
    IMGBB_API_KEY = "5782785615360e1e2a06975cdf8f6de5"
    IMGBB_API_URL = "https://api.imgbb.com/1/upload"
    
    try:
        # Download the image from the original URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        
        # Upload to ImgBB
        files = {
            'image': ('image.jpg', image_data, 'image/jpeg')
        }
        params = {
            'key': IMGBB_API_KEY,
        }
        
        upload_response = requests.post(IMGBB_API_URL, files=files, params=params)
        upload_response.raise_for_status()
        
        result = upload_response.json()
        if result.get('success'):
            return result['data']['url']
        else:
            app.logger.error(f"ImgBB upload failed: {result}")
            return None
            
    except Exception as e:
        app.logger.error(f"Error uploading to ImgBB: {str(e)}")
        return None

def generate_image(prompt: str, model: str = 'realistic') -> tuple:
    """Generate image from prompt with improved error handling and validation"""
    if not API_KEY:
        app.logger.error("API key not configured")
        return None, "API configuration error"

    # Validate prompt length
    if len(prompt) > 500:
        return None, "Prompt too long (maximum 500 characters)"

    # Translate prompt to English if it's not in English
    try:
        # Get current language from session
        current_lang = session.get('lang', 'fa')
        if current_lang != 'en':
            app.logger.info(f"Translating prompt from {current_lang} to English")
            prompt = translate_to_english(prompt)
            app.logger.info(f"Translated prompt: {prompt}")
    except TranslationError as e:
        app.logger.error(f"Translation error: {str(e)}")
        # Continue with original prompt if translation fails
        pass

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {API_KEY}"
    }
    
    # Model-specific prompt enhancements with safety checks
    model_prompts = {
        'realistic': f"{prompt}, high quality, detailed photograph",
        'anime': f"{prompt}, anime style art",
        'painting': f"{prompt}, digital painting style",
        'pixel': f"{prompt}, pixel art style",
        'minimal': f"{prompt}, minimalist style",
        '3d': f"{prompt}, 3D render style"
    }
    
    enhanced_prompt = model_prompts.get(model, model_prompts['realistic'])
    
    data = {
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "prompt": enhanced_prompt,
        "steps": 4,
        "n": 1,
        "height": 1024,
        "width": 1024
    }

    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if 'data' in result and len(result['data']) > 0:
            temp_image_url = result['data'][0]['url']
            
            # Upload to ImgBB for permanent storage
            permanent_url = upload_to_imgbb(temp_image_url)
            if permanent_url:
                return permanent_url, None
            else:
                return None, "Error storing image"
        else:
            return None, "No image generated"
            
    except requests.exceptions.Timeout:
        return None, "Image generation timed out"
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Request failed: {str(e)}")
        return None, "Network error during image generation"
    except Exception as e:
        app.logger.exception("Unexpected error in generate_image")
        return None, f"Unexpected error: {str(e)}"

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

@app.route('/subscriptions')
@login_required
def subscriptions():
    try:
        # Get all subscription plans
        plans = Subscription.query.all()
        return render_template('subscriptions.html', plans=plans)
    except Exception as e:
        app.logger.error(f"Error loading subscription plans: {str(e)}")
        flash(_('Error loading subscription plans'), 'error')
        return redirect(url_for('dashboard'))

@app.route('/generate', methods=['POST'])
@login_required
def generate():
    """Handle image generation request with improved error handling"""
    try:
        # Input validation
        if current_user.credits <= 0:
            return jsonify({'error': _("You don't have enough credits")}), 400

        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': _("Please enter an image description")}), 400

        model = request.form.get('model', 'realistic')
        if model not in ['realistic', 'anime', 'painting', 'pixel', 'minimal', '3d']:
            model = 'realistic'

        # Generate image
        image_url, error = generate_image(prompt, model)
        
        if error:
            return jsonify({'error': f"{_('Error generating image')}: {error}"}), 500

        # Database transaction with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Start a new transaction
                db.session.begin_nested()
                
                # Create image record
                image = Image(
                    prompt=prompt[:500],  # Ensure prompt fits in database
                    image_url=image_url[:500],  # Ensure URL fits in database
                    user_id=current_user.id
                )
                db.session.add(image)
                
                # Update user credits
                current_user.credits -= 1
                
                # Commit transaction
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'image_url': image_url,
                    'image_id': image.id,
                    'credits_remaining': current_user.credits,
                    'message': _("Image generated successfully")
                })

            except SQLAlchemyError as e:
                db.session.rollback()
                if attempt == max_retries - 1:  # Last attempt
                    app.logger.error(f"Database error after {max_retries} attempts: {str(e)}")
                    return jsonify({'error': _("Error saving image. Please try again")}), 500
                continue

    except Exception as e:
        app.logger.exception("Unexpected error in generate route")
        return jsonify({'error': _("An unexpected error occurred")}), 500

@app.route('/add_credits', methods=['POST'])
@login_required
def add_credits():
    # In a real application, verify ad was watched
    current_user.credits += 3
    db.session.commit()
    return jsonify({'credits': current_user.credits})

def get_proxy_url(url: str) -> str:
    """Convert an image URL to a proxied URL that doesn't expire"""
    try:
        # If it's an S3 URL with query parameters, remove them
        if "amazonaws.com" in url and "?" in url:
            # Keep everything before the query parameters
            base_url = url.split('?')[0]
            # Get the image ID from the URL
            image_id = base_url.split('/')[-1]
            # Construct the imgproxy URL
            return f"https://api.together.ai/imgproxy/{image_id}/format:jpeg/{base_url}"
        return url
    except Exception as e:
        app.logger.error(f"Error creating proxy URL: {str(e)}")
        return url

@app.route('/images')
@login_required
def images():
    user_images = Image.query.filter_by(user_id=current_user.id).order_by(Image.created_at.desc()).all()
    # Convert image URLs to proxy URLs
    for image in user_images:
        image.image_url = get_proxy_url(image.image_url)
    return render_template('images.html', images=user_images)

@app.route('/serve-image/<int:image_id>')
@login_required
def serve_image(image_id):
    """Serve an image from the database"""
    image = Image.query.get_or_404(image_id)
    
    # Verify the image belongs to the current user
    if image.user_id != current_user.id:
        abort(403)
    
    try:
        # Get the proxy URL
        proxy_url = get_proxy_url(image.image_url)
        
        # Download the image from the proxy URL with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(proxy_url, timeout=10)
                response.raise_for_status()
                break
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:  # Last attempt
                    app.logger.error(f"Error serving image {image_id}: {str(e)}")
                    abort(500)
                continue
        
        # Serve the image with caching headers
        return Response(
            response.content,
            mimetype='image/jpeg',
            headers={
                'Cache-Control': 'public, max-age=31536000',  # Cache for 1 year
                'Content-Type': 'image/jpeg',
                'X-Content-Type-Options': 'nosniff'  # Prevent MIME type sniffing
            }
        )
    except Exception as e:
        app.logger.error(f"Error serving image {image_id}: {str(e)}")
        abort(500)

@app.route('/download-image/<int:image_id>')
@login_required
def download_image(image_id):
    """Download an image from the database"""
    image = Image.query.get_or_404(image_id)
    
    # Verify the image belongs to the current user
    if image.user_id != current_user.id:
        abort(403)
    
    try:
        # Get the proxy URL
        proxy_url = get_proxy_url(image.image_url)
        
        # Download the image from the proxy URL with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(proxy_url, timeout=10)
                response.raise_for_status()
                break
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:  # Last attempt
                    app.logger.error(f"Error downloading image {image_id}: {str(e)}")
                    abort(500)
                continue
        
        # Create a BytesIO object from the image data
        image_data = BytesIO(response.content)
        
        # Generate a filename based on the prompt
        safe_prompt = "".join(x for x in image.prompt[:30] if x.isalnum() or x in (' ', '-', '_')).strip()
        filename = f"image-{safe_prompt}-{image.id}.jpg"
        
        return send_file(
            image_data,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=filename,
            max_age=300  # Cache for 5 minutes
        )
    except Exception as e:
        app.logger.error(f"Error downloading image {image_id}: {str(e)}")
        abort(500)

@app.route('/subscribe/<int:plan_id>', methods=['POST'])
@login_required
def subscribe(plan_id):
    # In a real application, this would integrate with a payment processor
    # For now, we'll just update the subscription
    subscription = Subscription.query.get_or_404(plan_id)
    current_user.subscription = subscription
    current_user.subscription_end_date = datetime.utcnow().replace(day=1) + timedelta(days=32)  # Set to end of next month
    if subscription.monthly_credits > 0:
        current_user.credits += subscription.monthly_credits
    db.session.commit()
    flash(_('Successfully subscribed to {} plan').format(subscription.name), 'success')
    return redirect(url_for('dashboard'))

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    """Create a Stripe Checkout Session for subscription"""
    try:
        plan_id = request.form.get('plan_id')
        if not plan_id:
            return jsonify({'error': _('Please select a plan')}), 400
            
        subscription = Subscription.query.get_or_404(plan_id)
        if not subscription.stripe_price_id:
            app.logger.error(f"No Stripe price ID for plan {plan_id}")
            return jsonify({'error': _('Invalid subscription plan')}), 400
            
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user)
        if not customer:
            app.logger.error(f"Failed to create/get Stripe customer for user {current_user.id}")
            return jsonify({'error': _('Error creating customer')}), 500
            
        # Create checkout session
        try:
            checkout_session = stripe.checkout.Session.create(
                customer=customer.id,
                payment_method_types=['card'],
                line_items=[{
                    'price': subscription.stripe_price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=urljoin(get_base_url(), '/subscription-success?session_id={CHECKOUT_SESSION_ID}'),
                cancel_url=urljoin(get_base_url(), '/subscriptions'),
                metadata={
                    'user_id': current_user.id,
                    'plan_id': plan_id
                }
            )
            
            return jsonify({'sessionId': checkout_session.id})
            
        except stripe.error.StripeError as e:
            app.logger.error(f"Stripe error creating checkout session: {str(e)}")
            return jsonify({'error': _('Payment processing error')}), 500
            
    except Exception as e:
        app.logger.error(f"Error creating checkout session: {str(e)}")
        return jsonify({'error': _('An unexpected error occurred')}), 500

@app.route('/subscription-success')
@login_required
def subscription_success():
    """Handle successful subscription checkout"""
    session_id = request.args.get('session_id')
    if not session_id:
        flash(_('Invalid session'), 'error')
        return redirect(url_for('subscriptions'))
        
    try:
        # Retrieve the session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        # Verify the user
        if str(checkout_session.metadata.get('user_id')) != str(current_user.id):
            app.logger.error(f"User ID mismatch in checkout session")
            flash(_('Invalid session'), 'error')
            return redirect(url_for('subscriptions'))
        
        # Update user's subscription
        subscription = Subscription.query.get(checkout_session.metadata['plan_id'])
        if subscription:
            try:
                current_user.subscription = subscription
                current_user.stripe_subscription_id = checkout_session.subscription
                current_user.subscription_status = 'active'
                current_user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
                current_user.credits += subscription.monthly_credits
                db.session.commit()
                
                flash(_('Successfully subscribed to {} plan').format(subscription.name), 'success')
            except SQLAlchemyError as e:
                app.logger.error(f"Database error updating subscription: {str(e)}")
                db.session.rollback()
                flash(_('Error activating subscription'), 'error')
        else:
            app.logger.error(f"Subscription plan not found: {checkout_session.metadata['plan_id']}")
            flash(_('Error activating subscription'), 'error')
            
    except stripe.error.StripeError as e:
        app.logger.error(f"Stripe error processing successful subscription: {str(e)}")
        flash(_('Error processing subscription'), 'error')
    except Exception as e:
        app.logger.error(f"Error processing successful subscription: {str(e)}")
        flash(_('Error processing subscription'), 'error')
        
    return redirect(url_for('dashboard'))

@app.route('/cancel-subscription', methods=['POST'])
@login_required
def cancel_subscription():
    """Cancel user's subscription"""
    try:
        if not current_user.stripe_subscription_id:
            return jsonify({'error': _('No active subscription')}), 400
            
        try:
            # Cancel the subscription at period end
            stripe.Subscription.modify(
                current_user.stripe_subscription_id,
                cancel_at_period_end=True
            )
            
            current_user.subscription_status = 'canceled'
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': _('Subscription will be canceled at the end of the billing period')
            })
        except stripe.error.StripeError as e:
            app.logger.error(f"Stripe error canceling subscription: {str(e)}")
            return jsonify({'error': _('Error canceling subscription')}), 500
            
    except Exception as e:
        app.logger.error(f"Error canceling subscription: {str(e)}")
        return jsonify({'error': _('An unexpected error occurred')}), 500

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.environ.get('STRIPE_WEBHOOK_SECRET')
        )
    except ValueError as e:
        app.logger.error(f"Invalid payload: {str(e)}")
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e:
        app.logger.error(f"Invalid signature: {str(e)}")
        return 'Invalid signature', 400
        
    try:
        if event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            user = User.query.filter_by(stripe_customer_id=subscription['customer']).first()
            if user:
                user.subscription_status = subscription['status']
                if subscription['status'] == 'active':
                    # Update subscription end date
                    user.subscription_end_date = datetime.fromtimestamp(subscription['current_period_end'])
                db.session.commit()
                
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            user = User.query.filter_by(stripe_customer_id=subscription['customer']).first()
            if user:
                user.subscription_id = None
                user.subscription_status = 'none'
                user.stripe_subscription_id = None
                db.session.commit()
                
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            user = User.query.filter_by(stripe_customer_id=invoice['customer']).first()
            if user:
                user.subscription_status = 'past_due'
                db.session.commit()
                # TODO: Send email notification to user
                
        return jsonify({'status': 'success'})
        
    except Exception as e:
        app.logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            create_subscription_plans()  # Initialize subscription plans
            app.logger.info("Database tables and subscription plans created successfully")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {e}")
            sys.exit(1)
    
    debug_mode = os.environ.get('FLASK_DEBUG') == '1'
    if debug_mode:
        app.run(debug=True, port=5000)
    else:
        app.run(ssl_context='adhoc', port=5000)  # Enable HTTPS for OAuth 