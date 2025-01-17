import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from oauthlib.oauth2 import WebApplicationClient
import requests
from datetime import datetime
import json
from googletrans import Translator
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "your-secret-key")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", None)
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", None)
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# OAuth 2 client setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# API Configuration
API_KEY = os.environ.get('FLUX_API_KEY', "381ba66d510aa8ae2c328a5c20b1a7a741e7727bb248b8af2b7be24cd6b767f1")
API_URL = "https://api.together.xyz/v1/images/generations"

# Initialize translator
translator = Translator()

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

# Test login route
@app.route('/test-login')
def test_login():
    # Create or get test user
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

def translate_to_english(text):
    try:
        translation = translator.translate(text, src='fa', dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def generate_image(prompt):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {API_KEY}"
    }
    
    # Translate prompt to English
    english_prompt = translate_to_english(prompt)
    print(f"Translated prompt: {english_prompt}")
    
    # Enhanced prompt with better guidance
    enhanced_prompt = f"professional high-quality detailed {english_prompt}, cinematic lighting, 8k uhd, highly detailed"
    
    data = {
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "prompt": enhanced_prompt,
        "steps": 4,  # Maximum allowed steps for this model
        "n": 1,
        "height": 1024,
        "width": 1024
    }
    
    try:
        print(f"Sending request to API with prompt: {enhanced_prompt}")
        response = requests.post(API_URL, json=data, headers=headers)
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Content: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        
        if 'data' in result and len(result['data']) > 0 and 'url' in result['data'][0]:
            return result['data'][0]['url']
        else:
            print(f"Unexpected API response format: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
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
        return jsonify({'error': 'اعتبار شما کافی نیست'}), 400
    
    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'لطفاً توضیحات تصویر را وارد کنید'}), 400
    
    image_url = generate_image(prompt)
    if not image_url:
        return jsonify({'error': 'خطا در تولید تصویر. لطفاً دوباره تلاش کنید'}), 500
    
    # Create new image record
    image = Image(prompt=prompt, image_url=image_url, user_id=current_user.id)
    current_user.credits -= 1
    
    db.session.add(image)
    db.session.commit()
    
    return jsonify({
        'image_url': image_url,
        'credits_remaining': current_user.credits
    })

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
    app.run(debug=True, ssl_context="adhoc")  # Enable HTTPS for OAuth 