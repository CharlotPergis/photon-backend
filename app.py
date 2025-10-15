# ============================================================
# app.py — Photon Backend (OCR + AI + Google Login + Mongo)
# ============================================================
# Features:
# - Google Vision OCR with REAL image quality analysis (PIL-based)
# - Image preprocessing for better OCR results
# - GPT indentation and Python tutor
# - MongoDB for users/sessions
# - Google OAuth2 login + JWT session
# - Chat history management
# ============================================================

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token
)
from pymongo import MongoClient
from datetime import datetime, timezone
import os, io, numpy as np, requests, re, base64
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from google.cloud import vision
from openai import OpenAI
from bson import ObjectId
import os

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/auth/callback")
FRONTEND_URL = os.getenv("CORS_ORIGIN", "http://localhost:3000")

# ============================================================
# INITIALIZATION
# ============================================================

app = Flask(__name__)
CORS(app, origins=["*"])
bcrypt = Bcrypt(app)
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "photon-secret-key")
jwt_manager = JWTManager(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# IMAGE ANALYSIS SETUP (Using PIL instead of OpenCV for Python 3.13 compatibility)
# ============================================================

print("🔧 Setting up image quality analysis...")
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageStat, ImageFilter, ImageChops, ImageEnhance, ImageOps
    import io
    PIL_AVAILABLE = True
    print("✅ PIL (Pillow) initialized for image quality analysis and preprocessing")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")
    print("⚠️ Install Pillow: pip install Pillow")

# ============================================================
# ENHANCED DATABASE SETUP (MongoDB)
# ============================================================

print("🔧 Setting up MongoDB database...")
try:
    # MongoDB connection
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/photon")
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client.get_database()
    
    # Collections
    users_col = db.users
    chats_col = db.chats
    messages_col = db.messages
    
    # Create indexes for better performance
    chats_col.create_index([("user_email", 1), ("updatedAt", -1)])
    messages_col.create_index([("chat_id", 1), ("timestamp", 1)])
    users_col.create_index([("email", 1)], unique=True)
    
    print("✅ MongoDB initialized successfully")
    
    # Test the connection
    db.command('ping')
    print("✅ MongoDB connection test passed")
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    print("⚠️ Falling back to in-memory storage")
    
    # Fallback to in-memory storage
    class MemoryStorage:
        def __init__(self):
            self.data = {'chats': [], 'users': [], 'messages': []}
            self.chat_counter = 0
            self.message_counter = 0
        
        def __getitem__(self, name):
            return self
        
        def insert_one(self, document):
            if 'chat' in str(document).lower():
                self.chat_counter += 1
                doc_id = f"mem-{self.chat_counter}"
                document['_id'] = doc_id
                self.data['chats'].append(document)
                return type('obj', (object,), {'inserted_id': doc_id})()
            elif 'message' in str(document).lower():
                self.message_counter += 1
                doc_id = f"msg-{self.message_counter}"
                document['_id'] = doc_id
                self.data['messages'].append(document)
                return type('obj', (object,), {'inserted_id': doc_id})()
            else:
                doc_id = f"usr-{len(self.data['users']) + 1}"
                document['_id'] = doc_id
                self.data['users'].append(document)
                return type('obj', (object,), {'inserted_id': doc_id})()
        
        def find_one(self, query=None):
            if query and 'email' in query:
                for user in self.data['users']:
                    if user.get('email') == query['email']:
                        return user
            elif query and '_id' in query:
                # Search in all collections
                for chat in self.data['chats']:
                    if chat.get('_id') == query['_id']:
                        return chat
                for message in self.data['messages']:
                    if message.get('_id') == query['_id']:
                        return message
            return None
        
        def find(self, query=None):
            if query and 'user_email' in query:
                return [chat for chat in self.data['chats'] if chat.get('user_email') == query['user_email']]
            elif query and 'chat_id' in query:
                return [msg for msg in self.data['messages'] if msg.get('chat_id') == query['chat_id']]
            return []
        
        def update_one(self, query, update):
            if query and '_id' in query:
                # Find and update document
                for chat in self.data['chats']:
                    if chat.get('_id') == query['_id']:
                        if '$set' in update:
                            chat.update(update['$set'])
                        return type('obj', (object,), {'modified_count': 1})()
            return type('obj', (object,), {'modified_count': 0})()
        
        def delete_one(self, query):
            if query and '_id' in query:
                # Remove from chats
                self.data['chats'] = [chat for chat in self.data['chats'] if chat.get('_id') != query['_id']]
                # Remove associated messages
                self.data['messages'] = [msg for msg in self.data['messages'] if msg.get('chat_id') != query['_id']]
            return type('obj', (object,), {'deleted_count': 1})()
        
        def delete_many(self, query):
            if query and 'chat_id' in query:
                initial_count = len(self.data['messages'])
                self.data['messages'] = [msg for msg in self.data['messages'] if msg.get('chat_id') != query['chat_id']]
                deleted_count = initial_count - len(self.data['messages'])
                return type('obj', (object,), {'deleted_count': deleted_count})()
            return type('obj', (object,), {'deleted_count': 0})()
        
        def count_documents(self, query):
            if query and 'chat_id' in query:
                return len([msg for msg in self.data['messages'] if msg.get('chat_id') == query['chat_id']])
            return 0
    
    db = MemoryStorage()
    users_col = db
    chats_col = db
    messages_col = db

# ============================================================
# GOOGLE VISION CLIENT
# ============================================================

try:
    vision_client = vision.ImageAnnotatorClient()
    print("✅ Google Vision initialized")
except Exception as e:
    print("❌ Failed to initialize Google Vision:", e)

# ============================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_image(image_bytes, preprocessing_steps=None):
    """
    Enhance image quality for better OCR results
    Returns: processed_image_bytes, preprocessing_log
    """
    if not PIL_AVAILABLE:
        return image_bytes, ["Preprocessing unavailable - PIL not installed"]
    
    preprocessing_log = []
    
    try:
        # Open original image
        original_img = Image.open(io.BytesIO(image_bytes))
        img = original_img.copy()
        
        # Default preprocessing steps
        if preprocessing_steps is None:
            preprocessing_steps = {
                'auto_rotate': True,
                'enhance_contrast': True,
                'enhance_sharpness': True,
                'remove_noise': True,
                'binarize': False  # Can help but might remove syntax highlighting
            }
        
        # Step 1: Auto-rotate if needed (detect orientation)
        if preprocessing_steps.get('auto_rotate'):
            try:
                # Simple orientation detection based on aspect ratio
                width, height = img.size
                if height > width * 1.5:  # Portrait image for landscape text
                    img = img.rotate(90, expand=True)
                    preprocessing_log.append("🔄 Auto-rotated to landscape orientation")
            except Exception as e:
                preprocessing_log.append(f"⚠️ Auto-rotate failed: {str(e)}")
        
        # Step 2: Convert to grayscale for better OCR
        if img.mode != 'L':
            img = img.convert('L')
            preprocessing_log.append("⚫ Converted to grayscale")
        
        # Step 3: Enhance contrast
        if preprocessing_steps.get('enhance_contrast'):
            # Analyze current contrast
            stats = ImageStat.Stat(img)
            current_contrast = stats.stddev[0]
            
            if current_contrast < 40:  # Low contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)  # Double the contrast
                preprocessing_log.append("🎨 Enhanced contrast (2.0x)")
            elif current_contrast < 60:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)  # Moderate contrast boost
                preprocessing_log.append("🎨 Enhanced contrast (1.5x)")
        
        # Step 4: Enhance sharpness
        if preprocessing_steps.get('enhance_sharpness'):
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)  # Slight sharpening
            preprocessing_log.append("🔍 Enhanced sharpness")
        
        # Step 5: Remove noise (light denoising)
        if preprocessing_steps.get('remove_noise'):
            img = img.filter(ImageFilter.MedianFilter(size=3))
            preprocessing_log.append("🧹 Applied noise reduction")
        
        # Step 6: Optional binarization (black & white)
        if preprocessing_steps.get('binarize'):
            # Adaptive thresholding
            img = img.point(lambda x: 0 if x < 128 else 255, '1')
            preprocessing_log.append("⚪ Applied binarization")
        
        # Step 7: Auto-crop borders if they're mostly empty
        try:
            # Convert to numpy array for border detection
            np_img = np.array(img)
            # Simple border crop (remove completely white borders)
            non_white_rows = np.where(np_img < 250)[0]
            non_white_cols = np.where(np_img < 250)[1]
            
            if len(non_white_rows) > 0 and len(non_white_cols) > 0:
                crop_box = (
                    max(0, non_white_cols[0] - 10),
                    max(0, non_white_rows[0] - 10),
                    min(img.width, non_white_cols[-1] + 10),
                    min(img.height, non_white_rows[-1] + 10)
                )
                
                # Only crop if we're removing significant empty space
                original_area = img.width * img.height
                cropped_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
                
                if cropped_area < original_area * 0.8:  # Only if removing >20% empty space
                    img = img.crop(crop_box)
                    preprocessing_log.append("✂️ Auto-cropped empty borders")
        except Exception as e:
            preprocessing_log.append(f"⚠️ Auto-crop failed: {str(e)}")
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='PNG', quality=95)
        processed_bytes = output_buffer.getvalue()
        
        preprocessing_log.append(f"✅ Preprocessing complete - {len(processed_bytes)} bytes")
        
        return processed_bytes, preprocessing_log
        
    except Exception as e:
        preprocessing_log.append(f"❌ Preprocessing error: {str(e)}")
        return image_bytes, preprocessing_log

def compare_image_quality(original_bytes, processed_bytes):
    """Compare quality metrics before and after preprocessing"""
    if not PIL_AVAILABLE:
        return {"improvement": "Cannot compare - PIL unavailable"}
    
    try:
        original_img = Image.open(io.BytesIO(original_bytes)).convert('L')
        processed_img = Image.open(io.BytesIO(processed_bytes)).convert('L')
        
        original_stats = ImageStat.Stat(original_img)
        processed_stats = ImageStat.Stat(processed_img)
        
        # Calculate metrics
        original_contrast = original_stats.stddev[0]
        processed_contrast = processed_stats.stddev[0]
        
        # Edge detection for sharpness comparison
        original_edges = original_img.filter(ImageFilter.FIND_EDGES)
        processed_edges = processed_img.filter(ImageFilter.FIND_EDGES)
        
        original_sharpness = ImageStat.Stat(original_edges).mean[0]
        processed_sharpness = ImageStat.Stat(processed_edges).mean[0]
        
        contrast_improvement = ((processed_contrast - original_contrast) / original_contrast * 100) if original_contrast > 0 else 0
        sharpness_improvement = ((processed_sharpness - original_sharpness) / original_sharpness * 100) if original_sharpness > 0 else 0
        
        return {
            "contrast_improvement": f"{contrast_improvement:+.1f}%",
            "sharpness_improvement": f"{sharpness_improvement:+.1f}%",
            "original_contrast": int(original_contrast),
            "processed_contrast": int(processed_contrast),
            "original_sharpness": int(original_sharpness * 10),
            "processed_sharpness": int(processed_sharpness * 10)
        }
        
    except Exception as e:
        return {"error": f"Comparison failed: {str(e)}"}

# ============================================================
# ENHANCED IMAGE QUALITY ANALYSIS FUNCTIONS (PIL-based)
# ============================================================

def analyze_image_quality(image_bytes, filename=""):
    """Enhanced image quality analysis with specific, actionable feedback"""
    quality_metrics = {
        "quality_issues": [],
        "suggestions": [],
        "analysis_available": False,
        "analysis_engine": "Enhanced PIL"
    }
    
    if not PIL_AVAILABLE:
        quality_metrics["quality_issues"].append("Image analysis not available")
        quality_metrics["suggestions"].append("Install Pillow for image quality feedback")
        return quality_metrics
    
    try:
        # Open and analyze image
        img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        width, height = img.size
        stats = ImageStat.Stat(img)
        
        quality_metrics["analysis_available"] = True
        quality_metrics["resolution"] = f"{width}x{height}"
        quality_metrics["file_size_kb"] = len(image_bytes) / 1024
        
        # Get image characteristics for specific feedback
        brightness = stats.mean[0]
        contrast = stats.stddev[0]
        quality_metrics["brightness_score"] = int(brightness)
        quality_metrics["contrast_score"] = int(contrast)
        
        # Enhanced sharpness analysis
        edges = img.filter(ImageFilter.FIND_EDGES)
        edge_strength = ImageStat.Stat(edges).mean[0]
        sharpness_score = int(edge_strength * 12)  # Better scaling
        quality_metrics["sharpness_score"] = min(100, sharpness_score)
        
        # Resolution analysis
        total_pixels = width * height
        
        # ============================================================
        # SPECIFIC, ACTIONABLE FEEDBACK BASED ON ACTUAL IMAGE CONDITIONS
        # ============================================================
        
        # BRIGHTNESS ANALYSIS - Give specific lighting advice
        if brightness < 30:
            quality_metrics["quality_issues"].append("Image is very dark - text is hard to read")
            quality_metrics["suggestions"].append("💡 Use direct lighting or flash - current lighting is insufficient")
        elif brightness < 60:
            quality_metrics["quality_issues"].append("Image is somewhat dark")
            quality_metrics["suggestions"].append("💡 Add more light source from the side to reduce shadows")
        elif brightness < 100:
            quality_metrics["suggestions"].append("🌅 Lighting is adequate but could be brighter")
        elif brightness > 220:
            quality_metrics["quality_issues"].append("Image is overexposed with glare")
            quality_metrics["suggestions"].append("🕶️ Reduce direct light or move away from bright sources")
        elif 120 <= brightness <= 180:
            quality_metrics["suggestions"].append("✅ Perfect lighting level for text recognition")
        
        # CONTRAST ANALYSIS - Specific contrast advice
        if contrast < 20:
            quality_metrics["quality_issues"].append("Very low contrast - text blends into background")
            quality_metrics["suggestions"].append("⚫ Use black text on white background for maximum contrast")
        elif contrast < 35:
            quality_metrics["quality_issues"].append("Low contrast affects readability")
            quality_metrics["suggestions"].append("📝 Ensure text color strongly contrasts with background")
        elif contrast > 65:
            quality_metrics["suggestions"].append("✅ Excellent contrast - text stands out clearly")
        
        # SHARPNESS ANALYSIS - Specific focus advice
        if sharpness_score < 25:
            quality_metrics["quality_issues"].append("Very blurry - text edges are unclear")
            quality_metrics["suggestions"].append("📷 Hold camera steady or use tripod - current image is too blurry")
        elif sharpness_score < 45:
            quality_metrics["quality_issues"].append("Slightly blurry")
            quality_metrics["suggestions"].append("🎯 Tap to focus on the text before capturing")
        elif sharpness_score > 75:
            quality_metrics["suggestions"].append("✅ Perfect focus - text edges are sharp and clear")
        
        # RESOLUTION ANALYSIS - Specific resolution advice
        if total_pixels < 80000:  # Less than 80K pixels
            quality_metrics["quality_issues"].append("Very low resolution")
            quality_metrics["suggestions"].append("🔍 Move camera closer or use higher resolution - text is too small")
        elif total_pixels < 200000:  # Less than 200K pixels
            quality_metrics["quality_issues"].append("Low resolution for detailed text")
            quality_metrics["suggestions"].append("📐 Capture closer to fill frame with code")
        elif total_pixels > 800000:  # More than 800K pixels
            quality_metrics["suggestions"].append("✅ Good resolution for detailed code recognition")
        
        # ASPECT RATIO & COMPOSITION
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 2.5:
            quality_metrics["quality_issues"].append("Image appears skewed or angled")
            quality_metrics["suggestions"].append("📐 Hold camera directly above text, parallel to surface")
        
        # TEXT DENSITY ANALYSIS
        edge_density = edge_strength * 100
        if edge_density < 3:
            quality_metrics["quality_issues"].append("Very little text detected")
            quality_metrics["suggestions"].append("📄 Ensure image contains visible code or text")
        elif edge_density > 8:
            quality_metrics["suggestions"].append("✅ Good text density detected")
        
        # FILE SIZE & COMPRESSION
        file_size_kb = len(image_bytes) / 1024
        if file_size_kb < 10 and total_pixels > 100000:
            quality_metrics["quality_issues"].append("Heavy compression may reduce quality")
            quality_metrics["suggestions"].append("🖼️ Use higher quality settings if available")
        
        # ============================================================
        # OVERALL ASSESSMENT WITH PRIORITIZED RECOMMENDATIONS
        # ============================================================
        
        # Calculate overall score
        overall_score = min(100, int(
            (min(contrast, 80) * 0.4) +  # Contrast importance
            (min(brightness / 2.55, 40) if 50 <= brightness <= 200 else 
             max(0, 40 - abs(brightness - 125) * 0.3)) +  # Brightness importance
            (min(sharpness_score, 20))  # Sharpness importance
        ))
        
        quality_metrics["overall_quality_score"] = overall_score
        
        # Overall quality summary
        if overall_score >= 85:
            quality_metrics["quality_summary"] = "Excellent quality - should produce accurate OCR results"
        elif overall_score >= 70:
            quality_metrics["quality_summary"] = "Good quality - minor adjustments could improve accuracy"
        elif overall_score >= 50:
            quality_metrics["quality_summary"] = "Fair quality - consider the suggestions below"
        else:
            quality_metrics["quality_summary"] = "Poor quality - significant improvements needed"
        
        # ============================================================
        # PRIORITIZED QUICK FIXES
        # ============================================================
        
        # Add quick fixes based on the most critical issues
        quick_fixes = []
        
        if brightness < 60:
            quick_fixes.append("Add more light immediately")
        if sharpness_score < 30:
            quick_fixes.append("Hold camera steadier or use support")
        if contrast < 25:
            quick_fixes.append("Improve text-background contrast")
        if total_pixels < 100000:
            quick_fixes.append("Move camera closer to text")
        
        if quick_fixes:
            quality_metrics["quick_fixes"] = quick_fixes
        
        return quality_metrics
        
    except Exception as e:
        print(f"Enhanced image analysis error: {e}")
        quality_metrics["quality_issues"].append("Analysis temporarily unavailable")
        quality_metrics["suggestions"].append("Check image manually for clarity and contrast")
        return quality_metrics

def analyze_ocr_quality(text, confidence, image_quality_metrics, filename=""):
    """Enhanced OCR quality analysis using actual image metrics"""
    feedback = {
        "confidence_score": confidence,
        "quality_issues": [],
        "suggestions": [],
        "estimated_readability": "good",
        "character_count": len(text) if text else 0,
        "line_count": len(text.split('\n')) if text else 0,
        "image_quality": image_quality_metrics
    }
    
    # Check for common OCR issues
    if not text or len(text.strip()) == 0:
        feedback["quality_issues"].append("No text detected in image")
        feedback["estimated_readability"] = "poor"
        
        # Use image quality to provide specific suggestions
        if image_quality_metrics.get("brightness_score", 0) < 50:
            feedback["suggestions"].append("Image is too dark - text cannot be read")
        elif image_quality_metrics.get("contrast_score", 0) < 20:
            feedback["suggestions"].append("Contrast is too low - text blends with background")
        elif image_quality_metrics.get("sharpness_score", 0) < 30:
            feedback["suggestions"].append("Image is too blurry - text is not clear")
        else:
            feedback["suggestions"].append("Try a clearer image with visible text")
        
        return feedback
    
    # Analyze text characteristics
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
    
    # Check for common problems - now based on actual image quality
    if confidence < 0.5:
        feedback["quality_issues"].append("Low confidence in text recognition")
        
        # Provide specific reasons based on image quality
        if image_quality_metrics.get("sharpness_score", 0) < 50:
            feedback["suggestions"].append(f"Image blurriness (score: {image_quality_metrics.get('sharpness_score', 0)}/100) is affecting accuracy")
        if image_quality_metrics.get("contrast_score", 0) < 40:
            feedback["suggestions"].append(f"Low contrast (score: {image_quality_metrics.get('contrast_score', 0)}/100) makes text hard to read")
    
    # Only show character warnings if image quality suggests they're likely
    if (image_quality_metrics.get("sharpness_score", 0) < 50 or 
        image_quality_metrics.get("contrast_score", 0) < 40 or
        image_quality_metrics.get("overall_quality_score", 0) < 60):
        
        common_OCR_errors = {
            'O': '0', 'l': '1', 'I': '1', 'Z': '2', 'S': '5', 
            'B': '8', 'rn': 'm', 'cl': 'd', 'vv': 'w'
        }
        
        detected_errors = []
        for error, correction in common_OCR_errors.items():
            if error in text:
                detected_errors.append(f"'{error}' might be misread as '{correction}'")
        
        if detected_errors:
            feedback["quality_issues"].extend(detected_errors[:3])
            feedback["suggestions"].append("Review characters that are commonly misread - image quality may cause errors")
    
    # Check line length issues
    if avg_line_length > 150:
        feedback["quality_issues"].append("Very long lines detected")
        feedback["suggestions"].append("Try capturing code in smaller chunks")
    
    # Estimate readability based on confidence AND image quality
    readability_score = confidence
    
    # Adjust based on image quality
    image_quality_factor = image_quality_metrics.get("overall_quality_score", 50) / 100
    readability_score *= (0.3 + 0.7 * image_quality_factor)  # Image quality affects 70% of score
    
    # Penalize for many issues
    if len(feedback["quality_issues"]) > 2:
        readability_score *= 0.7
    
    # Determine readability level
    if readability_score > 0.8:
        feedback["estimated_readability"] = "excellent"
    elif readability_score > 0.6:
        feedback["estimated_readability"] = "good"
    elif readability_score > 0.4:
        feedback["estimated_readability"] = "fair"
    else:
        feedback["estimated_readability"] = "poor"
    
    # Add positive feedback for good results
    if (confidence > 0.8 and 
        image_quality_metrics.get("overall_quality_score", 0) > 70 and 
        len(feedback["quality_issues"]) == 0):
        feedback["suggestions"].append("Excellent quality! Text should be accurately recognized")
    
    return feedback

def gpt_indent(raw_code: str) -> str:
    """Fix indentation using GPT."""
    if not client:
        return raw_code
    try:
        msg = (
            "Fix only the indentation of this Python code. Do not change any logic.\n\n"
            f"```python\n{raw_code}\n```"
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python formatter."},
                {"role": "user", "content": msg},
            ],
            temperature=0,
            max_tokens=800,
        )
        text = resp.choices[0].message.content or ""
        return text.replace("```python", "").replace("```", "").strip()
    except Exception as e:
        print("GPT indent error:", e)
        return raw_code

def vision_document_text(image_bytes: bytes):
    """Run OCR using Google Vision API."""
    try:
        img = vision.Image(content=image_bytes)
        res = vision_client.document_text_detection(image=img)
        if res.error and res.error.message:
            return "", 0.0
        text, confs = "", []
        if res.full_text_annotation and res.full_text_annotation.text:
            text = res.full_text_annotation.text
            for p in res.full_text_annotation.pages:
                for b in p.blocks:
                    for para in b.paragraphs:
                        for w in para.words:
                            confs.append(w.confidence or 0.0)
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        return text.strip(), float(avg_conf)
    except Exception as e:
        print("Vision error:", e)
        return "", 0.0
    


# ============================================================
# GOOGLE LOGIN (OAUTH2)
# ============================================================

@app.route("/auth/google")
def google_login():
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=code"
        f"&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        "&scope=openid%20email%20profile"
        "&access_type=offline"
    )
    return redirect(auth_url)

@app.route("/auth/callback")
def google_callback():
    """Google OAuth callback - updated to handle password users"""
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Missing code"}), 400

    try:
        token_data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        token_resp = requests.post("https://oauth2.googleapis.com/token", data=token_data)
        tokens = token_resp.json()

        if "id_token" not in tokens:
            return jsonify({"error": "Invalid Google token"}), 400

        idinfo = id_token.verify_oauth2_token(tokens["id_token"], grequests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        name = idinfo.get("name", "Photon User")
        picture = idinfo.get("picture", "")

        # Find or create user
        user = users_col.find_one({"email": email})
        if not user:
            user_data = {
                "email": email,
                "name": name,
                "picture": picture,
                "login_method": "google",
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc),
            }
            result = users_col.insert_one(user_data)
            user_id = str(result.inserted_id)
        else:
            user_id = str(user["_id"])
            # Update user info if needed
            users_col.update_one(
                {"email": email},
                {"$set": {
                    "name": name,
                    "picture": picture,
                    "updatedAt": datetime.now(timezone.utc)
                }}
            )

        # Create JWT token
        access_token = create_access_token(
            identity=email,
            additional_claims={"user_id": user_id, "name": name}
        )
        
        # Redirect to frontend with user info
        return redirect(f"{FRONTEND_URL}/workspace?token={access_token}&email={email}&name={name}&picture={picture}")

    except Exception as e:
        print(f"Google callback error: {e}")
        return jsonify({"error": "Authentication failed"}), 500
    
# ============================================================
# EMAIL/PASSWORD AUTHENTICATION ROUTES
# ============================================================

@app.route("/auth/register", methods=["POST"])
def register():
    """Register a new user with email and password"""
    try:
        data = request.json
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        name = data.get("name", "").strip()

        # Validation
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return jsonify({"error": "Invalid email format"}), 400

        # Check if user already exists
        existing_user = users_col.find_one({"email": email})
        if existing_user:
            return jsonify({"error": "User already exists"}), 409

        # Hash password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Create user
        user_data = {
            "email": email,
            "name": name or email.split('@')[0],
            "password_hash": hashed_password,
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc),
            "login_method": "email"  # To distinguish from Google login
        }

        result = users_col.insert_one(user_data)
        user_id = str(result.inserted_id)

        # Create JWT token
        access_token = create_access_token(
            identity=email,
            additional_claims={"user_id": user_id, "name": user_data["name"]}
        )

        return jsonify({
            "message": "User registered successfully",
            "user": {
                "id": user_id,
                "email": email,
                "name": user_data["name"]
            },
            "access_token": access_token
        }), 201

    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"error": "Registration failed"}), 500

@app.route("/auth/login", methods=["POST"])
def login():
    """Login with email and password"""
    try:
        data = request.json
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Find user
        user = users_col.find_one({"email": email})
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        # Check password
        if user.get("login_method") == "email":
            if not bcrypt.check_password_hash(user.get("password_hash", ""), password):
                return jsonify({"error": "Invalid email or password"}), 401
        else:
            # User registered with Google, can't use password login
            return jsonify({"error": "Please use Google login for this account"}), 401

        # Create JWT token
        access_token = create_access_token(
            identity=email,
            additional_claims={
                "user_id": str(user["_id"]), 
                "name": user.get("name", "User")
            }
        )

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": str(user["_id"]),
                "email": user["email"],
                "name": user.get("name", "User"),
                "picture": user.get("picture", "")
            },
            "access_token": access_token
        })

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": "Login failed"}), 500

@app.route("/auth/me", methods=["GET"])
@jwt_required()
def get_current_user():
    """Get current user info"""
    try:
        current_user_email = get_jwt_identity()
        user = users_col.find_one({"email": current_user_email})
        
        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "user": {
                "id": str(user["_id"]),
                "email": user["email"],
                "name": user.get("name", "User"),
                "picture": user.get("picture", ""),
                "login_method": user.get("login_method", "email"),
                "createdAt": user.get("createdAt", "").isoformat() if user.get("createdAt") else None
            }
        })
    except Exception as e:
        return jsonify({"error": "Unable to fetch user data"}), 500

@app.route("/auth/logout", methods=["POST"])
@jwt_required()
def logout():
    """Logout user (client-side token removal)"""
    return jsonify({"message": "Logout successful"})

# ============================================================
# IMAGE CROPPING FUNCTIONS
# ============================================================

@app.route("/crop_image", methods=["POST"])
def crop_image():
    """Crop image based on user-defined coordinates"""
    if not PIL_AVAILABLE:
        return jsonify({"error": "Image processing unavailable"}), 500
    
    try:
        # Get image and crop coordinates from request
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image provided"}), 400
        
        data = request.form
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        
        if width <= 0 or height <= 0:
            return jsonify({"error": "Invalid crop dimensions"}), 400
        
        # Read and crop image
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # Ensure crop coordinates are within image bounds
        img_width, img_height = img.size
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = min(width, img_width - x)
        height = min(height, img_height - y)
        
        # Perform crop
        cropped_img = img.crop((x, y, x + width, y + height))
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        cropped_img.save(output_buffer, format='PNG', quality=95)
        cropped_bytes = output_buffer.getvalue()
        
        return jsonify({
            "success": True,
            "cropped_size": f"{width}x{height}",
            "original_size": f"{img_width}x{img_height}",
            "cropped_image": base64.b64encode(cropped_bytes).decode('utf-8'),
            "crop_area": {
                "x": x, "y": y, "width": width, "height": height
            }
        })
        
    except Exception as e:
        print(f"Crop image error: {e}")
        return jsonify({"error": f"Crop failed: {str(e)}"}), 500

@app.route("/analyze_image_for_crop", methods=["POST"])
def analyze_image_for_crop():
    """Analyze image to suggest optimal crop areas for text"""
    if not PIL_AVAILABLE:
        return jsonify({"error": "Image processing unavailable"}), 500
    
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image provided"}), 400
        
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        width, height = img.size
        
        # Convert to numpy array for analysis
        np_img = np.array(img)
        
        # Find text-rich regions (areas with high edge density)
        edges = img.filter(ImageFilter.FIND_EDGES)
        np_edges = np.array(edges)
        
        # Suggest crop regions based on edge density
        suggestions = []
        
        # Method 1: Auto-detect text regions
        try:
            # Find non-empty regions (where pixel values indicate text)
            text_threshold = 200  # Pixels darker than this are likely text
            text_mask = np_img < text_threshold
            
            if np.any(text_mask):
                # Find bounding box of text content
                non_zero_rows = np.where(text_mask)[0]
                non_zero_cols = np.where(text_mask)[1]
                
                if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
                    min_row, max_row = non_zero_rows[0], non_zero_rows[-1]
                    min_col, max_col = non_zero_cols[0], non_zero_cols[-1]
                    
                    # Add padding
                    padding = 20
                    crop_x = max(0, min_col - padding)
                    crop_y = max(0, min_row - padding)
                    crop_width = min(width - crop_x, max_col - min_col + 2 * padding)
                    crop_height = min(height - crop_y, max_row - min_row + 2 * padding)
                    
                    suggestions.append({
                        "type": "auto_text_region",
                        "x": int(crop_x),
                        "y": int(crop_y),
                        "width": int(crop_width),
                        "height": int(crop_height),
                        "confidence": "high",
                        "description": "Auto-detected text region"
                    })
        except Exception as e:
            print(f"Auto text detection failed: {e}")
        
        # Method 2: Suggest common aspect ratios for code
        common_ratios = [
            {"width": width, "height": min(800, height), "desc": "Full width, content height"},
            {"width": min(600, width), "height": height, "desc": "Content width, full height"},
            {"width": min(800, width), "height": min(600, height), "desc": "Standard code block"},
        ]
        
        for ratio in common_ratios:
            if ratio["width"] > 100 and ratio["height"] > 100:  # Minimum size
                suggestions.append({
                    "type": "aspect_ratio",
                    "x": 0,
                    "y": 0,
                    "width": ratio["width"],
                    "height": ratio["height"],
                    "confidence": "medium",
                    "description": ratio["desc"]
                })
        
        return jsonify({
            "success": True,
            "original_size": f"{width}x{height}",
            "suggestions": suggestions,
            "analysis": {
                "edge_density": f"{np_edges.mean():.2f}",
                "text_areas_found": len([s for s in suggestions if s["type"] == "auto_text_region"])
            }
        })
        
    except Exception as e:
        print(f"Analyze for crop error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ============================================================
# ENHANCED OCR PROCESSING WITH IMAGE PREPROCESSING (ALWAYS ENABLED)
# ============================================================

@app.route("/process_images", methods=["POST"])
def process_images():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    auto_indent = (request.form.get("auto_indent", "false").lower() == "true")
    # PREPROCESSING IS NOW ALWAYS ENABLED - NO USER TOGGLE NEEDED
    enable_preprocessing = True
    
    combined = []
    vision_conf = 0.0
    quality_feedback = []
    total_characters = 0
    preprocessing_results = []

    for f in files:
        raw = f.read()
        
        # Analyze original image quality FIRST
        original_quality = analyze_image_quality(raw, f.filename)
        
        # ALWAYS preprocess image (if PIL is available)
        processed_bytes = raw
        preprocessing_log = ["Preprocessing enabled and applied"]
        quality_comparison = {}
        
        if PIL_AVAILABLE:
            processed_bytes, preprocessing_log = preprocess_image(raw)
            quality_comparison = compare_image_quality(raw, processed_bytes)
        else:
            preprocessing_log = ["Preprocessing unavailable - PIL not installed"]
        
        # Then run OCR on processed image
        text, conf = vision_document_text(processed_bytes)
        vision_conf += conf
        
        # Get enhanced quality analysis using actual image metrics
        file_feedback = analyze_ocr_quality(text, conf, original_quality, f.filename)
        file_feedback["filename"] = f.filename
        file_feedback["image_quality"] = original_quality
        file_feedback["preprocessing_applied"] = True  # Always true now
        file_feedback["preprocessing_log"] = preprocessing_log
        file_feedback["quality_improvement"] = quality_comparison
        
        quality_feedback.append(file_feedback)
        
        total_characters += len(text) if text else 0
        combined.append(text or f"# (No text detected in {f.filename})")

    merged = "\n".join(combined)
    if auto_indent and merged:
        merged = gpt_indent(merged)

    avg_conf = (vision_conf / len(files)) if files else 0
    
    # Overall quality assessment
    overall_readability = "good"
    if avg_conf < 0.5:
        overall_readability = "poor"
    elif avg_conf < 0.7:
        overall_readability = "fair"
    elif avg_conf < 0.9:
        overall_readability = "good"
    else:
        overall_readability = "excellent"
    
    return jsonify({
        "extracted_text": merged,
        "estimated_accuracy": int(avg_conf * 100),
        "engine": "vision",
        "quality_feedback": quality_feedback,
        "overall_readability": overall_readability,
        "file_count": len(files),
        "total_characters": total_characters,
        "success": True,
        "image_analysis_available": PIL_AVAILABLE,
        "preprocessing_enabled": True  # Always true
    })

# ============================================================
# OCR TIPS ENDPOINT
# ============================================================

@app.route("/ocr_tips", methods=["GET"])
def get_ocr_tips():
    """Get general OCR improvement tips"""
    tips = {
        "capture_tips": [
            "Ensure good lighting without shadows or glare",
            "Position camera directly above the code",
            "Use high contrast (black text on white background works best)",
            "Keep the image focused and avoid blur",
            "Capture code in smaller, well-defined blocks"
        ],
        "common_issues": [
            "Handwritten text is harder to recognize than printed",
            "Low resolution images reduce accuracy",
            "Curved pages or poor angles distort text",
            "Syntax highlighting colors can interfere",
            "Small font sizes may not be recognized well"
        ],
        "best_practices": [
            "Use screenshots instead of photos when possible",
            "Crop images to show only the code",
            "Increase font size in your code editor before capturing",
            "Use monospace fonts for better character recognition",
            "Check for common misreads like '0' vs 'O' or '1' vs 'l'"
        ],
        "troubleshooting": [
            "If accuracy is low, try retaking the photo with better lighting",
            "For long code, split into multiple images",
            "Avoid glossy surfaces that create reflections",
            "Ensure text is horizontal in the image",
            "Clean camera lens before capturing"
        ],
        "preprocessing_benefits": [
            "Auto-enhancement improves contrast and sharpness",
            "Noise reduction removes grain and artifacts",
            "Auto-rotation corrects orientation issues",
            "Grayscale conversion improves character recognition",
            "Border cropping focuses on relevant content"
        ]
    }
    return jsonify(tips)

# ============================================================
# ASK AI (CODE ASSISTANT)
# ============================================================

@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    data = request.json or {}
    prompt = data.get("message", "")
    context = data.get("context_code", "")
    if not client:
        return jsonify({'response': "OpenAI key not configured."})
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python tutor."},
                {"role": "user", "content": f"{prompt}\n\nCode:\n```python\n{context}\n```"}
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return jsonify({'response': resp.choices[0].message.content})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"}), 500

# ============================================================
# ENHANCED CHAT HISTORY MANAGEMENT
# ============================================================

@app.route("/chats", methods=["POST"])
@jwt_required()
def create_chat():
    """Create a new chat session"""
    try:
        data = request.json
        user_email = data.get("user_email")
        
        if not user_email:
            return jsonify({"error": "User email is required"}), 400
        
        chat = {
            "user_email": user_email,
            "title": data.get("title", "New Chat"),
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc),
            "message_count": 0
        }
        
        result = chats_col.insert_one(chat)
        chat["_id"] = str(result.inserted_id)
        
        return jsonify(chat)
        
    except Exception as e:
        print("Error creating chat:", e)
        return jsonify({"error": "Failed to create chat"}), 500

@app.route("/chats/<user_email>", methods=["GET"])
@jwt_required()
def get_user_chats(user_email):
    """Get all chats for a user"""
    try:
        if user_email == "undefined" or user_email == "null":
            return jsonify([])
        
        # Get chats sorted by most recent
        chats_cursor = chats_col.find({"user_email": user_email}).sort("updatedAt", -1)
        chats = list(chats_cursor)
        
        # Convert ObjectId to string and get message counts
        for chat in chats:
            chat["_id"] = str(chat["_id"])
            # Get message count for each chat
            chat["message_count"] = messages_col.count_documents({"chat_id": str(chat["_id"])})
        
        return jsonify(chats)
        
    except Exception as e:
        print("Error fetching chats:", e)
        return jsonify([])

@app.route("/chats/<chat_id>", methods=["PUT"])
@jwt_required()
def update_chat(chat_id):
    """Update chat title"""
    try:
        data = request.json
        new_title = data.get("title")
        
        if not new_title:
            return jsonify({"error": "Title is required"}), 400
        
        result = chats_col.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": {"title": new_title, "updatedAt": datetime.now(timezone.utc)}}
        )
        
        if result.modified_count == 0:
            return jsonify({"error": "Chat not found"}), 404
            
        return jsonify({"success": True, "title": new_title})
        
    except Exception as e:
        print("Error updating chat:", e)
        return jsonify({"error": "Failed to update chat"}), 500

@app.route("/chats/<chat_id>", methods=["DELETE"])
@jwt_required()
def delete_chat(chat_id):
    """Delete a chat and all its messages"""
    try:
        # Delete chat
        chat_result = chats_col.delete_one({"_id": ObjectId(chat_id)})
        
        if chat_result.deleted_count == 0:
            return jsonify({"error": "Chat not found"}), 404
        
        # Delete all messages in this chat
        messages_col.delete_many({"chat_id": chat_id})
        
        return jsonify({"success": True, "message": "Chat deleted successfully"})
        
    except Exception as e:
        print("Error deleting chat:", e)
        return jsonify({"error": "Failed to delete chat"}), 500

@app.route("/chats/<chat_id>/messages", methods=["POST"])
@jwt_required()
def add_message(chat_id):
    """Add a message to a chat"""
    try:
        data = request.json
        role = data.get("role")  # "user" or "assistant"
        content = data.get("content")
        
        if not role or not content:
            return jsonify({"error": "Role and content are required"}), 400
        
        # Verify chat exists
        chat = chats_col.find_one({"_id": ObjectId(chat_id)})
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        message = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        }
        
        result = messages_col.insert_one(message)
        message_id = str(result.inserted_id)
        
        # Update chat's updatedAt timestamp and increment message count
        chats_col.update_one(
            {"_id": ObjectId(chat_id)},
            {
                "$set": {"updatedAt": datetime.now(timezone.utc)},
                "$inc": {"message_count": 1}
            }
        )
        
        # Auto-generate chat title from first user message if it's the first message
        message_count = messages_col.count_documents({"chat_id": chat_id})
        if message_count == 1 and role == "user":
            # Use first 50 characters of first message as title
            auto_title = content[:50] + ("..." if len(content) > 50 else "")
            chats_col.update_one(
                {"_id": ObjectId(chat_id)},
                {"$set": {"title": auto_title}}
            )
        
        return jsonify({
            "success": True, 
            "message_id": message_id,
            "timestamp": message["timestamp"].isoformat()
        })
        
    except Exception as e:
        print("Error adding message:", e)
        return jsonify({"error": "Failed to add message"}), 500

@app.route("/chats/<chat_id>/messages", methods=["GET"])
@jwt_required()
def get_chat_messages(chat_id):
    """Get all messages for a chat"""
    try:
        if not chat_id or chat_id == "undefined" or chat_id == "null":
            return jsonify([])
        
        # Verify chat exists
        chat = chats_col.find_one({"_id": ObjectId(chat_id)})
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        # Get messages sorted by timestamp
        messages_cursor = messages_col.find({"chat_id": chat_id}).sort("timestamp", 1)
        messages = list(messages_cursor)
        
        # Convert ObjectId to string and format timestamp
        for message in messages:
            message["_id"] = str(message["_id"])
            message["timestamp"] = message["timestamp"].isoformat()
        
        return jsonify(messages)
        
    except Exception as e:
        print("Error fetching messages:", e)
        return jsonify([])

# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

@app.route("/test-db", methods=["GET"])
def test_db():
    """Test MongoDB connection and return status"""
    try:
        # Try to perform a simple operation
        db.command('ping')
        
        # Get some stats
        user_count = users_col.count_documents({})
        chat_count = chats_col.count_documents({})
        message_count = messages_col.count_documents({})
        
        return jsonify({
            "status": "connected",
            "database": "MongoDB",
            "stats": {
                "users": user_count,
                "chats": chat_count,
                "messages": message_count
            },
            "message": "Database connection is healthy"
        })
    except Exception as e:
        return jsonify({
            "status": "disconnected", 
            "database": "In-Memory Fallback",
            "message": f"MongoDB unavailable: {str(e)}"
        })
    
@app.route("/health", methods=["GET"])
def health_check():
    """Comprehensive health check"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {}
    }
    
    # Check MongoDB
    try:
        db.command('ping')
        status["services"]["mongodb"] = "healthy"
    except Exception as e:
        status["services"]["mongodb"] = f"unhealthy: {str(e)}"
        status["status"] = "degraded"
    
    # Check Google Vision
    status["services"]["google_vision"] = "available" if 'vision_client' in globals() else "unavailable"
    
    # Check PIL
    status["services"]["image_processing"] = "available" if PIL_AVAILABLE else "unavailable"
    
    # Check OpenAI
    status["services"]["openai"] = "available" if client else "unavailable"
    
    return jsonify(status)
# ============================================================
# ROOT ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return "✅ Photon Backend with ENHANCED Image Quality Analysis & Preprocessing Ready!"

# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)