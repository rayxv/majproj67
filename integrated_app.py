import os
import json
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import requests
from flask import Flask, render_template, request, jsonify, Response
from pyzbar.pyzbar import decode as pyzbar_decode
from dotenv import load_dotenv
import google.generativeai as genai
import re

# --- Setup ---
load_dotenv()

# --- Configuration Constants for Fruit Recognition ---
WEIGHTS_FILE_NAME = "clean_weights.weights.h5"  
LABELS_FILE_NAME = "model_labels.json"
NUTRI_DATA_FILE = "nutri_data.json" 
IMG_SIZE = (180, 180) 
CWD = os.getcwd()
MODEL_PATH = os.path.join(CWD, WEIGHTS_FILE_NAME) 
LABELS_PATH = os.path.join(CWD, LABELS_FILE_NAME)
NUTRI_DATA_PATH = os.path.join(CWD, NUTRI_DATA_FILE)

# Global variables for fruit recognition
fruit_model = None
LABELS = []
NUTRI_DATA = {} 
camera = None 
ACTIVE_DEVICE_INDEX = -1 

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates')

# --- Gemini Setup for Barcode Scanner ---
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-pro-latest')

# ==================== FRUIT RECOGNITION FUNCTIONS ====================

def initialize_camera():
    """Tries indices 0-4 until a working camera is found."""
    global camera, ACTIVE_DEVICE_INDEX
    
    for index in range(5):
        try:
            temp_camera = cv2.VideoCapture(index)
            if temp_camera.isOpened():
                temp_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                temp_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera = temp_camera
                ACTIVE_DEVICE_INDEX = index
                print(f"Camera initialized successfully on device index {index}.")
                return True
            else:
                temp_camera.release()
        except Exception:
            continue

    print("WARNING: Could not initialize camera for fruit recognition.")
    return False

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Applies saturation boost for better image isolation."""
    if frame is None or frame.size == 0: 
        return frame
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.float32)
    s = s * 1.5  
    s = np.clip(s, 0, 255).astype(np.uint8)
    processed_hsv = cv2.merge([h, s, v])
    processed_frame = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2BGR)
    return processed_frame

def create_inference_model(num_classes):
    """Manually reconstructs the core CNN architecture."""
    input_tensor = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = Rescaling(1./255)(input_tensor) 
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    x = Flatten()(x)
    x = Dropout(0.2)(x) 
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_tensor, outputs=predictions, name="Inference_Model")

def load_recognition_artifacts():
    """Loads model architecture, weights, labels, and nutritional data."""
    global fruit_model, LABELS, NUTRI_DATA
    
    # 1. Load Labels
    try:
        with open(LABELS_PATH, 'r') as f:
            LABELS = json.load(f)
        num_classes = len(LABELS)
        print(f"Labels loaded successfully: {num_classes} classes found.")
    except Exception as e:
        print(f"ERROR loading labels: {e}.")
        return False

    # 2. Load Local Nutritional Data
    try:
        with open(NUTRI_DATA_PATH, 'r') as f:
            nutri_list = json.load(f)
            NUTRI_DATA = {item['name'].lower(): item for item in nutri_list}
        print(f"Nutritional data loaded successfully: {len(NUTRI_DATA)} items.")
    except Exception as e:
        print(f"ERROR loading nutritional data: {e}.")
        return False

    # 3. Load Weights
    if not os.path.exists(MODEL_PATH):
         print(f"WARNING: Model weights file '{WEIGHTS_FILE_NAME}' not found. Fruit recognition disabled.")
         return False

    try:
        inference_model = create_inference_model(num_classes)
        inference_model.load_weights(MODEL_PATH)
        fruit_model = inference_model
        print(f"Fruit recognition model loaded from {WEIGHTS_FILE_NAME}.")
    except Exception as e:
        print(f"ERROR during model loading: {e}")
        fruit_model = None
        return False
    
    return True

def classify_fruit(frame: np.ndarray) -> str:
    """Performs model inference on a single frame."""
    if fruit_model is None or not LABELS or frame is None or frame.size == 0:
        return "Unknown"
    
    enhanced_frame = preprocess_frame(frame)
    processed_frame = cv2.resize(enhanced_frame, IMG_SIZE) 
    input_tensor = np.expand_dims(processed_frame, axis=0) 
    predictions = fruit_model.predict(input_tensor, verbose=0) 
    score = tf.nn.softmax(predictions) 
    predicted_class_index = np.argmax(score)
    
    if predicted_class_index < len(LABELS):
        fruit_name = LABELS[predicted_class_index].lower().split()[0].replace('-', '')
    else:
        fruit_name = "Unknown"
        
    return fruit_name

def get_nutritional_info_local(fruit_name: str) -> str:
    """Performs a local lookup for nutritional information and formats output."""
    nutri_item = NUTRI_DATA.get(fruit_name.lower())
    
    if not nutri_item:
        return f"Error: Nutritional data for '{fruit_name}' not found in local JSON file."

    intro = f"The nutritional breakdown per 100g serving for a {fruit_name.title()} is as follows:"
    
    table = "| Nutrient | Amount (per 100g) |\n"
    table += "|---|---|\n"
    table += f"| Calories | {nutri_item['calories']} kcal |\n"
    table += f"| Total Carbs | {nutri_item['carbs']} g |\n"
    table += f"| Fiber | {nutri_item['fiber']} g |\n"
    table += f"| Sugar | {nutri_item['sugar']} g |\n"
    table += f"| Protein | {nutri_item['protein']} g |\n"
    table += f"| Key Vitamins | {nutri_item['vitamins']} |\n"
    
    return f"{intro}\n\n{table}"

def generate_frames():
    """Video stream generator for fruit recognition."""
    global camera
    if camera is None or not camera.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not initialized.\r\r\n'
        return
        
    while True:
        success, frame = camera.read()
        if not success: 
            break 
        else:
            processed_frame = preprocess_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\r\n')

# ==================== BARCODE SCANNER FUNCTIONS ====================

def process_barcode_image(image):
    """Process image to improve barcode detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
    
    processed_images = [
        gray,
        thresh,
        blurred,
        cv2.bitwise_not(blurred),
    ]
    
    return processed_images

def fetch_product_info(barcode):
    """Fetch product info from OpenFoodFacts API"""
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get("status") == 1:
            product = data["product"]
            ingredients = product.get("ingredients_text") or product.get("ingredients_text_en") or "N/A"
            
            return {
                "status": "success",
                "product_name": product.get("product_name", "N/A"),
                "brands": product.get("brands", "N/A"),
                "sugar": product.get("nutriments", {}).get("sugars_100g", "N/A"),
                "ingredients_text": ingredients,
                "nutriments": product.get("nutriments", {}),
                "nutriscore_grade": product.get("nutriscore_grade", "").lower()
            }
        return {"status": "not_found", "message": "Product not found in database"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def analyze_ingredients(ingredients_text):
    """Use Gemini to analyze ingredients for health, allergens, and environmental impact"""
    if not ingredients_text or ingredients_text == "N/A":
        return {
            "health_warnings": [],
            "allergens": [],
            "health_concerns": [],
            "environmental_impact": {
                "score": 3,
                "label": "Moderate",
                "summary": "No ingredient information available",
                "main_concern": "",
                "suggestion": ""
            },
            "analysis": "No ingredient information available"
        }
    
    try:
        prompt = f"""
        Analyze these food product ingredients for health and allergens:
        {ingredients_text}

        Respond ONLY with valid JSON in this exact format:
        {{
            "health_warnings": ["warning1", "warning2"],
            "allergens": ["allergen1", "allergen2"],
            "analysis": "brief health summary"
        }}
        
        Rules:
        - Maximum 3 health warnings
        - Maximum 3 allergens
        - Analysis must be under 50 words
        - Only return the JSON object, no other text
        - No markdown formatting
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"Raw Gemini response: {response_text}")
        
        json_text = response_text
        if response_text.startswith('```'):
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_text = response_text[start:end+1]
        
        try:
            result = json.loads(json_text)
            print(f"Parsed JSON: {result}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return extract_info_manually(response_text)
        
        result = {
            "health_warnings": result.get("health_warnings", [])[:3],
            "allergens": result.get("allergens", [])[:3],
            "analysis": result.get("analysis", "Analysis unavailable")
        }
        
        return result
        
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return {
            "health_warnings": [],
            "allergens": [],
            "analysis": f"Analysis failed: {str(e)}"
        }

def extract_info_manually(response_text):
    """Fallback method to extract information if JSON parsing fails"""
    try:
        health_warnings = []
        allergens = []
        
        warning_patterns = [
            r'palm oil', r'high sodium', r'artificial', r'preservative', 
            r'msg', r'trans fat', r'high sugar', r'processed'
        ]
        
        allergen_patterns = [
            r'wheat', r'gluten', r'soy', r'milk', r'egg', r'peanut', 
            r'tree nut', r'fish', r'shellfish', r'sesame'
        ]
        
        text_lower = response_text.lower()
        
        for pattern in warning_patterns:
            if re.search(pattern, text_lower):
                health_warnings.append(pattern.replace(r'', '').title())
        
        for pattern in allergen_patterns:
            if re.search(pattern, text_lower):
                allergens.append(pattern.replace(r'', '').title())
        
        return {
            "health_warnings": health_warnings[:3],
            "allergens": allergens[:3],
            "analysis": "Basic analysis completed"
        }
    except:
        return {
            "health_warnings": [],
            "allergens": [],
            "analysis": "Analysis extraction failed"
        }

def calculate_nutriscore_with_gemini(nutriments, product_name=""):
    """Use Gemini to calculate Nutri-Score based on nutritional data"""
    
    energy_kj = nutriments.get('energy_100g') or nutriments.get('energy-kj_100g', 0)
    energy_kcal = nutriments.get('energy-kcal_100g', 0)
    saturated_fat = nutriments.get('saturated-fat_100g', 0)
    total_fat = nutriments.get('fat_100g', 0)
    sugars = nutriments.get('sugars_100g', 0)
    sodium = nutriments.get('sodium_100g', 0)
    salt = nutriments.get('salt_100g', 0)
    fiber = nutriments.get('fiber_100g', 0)
    proteins = nutriments.get('proteins_100g', 0)
    
    if sodium == 0 and salt > 0:
        sodium = salt * 0.4
    
    if energy_kj == 0 and energy_kcal > 0:
        energy_kj = energy_kcal * 4.184

    try:
        prompt = f"""
        Calculate the Nutri-Score (A, B, C, D, or E) for this food product using the official algorithm.

        Product: {product_name}
        Nutritional values per 100g:
        - Energy: {energy_kj} kJ ({energy_kcal} kcal)
        - Saturated fat: {saturated_fat}g
        - Total fat: {total_fat}g
        - Sugars: {sugars}g
        - Sodium: {sodium}g
        - Fiber: {fiber}g
        - Proteins: {proteins}g

        Use the official Nutri-Score algorithm and respond in JSON format:
        {{
            "nutriscore_grade": "X",
            "score": X,
            "negative_points": X,
            "positive_points": X,
            "calculation_details": "brief explanation"
        }}
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"Nutri-Score calculation response: {response_text}")
        
        json_text = response_text
        if response_text.startswith('```'):
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
        
        try:
            result = json.loads(json_text)
            grade = result.get('nutriscore_grade', '').upper()
            if grade in ['A', 'B', 'C', 'D', 'E']:
                return {
                    'nutriscore_grade': grade.lower(),
                    'score': result.get('score', 0),
                    'calculation_source': 'gemini',
                    'details': result.get('calculation_details', '')
                }
        except json.JSONDecodeError as e:
            print(f"JSON decode error for Nutri-Score: {e}")
    
    except Exception as e:
        print(f"Gemini Nutri-Score calculation error: {e}")
    
    return calculate_simple_nutriscore(nutriments)

def calculate_simple_nutriscore(nutriments):
    """Fallback simple Nutri-Score calculation"""
    try:
        energy_kcal = float(nutriments.get('energy-kcal_100g', 0))
        saturated_fat = float(nutriments.get('saturated-fat_100g', 0))
        sugars = float(nutriments.get('sugars_100g', 0))
        sodium = float(nutriments.get('sodium_100g', 0)) * 1000
        salt = float(nutriments.get('salt_100g', 0))
        fiber = float(nutriments.get('fiber_100g', 0))
        proteins = float(nutriments.get('proteins_100g', 0))
        
        if sodium == 0 and salt > 0:
            sodium = salt * 400
        
        energy_points = min(10, max(0, int((energy_kcal - 80) / 67)))
        sat_fat_points = min(10, max(0, int((saturated_fat - 1) / 1)))
        sugar_points = min(15, max(0, int((sugars - 4.5) / 2.7)))
        sodium_points = min(20, max(0, int((sodium - 90) / 45)))
        
        negative_points = energy_points + sat_fat_points + sugar_points + sodium_points
        
        fiber_points = min(5, max(0, int((fiber - 0.9) / 0.95)))
        protein_points = min(5, max(0, int((proteins - 1.6) / 1.28)))
        
        positive_points = fiber_points + protein_points
        final_score = negative_points - positive_points
        
        if final_score <= -1:
            grade = 'a'
        elif final_score <= 2:
            grade = 'b'
        elif final_score <= 10:
            grade = 'c'
        elif final_score <= 18:
            grade = 'd'
        else:
            grade = 'e'
        
        return {
            'nutriscore_grade': grade,
            'score': final_score,
            'calculation_source': 'calculated',
            'details': f'Score: {final_score} (Negative: {negative_points}, Positive: {positive_points})'
        }
    
    except Exception as e:
        print(f"Simple Nutri-Score calculation error: {e}")
        return {
            'nutriscore_grade': '',
            'score': 0,
            'calculation_source': 'failed',
            'details': 'Calculation failed'
        }

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """Main landing page with both features"""
    return render_template('index.html', active_index=ACTIVE_DEVICE_INDEX)

# --- Barcode Scanner Routes ---
@app.route('/barcode')
def barcode_scanner():
    """Barcode scanner page"""
    return render_template('barcode_scanner.html')

@app.route('/scan', methods=['POST'])
def scan():
    """Handle barcode scanning"""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    try:
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"status": "error", "message": "Could not read image"}), 400

        processed_images = process_barcode_image(img)
        
        barcodes_found = []
        for processed_img in processed_images:
            try:
                barcodes = pyzbar_decode(processed_img)
                for barcode in barcodes:
                    barcode_data = barcode.data.decode('utf-8')
                    if len(barcode_data) >= 8:
                        clean_barcode = ''.join(c for c in barcode_data if c.isdigit())
                        if clean_barcode not in [b[0] for b in barcodes_found]:
                            barcodes_found.append((clean_barcode, barcode.type))
            except Exception as e:
                print(f"Barcode decode error: {str(e)}")
                continue
        
        if not barcodes_found:
            return jsonify({
                "status": "not_found", 
                "message": "No valid barcode detected"
            })

        barcodes_found.sort(key=lambda x: len(x[0]), reverse=True)
        barcode_data, barcode_type = barcodes_found[0]
        
        print(f"Detected barcode: {barcode_data} (Type: {barcode_type})")
        
        product_info = fetch_product_info(barcode_data)
        
        response_data = {
            "status": product_info["status"],
            "barcode": barcode_data,
            "barcode_type": str(barcode_type),
            "product": product_info
        }

        if product_info["status"] == "success":
            if not product_info.get("nutriscore_grade") or product_info["nutriscore_grade"] == "":
                if product_info.get("nutriments"):
                    print("Calculating Nutri-Score with Gemini...")
                    nutriscore_result = calculate_nutriscore_with_gemini(
                        product_info["nutriments"], 
                        product_info.get("product_name", "")
                    )
                
                    product_info["nutriscore_grade"] = nutriscore_result["nutriscore_grade"]
                    product_info["nutriscore_source"] = nutriscore_result["calculation_source"]
                    product_info["nutriscore_details"] = nutriscore_result["details"]
                
                    print(f"Calculated Nutri-Score: {nutriscore_result['nutriscore_grade']}")
        
            if product_info.get("ingredients_text") and product_info["ingredients_text"] != "N/A":
                analysis = analyze_ingredients(product_info["ingredients_text"])
                response_data["product"]["ingredient_analysis"] = analysis
            
            return jsonify(response_data)
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Scan endpoint error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500

# --- Fruit Recognition Routes ---
@app.route('/fruit')
def fruit_recognition():
    """Fruit recognition page"""
    return render_template('fruit_recognition.html', active_index=ACTIVE_DEVICE_INDEX)

@app.route('/video_feed')
def video_feed():
    """Video stream for fruit recognition"""
    if camera is None or not camera.isOpened():
         return Response("Camera Not Available. Check permissions and device index.", status=503)

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan_and_get_info')
def scan_and_get_info():
    """Scan fruit and get nutritional info"""
    global camera
    
    if camera is None or not camera.isOpened():
         return jsonify({'detected_fruit': 'Error', 'info': 'Camera is unavailable. Check system permissions.'})

    success, frame = camera.read()
    if not success:
        return jsonify({'detected_fruit': 'Error', 'info': 'Failed to capture frame from camera.'})

    fruit_name = classify_fruit(frame)
    
    if fruit_name != "Unknown":
        info = get_nutritional_info_local(fruit_name)
    else:
        info = "Classification failed. Please ensure the fruit is clearly visible in the camera feed."

    return jsonify({
        'detected_fruit': fruit_name,
        'info': info
    })

# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("INTEGRATED NUTRITION APP - Starting Up")
    print("=" * 60)
    
    # Initialize fruit recognition components (optional)
    print("\n--- Loading Fruit Recognition Module ---")
    fruit_loaded = load_recognition_artifacts()
    camera_loaded = initialize_camera()
    
    if fruit_loaded and camera_loaded:
        print("✓ Fruit recognition module ready")
    else:
        print("⚠ Fruit recognition module disabled (missing files or camera)")
    
    # Barcode scanner is always available
    print("\n--- Barcode Scanner Module ---")
    print("✓ Barcode scanner ready")
    
    print("\n" + "=" * 60)
    print("Server starting on http://127.0.0.1:5001")
    print("=" * 60)
    
    os.makedirs('uploads', exist_ok=True)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)
    finally:
        if camera is not None and camera.isOpened():
            camera.release()
            print("\nCamera released successfully")