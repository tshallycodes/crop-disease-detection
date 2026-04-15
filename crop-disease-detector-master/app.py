import os, json, uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ── Disease info ──────────────────────────────────────────
DISEASE_INFO = {
    'healthy': {
        'status': 'healthy',
        'advice': 'Your plant looks healthy! Keep up the good care — water regularly and ensure adequate sunlight.'
    },
    'Tomato_Bacterial_spot': {
        'status': 'diseased',
        'advice': 'Remove infected leaves immediately. Apply copper-based fungicide. Avoid overhead watering.'
    },
    'Tomato_Early_blight': {
        'status': 'diseased',
        'advice': 'Remove lower infected leaves. Apply fungicide containing chlorothalonil. Rotate crops next season.'
    },
    'Tomato_Late_blight': {
        'status': 'diseased',
        'advice': 'Serious disease — act fast. Remove all infected parts. Apply mancozeb or copper fungicide immediately.'
    },
    'Tomato_Leaf_Mold': {
        'status': 'diseased',
        'advice': 'Improve air circulation. Reduce humidity. Apply fungicide. Avoid wetting leaves when watering.'
    },
    'Tomato_Septoria_leaf_spot': {
        'status': 'diseased',
        'advice': 'Remove infected leaves. Apply fungicide early. Mulch around base to prevent soil splash.'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'status': 'diseased',
        'advice': 'Spray plants with water to dislodge mites. Apply neem oil or insecticidal soap spray.'
    },
    'Tomato__Target_Spot': {
        'status': 'diseased',
        'advice': 'Remove and destroy infected leaves. Apply fungicide. Ensure good air circulation.'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'status': 'diseased',
        'advice': 'No cure — remove and destroy infected plants. Control whiteflies which spread this virus.'
    },
    'Tomato__Tomato_mosaic_virus': {
        'status': 'diseased',
        'advice': 'No cure — remove infected plants. Wash hands and tools. Control aphids to prevent spread.'
    },
    'Potato___Early_blight': {
        'status': 'diseased',
        'advice': 'Apply fungicide at first signs. Remove infected foliage. Ensure proper plant spacing.'
    },
    'Potato___Late_blight': {
        'status': 'diseased',
        'advice': 'Very serious — can destroy entire crop. Apply fungicide immediately. Remove all infected plants.'
    },
    'Pepper__bell___Bacterial_spot': {
        'status': 'diseased',
        'advice': 'Apply copper-based bactericide. Remove infected leaves. Avoid working with wet plants.'
    },
}

def get_disease_info(class_name):
    # Check if healthy
    if 'healthy' in class_name.lower():
        return DISEASE_INFO['healthy']
    # Check specific disease
    for key, info in DISEASE_INFO.items():
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return info
    # Default for unknown disease
    return {
        'status': 'diseased',
        'advice': 'Disease detected. Consult a local agricultural expert for specific treatment advice.'
    }

def format_class_name(class_name):
    """Convert class name like Tomato_Early_blight to Tomato — Early Blight"""
    parts = class_name.replace('__', '_').split('_')
    plant = parts[0]
    condition = ' '.join(parts[1:]).replace('  ', ' ').strip()
    condition = condition.title()
    return plant, condition

# ── Load model ────────────────────────────────────────────
print("[INFO] Loading model...")
model = tf.keras.models.load_model('crop_disease_model.keras')
class_names = json.load(open('class_names.json'))
print(f"[INFO] Model loaded. Classes: {len(class_names)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    results = []
    for idx in top3_idx:
        plant, condition = format_class_name(class_names[idx])
        results.append({
            'class': class_names[idx],
            'plant': plant,
            'condition': condition,
            'confidence': round(float(preds[idx]) * 100, 1)
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return render_template('index.html', error='No file selected.')
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error='Please upload a JPG, PNG or WEBP image.')

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        results = predict(save_path)
    except Exception as e:
        os.remove(save_path)
        return render_template('index.html', error=f'Could not process image: {str(e)}')

    top = results[0]
    disease_info = get_disease_info(top['class'])
    image_url = url_for('static', filename=f'../uploads/{filename}')

    return render_template('index.html',
                           results=results,
                           top=top,
                           disease_info=disease_info,
                           image_url=image_url)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5006)
