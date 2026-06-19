import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from flask import Flask, request, jsonify, render_template

# CONFIG
IMG_SIZE = (320, 320)
MODEL_PATH = "best_densenet_model.keras"
LAST_CONV_LAYER = "conv5_block16_concat"

app = Flask(__name__)

# Load Keras Model once at startup
print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# CLAHE Preprocessing (Matching original app.py)
def apply_clahe(img):
    img = img.astype('uint8')
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return final_img / 255.0

# Preprocess Image (Matching original app.py)
def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = apply_clahe(img)
    return np.expand_dims(img, axis=0)

# Grad-CAM Heatmap Generation (Matching original app.py)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        loss = preds[0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    try:
        # Read the uploaded image file in memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'error': 'Invalid image file format'}), 400

        # Convert to RGB to match Pillow output used in original training/inference
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Keep copy of RGB for output overlay sizing
        img_for_overlay = img_rgb.copy()

        # Run preprocessing
        img_array = preprocess(img_rgb)

        # Model Predict
        pred_val = model.predict(img_array)[0][0]
        confidence = float(pred_val)
        label = "PNEUMONIA" if confidence >= 0.5 else "NORMAL"

        # Generate Grad-CAM Heatmap
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        heatmap = np.uint8(255 * heatmap)

        # Apply Jet Color Map
        jet = colormaps["jet"]
        jet_heatmap = jet(heatmap)[:, :, :3]
        
        # Resize heatmap to match the original uploaded image shape
        jet_heatmap = cv2.resize(jet_heatmap, (img_for_overlay.shape[1], img_for_overlay.shape[0]))

        # Superimpose Grad-CAM heatmap on original image
        img_color = img_for_overlay / 255.0
        superimposed = jet_heatmap * 0.5 + img_color
        superimposed = np.clip(superimposed, 0, 1)
        superimposed_uint8 = np.uint8(255 * superimposed)

        # Base64 Encode Original Image (RGB -> BGR for opencv encoding)
        img_bgr_out = cv2.cvtColor(img_for_overlay, cv2.COLOR_RGB2BGR)
        _, buffer_orig = cv2.imencode('.jpg', img_bgr_out)
        orig_base64 = base64.b64encode(buffer_orig).decode('utf-8')

        # Base64 Encode Superimposed Image (RGB -> BGR for opencv encoding)
        superimposed_bgr = cv2.cvtColor(superimposed_uint8, cv2.COLOR_RGB2BGR)
        _, buffer_super = cv2.imencode('.jpg', superimposed_bgr)
        super_base64 = base64.b64encode(buffer_super).decode('utf-8')

        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence,
            'original_image': f"data:image/jpeg;base64,{orig_base64}",
            'heatmap_image': f"data:image/jpeg;base64,{super_base64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    # Bind to localhost
    app.run(host='127.0.0.1', port=5000, debug=True)
