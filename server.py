import os
import base64
import numpy as np
import cv2
from matplotlib import colormaps
from flask import Flask, request, jsonify, render_template

# Try loading tflite-runtime, fallback to tensorflow.lite if running in developer env
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# CONFIG
IMG_SIZE = (320, 320)
MODEL_PATH = "best_densenet_model.tflite"
WEIGHTS_PATH = "gradcam_weights.npz"

app = Flask(__name__)

# Load TFLite model at startup
print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("Model loaded successfully!")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load saved Grad-CAM weights for fast mathematical execution
print("Loading Grad-CAM weights...")
weights = np.load(WEIGHTS_PATH)
W1 = weights['W1']
W2 = weights['W2']
scale_bn = weights['scale_bn']
print("Weights loaded successfully!")

# Dynamically map TFLite outputs to avoid dependency on order/names
preds_idx = None
conv_idx = None
dense_idx = None
relu_idx = None

# 1. Try suffix matching based on Keras outputs order (StatefulPartitionedCall_1:X -> X)
for detail in output_details:
    name = detail['name']
    idx = detail['index']
    parts = name.split(':')
    if len(parts) > 1 and parts[-1].isdigit():
        k_idx = int(parts[-1])
        if k_idx == 0: preds_idx = idx
        elif k_idx == 1: conv_idx = idx
        elif k_idx == 2: dense_idx = idx
        elif k_idx == 3: relu_idx = idx

# 2. Fallback matching by shape and name keywords if suffix matching is incomplete
for detail in output_details:
    idx = detail['index']
    shape = list(detail['shape'])
    name = detail['name'].lower()
    
    if shape == [1, 1] and preds_idx is None:
        preds_idx = idx
    elif shape == [1, 128] and dense_idx is None:
        dense_idx = idx
    elif shape == [1, 10, 10, 1024]:
        if conv_idx is None and ('concat' in name or ':1' in name):
            conv_idx = idx
        elif relu_idx is None and ('relu' in name or ':3' in name):
            relu_idx = idx

# 3. Secondary fallback for conv/relu shape mismatch
for detail in output_details:
    idx = detail['index']
    shape = list(detail['shape'])
    if shape == [1, 10, 10, 1024]:
        if idx != conv_idx and relu_idx is None:
            relu_idx = idx
        elif idx != relu_idx and conv_idx is None:
            conv_idx = idx

if any(x is None for x in [preds_idx, conv_idx, dense_idx, relu_idx]):
    raise RuntimeError(f"Could not map all TFLite outputs! preds={preds_idx}, conv={conv_idx}, dense={dense_idx}, relu={relu_idx}")

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

        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()

        # Extract output tensors
        pred_val = float(interpreter.get_tensor(preds_idx)[0][0])
        conv_out = interpreter.get_tensor(conv_idx)
        dense_out = interpreter.get_tensor(dense_idx)
        relu_out = interpreter.get_tensor(relu_idx)

        confidence = pred_val
        label = "PNEUMONIA" if confidence >= 0.5 else "NORMAL"

        # Generate Grad-CAM Heatmap mathematically via NumPy
        # 1. Compute active nodes in the first Dense layer
        d1_active = (dense_out > 0).astype(np.float32)
        # 2. Backpropagate Dense layers: d(logit) / d(GAP_output)
        dlogit_dg = np.sum(W1 * (W2.T * d1_active), axis=1)
        # 3. Apply active mask from Conv block ReLU
        x_bn_active = (relu_out > 0).astype(np.float32)
        # 4. Multiply with BN scale parameters
        grads_manual = dlogit_dg * scale_bn * x_bn_active
        # 5. Global Average Pooling of gradients
        pooled_grads = np.mean(grads_manual, axis=(0, 1, 2))
        
        # 6. Apply weights to conv feature maps
        heatmap = conv_out[0] @ pooled_grads[..., np.newaxis]
        heatmap = heatmap.squeeze()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
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
