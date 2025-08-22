import joblib
import numpy as np
from tensorflow.keras.preprocessing import image

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = (150, 150)
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}   # dataset labels

# ---------------------------
# LOAD MODEL
# ---------------------------
# Choose one of your trained models
# MODEL_PATH = "saved_trained_model/pneumonia_xgb.pkl"   # or "pneumonia_log_reg.pkl"
MODEL_PATH = "saved_trained_model/pneumonia_log_reg.pkl"   # or "pneumonia_log_reg.pkl"
model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# ---------------------------
# PREPROCESS FUNCTION
# ---------------------------
def preprocess_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # normalize
    img_flat = img_array.reshape(1, -1)          # flatten for model
    return img_flat

# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict_image(img_path):
    img_flat = preprocess_image(img_path)
    pred = model.predict(img_flat)[0]   # get prediction (0 or 1)
    return CLASS_NAMES[int(pred)]

# ---------------------------
# TEST INFERENCE
# ---------------------------
# Example: pass any test image
# test_image_path = "chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"  # change path
test_image_path = "chest_xray/test/NORMAL/IM-0001-0001.jpeg"  # change path
result = predict_image(test_image_path)
print(f"🔍 Prediction for {test_image_path}: {result}")
