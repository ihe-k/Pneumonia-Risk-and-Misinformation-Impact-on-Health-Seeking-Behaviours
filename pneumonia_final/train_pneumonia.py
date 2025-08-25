# train_pneumonia.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# ---------------------------
# CONFIG
# ---------------------------
DATASET_DIR = "chest_xray"   # root folder with train, val, test
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SAVE_DIR = "saved_trained_model"  # folder to save trained models

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# DATA LOADING
# ---------------------------
datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_gen = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ---------------------------
# EXTRACT ARRAYS
# ---------------------------
def generator_to_numpy(gen):
    X, y = [], []
    for i in range(len(gen)):
        imgs, labels = gen[i]
        X.append(imgs)
        y.append(labels)
    return np.vstack(X), np.hstack(y)

print("Loading datasets into memory...")
X_train, y_train = generator_to_numpy(train_gen)
X_val, y_val = generator_to_numpy(val_gen)
X_test, y_test = generator_to_numpy(test_gen)

print("Data shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# Flatten images for classical ML
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat   = X_val.reshape(len(X_val), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)

# ---------------------------
# MODEL 1: Logistic Regression
# ---------------------------
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, verbose=1)
log_reg.fit(X_train_flat, y_train)

y_val_pred = log_reg.predict(X_val_flat)
print("LogReg Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

joblib.dump(log_reg, os.path.join(SAVE_DIR, "pneumonia_log_reg.pkl"))
print(f"✅ Saved Logistic Regression model as {os.path.join(SAVE_DIR, 'pneumonia_log_reg.pkl')}")

# ---------------------------
# MODEL 2: XGBoost
# ---------------------------
print("\nTraining XGBoost...")
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
xgb_clf.fit(X_train_flat, y_train)

y_val_pred = xgb_clf.predict(X_val_flat)
print("XGBoost Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

joblib.dump(xgb_clf, os.path.join(SAVE_DIR, "pneumonia_xgb.pkl"))
print(f"✅ Saved XGBoost model as {os.path.join(SAVE_DIR, 'pneumonia_xgb.pkl')}")

# ---------------------------
# Final Test Evaluation
# ---------------------------
print("\nEvaluating on test set...")

y_test_pred_lr = log_reg.predict(X_test_flat)
print("LogReg Test Accuracy:", accuracy_score(y_test, y_test_pred_lr))

y_test_pred_xgb = xgb_clf.predict(X_test_flat)
print("XGBoost Test Accuracy:", accuracy_score(y_test, y_test_pred_xgb))
