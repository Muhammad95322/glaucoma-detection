import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import gdown


st.set_page_config(page_title="Glaucoma Detection Dashboard", layout="centered")

# ===============================
# BASE DIRECTORY
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# PREPROCESS
# ===============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "glaucoma_model.h5")

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1vpuMknEKh9TdVsoURvxwEYuxC6mD5l5T"
        gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

model = load_model()
# ===============================
# LOAD FILES
# ===============================
metrics_path = os.path.join(BASE_DIR, "metrics.json")
cm_path = os.path.join(BASE_DIR, "confusion_matrix.npy")
data_path = os.path.join(BASE_DIR, "data_overview.json")

metrics = None
cm = None
data_overview = None

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

if os.path.exists(cm_path):
    cm = np.load(cm_path)

if os.path.exists(data_path):
    with open(data_path, "r") as f:
        data_overview = json.load(f)

# ===============================
# UI TITLE
# ===============================
st.title("Glaucoma Artificial Intelligence Analysis (GAIA)")
st.markdown("### Retinal Fundus Image Screening")

# ===============================
# PREDICTION
# ===============================
st.subheader("Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        if prediction > 0.5:
            label, confidence = "Glaucoma", prediction
        else:
            label, confidence = "Normal", 1 - prediction

        st.subheader("Result")
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.4f}")

        if confidence < 0.6:
            st.warning("Low confidence prediction")

        st.info("For research/education only, not medical diagnosis")

# ===============================
# DATA OVERVIEW 
# ===============================
if data_overview:
    st.subheader("Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Images", data_overview["total_images"])
    col2.metric("Total Patients", data_overview["total_patients"])
    col3.metric("Glaucoma", data_overview["num_glaucoma"])
    col4.metric("Normal", data_overview["num_normal"])

    st.write(f"Average Quality Score: {data_overview['avg_quality']:.2f}")

else:
    st.warning("data_overview.json not found")

# ===============================
# MODEL PERFORMANCE
# ===============================
if metrics:
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")
else:
    st.warning("metrics.json not found")

