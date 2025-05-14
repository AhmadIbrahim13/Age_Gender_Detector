import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = tf.keras.models.load_model("gender_age_model.h5")
gender_dict = {0: "Male", 1: "Female"}

# ---- SIDEBAR ----
st.sidebar.title("🧠 App Settings")
st.sidebar.markdown("""
Upload a clear face photo.  
Model predicts age (as a number) and gender (Male/Female).  
""")

# ---- MAIN TITLE ----
st.title("👤 Gender and Age Predictor")
st.markdown("Using a CNN trained on UTKFace")

# ---- IMAGE UPLOAD ONLY ----
uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")

# ---- PREDICTION ----
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    image_resized = image.resize((128, 128))
    img_array = img_to_array(image_resized).reshape(1, 128, 128, 1) / 255.0

    # Predict
    start_time = time.time()
    gender_pred, age_pred = model.predict(img_array)
    duration = time.time() - start_time

    predicted_gender = gender_dict[round(gender_pred[0][0])]
    predicted_age = round(age_pred[0][0])

    # Display result in columns
    col1, col2 = st.columns(2)
    col1.metric("🧑 Gender", predicted_gender)
    col2.metric("🎂 Age", predicted_age)

    st.caption(f"⏱️ Prediction took {duration:.2f} seconds")

# ---- ABOUT SECTION ----
with st.expander("📘 About This Model"):
    st.markdown("""
    **🧠 Model Overview**  
    - Model: Convolutional Neural Network (CNN)  
    - Inputs: 128x128 grayscale face image  
    - Outputs:  
        - Gender: Binary classification (Male/Female)  
        - Age: Regression (exact age prediction)

    **📚 Dataset Used**  
    - [UTKFace](https://susanqq.github.io/UTKFace/)  
    - ~20K face images with age, gender, and ethnicity labels

    **📏 Metrics**  
    - Gender Accuracy: ~94% (val)  
    - Age MAE: ~4–6 years depending on lighting and input quality

    **👨‍💻 Developed By**  
    - Ahmad Ibrahim & Hadi Sabra  
    - 2025
    """)
