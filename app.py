import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model only once using caching
@st.cache_resource
def get_model():
    model = load_model(r'Model 1\Main Project\model.h5')
    return model

# Image preprocessing function
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((180, 180))  # Resize to model input
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# App UI
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.markdown("Upload a chest X-ray image and let the AI model determine if it shows signs of pneumonia.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Chest X-Ray (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# Prediction threshold slider
threshold = st.slider("🔍 Prediction Confidence Threshold", 0.0, 1.0, 0.5, 0.01,
                      help="Adjust this to control the sensitivity of the classification.")

# Process uploaded image
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded X-Ray", use_column_width=True)

        img_array = preprocess_image(image)

        with st.spinner("🧠 Analyzing X-Ray..."):
            model = get_model()
            prediction = model.predict(img_array)[0][0]

        st.subheader("🧾 Prediction Result")
        if prediction < threshold:
            st.success(f"🟢 Result: Normal (Confidence: {(1 - prediction) * 100:.2f}%)")
        else:
            st.error(f"🔴 Result: Pneumonia Detected (Confidence: {prediction * 100:.2f}%)")

        st.caption(f"🧪 Raw Model Score: {prediction:.4f}")

    except Exception as e:
        st.error("⚠️ Error processing image. Please try again with a valid image file.")
        st.exception(e)
else:
    st.info("📎 Please upload a chest X-ray image to begin.")
