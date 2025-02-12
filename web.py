import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import gdown
import os


st.set_page_config(page_title="Plant Disease Detector", page_icon="🌱", layout="wide")


file_id = "1ulkitYFN3R674IYTrIcvSi7xaqibCnVE"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"



if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)


st.markdown("""
    <style>
        /* Background Gradient */
        body {
            background: linear-gradient(to right, #56ab2f, #a8e063);
            font-family: 'Arial', sans-serif;
        }
        
        /* Header Title */
        .main-title {
            text-align: center;
            font-size: 42px;
            color: white;
            font-weight: bold;
            text-shadow: 2px 2px 5px black;
            padding: 10px;
        }

        /* Sidebar Design */
        .css-1aumxhk {
            background-color: #2E8B57 !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #008080 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 12px !important;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #1abc9c !important;
        }

        /* Image Styling */
        .image-container {
            text-align: center;
            padding: 15px;
        }

        /* Footer */
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            color: white;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


st.sidebar.title("🌱 Plant Disease Detector")
app_mode = st.sidebar.radio("📌 Select Page", ['🏠 Home', '🔍 Disease Recognition'])


col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    img = Image.open('Diseases.png')
    st.image(img, caption="🌿 Plant Disease Detection System", use_container_width=True)


if app_mode == '🏠 Home':
    st.markdown("<h1 class='main-title'>🌱 Welcome to Plant Disease Detection 🌱</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: white; font-size: 20px;">
            This system helps in recognizing plant diseases for sustainable agriculture.<br>
            Upload an image of a plant leaf and get an instant diagnosis! 🌍
        </div>
    """, unsafe_allow_html=True)


elif app_mode == '🔍 Disease Recognition':
    st.markdown("<h1 class='main-title'>🔍 Disease Recognition</h1>", unsafe_allow_html=True)

    
    test_image = st.file_uploader("📸 Upload an image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(test_image, caption="🌿 Uploaded Image", use_container_width=True, output_format="auto")

        
        if st.button("🖼️ Show Image"):
            with col2:
                st.image(test_image, caption="📷 Uploaded Image", use_container_width=True)

        
        if st.button("🔍 Predict"):
            with st.spinner("🔬 Analyzing the image..."):
                time.sleep(2)  
                result_index = model_prediction(test_image)
                class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']

            
            st.success(f"✅ Model Prediction: **{class_names[result_index]}**")
            st.balloons()


st.markdown("""
    <div class="footer">
        Developed with ❤️ for Sustainable Agriculture 🌿
    </div>
""", unsafe_allow_html=True)
