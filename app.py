import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

st.set_page_config(page_title="Waste Identifier", layout="centered")

# CSS to change background color based on prediction
def set_bg_color(hex_color):
    st.markdown(f"""<style>.stApp {{background-color: {hex_color}; transition: background-color 0.5s ease; color: white !important;}}</style>""", unsafe_allow_html=True)

st.title("♻️ Smart Waste Identifier")

# Load model and labels
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
    return model, class_names

model, class_names = load_my_model()

img_file = st.camera_input("Point at some waste!")

if img_file:
    image = Image.open(img_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = (np.asarray(image).astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    label = class_names[index].lower()
    confidence = prediction[0][index]

    if confidence > 0.80: # Slightly lower threshold since there's no neutral
        if "Paper" in label:
            set_bg_color("#8B4513") # Brown
            st.header("Result: PAPER 📦")
        elif "Glass" in label:
            set_bg_color("#0077be") # Blue
            st.header("Result: GLASS 🍾")
        elif "Plastic" in label:
            set_bg_color("#FF69B4") # Pink
            st.header("Result: PLASTIC 🥤")
    else:
        # If confidence is low, keep it gray
        set_bg_color("#2E2E2E")
        st.write("Scan an item clearly...")