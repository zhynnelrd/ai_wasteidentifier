import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

st.set_page_config(page_title="Waste Identifier", layout="centered")

# CSS to change background color based on prediction
def set_bg_color(hex_color):
    st.markdown(f"""<style>.stApp {{background-color: {hex_color}; transition: background-color 0.5s ease; color: white !important;}}</style>""", unsafe_allow_html=True)

st.title("♻️ Smart Waste Identifier")
file_input = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png", "webp", "avif"])    
# Load model and labels
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    #this is the reader of your labels.txt file, it reads each line and strips any whitespace
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
    return model, class_names

model, class_names = load_my_model()

img_file = file_input

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

    if "0Paper" in label:
        set_bg_color("#8B4513") # Brown
        st.header("Result: PAPER 📦")
    elif "1Glass" in label:
        set_bg_color("#0077be") # Blue
        st.header("Result: GLASS 🍾")
    elif "2Plastic" in label:
        set_bg_color("#FF69B4") # Pink
        st.header("Result: PLASTIC 🥤")
