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
    with open("labels.txt", "r") as f:
        class_names = [line.strip()[2:] for line in f.readlines()]
    return model, class_names

model, class_names = load_my_model()

img_file = file_input

if img_file:
    image = Image.open(img_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="Uploaded Image", use_column_width=True) 
     # Correct orientation based on EXIF data
    with st.spinner("Processing..."):
        size = (224, 224)
        cropped_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(cropped_image).astype(np.float32)
        normalized_img_array = (img_array / 127.5) - 1
        data = np.expand_dims(normalized_img_array, axis=0)

        prediction = model.predict(data)
        index = np.argmax(prediction)

        full_label = class_names[index]
        clean_label = full_label.lower()  # Get the first word (e.g., "Paper", "Glass", "Plastic")
        confidence = prediction[0][index]

        if "Paper" in clean_label:
            set_bg_color("#8B4513") # Brown
            st.success("Result: PAPER 📦")
        elif "Glass" in clean_label:
            set_bg_color("#0077be") # Blue
            st.success("Result: GLASS 🍾")
        elif "Plastic" in clean_label:
            set_bg_color("#FF69B4") # Pink
            st.success("Result: PLASTIC 🥤")
        
        st.subheader(f"Confidence: {confidence:.2%}")
