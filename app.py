import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

# 1. Page Config must be the very first Streamlit command
st.set_page_config(page_title="Waste Identifier", layout="centered")

# CSS to change background color based on prediction
def set_bg_color(hex_color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {hex_color};
            transition: background-color 0.5s ease;
        }}
        /* Ensure text remains readable against colored backgrounds */
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label {{
            color: white !important;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

st.title("♻️ Smart Waste Identifier")

# 2. Load model and labels (Cached for performance)
@st.cache_resource
def load_my_model():
    # Ensure these files are in the same directory as your script
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_my_model()

# 3. File Uploader
file_input = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png", "webp", "avif"]) 

if file_input:
    # Open and correct image orientation
    image = Image.open(file_input).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="Uploaded Image", use_container_width=True) 
    
    with st.spinner("Classifying..."):
        # Preprocessing to match Teachable Machine / Keras expectations
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.asarray(image_resized).astype(np.float32)
        normalized_img_array = (img_array / 127.5) - 1  # Standard Keras normalization
        
        # Create the payload for the model
        data = np.expand_dims(normalized_img_array, axis=0)

        # Run Prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        
        # Clean the label (Teachable Machine usually saves as "0 Paper")
        full_label = class_names[index]
        label_clean = full_label.lower() 
        confidence_score = prediction[0][index]

        # 4. Logic for UI updates
        # Using "in" handles cases where labels are "0 Paper" or "1 Glass"
        if "paper" in label_clean:
            set_bg_color("#8B4513") # Brown
            st.header(f"Result: PAPER 📦 ({confidence_score:.2%})")
        elif "glass" in label_clean:
            set_bg_color("#0077be") # Blue
            st.header(f"Result: GLASS 🍾 ({confidence_score:.2%})")
        elif "plastic" in label_clean:
            set_bg_color("#FF69B4") # Pink
            st.header(f"Result: PLASTIC 🥤 ({confidence_score:.2%})")
        else:
            st.header(f"Result: {full_label} ({confidence_score:.2%})")