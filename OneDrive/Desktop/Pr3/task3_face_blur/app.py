# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from person_blur import blur_persons

st.set_page_config(page_title="Person Blur", page_icon="🕵️", layout="centered")

st.title("🔒 Person Anonymizer (Blur Humans)")
st.write("Upload an image — humans will be detected and blurred to protect identity.")

# Upload image
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Slider for blur strength
blur_strength = st.slider("Blur strength", min_value=15, max_value=101, value=35, step=2)

if uploaded:
    # Load image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original", use_column_width=True)

    # Convert to numpy RGB
    img_np = np.array(image)

    # Process
    with st.spinner("Detecting persons and applying blur..."):
        result = blur_persons(img_np, blur_strength=blur_strength)

    # Show blurred image
    st.image(result, caption="Blurred Output", use_column_width=True)
