import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Load YOLO model
model = YOLO("best.pt")  # Replace with your custom-trained model if available

st.title("🔥 Forest Fire Detection with YOLOv8")
st.markdown("Upload an image to detect forest fires.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    # Run YOLO inference
    st.write("Detecting forest fire...")
    results = model(temp_path)

    # Get annotated image from results
    annotated_frame = results[0].plot()  # numpy array (BGR)

    # Convert BGR to RGB for display
    annotated_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    st.image(annotated_image, caption="Detection Result", use_column_width=True)

    # Optional: clean up temp file
    os.remove(temp_path)
