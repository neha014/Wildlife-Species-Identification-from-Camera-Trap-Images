import os
import time
import logging
import tempfile
import cv2 as cv
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# -------------------------------------------------
# Setup logging directory
# -------------------------------------------------
os.makedirs('./logs', exist_ok=True)

logging.basicConfig(
    filename="./logs/log.log",
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
MODEL_DIR = 'runs/classify/train3/weights/best.pt'  # <- update if you retrain later
model = YOLO(MODEL_DIR)

# -------------------------------------------------
# Streamlit App
# -------------------------------------------------
def main():
    st.sidebar.header("**Animal Classes**")
    for name in model.names.values():
        st.sidebar.markdown(f"- *{name.capitalize()}*")

    st.title("Wildlife Animal Classification")
    st.write(
        "This project classifies animals into their species using a trained YOLOv8 classification model. "
        "Upload an image or video below to test real-time classification."
    )

    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file)
        elif uploaded_file.type.startswith('video'):
            inference_video(uploaded_file)

# -------------------------------------------------
# Image Inference
# -------------------------------------------------
def inference_images(uploaded_file):
    image = Image.open(uploaded_file)
    # Run prediction
    pred = model(image)

    pred_class_idx = pred[0].probs.top1
    conf = pred[0].probs.top1conf.item()
    class_name = model.names[pred_class_idx]

    st.image(image, caption=f"Prediction: {class_name} ({conf:.2f} confidence)", width=600)
    logging.info(f"Detected {class_name} with confidence {conf:.2f}")

# -------------------------------------------------
# Video Inference
# -------------------------------------------------
def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop")

    while True:
        ret, frame = cap.read()
        if not ret or stop_placeholder:
            break

        pred = model(frame)
        pred_class_idx = pred[0].probs.top1
        conf = pred[0].probs.top1conf.item()
        class_name = model.names[pred_class_idx]

        cv.putText(frame, f"{class_name} ({conf:.2f})", (20, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR", caption="Classified Frame")

    cap.release()
    os.unlink(temp_file.name)

# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == '__main__':
    main()
    