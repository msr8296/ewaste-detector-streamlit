import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os
import json
import requests

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Required libraries not installed: {e}")
    st.stop()

# 🔗 Hugging Face hosted model paths
MODEL_URL = "https://huggingface.co/msr8296/ewaste-models/resolve/main/e_waste_model.h5"
CLASS_URL = "https://huggingface.co/msr8296/ewaste-models/resolve/main/class_names.json"

@st.cache_resource
def load_models_and_classes():
    """Download & load YOLO + Classification model + classes"""

    # Download classification model if not present
    model_path = "model.h5"
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading classification model..."):
            with open(model_path, "wb") as f:
                f.write(requests.get(MODEL_URL).content)

    # Download class names if not present
    class_path = "class_names.json"
    if not os.path.exists(class_path):
        with st.spinner("📥 Downloading class names..."):
            with open(class_path, "wb") as f:
                f.write(requests.get(CLASS_URL).content)

    # Load class names
    with open(class_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            class_names = data
        elif isinstance(data, dict):
            if "class_names" in data:
                class_names = data["class_names"]
            elif "classes" in data:
                class_names = data["classes"]
            else:
                sorted_keys = sorted([int(k) for k in data.keys() if k.isdigit()])
                class_names = [data[str(k)] for k in sorted_keys]
        else:
            class_names = ["Unknown"]

    # Load YOLOv8 + classification model
    yolo_model = YOLO("yolov8n.pt")
    classification_model = keras.models.load_model(model_path)

    return yolo_model, classification_model, class_names


def preprocess_image(image, target_size=(224, 224)):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def classify_image(image, model, class_names):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx])
    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"unknown_{predicted_idx}"
    return predicted_class, confidence


def detect_and_classify(frame, yolo_model, classification_model, class_names):
    results = yolo_model(frame, conf=0.3, verbose=False)
    annotated_frame = frame.copy()
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped = frame[y1:y2, x1:x2]

            if cropped.size > 0:
                class_name, confidence = classify_image(cropped, classification_model, class_names)
                color = (0, 255, 0) if "e_waste" in class_name.lower() else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detections.append({"class": class_name, "confidence": confidence, "bbox": [x1, y1, x2, y2]})
    return annotated_frame, detections


def main():
    st.set_page_config(page_title="🔋 E-Waste Detector", page_icon="🔋", layout="wide")
    st.title("🔋 E-Waste Detection System")
    st.write("**Upload an image or use your webcam to detect and classify e-waste items**")

    # Load models
    yolo_model, classification_model, class_names = load_models_and_classes()

    # Sidebar status
    st.sidebar.header("System Status")
    st.sidebar.success("✅ Models Loaded")
    st.sidebar.success(f"✅ {len(class_names)} Classes Loaded")

    st.sidebar.subheader("📋 Classes")
    for i, cname in enumerate(class_names):
        icon = "♻️" if "e_waste" in cname.lower() else "🗑️"
        st.sidebar.text(f"{i}: {icon} {cname}")

    # Tabs: Upload vs Webcam
    tab1, tab2 = st.tabs(["📂 Upload Image", "📹 Webcam Detection"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            col1.subheader("Original Image")
            col1.image(image, use_column_width=True)

            with col2:
                col2.subheader("Detection Results")
                opencv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                annotated_frame, detections = detect_and_classify(opencv_img, yolo_model, classification_model, class_names)
                result_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(result_img, use_column_width=True)

                if detections:
                    st.subheader("Detected Objects")
                    for i, det in enumerate(detections, 1):
                        status = "♻️ E-WASTE" if "e_waste" in det["class"].lower() else "🗑️ NON-E-WASTE"
                        color = "green" if "e_waste" in det["class"].lower() else "orange"
                        st.markdown(f"**Object {i}:** :{color}[{status}] - {det['class']} ({det['confidence']:.2f})")
                else:
                    st.info("No objects detected in this image.")

    with tab2:
        st.subheader("⚡ Real-time Webcam Detection")
        st.warning("⚠️ Webcam may not work on Streamlit Cloud. Works best in local runs.")
        st.info("To use webcam locally: `streamlit run app.py`")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "<p>🔋 E-Waste Detection System | Powered by YOLOv8 + TensorFlow</p>"
        "</div>", unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
