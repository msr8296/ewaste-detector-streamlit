import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import json
import requests
import tempfile
import os
import time

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Required libraries not installed: {e}")
    st.info("Please run: pip install -r requirements_simple.txt")
    st.stop()

# Hugging Face URLs
YOLO_MODEL_HF = "https://huggingface.co/msr8296/ewaste-models/resolve/main/yolov8n.pt"
CLASSIFICATION_MODEL_HF = "https://huggingface.co/msr8296/ewaste-models/resolve/main/e_waste_model.h5"
CLASS_NAMES_HF = "https://huggingface.co/msr8296/ewaste-models/resolve/main/class_names.json"


@st.cache_resource
def load_models_and_classes():
    """Load YOLO, classification model and class names from Hugging Face"""
    try:
        # YOLO model
        yolo_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        yolo_temp.write(requests.get(YOLO_MODEL_HF).content)
        yolo_temp.flush()
        yolo_model = YOLO(yolo_temp.name)

        # Classification model
        clf_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        clf_temp.write(requests.get(CLASSIFICATION_MODEL_HF).content)
        clf_temp.flush()
        classification_model = keras.models.load_model(clf_temp.name)

        # Class names
        response = requests.get(CLASS_NAMES_HF)
        data = response.json()
        if isinstance(data, list):
            class_names = data
        elif isinstance(data, dict):
            if 'class_names' in data:
                class_names = data['class_names']
            elif 'classes' in data:
                class_names = data['classes']
            else:
                sorted_keys = sorted([int(k) for k in data.keys() if k.isdigit()])
                class_names = [data[str(k)] for k in sorted_keys]
        else:
            class_names = []

        return yolo_model, classification_model, class_names
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()


def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def classify_image(image, model, class_names):
    try:
        processed = preprocess_image(image)
        preds = model.predict(processed, verbose=0)
        idx = np.argmax(preds[0])
        conf = float(preds[0][idx])
        return class_names[idx] if idx < len(class_names) else f"unknown_{idx}", conf
    except Exception as e:
        return f"Error: {e}", 0.0


def detect_and_classify(frame, yolo_model, classification_model, class_names):
    results = yolo_model(frame, conf=0.3, verbose=False)
    annotated = frame.copy()
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box, cls_id in zip(boxes.xyxy, boxes.cls):
                # Skip "person" class
                if yolo_model.names[int(cls_id)] == "person":
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    cls_name, conf = classify_image(cropped, classification_model, class_names)
                    color = (0, 255, 0) if 'e_waste' in cls_name.lower() else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} ({conf:.2f})"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    detections.append({'class': cls_name, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})
    return annotated, detections


def main():
    st.set_page_config(page_title="🔋 E-Waste Detector", layout="wide")
    st.title("🔋 E-Waste Detection System")
    st.write("*Upload an image or use your webcam for e-waste detection.* "
             "**(For best accuracy, try uploading clear images.)**")

    yolo_model, classification_model, class_names = load_models_and_classes()

    # Sidebar
    st.sidebar.header("System Status")
    st.sidebar.success("Models: Ready")
    st.sidebar.success(f"Classes: {len(class_names)} loaded")

    st.sidebar.subheader("Loaded Classes")
    for i, cls in enumerate(class_names):
        icon = "♻" if 'e_waste' in cls.lower() else "🗑"
        st.sidebar.text(f"{i}: {icon} {cls}")

    st.sidebar.subheader("Configuration")
    st.sidebar.code(YOLO_MODEL_HF, language="text")
    st.sidebar.code(CLASSIFICATION_MODEL_HF, language="text")
    st.sidebar.code(CLASS_NAMES_HF, language="text")

    # Tabs
    tab1, tab2 = st.tabs(["Upload Image", "📸 Browser Webcam Detection"])

    # ---------- Upload Image ----------
    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Detection Results")
                annotated, detections = detect_and_classify(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
                                                            yolo_model, classification_model, class_names)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                if detections:
                    st.subheader("Detected Objects:")
                    for i, det in enumerate(detections, 1):
                        status = "♻ E-WASTE" if 'e_waste' in det['class'].lower() else "🗑 NON-E-WASTE"
                        color = "green" if 'e_waste' in det['class'].lower() else "orange"
                        st.markdown(f"*Object {i}:* :{color}[{status}] - "
                                    f"{det['class']} ({det['confidence']:.2f})")
                else:
                    st.info("No objects detected in this image.")

    # ---------- Browser Webcam (st.camera_input) ----------
    with tab2:
        st.subheader("📹 Browser-Based Webcam Detection")
        st.write("Use your browser's camera below to capture an image.")

        img_data = st.camera_input("Click below to capture a photo")

        if img_data is not None:
            image = Image.open(img_data)
            img_array = np.array(image)
            annotated, detections = detect_and_classify(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
                                                        yolo_model, classification_model, class_names)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
            if detections:
                st.subheader("Detected Objects:")
                for i, det in enumerate(detections, 1):
                    status = "♻ E-WASTE" if 'e_waste' in det['class'].lower() else "🗑 NON-E-WASTE"
                    color = "green" if 'e_waste' in det['class'].lower() else "orange"
                    st.markdown(f"*Object {i}:* :{color}[{status}] - "
                                f"{det['class']} ({det['confidence']:.2f})")
            else:
                st.info("No objects detected in this frame.")
        else:
            st.info("📸 Click 'Take Photo' above to capture an image.")

    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#666;">
    🔋 E-Waste Detection System | Powered by Hugging Face Models
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
