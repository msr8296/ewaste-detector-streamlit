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
                # Skip person class
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
    st.write("*Upload an image or use your webcam for continuous e-waste detection."
             " **(For better prediction, I suggest you upload images.)***")

    yolo_model, classification_model, class_names = load_models_and_classes()

    # Sidebar info
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

    tab1, tab2 = st.tabs(["Upload Image", "📹 Webcam Detection"])

    # ===================== Upload Image =====================
    with tab1:
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
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
                        st.markdown(f"*Object {i}:* :{color}[{status}] - {det['class']} ({det['confidence']:.2f})")
                else:
                    st.info("No objects detected in the image")

    # ===================== Webcam Detection =====================
    with tab2:
        st.subheader("Continuous Webcam Detection")
        col1, col2 = st.columns([2, 1])
        video_placeholder = col1.empty()
        live_results = col2.empty()

        if 'cap' not in st.session_state:
            st.session_state.cap = None
        if 'current_detections' not in st.session_state:
            st.session_state.current_detections = []
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = None

        start = st.button("🟢 Start Camera")
        stop = st.button("🔴 Stop Camera")
        capture = st.button("📸 Capture Frame")

        # Start camera
        if start:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("❌ Unable to access the webcam.")
                st.session_state.cap = None

        # Stop camera
        if stop and st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
            video_placeholder.empty()
            live_results.empty()
            st.session_state.current_detections = []
            st.session_state.last_frame = None

        # Continuous detection loop
        if st.session_state.cap:
            ret, frame = st.session_state.cap.read()
            if not ret or frame is None:
                st.warning("⚠️ Failed to read from webcam. Please check your camera.")
            else:
                annotated, detections = detect_and_classify(frame, yolo_model, classification_model, class_names)
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.session_state.current_detections = detections
                st.session_state.last_frame = frame

        # Capture frame safely
        if capture:
            if st.session_state.last_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"captured_{timestamp}.jpg"
                cv2.imwrite(save_path, st.session_state.last_frame)
                st.success(f"✅ Frame captured and saved as `{save_path}`")
            else:
                st.warning("⚠️ No frame available to capture. Start the webcam first.")

        # Live detection results
        if st.session_state.current_detections:
            live_results.subheader("Live Detection Results")
            for i, det in enumerate(st.session_state.current_detections, 1):
                status = "♻ E-WASTE" if 'e_waste' in det['class'].lower() else "🗑 NON-E-WASTE"
                color = "green" if 'e_waste' in det['class'].lower() else "orange"
                live_results.markdown(f"*Object {i}:* :{color}[{status}] - {det['class']} ({det['confidence']:.2f})")
        else:
            live_results.info("No objects currently detected.")

    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#666;">
    🔋 E-Waste Detection System | Production Ready | Hugging Face Models
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
