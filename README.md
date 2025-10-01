🔋 E-Waste Detection System

This is a real-time E-Waste Detection System built with Streamlit, YOLOv8 object detection, and a TensorFlow classification model. It can detect and classify e-waste items from uploaded images or live webcam feed.

The system is optimized for continuous real-time detection and ignores people in the frame. Detected objects are displayed with bounding boxes, classification labels, and confidence scores.

Features

✅ Upload Image Detection: Detect and classify e-waste items in any uploaded image.

✅ Continuous Webcam Detection: Real-time detection using your webcam.

✅ Person Detection Ignored: The system does not classify or highlight humans.

✅ Classification in Bounding Boxes: Shows predicted class + confidence on detected objects.

✅ Live Detection Panel: See a live summary of detected objects and their types.

✅ Capture Frame: Save any frame from the webcam feed with predictions.

✅ Hugging Face Integration: Models and class files are downloaded directly from Hugging Face.



1. Install Python

Make sure you have Python 3.10–3.13 installed. You can check your version with:

python --version

2. Clone the Repository

If your project is in GitHub:

git clone https://github.com/yourusername/e-waste-detector.git
cd e-waste-detector


If it’s a local folder, just navigate to it:

cd path/to/your/project

3. Create a Virtual Environment (Recommended)

python -m venv venv

Activate it:

Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate

4. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

5.Run

streamlit run app.py

http://localhost:8501



