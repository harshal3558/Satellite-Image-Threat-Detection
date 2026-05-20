import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.SITD.pipelines.prediction_pipeline import PredictionPipeline
from src.SITD.logger import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Default model path
DEFAULT_MODEL_PATH = r"c:\Users\harsh\OneDrive\Desktop\Satellite image\yolov8n.pt"
if not os.path.exists(DEFAULT_MODEL_PATH):
    DEFAULT_MODEL_PATH = "yolov8n.pt"  # Will auto-download via ultralytics if missing

# Class names mapping list (same as our xView class mapping)
XVIEW_CLASSES = [
    'Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter',
    'Passenger Vehicle', 'Small Car', 'Bus', 'Pickup Truck',
    'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box',
    'Truck Tractor', 'Trailer', 'Truck w/Flatbed',
    'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle',
    'Passenger Car', 'Cargo Car', 'Flat Car', 'Tank car',
    'Locomotive', 'Maritime Vessel', 'Motorboat',
    'Sailboat', 'Tugboat', 'Barge', 'Fishing Vessel',
    'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker',
    'Engineering Vehicle', 'Tower crane', 'Container Crane',
    'Reach Stacker', 'Straddle Carrier', 'Mobile Crane',
    'Dump Truck', 'Haul Truck', 'Scraper/Tractor',
    'Front loader/Bulldozer', 'Excavator', 'Cement Mixer',
    'Ground Grader', 'Hut/Tent', 'Shed', 'Building',
    'Aircraft Hangar', 'Damaged Building', 'Facility',
    'Construction Site', 'Vehicle Lot', 'Helipad',
    'Storage Tank', 'Shipping container lot',
    'Shipping Container', 'Pylon', 'Tower', 
    'Class 60 Placeholder', 'Class 61 Placeholder'
]

def get_threat_level(count: int) -> str:
    if count == 0:
        return "CLEAR"
    elif count < 5:
        return "LOW"
    elif count < 15:
        return "MEDIUM"
    else:
        return "HIGH"

def draw_predictions(image_path: str, predictions: list) -> str:
    """
    Draws predictions (bounding boxes and labels) on the image and returns base64 string.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return ""
    
    # Calculate scale to avoid rendering huge 3000px+ images on screen
    max_dim = 1200
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Draw boxes
    for pred in predictions:
        box = pred['box']
        conf = pred['confidence']
        class_id = pred['class_id']
        
        # Scale bounding boxes
        x1 = int(box[0] * scale)
        y1 = int(box[1] * scale)
        x2 = int(box[2] * scale)
        y2 = int(box[3] * scale)

        class_name = XVIEW_CLASSES[class_id] if class_id < len(XVIEW_CLASSES) else f"Class_{class_id}"
        label = f"{class_name} {conf:.2f}"

        # Draw Rectangle (Cyan color)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 242, 0), 2)
        
        # Put Text
        cv2.putText(
            img, label, (x1, max(y1 - 10, 15)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 242, 0), 2
        )

    # Encode to base64 jpeg
    _, buffer = cv2.imencode('.jpg', img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    logging.info("Flask detect route hit.")
    try:
        # Check if file part is in request
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part in request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        logging.info(f"File uploaded successfully to: {upload_path}")

        # Initialize prediction pipeline
        pipeline = PredictionPipeline(model_path=DEFAULT_MODEL_PATH)
        
        # Run sliding window prediction
        predictions = pipeline.predict_large_image(
            image_path=upload_path,
            tile_size=800,
            overlap=100,
            iou_threshold=0.45,
            conf_threshold=0.25
        )

        # Draw predictions on image and convert to base64
        base64_image = draw_predictions(upload_path, predictions)

        # Cleanup uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)

        threat_level = get_threat_level(len(predictions))

        return jsonify({
            "status": "success",
            "filename": filename,
            "detections": predictions,
            "threat_level": threat_level,
            "image": base64_image
        })

    except Exception as e:
        logging.error(f"Error during flask detection endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
