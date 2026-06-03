# Satellite Image Threat Detection (SITP)

An end-to-end Machine Learning pipeline and web application designed to ingest the **xView dataset**, preprocess large-format GeoTIFF imagery into training chips, fine-tune a **YOLO11** model, and run tiled object detection to identify threat classes in satellite imagery.

---

## 📡 System Architecture

The pipeline consists of four major stages, structured cleanly under the `src/SITP` package:

```
[ Data Ingestion ] ──> [ Data Transformation ] ──> [ Model Training ] ──> [ Web App Inference ]
 (Split images)          (Sliding-window chips)     (YOLO11 fine-tuning)   (Tiled NMS & visualization)
```

1. **Data Ingestion** (`data_ingestion.py`): Parses the raw xView GeoJSON labels, verifies images on disk, and splits the dataset at the *image* level (80/20 train/validation split) to prevent data leakage.
2. **Data Transformation** (`data_transformation.py`): Uses sliding-window tiling (512×512 pixels, stride 364) to slice large satellite images. Applies training-time augmentations (CLAHE, brightness, flips) via Albumentations.
3. **Model Training** (`model_trainer.py`): Ingests the chipped dataset and fine-tunes a `yolo11m.pt` model, tracking mAP50 and mAP50-95 metrics.
4. **Web App Interface** (`application.py`): Allows users to upload a GeoTIFF image, choose custom confidence & IoU thresholds, run tiled model predictions, deduplicate boxes with batched non-maximum suppression (NMS), and view detailed detection logs.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- (Optional) Docker

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/harshal3558/Satellite-Image-Threat-Detection.git
cd Satellite-Image-Threat-Detection

# Create a virtual environment
python -m venv senv
senv\Scripts\activate  # On Windows
source senv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Placement
Your local data folder structure should look like this:
```
data/
├── train_images/
│   └── train_images/          # Contains GeoTIFF (.tif) files
└── train_labels/
    └── xView_train.geojson    # Annotations file
```

---

## 🚀 Running the Pipeline

### Step 1: Model Training
To execute the end-to-end training pipeline (Ingestion ➔ Transformation ➔ YOLO11 Training):
```bash
python main.py
```
This will:
- Partition your images into train/validation sets.
- Generate YOLO chips inside the `xview_yolo/` directory.
- Run YOLO11 training and save checkpoints under `xview_yolo/satellite_detector/weights/best.pt`.

### Step 2: Launch the Web App
After training is complete (or after obtaining a checkpoint):
```bash
python application.py
```
Open `http://localhost:5000` in your web browser. You will see a dark military-themed HUD interface where you can upload `.tif` images and run threat scans.

---

## 🐳 Running with Docker

You can containerize the web application using the provided Docker configuration.

### 1. Build the Docker Image
```bash
docker build -t satellite-threat-detection .
```

### 2. Run the Container
```bash
docker run -p 5000:5000 satellite-threat-detection
```
Navigate to `http://localhost:5000`.

---

## 📂 Project Directory Structure

```
├── data/                      # Raw xView dataset directory
├── src/
│   └── SITP/
│       ├── components/        # Pipeline modular stages
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_monitoring.py
│       ├── pipelines/         # Pipelines runner definitions
│       │   ├── training_pipeline.py
│       │   └── prediction_pipeline.py
│       ├── exception.py       # Custom exception handler
│       ├── logger.py          # Logger module
│       └── utils.py           # Helper utilities
├── templates/
│   └── index.html             # Web app user interface
├── static/
│   └── css/
│       └── style.css          # Space/military tactical styles
├── application.py             # Flask web server
├── main.py                    # Training entry point
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python library dependencies
└── setup.py                   # Packaging configuration
```