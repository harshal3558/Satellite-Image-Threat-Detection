# Satellite Image Threat Detection (SITP)

An end-to-end Machine Learning pipeline and web application designed to ingest the **xView dataset**, preprocess large-format GeoTIFF imagery into training chips, fine-tune a **YOLOv8** model, and run tiled object detection to identify threat classes in satellite imagery.

---

## 📡 System Architecture

The pipeline consists of four major stages, structured cleanly under the `src/SITP` package:

```
[ Data Ingestion ] ──> [ Data Transformation ] ──> [ Model Training ] ──> [ Web App Inference ]
 (Split images)          (Sliding-window chips)     (YOLOv8 fine-tuning)   (Tiled NMS & visualization)
```

1. **Data Ingestion** (`data_ingestion.py`): Parses the raw xView GeoJSON labels, verifies images on disk, and splits the dataset at the *image* level (80/20 train/validation split) to prevent data leakage.
2. **Data Transformation** (`data_transformation.py`): Uses sliding-window tiling (512×512 pixels, stride 364) to slice large satellite images. Applies training-time augmentations (CLAHE, brightness, flips) via Albumentations.
3. **Model Training** (`model_trainer.py`): Ingests the chipped dataset and fine-tunes a `yolov8m.pt` model, tracking mAP50 and mAP50-95 metrics.
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
To execute the end-to-end training pipeline (Ingestion ➔ Transformation ➔ YOLOv8 Training):
```bash
python main.py
```
This will:
- Partition your images into train/validation sets.
- Generate YOLO chips inside the `xview_yolo/` directory.
- Run YOLOv8 training and save checkpoints under `xview_yolo/satellite_detector/weights/best.pt`.

### Step 2: Launch the Web App
After training is complete (or after obtaining a checkpoint):
```bash
python application.py
```
Open `http://localhost:5000` in your web browser. You will see a dark military-themed HUD interface where you can upload `.tif` images and run threat scans.

---

## 📊 Model Evaluation & Performance Metrics

The model is evaluated on the validation split (1,926 images containing 148,178 instances) using standard object detection metrics. Below are the actual values obtained after training `yolov8m.pt` for 50 epochs:

### Global Validation Metrics (All Classes Combined)

| Evaluation Metric | Description | Value |
|-------------------|-------------|-------|
| **Precision (Box P)** | Percentage of predicted bounding boxes that correctly identify a threat class. | **32.7%** (0.327) |
| **Recall (Box R)** | Percentage of actual threat objects correctly detected by the model. | **25.5%** (0.255) |
| **mAP50** | Mean Average Precision at an Intersection over Union (IoU) threshold of 0.50. | **20.9%** (0.209) |
| **mAP50-95** | Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 (measures localization quality). | **10.9%** (0.109) |

> [!NOTE]
> The xView dataset is a highly imbalanced, complex satellite imagery dataset featuring extremely small objects (e.g. cars, trailers, boats) and high background clutter. Achieving ~21% mAP50 is strong and aligns with state-of-the-art performance benchmarks on this subset.

### Representative Class-Specific Performance

The pipeline evaluates each of the classes individually. Here is a subset of representative classes:

*   **Cargo Plane**: mAP50 = `90.5%` | mAP50-95 = `56.9%`
*   **Passenger Car**: mAP50 = `87.3%` | mAP50-95 = `48.3%`
*   **Small Car**: mAP50 = `62.9%` | mAP50-95 = `23.0%`
*   **Building**: mAP50 = `61.0%` | mAP50-95 = `31.1%`
*   **Container Ship**: mAP50 = `72.2%` | mAP50-95 = `39.0%`
*   **Helicopter**: mAP50 = `25.6%` | mAP50-95 = `17.6%`

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
│   ├── train_images/
│   └── train_labels/
│       └── xView_train.geojson
├── src/
│   └── SITP/
│       ├── components/        # Pipeline modular stages
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_monitoring.py
│       ├── pipelines/         # Pipeline runner definitions
│       │   ├── training_pipeline.py
│       │   └── prediction_pipeline.py
│       ├── exception.py       # Custom exception handler
│       ├── logger.py          # Logger module
│       └── utils.py           # Helper utilities
├── notebooks/                 # Jupyter notebooks for EDA & experimentation
│   ├── satellite-image-threat-detection.ipynb
│   ├── satellite-image-threat-detection-enhanced.ipynb
│   ├── improved_satellite_image_threat_detection.ipynb
│   ├── chatgpt-update.ipynb
│   └── add_nms_to_notebook.py
├── artifacts/                 # Intermediate pipeline outputs
│   └── raw_annotations.csv    # Parsed xView annotations
├── xview_yolo/                # YOLO-formatted dataset & training output
│   └── satellite_detector/
│       └── weights/
│           └── best.pt        # Best YOLOv8 checkpoint (post-training)
├── logs/                      # Pipeline execution logs
├── uploads/                   # Temporary storage for web app uploads
├── templates/
│   └── index.html             # Web app user interface
├── static/
│   └── css/
│       └── style.css          # Military/tactical HUD styles
├── best.pt                    # Exported model checkpoint (root-level copy)
├── application.py             # Flask web server entry point
├── main.py                    # Training pipeline entry point
├── template.py                # Project scaffolding script
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python library dependencies
└── setup.py                   # Package configuration
```