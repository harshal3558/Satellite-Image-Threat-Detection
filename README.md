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

## 📊 Model Evaluation & Performance Metrics & Their Operational Importance

Evaluating object detection models on high-resolution satellite imagery presents unique challenges due to extreme scale variations (objects ranging from 10 to 500 pixels), high background clutter (forests, desert terrain, urban shadows), and severe class imbalance. 

The **SITP pipeline** evaluates model performance on the validation split (**1,926 image chips** containing **148,178 ground-truth target instances**) using standard computer vision metrics alongside custom pipeline logging (`src/SITP/components/model_monitoring.py`).

---

### 1. Global Performance Metrics Summary

Below are the key benchmark metrics obtained after fine-tuning **YOLOv8m** for 50 epochs on the xView dataset:

| Metric | Formula / Standard | Empirical Value | Operational Role & Importance in Project |
|---|---|---|---|
| **Precision (Box P)** | $\frac{TP}{TP + FP}$ | **32.7%** (`0.327`) | **Minimizes False Alarms:** Ensures military/defense operators are not overwhelmed by false detections across large-scale satellite surveys. |
| **Recall (Box R)** | $\frac{TP}{TP + FN}$ | **25.5%** (`0.255`) | **Minimizes Missed Threats:** Measures the pipeline's ability to detect high-value target assets (planes, ships, vehicles) in complex terrain. |
| **mAP50** | Mean AP at $\text{IoU} = 0.50$ | **20.9%** (`0.209`) | **Primary Detection Benchmark:** Overall object recognition accuracy across all 60+ classes at standard 50% overlap. |
| **mAP50-95** | Mean AP over $\text{IoU} \in [0.50, 0.95]$ | **10.9%** (`0.109`) | **Localization Precision:** Evaluates exact bounding box alignment, critical for geospatial positioning and intelligence mapping. |

> [!NOTE]
> **Benchmarking Context:** xView is widely recognized as one of the most challenging overhead satellite benchmarks. Objects are extremely small (down to $10 \times 10$ pixels), with intense class imbalance. An mAP50 of **~21%** aligns with state-of-the-art performance on this benchmark subset.

---

### 2. In-Depth Metric Analysis & Importance in Satellite Threat Detection

#### 🎯 A. Precision ($\text{Box P}$) — *False Alarm Rate Control*
* **Definition:** The proportion of predicted threat bounding boxes that truly correspond to ground-truth threat objects.
* **Importance in the Project:**
  * **Analyst Workload Management:** When scanning gigapixel satellite images, a low-precision model generates thousands of false alarms (e.g., mistaking building shadows, container stacks, or terrain patterns for threats). High precision ensures analyst time is focused on genuine targets.
  * **System Trust:** In defense intelligence, frequent false alarms lead to alarm fatigue and reduced user trust in automated AI systems.

#### 🔍 B. Recall ($\text{Box R}$) — *Missed Threat Prevention*
* **Definition:** The proportion of actual ground-truth threat objects correctly detected by the system.
* **Importance in the Project:**
  * **Strategic Risk Reduction:** Missing a critical threat asset (e.g., an undetected stealth aircraft, mobile missile launcher, or warship) has severe tactical consequences. 
  * **Confidence Tuning:** In high-stakes intelligence operations, operators can intentionally lower the confidence threshold (`conf`) in the web application (`application.py`) to boost Recall, accepting more candidate detections to ensure zero missed threats.

#### ⚖️ C. Precision-Recall Trade-off & Operational Slider Controls
* **Definition:** The inverse relationship between Precision and Recall as the decision confidence threshold ($\tau_{\text{conf}}$) is adjusted.
* **Importance in the Project:**
  * The Flask web interface (`application.py`) exposes interactive control sliders for **Confidence Threshold** and **IoU Threshold**.
  * **Surveillance Scenario (High Sensitivity):** Lower `conf` (e.g., `0.15`) $\rightarrow$ Higher Recall $\rightarrow$ Scans every potential threat.
  * **Target Verification (High Specificity):** Higher `conf` (e.g., `0.50`) $\rightarrow$ Higher Precision $\rightarrow$ Reports only confirmed, high-certainty targets.

#### 📐 D. Intersection over Union (IoU) & Tiled Deduplication
* **Definition:** Spatial overlap ratio between predicted ($\mathbf{B}_p$) and ground truth ($\mathbf{B}_g$) boxes:
  $$\text{IoU} = \frac{\text{Area}(\mathbf{B}_p \cap \mathbf{B}_g)}{\text{Area}(\mathbf{B}_p \cup \mathbf{B}_g)}$$
* **Importance in the Project:**
  * **Tile Border Deduplication:** The pipeline processes large GeoTIFF imagery using sliding windows ($512 \times 512$ tiles with 100px overlap). Objects lying on tile boundaries are detected multiple times.
  * **Batched Non-Maximum Suppression (NMS):** Using `torchvision.ops.batched_nms` with an IoU threshold (`iou=0.45` default), the system merges overlapping bounding boxes across tile borders into single global-coordinate detections.

#### 📈 E. Mean Average Precision (mAP50 & mAP50-95)
* **mAP50 (Detection Completeness):** Measures average precision across all recall levels at a standard 50% IoU match score. Gives an overall rating of object detection capability across diverse target types.
* **mAP50-95 (Localization Quality):** Averaged over 10 IoU thresholds ($0.50, 0.55, \dots, 0.95$). Crucial for precise geospatial coordinates: a box with 50% IoU confirms presence, but 90% IoU provides exact coordinates needed for geolocation mapping and targeted tracking.

#### 🏷️ F. Per-Class Average Precision (Class Imbalance Mitigation)
* **Definition:** Class-specific AP metrics evaluated independently for each xView object category (`ModelMonitoring.inspect_per_class_performance`).
* **Importance in the Project:**
  * Global mAP can mask poor performance on rare strategic targets if common classes (like passenger cars) dominate the dataset.
  * Per-class tracking allows monitoring specific high-value assets independently.

---

### 3. Representative Class-Specific Performance

| Threat / Asset Class | mAP50 | mAP50-95 | Tactical Relevance & Characteristic |
|---|---|---|---|
| ✈️ **Cargo Plane** | **90.5%** | **56.9%** | Large strategic asset with distinct structural geometry and runway background contrast. |
| 🚗 **Passenger Car** | **87.3%** | **48.3%** | High training density; distinct vehicle silhouette on paved roads. |
| 🚢 **Container Ship** | **72.2%** | **39.0%** | Distinct maritime signatures; clear water background separation. |
| 🚙 **Small Car** | **62.9%** | **23.0%** | Highly dense in urban parking lots; prone to occlusion by nearby buildings. |
| 🏢 **Building** | **61.0%** | **31.1%** | Fixed infrastructure; large spatial variation requiring precise box boundaries. |
| 🚁 **Helicopter** | **25.6%** | **17.6%** | Low sample count; rotor shadows and camouflage make detection challenging. |

---

### 4. Training Loss Metrics & Convergence Tracking

During model training (`ModelTrainer`), YOLOv8 tracks three primary loss components:

1. **Box Loss ($\mathcal{L}_{\text{box}}$ - Complete IoU Loss):** Optimizes bounding box location, size, and aspect ratio alignment for satellite targets.
2. **Class Loss ($\mathcal{L}_{\text{cls}}$ - BCE Loss):** Measures classification accuracy across threat classes.
3. **DFL Loss ($\mathcal{L}_{\text{dfl}}$ - Distribution Focal Loss):** Refines boundary regression for tiny or occluded objects in satellite imagery.

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