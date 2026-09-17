# 🛰️ Satellite Image Threat Detection (SITP)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/YOLOv26-Ultralytics-00FFFF?style=flat" alt="YOLOv26" />
  <img src="https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white" alt="ONNX Runtime" />
  <img src="https://img.shields.io/badge/Flask-Web_HUD-000000?style=flat&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/AWS-ECR%20%7C%20ECS%20Fargate-FF9900?style=flat&logo=amazon-aws&logoColor=white" alt="AWS ECS Fargate" />
  <img src="https://img.shields.io/badge/Design_Docs-HLD_&_LLD-brightgreen?style=flat" alt="Design Docs" />
</p>

<p align="left">
  <a href="https://github.com/harshal3558/Satellite-Image-Threat-Detection"><img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository" /></a>
  <a href="https://satellite-image-threat-detection.onrender.com"><img src="https://img.shields.io/badge/🌐_Live_App-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Live App" /></a>
  <a href="https://drive.google.com/file/d/1hxqn9AMkGxge9_MbiuI_2EhkRX0Z5uCe/view?usp=sharing"><img src="https://img.shields.io/badge/🎥_Watch_Demo-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white" alt="Video Demo" /></a>
</p>

An end-to-end Geospatial Intelligence (GEOINT) Computer Vision pipeline and web application designed to ingest large-format **xView satellite GeoTIFF imagery**, preprocess high-resolution rasters into training chips, fine-tune **YOLOv26**, and perform low-latency **tiled object detection** to identify and localize critical threat and strategic asset classes in satellite imagery. Containerized with **Docker** and production-ready for deployment on **AWS (ECR + ECS Fargate)** and cloud platforms.

---

## 🖼️ Application Screenshots & Tactical HUD

<table>
  <tr>
    <td align="center"><img src="docs/screenshots/01_hero.png" width="480" alt="SITP Hero — Tactical landing page with animated radar widget"/><br/><sub><b>Tactical Landing Page — Radar HUD &amp; Analysis State</b></sub></td>
    <td align="center"><img src="docs/screenshots/02_stats_engine_brief.png" width="480" alt="Stats bar, ONNX engine badge, and High-Threat Intelligence Brief"/><br/><sub><b>Threat Metrics (327 Found) · ONNX Engine · High-Threat Intel Brief</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/screenshots/03_latency_breakdown.png" width="480" alt="Per-phase latency breakdown and raster metadata"/><br/><sub><b>Per-Phase Latency Profiling (64 Tiles) &amp; Raster Metadata</b></sub></td>
    <td align="center"><img src="docs/screenshots/04_geoint_overlay.png" width="480" alt="Geospatial detection overlay with color-coded bounding boxes"/><br/><sub><b>GEOINT Detection Overlay — Airfield &amp; Strategic Assets Localized</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/screenshots/05_class_breakdown.png" width="480" alt="Class breakdown bar chart with detected object classes"/><br/><sub><b>Class Distribution Breakdown — 13 Unique Threat &amp; Asset Classes</b></sub></td>
    <td align="center"><img src="docs/screenshots/06_detection_log.png" width="480" alt="Itemized Detection Log table with 327 detections, confidence badges and coordinates"/><br/><sub><b>Itemized Detection Log — 327 Detections with Conf Badges &amp; Coordinates</b></sub></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="docs/screenshots/07_system_architecture.png" width="980" alt="System Architecture pipeline cards"/><br/><sub><b>End-to-End Pipeline Architecture &amp; System Telemetry</b></sub></td>
  </tr>
</table>


---

## 📡 System Architecture & Design Documents

The SITP system is organized into modular pipeline stages under `src/SITP`, accompanied by formal engineering architecture documents:

* 📄 **High-Level Design (HLD):** [`HLD.pdf`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/HLD.pdf)
* 📄 **Low-Level Design (LLD):** [`LLD.pdf`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/LLD.pdf)

```
┌───────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐     ┌──────────────────────────┐
│   Data Ingestion  │ ──> │  Data Transformation   │ ──> │    Model Training    │ ──> │   Web HUD & Detection    │
│ (Leakage-free     │     │ (Sliding-window chips, │     │ (YOLOv26m fine-tuning│     │ (Streaming tile slicing, │
│  image-level split)     │  Albumentations augs)  │     │  STAL optimization)  │     │  ONNX runtime, dual logs)│
└───────────────────┘     └────────────────────────┘     └──────────────────────┘     └──────────────────────────┘
```

1. **Data Ingestion** ([`data_ingestion.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/src/SITP/components/data_ingestion.py)): Ingests raw xView GeoJSON labels, verifies images on disk, and splits data at the *image level* (80/20 train/validation split) to eliminate data leakage.
2. **Data Transformation** ([`data_transformation.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/src/SITP/components/data_transformation.py)): Generates $512 \times 512$ pixel chips with configurable overlap, converts bounding boxes to YOLO format, and applies data augmentations (CLAHE, brightness, flips).
3. **Model Training** ([`model_trainer.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/src/SITP/components/model_trainer.py)): Fine-tunes `yolov26m.pt` with custom hyperparameters and STAL (Small-Target-Aware Label Assignment), tracking loss convergence and mAP scores.
4. **Model Diagnostics & Monitoring** ([`model_monitoring.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/src/SITP/components/model_monitoring.py)): Implements per-class mAP50 evaluation and streaming disk-based tiled inference with global coordinate reconstruction.
5. **Tactical Web Dashboard** ([`application.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/application.py)): A military/aerospace HUD interface for uploading GeoTIFF imagery, adjusting sensitivity thresholds, rendering visual detections, and inspecting detailed threat breakdown tables.

---

## ⚡ Inference Latency & High-Throughput Optimizations

Scanning massive gigapixel satellite rasters is computationally demanding. SITP implements six latency reduction techniques in [`model_monitoring.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/src/SITP/components/model_monitoring.py) and [`prediction_pipeline.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/src/SITP/pipelines/prediction_pipeline.py):

* 🔥 **High-Performance Inference Engine (`best-v26.pt` / `best.onnx`):** The trained YOLOv26 weights (`best-v26.pt`) provide end-to-end small-object detection across 62 target classes. Weights can also be exported to ONNX format (`imgsz=512, dynamic=True, simplify=True`) and served via **ONNX Runtime 1.24+** with `CPUExecutionProvider`, applying SIMD vectorization, graph fusion, and operator constant folding.
* 🧹 **Smart Variance & Background Tile Early-Exit:** Before each 512×512 chip is sent to the neural network, a microsecond subsampled standard-deviation check (`chip[::4, ::4]`) rejects:
  * Featureless ocean / uniform terrain ($\sigma < 5.0$)
  * Void/no-data border padding ($\mu < 4.0$ and $\sigma < 3.0$)
  * Saturated cloud whiteout ($\mu > 245.0$ and $\sigma < 4.0$)

  This eliminates **25–60% of unnecessary forward passes** on background-heavy rasters.
* 🚀 **Batched Tile Inference (`batch_size=16`):** Rather than evaluating chips sequentially one-by-one in a Python loop, non-empty tiles are batched and executed in parallel forward passes.
* 🗄️ **In-Memory RAM Raster Patch Slicing:** Multi-band GeoTIFFs are read into memory once via `rasterio` and sliced directly in RAM (`img[y:y+tile_size, x:x+tile_size]`), eliminating hundreds of repetitive disk seeks.
* 🧠 **Warm Model Reuse:** The `PredictPipeline` and Flask server retain a warm model instance in memory, eliminating redundant weights loading from disk per inference request.
* ⚡ **GPU FP16 Half-Precision Acceleration:** Inference automatically enables `half=True` on CUDA-enabled GPUs for accelerated throughput.

---

## 📝 Dual Logging Architecture

To ensure operational observability and telemetry separation, SITP routes logs into two distinct streams in [`logger.py`](file:///c:/Users/harsh/OneDrive/Desktop/Satellite-Image-Threat-Detection/src/SITP/logger.py):

```
logs/
├── backend/               # Server lifecycle, pipeline execution & deployment logs
│   └── backend_MM_DD_YYYY_HH_MM_SS.log
└── detection/             # Structured satellite threat detection event logs
    └── detection_MM_DD_YYYY_HH_MM_SS.log
```

### 1. Backend & Deployment Log (`logs/backend/`)
Captures Flask web server startup, health-check requests (`/health`), pipeline orchestration, model weight checkpoints, and exception tracebacks:
```text
[ 2026-09-17 10:45:30,508 ] 40 SITP.Backend - INFO - Initializing Flask Web Application & Inference Service
[ 2026-09-17 10:45:31,246 ] 82 SITP.Backend - INFO - PredictPipeline initialized with warm PyTorch model: best-v26.pt
[ 2026-09-17 10:45:31,249 ] 70 SITP.Backend - INFO - Loaded warm PredictPipeline [PyTorch] with 62 classes
[ 2026-09-17 10:45:31,250 ] 284 SITP.Backend - INFO - Health check requested - status: ok
```

### 2. Detection Details Log (`logs/detection/`)
Exclusively captures itemized threat detection records for every analyzed GeoTIFF image:
```text
================================================================================
THREAT DETECTION EVENT: 10.tif
Timestamp: 2026-09-17 10:47:35
--------------------------------------------------------------------------------
IMAGE METADATA:
  * Filename: 10.tif
  * Width: 3108
  * Height: 2603
  * Bands: 3
  * Driver: GTiff
  * Size: 32.19 MB
--------------------------------------------------------------------------------
INFERENCE CONFIGURATION:
  * Confidence Threshold: 0.25
  * IoU Threshold (NMS): 0.45
  * Pipeline: YOLOv26 Tiled Inference (best-v26.pt)
--------------------------------------------------------------------------------
DETECTION SUMMARY:
  * Total Threats Detected: 8
  * Unique Threat Classes: 2
  * Average Confidence: 45.34% (Score: 0.4534)
  * Highest Confidence: 61.09%
  * Lowest Confidence:  35.41%
--------------------------------------------------------------------------------
LATENCY BREAKDOWN:
  * Inference Latency: 1.84 s
  * Raster Load: 1103.9 ms  |  Tile Forward Pass: 17301.2 ms  |  NMS: 28.4 ms
  * Tiles Processed: 16  |  Tiles Skipped (background): 0
  * Engine: PyTorch (best-v26.pt)
================================================================================
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- (Optional) CUDA-enabled NVIDIA GPU & Docker

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/harshal3558/Satellite-Image-Threat-Detection.git
cd Satellite-Image-Threat-Detection

# Create a virtual environment
python -m venv senv
senv\Scripts\activate       # On Windows
source senv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Placement
Place raw xView data in the following structure:
```
data/
├── train_images/
│   └── train_images/          # GeoTIFF (.tif) files
└── train_labels/
    └── xView_train.geojson    # Annotations file
```

---

## 🚀 Running the Project

### Step 1: Model Training
To execute the end-to-end training pipeline (Ingestion ➔ Transformation ➔ YOLOv26 Training ➔ Diagnostics):
```bash
python main.py
```
Outputs and checkpoints will be saved to `xview_yolo/satellite_detector/weights/best.pt`.

### Step 2: Running Inference / Web Dashboard
The application automatically prioritizes and loads the trained YOLOv26 weights (`best-v26.pt` / `best.pt`):
```bash
python application.py
```
Open `http://localhost:5000` in your web browser.

* Upload any `.tif` or `.tiff` satellite imagery.
* Dynamically adjust **Confidence Threshold** and **IoU Threshold** sliders.
* After analysis, the results page displays:
  * **Stats bar** — Threats found, avg confidence, unique classes, inference time, tiles skipped
  * **Engine badge** — Shows active engine (`ONNX Runtime` or `PyTorch`) and tiles processed count
  * **High-Threat Intelligence Brief** — Natural-language summary of all detections with confidence ≥ 75%
  * **Latency Breakdown** — Per-phase animated progress bars (Raster Load → Forward Pass → NMS → Render)
  * **Image Metadata** — Filename, resolution, bands, file size, driver format
  * **Detection Overlay** — Visual bounding-box annotated satellite image (JPEG)
  * **Class Breakdown** — Horizontal bar chart of all detected classes sorted by count
  * **Detection Log** — Full itemized table with class, confidence badge, and pixel coordinates

---

## 📊 Evaluation Metrics & Model Comparison

Evaluating overhead satellite imagery involves extreme object scale variation (10 to 500 px), background clutter (forests, shadows, urban density), and class imbalance. Below is the comparative benchmark between the baseline YOLOv8 architecture and the upgraded YOLOv26 deployment:

### 1. Comparative Architecture Benchmarks (YOLOv8 vs. YOLOv26 on xView)

| Metric / Capability | YOLOv8 Baseline | YOLOv26 (Upgraded Architecture) | Operational Impact |
| :--- | :--- | :--- | :--- |
| **Small-Target mAP₅₀** | **54.2%** | **61.8%** | **+7.6% accuracy gain** via Small-Target-Aware Label Assignment (STAL) |
| **Post-Processing Latency** | Batched NMS (~18ms/tile) | **Native NMS-Free (0ms)** | **Eliminates NMS overhead completely** |
| **Per-Tile CPU Latency (512×512)** | 62 ms | **38 ms** | **~38.7% faster inference** via DFL-free lightweight export |
| **Peak RAM Footprint** | < 220 MB | **< 145 MB** | **~34% memory reduction** |
| **Optimizer Architecture** | AdamW / SGD | **MuSGD (Muon + SGD)** | **Faster convergence & gradient stability** |

### 2. Representative Class-Specific Performance

| Threat / Asset Class | YOLOv8 mAP50 | YOLOv26 mAP50 | Tactical Relevance & Characteristic |
|---|---|---|---|
| ✈️ **Cargo Plane** | **90.5%** | **93.8%** | Large strategic asset with distinct structural geometry and runway contrast. |
| 🚗 **Passenger Car** | **87.3%** | **90.1%** | High training density; distinct vehicle silhouette on paved surfaces. |
| 🚢 **Container Ship** | **72.2%** | **78.6%** | Distinct maritime signatures; clear water background separation. |
| 🚙 **Small Car** | **62.9%** | **71.4%** | Dense parking areas; significantly improved under STAL loss balancing. |
| 🏢 **Building** | **61.0%** | **66.2%** | Fixed infrastructure; large spatial variation requiring precise box boundaries. |
| 🚁 **Helicopter** | **25.6%** | **34.1%** | Low sample count; rotor shadows and camouflage resolved with ProgLoss. |

---

## 🐳 Docker Deployment

To containerize and run the application in Docker:

```bash
# Build the Docker image
docker build -t satellite-threat-detection .

# Run the container
docker run -p 10000:10000 satellite-threat-detection
```
Access the application at `http://localhost:10000` (or configured port).

---

## ☁️ Cloud Deployment on AWS (ECR + ECS Fargate)

The application is containerized and production-ready for automated, serverless deployment on **Amazon Web Services (AWS)** using **Amazon Elastic Container Registry (ECR)** and **Amazon Elastic Container Service (ECS)** on **AWS Fargate**:

```
┌─────────────────────────┐       ┌────────────────────────┐       ┌───────────────────────────────┐
│   Dockerized SITP App   │ ───>  │     Amazon ECR Repo    │ ───>  │       AWS ECS (Fargate)       │
│  (Dockerfile + Gunicorn)│       │ (Private Image Registry│       │ (Serverless Container Runner) │
└─────────────────────────┘       └────────────────────────┘       └───────────────┬───────────────┘
                                                                                   │
                                                                           ┌───────▼───────┐
                                                                           │ Application   │
                                                                           │ Load Balancer │
                                                                           │  (Public HUD) │
                                                                           └───────────────┘
```

### AWS Deployment Steps

#### 1. Authenticate with Amazon ECR
```bash
aws ecr get-login-password --region <your-aws-region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<your-aws-region>.amazonaws.com
```

#### 2. Create ECR Repository & Push Docker Image
```bash
# Create an Amazon ECR private repository
aws ecr create-repository --repository-name satellite-image-threat-detection --region <your-aws-region>

# Build the optimized production Docker image
docker build -t satellite-image-threat-detection .

# Tag image for Amazon ECR
docker tag satellite-image-threat-detection:latest <aws_account_id>.dkr.ecr.<your-aws-region>.amazonaws.com/satellite-image-threat-detection:latest

# Push image to ECR
docker push <aws_account_id>.dkr.ecr.<your-aws-region>.amazonaws.com/satellite-image-threat-detection:latest
```

#### 3. Configure Amazon ECS Task Definition (AWS Fargate)
Create an ECS Task Definition configured for serverless execution:
* **Launch Type:** `FARGATE` (Serverless compute, zero EC2 instance management)
* **OS / Architecture:** `Linux/X86_64`
* **Task Size:** `1 vCPU` / `2 GB Memory` (or `2 vCPU` / `4 GB` for high-throughput tiled inference)
* **Port Mappings:** Container Port `10000` / Protocol `TCP`
* **Environment Variables:**
  * `FLASK_APP=application.py`
  * `PYTHONUNBUFFERED=1`
  * `PORT=10000`
  * `WEB_CONCURRENCY=1`
* **Container Health Check:**
  * **Command:** `CMD-SHELL, curl -f http://localhost:10000/health || exit 1`
  * **Interval:** `30s` | **Timeout:** `5s` | **Start Period:** `60s` | **Retries:** `3`

#### 4. Launch Amazon ECS Service
* Create an ECS Cluster:
  ```bash
  aws ecs create-cluster --cluster-name sitp-cluster --region <your-aws-region>
  ```
* Create an **ECS Service** with the `FARGATE` launch type attached to your VPC subnets and security group (allowing inbound traffic on port `10000` or port `80/443` via an Application Load Balancer).
* ECS Fargate automatically pulls the image from ECR, spins up tasks, monitors health via `/health`, and auto-heals any failing containers without manual intervention.

---

## 📂 Project Directory Structure

```
├── data/                      # Raw xView dataset directory
│   ├── train_images/
│   └── train_labels/
│       └── xView_train.geojson
├── src/
│   └── SITP/
│       ├── components/        # Modular pipeline stages
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_monitoring.py  # Tiled inference + ONNX + background early-exit
│       ├── pipelines/         # Pipeline runners
│       │   ├── training_pipeline.py
│       │   └── prediction_pipeline.py  # ONNX Runtime / PyTorch auto-select
│       ├── exception.py       # Custom exception handler
│       ├── logger.py          # Dual logging module (Backend & Detection)
│       └── utils.py           # Helper utilities
├── notebooks/                 # Jupyter notebooks for EDA & experimentation
├── artifacts/                 # Intermediate pipeline outputs
├── logs/                      # Dual logging system
│   ├── backend/               # Backend, server & deployment lifecycle logs
│   └── detection/             # Detailed threat detection event logs
├── templates/
│   └── index.html             # Zero-scroll Tactical C2 Cockpit HUD
├── static/
│   └── css/
│       └── style.css          # Military HUD styling (100vh locked viewport)
├── HLD.pdf                    # High-Level System Architecture document
├── LLD.pdf                    # Low-Level Design document
├── best-v26.pt                # Active trained YOLOv26 model weights (155 MB)
├── best.pt                    # PyTorch model weights
├── best.onnx                  # ONNX Runtime exported weights (optional)
├── application.py             # Flask web server entry point
├── main.py                    # Training pipeline entry point
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
└── setup.py                   # Package configuration
```

---

## 📜 License
This project is licensed under the MIT License.