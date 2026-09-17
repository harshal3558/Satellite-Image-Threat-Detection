# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────────────────────
# - build-essential      : compilers for wheels that need compilation
# - libgl1, libglib2.0-0 : OpenCV runtime (libGL & libgthread)
# - libsm6, libxext6, libxrender1: X11/headless OpenCV compatibility
# - libgdal-dev / gdal-bin: rasterio / GDAL bindings
# - libgomp1             : OpenMP used by PyTorch & ultralytics
# - python3-dev          : headers needed for some native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Pre-install CPU-only PyTorch to save ~2.5 GB of download/disk and prevent OOM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# ── Application source ────────────────────────────────────────────────────────
# Copies: application.py, best.pt, best.onnx (if present), setup.py, src/, templates/, static/, uploads/
COPY . .

# Install the local src/SITP package so that `from src.SITP...` imports resolve
RUN pip install --no-cache-dir -e .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p uploads logs

# ── Environment variables ─────────────────────────────────────────────────────
ENV FLASK_APP=application.py
ENV PYTHONUNBUFFERED=1
# Route Ultralytics settings to writable directory in container
ENV YOLO_CONFIG_DIR=/tmp
# Prevent rasterio from using excessive memory cache
ENV GDAL_CACHEMAX=256
# ONNX Runtime engine preference (app auto-selects best.onnx > best.pt)
ENV ORT_DISABLE_ALL_LOGS=1

# ── Expose port ───────────────────────────────────────────────────────────────
# Render Web Services listen on port 10000 by default. Exposing 10000 ensures
# Render's internal port detection and health check probes match Gunicorn.
EXPOSE 10000

# ── Start the application via gunicorn (production-grade WSGI server) ─────────
# - Workers: use ${WEB_CONCURRENCY:-1} (Render defaults to 1 on 512MB tier to prevent OOM)
# - Threads: 2 threads share single-process memory (avoids duplicating model weights in RAM)
# - Port: binds to ${PORT:-10000} dynamically assigned by Render
# - Timeout: 300s to accommodate tiled GeoTIFF satellite image inference
# - Graceful Timeout: 120s to allow clean worker transitions
CMD exec gunicorn --workers ${WEB_CONCURRENCY:-1} --threads 2 --timeout 300 --graceful-timeout 120 --bind 0.0.0.0:${PORT:-10000} application:app