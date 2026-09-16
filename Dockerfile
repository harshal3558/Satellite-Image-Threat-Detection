# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────────────────────
# - build-essential      : compilers for wheels that need compilation
# - libgl1               : OpenCV runtime (libGL) — replaces libgl1-mesa-glx on Debian trixie
# - libglib2.0-0         : OpenCV runtime (libgthread)
# - libgdal-dev / gdal-bin: rasterio / GDAL bindings
# - libspatialindex-dev  : rtree / geopandas spatial index
# - libgomp1             : OpenMP used by PyTorch & ultralytics
# - python3-dev          : headers needed for some native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# ── Application source ────────────────────────────────────────────────────────
# Copies: application.py, best.pt, best.onnx (if present), setup.py, src/, templates/, static/, uploads/
COPY . .

# Install the local src/SITP package so that `from src.SITP...` imports resolve
RUN pip install --no-cache-dir -e .

# ── Auto-export ONNX model if best.pt exists but best.onnx does not ──────────
# ONNX Runtime (CPUExecutionProvider) is significantly faster than PyTorch CPU
RUN [ -f best.pt ] && [ ! -f best.onnx ] && \
    python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=512, dynamic=True, simplify=True)" || true

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p uploads logs

# ── Environment variables ─────────────────────────────────────────────────────
ENV FLASK_APP=application.py
ENV PYTHONUNBUFFERED=1
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
# - Threads: 4 threads share single-process memory (avoids duplicating model weights in RAM)
# - Port: binds to ${PORT:-10000} dynamically assigned by Render
# - Timeout: 300s to accommodate tiled GeoTIFF satellite image inference
CMD sh -c "gunicorn --workers ${WEB_CONCURRENCY:-1} --threads 4 --timeout 300 --bind 0.0.0.0:${PORT:-10000} application:app"