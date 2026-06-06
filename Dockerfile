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
# Copies: application.py, best.pt, setup.py, src/, templates/, static/, uploads/
COPY . .

# Install the local src/SITP package so that `from src.SITP...` imports resolve
RUN pip install --no-cache-dir -e .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p uploads logs

# ── Environment variables ─────────────────────────────────────────────────────
ENV FLASK_APP=application.py
ENV PYTHONUNBUFFERED=1
# Prevent rasterio from using excessive memory cache
ENV GDAL_CACHEMAX=256

# ── Expose Flask port ─────────────────────────────────────────────────────────
EXPOSE 5000

# ── Start the application via gunicorn (production-grade WSGI server) ─────────
# - 2 workers is safe for CPU-bound ML inference workload
# - timeout 300s to handle large image inference time
CMD ["gunicorn", "--workers", "2", "--timeout", "300", "--bind", "0.0.0.0:5000", "application:app"]
