FROM python:3.10-slim

# Install system dependencies for OpenCV and other binary wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=application.py
ENV PYTHONUNBUFFERED=1

# Start the Flask app
CMD ["python", "application.py"]
