FROM python:3.11-slim

# Install system dependencies for OpenCV & GStreamer
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Expose port
EXPOSE 8000

# Start Uvicorn production server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
