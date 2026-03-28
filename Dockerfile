FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for rasterio
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY shambaai/ shambaai/
COPY models/ models/
COPY setup.py .

RUN pip install -e . --no-deps

# Cloud Run uses PORT env variable
ENV PORT=8080
ENV MODEL_DIR=/app/models

EXPOSE 8080

CMD ["uvicorn", "shambaai.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--timeout-keep-alive", "30"]
