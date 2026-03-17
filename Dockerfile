FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV and Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD streamlit run dashboard.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.fileWatcherType none --browser.gatherUsageStats false
