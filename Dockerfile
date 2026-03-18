FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV, Tesseract OCR & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-fra \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for security
RUN groupadd -r verifdoc && useradd -r -g verifdoc -d /app -s /sbin/nologin verifdoc \
    && chown -R verifdoc:verifdoc /app
USER verifdoc

# AI layer — set ANTHROPIC_API_KEY at runtime to enable Claude Vision analysis
# docker run -e ANTHROPIC_API_KEY=sk-ant-... ...
ENV ANTHROPIC_API_KEY=""

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

CMD streamlit run dashboard.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.fileWatcherType none --browser.gatherUsageStats false
