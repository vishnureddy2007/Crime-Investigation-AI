# Single-stage image for the AI-Based Crime Investigation Assistant.
#
# Base: python:3.11-slim (one of the CI-tested versions).
# System deps below are needed for cv2.VideoWriter mp4v encoding.
# The whole thing ends up ~1.2 GB; first run downloads FLAN-T5 (~990 MB)
# to /app/.cache/huggingface at first request.
#
# Build:    docker build -t crime-ai .
# Run:      docker run -p 8501:8501 crime-ai
# Health:   curl http://localhost:8501/_stcore/health

FROM python:3.11-slim

# ---- OpenCV / video runtime deps ------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender1 \
        fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# ---- Python runtime knobs ------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# ---- Install Python deps first (better Docker layer caching) --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy the rest of the project ----------------------------------
COPY . /app

# ---- Streamlit listens on 0.0.0.0 in a headless container ----------
EXPOSE 8501

# ---- Healthcheck (Streamlit serves /_stcore/health -> 200 "ok") ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=true"]
