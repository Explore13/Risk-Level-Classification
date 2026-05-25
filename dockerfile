FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HF_HOME=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# - ffmpeg: audio decoding for openai-whisper
# - libsndfile1: audio file I/O
# - gcc, build-essential: compiling Python C extensions
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create cache dirs
RUN mkdir -p /app/.cache/huggingface /app/.cache/whisper

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install "setuptools<81" && \
    pip install --no-build-isolation --no-deps openai-whisper==20240930 && \
    pip install tiktoken more-itertools numba && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper medium model at build time (download only, no GPU/RAM needed)
RUN mkdir -p /app/.cache/whisper && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt', '/app/.cache/whisper/medium.pt')"

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]