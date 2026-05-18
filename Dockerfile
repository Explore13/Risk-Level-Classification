FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install common system deps (audio, ffmpeg). Remove if not needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project
COPY . .

# Expose the port HF Spaces typically uses; fallback to 7860 when PORT not provided
EXPOSE 7860

CMD ["bash", "-lc", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
