FROM python:3.10-slim
 
WORKDIR /app
 
# Install system dependencies
# ffmpeg: needed for audio processing
# gcc + build-essential: needed to compile some Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# Copy and install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy app code
COPY app.py .
 
# HuggingFace Spaces requires port 7860
EXPOSE 7860
 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]