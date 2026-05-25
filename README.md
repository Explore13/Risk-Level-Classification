# Risk Level Classification API

A real-time audio risk assessment API built for **women's safety applications**. When an SOS is triggered, the app captures audio and sends it here to determine the threat level.

## How It Works

```text
🎙️ Audio (WAV, ≤35s)
    │
    ▼
🎧 Whisper (language detection from audio signal)
    │
    ▼
🗣️ Google STT (transcription in native script — Bengali, Devanagari, etc.)
    │
    ▼
🔤 Romanization (native script → Latin characters)
    │
    ▼
🌐 Translation (→ English)
    │
    ▼
🤖 Risk Classification (Low / Medium / High + confidence score)
```

**Key design decision:** Whisper detects the language from audio acoustics (never confuses Hindi/Bengali), then Google STT transcribes using that language code (produces clean native script). Best of both worlds.

## Quick Start

```bash
curl -X POST "https://your-space.hf.space/analyze_audio" \
  -F "file=@recording.wav"
```

### Example Response

```json
{
  "original_text": "আমাকে আমার মত থাকতে দাও",
  "romanized_text": "amake amar moto thakte dao",
  "translated_text": "Let me be who I am",
  "risk_level": "Low",
  "score": 0.77
}
```

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

First run downloads Whisper medium (~1.42GB) and the risk classifier (~500MB).

**Important:** Do NOT use `--reload` — it restarts the process and re-downloads models.

## API Endpoints

| Method | Path             | Description                            |
| ------ | ---------------- | -------------------------------------- |
| GET    | `/`              | Status check                           |
| GET    | `/health`        | Health check (container orchestration) |
| GET    | `/docs`          | Swagger UI                             |
| POST   | `/analyze_audio` | Main endpoint — analyze audio file     |

## Tech Stack

| Component           | Technology                                              |
| ------------------- | ------------------------------------------------------- |
| Language detection  | OpenAI Whisper (medium, audio-based)                    |
| Speech-to-text      | Google Speech Recognition API                           |
| Translation         | deep-translator (Google Translate)                      |
| Romanization        | Google Translate romanization API                       |
| Risk classification | HuggingFace Transformers (sohampal0011/risk-classifier) |
| Web framework       | FastAPI + Uvicorn                                       |
| Runtime             | Python 3.10, PyTorch CPU                                |
| Deployment          | Docker on HuggingFace Spaces                            |

## Supported Languages

| Language   | Detection  | Transcription | Script     |
| ---------- | ---------- | ------------- | ---------- |
| Hindi      | ✅ Whisper | ✅ Google STT | Devanagari |
| Bengali    | ✅ Whisper | ✅ Google STT | Bengali    |
| English    | ✅ Whisper | ✅ Google STT | Latin      |
| 18+ others | ✅ Whisper | ✅ Google STT | Various    |

## Audio Requirements

| Constraint   | Value                   |
| ------------ | ----------------------- |
| Format       | WAV only                |
| Max duration | 35 seconds              |
| Sample rate  | Any (48kHz recommended) |
| Channels     | Mono or stereo          |

## Performance

| Metric                       | Value     |
| ---------------------------- | --------- |
| Cold start                   | ~20-30s   |
| Warm request (English)       | ~3-5s     |
| Warm request (Hindi/Bengali) | ~5-8s     |
| Memory                       | ~1.5-2 GB |

## Documentation

See [API_REQUIREMENTS.md](./API_REQUIREMENTS.md) for detailed technical documentation including:

- Full architecture diagrams
- Build pipeline details
- Deployment challenges & solutions
- Integration guide with code examples