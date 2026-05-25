# Risk-Level-Classification API — Technical Documentation

## Overview

The **Risk-Level-Classification API** is a FastAPI service built for a **women's safety SOS application**. When an SOS is activated, the mobile app captures up to 30 seconds of audio, sends it to this API, and receives a risk-level classification to determine the urgency of the situation.

---

## Architecture — How It Works

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT (WAV, ≤35s)                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Language Detection (Whisper — audio-based)                     │
│                                                                         │
│  • Uses OpenAI Whisper "medium" model                                   │
│  • Analyzes mel-spectrogram of audio signal                             │
│  • Returns language code (e.g. "hi", "bn", "en")                       │
│  • Maps to Google STT code (e.g. "hi-IN", "bn-IN")                     │
│                                                                         │
│  WHY: Whisper detects language from AUDIO, not text.                    │
│       This eliminates Hindi/Bengali confusion that text-based           │
│       detection suffers from.                                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ detected_lang = "bn-IN"
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Speech-to-Text (Google STT — native script)                    │
│                                                                         │
│  • Uses SpeechRecognition library (Google Web Speech API)               │
│  • Passes detected language code from Step 1                            │
│  • Returns text in NATIVE SCRIPT (Bengali, Devanagari, etc.)            │
│                                                                         │
│  OUTPUT: "আমাকে আমার মত থাকতে দাও" (Bengali script)                    │
│                                                                         │
│  WHY: Google STT produces clean native-script text when given           │
│       the correct language code. This feeds perfectly into              │
│       romanization and translation.                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ original_text (native script)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Romanization (Google Translate API — "rm" field)               │
│                                                                         │
│  • Calls translate.googleapis.com with dt=["t","rm"]                    │
│  • Extracts romanized pronunciation from response                       │
│  • Converts native script → Latin/Roman characters                      │
│                                                                         │
│  OUTPUT: "amake amar moto thakte dao" (Roman script)                    │
│                                                                         │
│  WHY: Readable by non-native speakers, useful for logging               │
│       and display in the mobile app.                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ romanized_text
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Translation (deep-translator → English)                        │
│                                                                         │
│  • Uses GoogleTranslator(source="auto", target="en")                    │
│  • Translates native script text to English                             │
│                                                                         │
│  OUTPUT: "Let me be who I am"                                           │
│                                                                         │
│  WHY: The risk classifier model is trained on English text.             │
│       All languages must be translated before classification.           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ translated_text (English)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Risk Classification (HuggingFace Transformer)                  │
│                                                                         │
│  • Model: sohampal0011/risk-classifier                                  │
│  • Input: English text from Step 4                                      │
│  • Output: Label (Low/Medium/High) + confidence score                   │
│  • Runs on CPU (device=-1)                                              │
│                                                                         │
│  OUTPUT: risk_level="High", score=0.99                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RESPONSE                                                               │
│  {                                                                      │
│    "original_text": "আমাকে আমার মত থাকতে দাও",                         │
│    "romanized_text": "amake amar moto thakte dao",                      │
│    "translated_text": "Let me be who I am",                             │
│    "risk_level": "Low",                                                 │
│    "score": 0.77                                                        │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture?

We tried multiple approaches before arriving at this design:

| Approach                                   | Problem                                                                                |
| ------------------------------------------ | -------------------------------------------------------------------------------------- |
| Google STT multi-pass (try all languages)  | Slow, unreliable language detection from text                                          |
| Whisper transcribe + translate             | `base` model garbles Indic scripts; `medium` is better but still imperfect for Bengali |
| Whisper detect + Whisper transcribe        | Good detection, but transcription quality worse than Google STT for Indic languages    |
| **Whisper detect + Google STT transcribe** | ✅ Best of both: reliable audio-based detection + clean native-script transcription    |

---

## Deployment Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        DOCKER BUILD (HF Spaces)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. python:3.10-slim base image                                         │
│  2. Install system deps (ffmpeg, libsndfile1, gcc, build-essential)     │
│  3. Install Python packages:                                            │
│     • setuptools<81 (provides pkg_resources for whisper build)          │
│     • openai-whisper (--no-build-isolation --no-deps to avoid GPU torch)│
│     • tiktoken, more-itertools, numba (whisper's runtime deps)          │
│     • requirements.txt (torch CPU, transformers, fastapi, etc.)         │
│  4. Download Whisper medium model (1.42GB) via urllib → disk only       │
│     (no RAM spike — avoids OOMKilled during build)                      │
│  5. COPY app code                                                       │
│                                                                         │
│  RESULT: ~4GB Docker image with model baked in                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        RUNTIME (Container Start)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. uvicorn starts FastAPI app                                          │
│  2. @app.on_event("startup") loads:                                     │
│     • Whisper medium from /app/.cache/whisper/medium.pt (~5s)           │
│     • Risk classifier from HuggingFace Hub (~10-15s first time)         │
│  3. Health check passes → container marked ready                        │
│  4. Accepts POST /analyze_audio requests                                │
│                                                                         │
│  COLD START: ~20-30s | WARM REQUEST: ~3-8s                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Local Development vs Production

```text
┌──────────────────────────────┐    ┌──────────────────────────────────────┐
│        LOCAL DEV             │    │        PRODUCTION (HF Spaces)        │
├──────────────────────────────┤    ├──────────────────────────────────────┤
│                              │    │                                      │
│  Run:                        │    │  Run:                                │
│  uvicorn app:app --port 7860 │    │  docker build + docker run           │
│                              │    │  (automated by HF Spaces)            │
│  Whisper model:              │    │  Whisper model:                      │
│  ~/.cache/whisper/medium.pt  │    │  /app/.cache/whisper/medium.pt       │
│  (downloaded on first run)   │    │  (baked into image at build time)    │
│                              │    │                                      │
│  Risk classifier:            │    │  Risk classifier:                    │
│  ~/.cache/huggingface/       │    │  /app/.cache/huggingface/            │
│  (downloaded on first run)   │    │  (downloaded on first startup)       │
│                              │    │                                      │
│  Internet needed:            │    │  Internet needed:                    │
│  • First run (model DL)      │    │  • Every request (Google STT +       │
│  • Every request (Google STT │    │    Translation + Romanization)       │
│    + Translation)            │    │  • First startup (risk classifier)   │
│                              │    │                                      │
│  DO NOT use --reload with    │    │  No --reload, single process         │
│  heavy models (restarts      │    │                                      │
│  download the model)         │    │                                      │
└──────────────────────────────┘    └──────────────────────────────────────┘
```

---

## API Endpoints

### 1. Health Check — `GET /health`

```json
{ "status": "OK" }
```

### 2. Analyze Audio — `POST /analyze_audio`

**Request:** `multipart/form-data` with field `file` (WAV audio)

**Constraints:**

- Format: WAV only (`.wav` extension required)
- Duration: ≤ 35 seconds
- MIME: `audio/wav`, `audio/x-wav`, `audio/wave`, or similar

**Success Response (200):**

```json
{
  "original_text": "मुझे डर लग रहा है",
  "romanized_text": "mujhe dar lag raha hai",
  "translated_text": "I am feeling scared",
  "risk_level": "High",
  "score": 1.0
}
```

**Response Fields:**

| Field             | Type   | Description                                                   |
| ----------------- | ------ | ------------------------------------------------------------- |
| `original_text`   | string | Transcribed text in native script (Bengali, Devanagari, etc.) |
| `romanized_text`  | string | Native text transliterated to Roman/Latin characters          |
| `translated_text` | string | English translation of the original text                      |
| `risk_level`      | string | `Low`, `Medium`, or `High`                                    |
| `score`           | float  | Model confidence (0.0–1.0), rounded to 2 decimal places       |

**Error Responses:**

| Code | Scenario                               | Detail                      |
| ---- | -------------------------------------- | --------------------------- |
| 400  | No file / audio too long / invalid WAV | Descriptive message         |
| 415  | Wrong extension or MIME type           | Descriptive message         |
| 500  | STT/pipeline failure                   | Error from failed component |
| 503  | Model not loaded                       | `"Model not loaded"`        |

---

## Supported Languages

### Primary (actively tested)

| Language | Whisper Code | STT Code | Script     |
| -------- | ------------ | -------- | ---------- |
| Hindi    | hi           | hi-IN    | Devanagari |
| Bengali  | bn           | bn-IN    | Bengali    |
| English  | en           | en-US    | Latin      |

### Secondary (supported via mapping, not extensively tested)

Tamil (ta-IN), Telugu (te-IN), Marathi (mr-IN), Gujarati (gu-IN), Kannada (kn-IN), Malayalam (ml-IN), Punjabi (pa-IN), Urdu (ur-PK), French (fr-FR), German (de-DE), Spanish (es-ES), Arabic (ar-SA), Chinese (zh-CN), Japanese (ja-JP), Korean (ko-KR), Portuguese (pt-BR), Russian (ru-RU), Italian (it-IT)

---

## Dependencies & Tech Stack

### Python Packages

| Package             | Version   | Purpose                        |
| ------------------- | --------- | ------------------------------ |
| `torch`             | 2.3.1+cpu | PyTorch (CPU-only, no CUDA)    |
| `transformers`      | 4.46.3    | HuggingFace model loading      |
| `openai-whisper`    | 20240930  | Language detection from audio  |
| `SpeechRecognition` | 3.10.4    | Google STT wrapper             |
| `fastapi`           | 0.111.0   | Web framework                  |
| `uvicorn`           | 0.30.1    | ASGI server                    |
| `deep-translator`   | 1.11.4    | Translation to English         |
| `requests`          | 2.32.3    | HTTP client (romanization API) |
| `numpy`             | 1.26.4    | Numerical computing            |
| `tiktoken`          | latest    | Whisper tokenizer              |
| `numba`             | latest    | Whisper audio processing       |

### System Dependencies (Dockerfile)

| Package                  | Purpose                    |
| ------------------------ | -------------------------- |
| `ffmpeg`                 | Audio decoding for Whisper |
| `libsndfile1`            | Audio file I/O             |
| `gcc`, `build-essential` | Compiling C extensions     |

---

## Environment Variables

| Variable         | Default                   | Description              |
| ---------------- | ------------------------- | ------------------------ |
| `PORT`           | 7860                      | Server port              |
| `HF_HOME`        | `/app/.cache/huggingface` | HuggingFace model cache  |
| `XDG_CACHE_HOME` | `/app/.cache`             | Whisper model cache root |

---

## Running Locally

### Prerequisites

- Python 3.10+
- ffmpeg installed (`choco install ffmpeg` on Windows, `brew install ffmpeg` on Mac)
- Internet connection (required for Google STT, Translation, and first-run model downloads)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd risk-level-classification

# Install dependencies
pip install -r requirements.txt

# Start the server (DO NOT use --reload with heavy models)
uvicorn app:app --host 0.0.0.0 --port 7860
```

First startup will download:

- Whisper medium model (~1.42GB) → `~/.cache/whisper/medium.pt`
- Risk classifier (~500MB) → `~/.cache/huggingface/`

### Testing

```bash
curl -X POST "http://localhost:7860/analyze_audio" \
  -F "file=@recording.wav"
```

---

## Docker Build & Deployment

### Build Locally

```bash
docker build -t risk-classifier .
docker run -p 7860:7860 risk-classifier
```

### HF Spaces Deployment

Push to the HF Spaces repo. The Dockerfile handles everything automatically.

### Build Challenges & Solutions

| Challenge                                                   | Solution                                                       |
| ----------------------------------------------------------- | -------------------------------------------------------------- |
| `setuptools>=81` removed `pkg_resources`                    | Pin `setuptools<81` before building whisper                    |
| `openai-whisper` pulls GPU torch (532MB) in build isolation | Use `--no-build-isolation --no-deps` + install deps separately |
| Loading whisper model at build time causes OOMKilled        | Download model file via `urllib` (streams to disk, no RAM)     |
| Whisper model re-downloads on every cold start              | Bake model file into Docker image at build time                |

---

## Performance

| Metric                                     | Value     |
| ------------------------------------------ | --------- |
| Cold start (first request after deploy)    | ~20-30s   |
| Warm request (English, short audio)        | ~3-5s     |
| Warm request (Hindi/Bengali, longer audio) | ~5-8s     |
| Memory usage (runtime)                     | ~1.5-2 GB |
| Docker image size                          | ~4 GB     |

---

## Project Structure

```text
├── app.py                 # Main application — all API logic
├── requirements.txt       # Python dependencies (pinned versions)
├── Dockerfile             # Production container definition
├── API_REQUIREMENTS.md    # This file — detailed technical docs
├── README.md              # Quick-start README for HF Spaces
├── .gitignore
└── .dockerignore
```

---

## Security Notes

**Current implementation** (suitable for HF Spaces demo):

- No authentication
- No rate limiting
- Temp files cleaned up after each request
- CPU-only inference

**Recommendations for production:**

1. Add API key or JWT authentication
2. Implement rate limiting
3. Enable HTTPS/TLS
4. Add request size limits at reverse proxy level
5. Monitor error rates
6. Add CORS configuration for the mobile app

---

## Integration Guide (Mobile App)

```text
SOS Triggered
    │
    ▼
Start Recording (30s max)
    │
    ▼
Save as WAV
    │
    ▼
POST /analyze_audio (multipart/form-data)
    │
    ▼
Read response.risk_level:
    ├── "High"   → Immediate alert to emergency contacts + authorities
    ├── "Medium" → Alert to emergency contacts with context
    └── "Low"    → Log event, no immediate escalation
```

### Android/Kotlin Example

```kotlin
val file = File(audioFilePath)
val requestBody = MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", file.name,
        file.asRequestBody("audio/wav".toMediaType()))
    .build()

val request = Request.Builder()
    .url("https://your-space.hf.space/analyze_audio")
    .post(requestBody)
    .build()

client.newCall(request).enqueue(object : Callback {
    override fun onResponse(call: Call, response: Response) {
        val json = JSONObject(response.body!!.string())
        val riskLevel = json.getString("risk_level")
        // Trigger appropriate alert
    }
    override fun onFailure(call: Call, e: IOException) { /* handle */ }
})
```

---

## Version History

| Version | Date         | Changes                                                                                                                                                                |
| ------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | May 18, 2026 | Initial release                                                                                                                                                        |
| 1.1     | May 25, 2026 | Fixed model loading (safetensors)                                                                                                                                      |
| 1.2     | May 25, 2026 | Multi-candidate STT with Hindi/Bengali disambiguation                                                                                                                  |
| 2.0     | May 25, 2026 | **Architecture overhaul**: Whisper (detect) + Google STT (transcribe). Removed text-based language detection. Added openai-whisper for reliable audio-based detection. |

---

## License

MIT

---

**Last Updated**: May 25, 2026
