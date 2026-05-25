import logging
import os
import shutil
import tempfile
import wave

import speech_recognition as sr
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# =========================================================
# Logging setup
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =========================================================
# Language map — Whisper language code → Google STT code
# =========================================================
WHISPER_TO_STT = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "en": "en-US",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "ur": "ur-PK",
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "ar": "ar-SA",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "pt": "pt-BR",
    "ru": "ru-RU",
    "it": "it-IT",
}

# =========================================================
# Models
# =========================================================
model_name = "sohampal0011/risk-classifier"
model_pipeline = None
whisper_model = None

app = FastAPI()


@app.on_event("startup")
def load_models():
    global model_pipeline, whisper_model

    # --- Load Whisper for language detection only ---
    logger.info("🚀 Loading Whisper model (medium) for language detection...")
    try:
        import whisper
        whisper_model = whisper.load_model("medium")
        logger.info("✅ Whisper model loaded successfully.")
    except Exception as e:
        whisper_model = None
        logger.error("⚠️ Failed to load Whisper model: %s", e)

    # --- Load risk classifier ---
    logger.info("🚀 Loading risk classifier: %s", model_name)
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        logger.info("📦 transformers==%s | torch==%s", transformers.__version__, torch.__version__)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
        )
        model_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )
        logger.info("✅ Risk classifier loaded successfully.")
    except Exception as e:
        model_pipeline = None
        logger.error("⚠️ Failed to load risk classifier: %s", e)


@app.get("/")
async def root():
    return {"message": "Risk Level Classification API is running!"}


@app.get("/health")
async def health_check():
    return {"status": "OK"}


class AudioResponse(BaseModel):
    original_text: str
    romanized_text: str
    translated_text: str
    risk_level: str
    score: float


# =========================================================
# Validate WAV
# =========================================================
def validate_wav(file_path):
    try:
        with wave.open(file_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            duration = frames / rate
            logger.info(
                "🎵 WAV info — duration: %.2fs | sample_rate: %dHz | channels: %d",
                duration, rate, channels,
            )
            if duration > 35:
                logger.warning("❌ Audio too long: %.2fs (max 35s)", duration)
                return False, "Audio too long (max 35 seconds)"
            return True, "Valid WAV file"
    except Exception as e:
        logger.error("❌ WAV validation failed: %s", e)
        return False, "Invalid WAV file"


# =========================================================
# Translate to English
# =========================================================
def translate_to_english(text):
    logger.info("🔄 Translating to English: %s", text)
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        logger.info("✅ Translated text: %s", translated)
        return translated
    except Exception as e:
        logger.error("❌ Translation failed: %s — returning original text", e)
        return text


# =========================================================
# Transliterate to Roman script
# =========================================================
def transliterate_to_roman(text, detected_lang_code):
    """Convert native script text to Roman script using Google Translate."""
    logger.info("🔤 Romanizing text (lang: %s): %s", detected_lang_code, text)

    latin_langs = {"en-US", "fr-FR", "de-DE", "es-ES", "pt-BR", "it-IT"}
    if detected_lang_code in latin_langs:
        logger.info("ℹ️ Already Roman script, returning as-is")
        return text

    try:
        import requests
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": detected_lang_code.split("-")[0],  # e.g. "bn" from "bn-IN"
            "tl": "en",
            "dt": ["t", "rm"],                       # rm = romanization
            "q": text,
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        romanized_parts = []
        for chunk in data[0]:
            if chunk and len(chunk) > 3 and chunk[3]:
                romanized_parts.append(chunk[3])

        if romanized_parts:
            romanized = " ".join(romanized_parts).strip()
            logger.info("✅ Romanized text: %s", romanized)
            return romanized
        else:
            logger.warning("⚠️ No romanization returned, using original text")
            return text

    except Exception as e:
        logger.warning("⚠️ Romanization failed: %s — returning original text", e)
        return text


# =========================================================
# Language detection via Whisper (audio-based, most reliable)
# =========================================================
def detect_language_whisper(wav_file):
    """
    Use Whisper's audio-based language detection.
    This is far more reliable than text-based detection because it
    operates on acoustic features, not transcribed text — so it never
    confuses Hindi/Bengali/English regardless of script or romanization.
    Returns a Google STT language code (e.g. 'bn-IN').
    """
    if whisper_model is None:
        logger.warning("⚠️ Whisper model not loaded — defaulting to en-US")
        return "en-US"

    try:
        import whisper
        logger.info("🎧 Detecting language from audio using Whisper...")
        audio = whisper.load_audio(wav_file)
        audio_clip = whisper.pad_or_trim(audio)          # first 30s is enough
        mel = whisper.log_mel_spectrogram(audio_clip).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang] * 100
        stt_code = WHISPER_TO_STT.get(detected_lang, "en-US")
        logger.info(
            "🌐 Whisper detected language: %s (confidence: %.2f%%) → STT code: %s",
            detected_lang, confidence, stt_code,
        )
        return stt_code
    except Exception as e:
        logger.error("❌ Whisper language detection failed: %s — defaulting to en-US", e)
        return "en-US"


# =========================================================
# Speech to text via Google STT (native script output)
# =========================================================
def wav_to_text(wav_file):
    """
    Step 1 — Detect language from audio using Whisper (audio-based, reliable).
    Step 2 — Transcribe with Google STT using the detected language code.
             Google STT returns native script (Bengali, Devanagari, etc.)
             which feeds cleanly into romanization and translation.
    """
    recognizer = sr.Recognizer()

    with sr.AudioFile(wav_file) as source:
        audio = recognizer.record(source)

    # ── Step 1: Whisper language detection ───────────────────────────────────
    detected_lang = detect_language_whisper(wav_file)

    # ── Step 2: Google STT with detected language → native script ────────────
    logger.info("📝 Transcribing with Google STT (lang=%s)...", detected_lang)
    try:
        text = recognizer.recognize_google(audio, language=detected_lang)
        logger.info("📄 Transcribed text (%s): %s", detected_lang, text)
        return text, detected_lang
    except sr.UnknownValueError:
        logger.error("❌ Google STT could not understand audio with lang=%s", detected_lang)
        raise Exception("Could not recognize speech - no audio detected")
    except sr.RequestError as e:
        logger.error("❌ Google STT API error: %s", e)
        raise Exception(f"Speech recognition service error: {e}")


# =========================================================
# API Endpoint
# =========================================================
@app.post("/analyze_audio", response_model=AudioResponse)
async def analyze_audio(file: UploadFile = File(...)):

    logger.info("📥 Incoming request — filename: %s | content_type: %s", file.filename, file.content_type)

    if not file.filename:
        logger.warning("❌ No filename provided")
        raise HTTPException(status_code=400, detail="No file uploaded. Please upload a WAV audio file.")

    if not file.filename.lower().endswith(".wav"):
        logger.warning("❌ Invalid file extension: %s", file.filename)
        raise HTTPException(status_code=415, detail="Invalid file type. Please upload a WAV audio file only.")

    allowed_mime_types = [
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/x-wave", "audio/vnd.wav", "audio/vnd.wave",
        "application/octet-stream",
    ]
    if file.content_type and file.content_type not in allowed_mime_types:
        if not file.content_type.startswith("audio/"):
            logger.warning("❌ Invalid MIME type: %s", file.content_type)
            raise HTTPException(
                status_code=415,
                detail=f"Invalid audio format. Detected MIME type: {file.content_type}. Only WAV audio files are supported.",
            )

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_file = tmp.name
            shutil.copyfileobj(file.file, tmp)
        file_size_kb = os.path.getsize(temp_file) / 1024
        logger.info("💾 File saved — temp path: %s | size: %.2f KB", temp_file, file_size_kb)
    except Exception as e:
        logger.error("❌ Error saving file: %s", e)
        raise HTTPException(status_code=500, detail="Error saving the uploaded file.")

    is_valid, message = validate_wav(temp_file)
    if not is_valid:
        os.remove(temp_file)
        raise HTTPException(status_code=400, detail=message)

    try:
        original_text, detected_lang = wav_to_text(temp_file)
        romanized_text = transliterate_to_roman(original_text, detected_lang)
        translated_text = translate_to_english(original_text)

        if model_pipeline is None:
            logger.error("❌ Model pipeline is not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")

        logger.info("🤖 Running risk classification on: %s", translated_text)
        result = model_pipeline(translated_text)
        risk = result[0]["label"]
        score = result[0]["score"]
        logger.info("🚨 Risk result — label: %s | score: %.4f (%.2f%%)", risk, score, score * 100)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_file)
        logger.info("🧹 Temp file cleaned up: %s", temp_file)

    logger.info(
        "✅ Request complete — risk: %s | score: %.2f | original: %s | romanized: %s | translated: %s",
        risk, round(score, 2), original_text, romanized_text, translated_text,
    )

    return AudioResponse(
        original_text=original_text,
        romanized_text=romanized_text,
        translated_text=translated_text,
        risk_level=risk,
        score=round(score, 2),
    )
