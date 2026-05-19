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
# Language map — langdetect code → Google STT code
# =========================================================
LANG_MAP = {
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
# Model
# =========================================================
model_name = "sohampal0011/risk-classifier"
model_pipeline = None

app = FastAPI()


@app.on_event("startup")
def load_model():
    global model_pipeline
    logger.info("🚀 Loading model: %s", model_name)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
        logger.info("✅ Model pipeline loaded successfully.")
    except Exception as e:
        model_pipeline = None
        logger.error("⚠️ Failed to load model: %s", e)


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
            if duration > 30:
                logger.warning("❌ Audio too long: %.2fs (max 30s)", duration)
                return False, "Audio too long (max 30 seconds)"
            return True, "Valid WAV file"
    except Exception as e:
        logger.error("❌ WAV validation failed: %s", e)
        return False, "Invalid WAV file"


# =========================================================
# Translate using deep-translator (stable, server-safe)
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
# Detect language using deep-translator
# =========================================================
def detect_language(text):
    """
    Use deep-translator's GoogleTranslator to detect language.
    Returns a Google STT language code like 'hi-IN'.
    """
    try:
        from deep_translator import GoogleTranslator
        detected = GoogleTranslator(source="auto", target="en").translate(text)
        # deep-translator doesn't expose detected lang directly,
        # so we use langdetect as fallback but with a Hinglish-aware check
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # makes langdetect deterministic
        lang_code = detect(text)
        logger.info("🌐 langdetect raw code: %s", lang_code)

        text_lower = text.lower()
        words = text_lower.split()

        # Hinglish (Hindi romanized) markers
        hindi_markers = [
            "hai", "mujhe", "koi", "raha", "lag", "dar", "nahi",
            "aur", "ko", "ka", "ki", "ke", "hoon", "tum", "yeh",
            "woh", "kya", "main", "mere", "tera", "apna", "bahut",
        ]
        hindi_count = sum(1 for w in hindi_markers if w in words)
        if hindi_count >= 2:
            logger.info("🌐 Hinglish detected via keyword match (%d markers) → hi-IN", hindi_count)
            return "hi-IN"

        # Banglish (Bengali romanized) markers
        bengali_markers = [
            "ami", "amake", "amar", "tumi", "tomake", "apni", "apnake",
            "dao", "thakte", "ache", "hobe", "koro", "kori", "jabo",
            "asbo", "ebar", "ekhane", "okhane", "bhalo", "moto", "kintu",
            "tahole", "diye", "niye", "jai", "chai", "ki", "ke", "ba",
        ]
        bengali_count = sum(1 for w in bengali_markers if w in words)
        if bengali_count >= 2:
            logger.info("🌐 Banglish detected via keyword match (%d markers) → bn-IN", bengali_count)
            return "bn-IN"

        stt_code = LANG_MAP.get(lang_code, "en-US")
        logger.info("🌐 Language detected: %s → STT code: %s", lang_code, stt_code)
        return stt_code

    except Exception as e:
        logger.warning("⚠️ Language detection failed: %s — defaulting to en-US", e)
        return "en-US"




# =========================================================
# Transliterate to Roman script
# =========================================================
def transliterate_to_roman(text, detected_lang_code):
    """Convert native script text to Roman script using Google Translate."""
    logger.info("🔤 Romanizing text (lang: %s): %s", detected_lang_code, text)

    # If already Roman script (English or other Latin-script languages), return as-is
    latin_langs = {"en-US", "fr-FR", "de-DE", "es-ES", "pt-BR", "it-IT"}
    if detected_lang_code in latin_langs:
        logger.info("ℹ️ Already Roman script, returning as-is")
        return text

    try:
        import requests

        # Google Translate returns pronunciation (romanization) in its raw API response
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl":  detected_lang_code.split("-")[0],  # e.g. "bn" from "bn-IN"
            "tl": "en",
            "dt": ["t", "rm"],   # rm = romanization
            "q": text,
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        # Romanization is in data[0][i][3] for each sentence chunk
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
# Speech to text
# =========================================================
def wav_to_text(wav_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(wav_file) as source:
        audio = recognizer.record(source)

    # Step 1: Rough English transcript for language detection
    logger.info("🎧 Step 1 — Running rough English STT for language detection...")
    try:
        rough_text = recognizer.recognize_google(audio, language="en-US")
        logger.info("📝 Rough text (en-US): %s", rough_text)
    except sr.UnknownValueError:
        rough_text = ""
        logger.warning("⚠️ Could not understand audio — no speech detected")
    except sr.RequestError as e:
        rough_text = ""
        logger.error("❌ Google STT API error in Step 1: %s", e)
    except Exception as e:
        rough_text = ""
        logger.error("❌ Unexpected error in Step 1: %s", e)

    # Step 2: Detect language from rough text
    if rough_text:
        detected_lang = detect_language(rough_text)
    else:
        detected_lang = "en-US"
        logger.info("🌐 Step 2 — No rough text, skipping detection, using en-US")

    # Step 3: Re-recognize with correct language
    logger.info("🎧 Step 3 — Re-running STT with language: %s", detected_lang)
    try:
        text = recognizer.recognize_google(audio, language=detected_lang)
        logger.info("📝 Final transcribed text: %s", text)
        return text, detected_lang
    except sr.UnknownValueError:
        if rough_text:
            logger.warning("⚠️ Re-recognition failed, using rough text: %s", rough_text)
            return rough_text, detected_lang
        logger.error("❌ Could not recognize speech in any language")
        raise Exception("Could not recognize speech - no audio detected")
    except sr.RequestError as e:
        logger.error("❌ Google STT API error in Step 3: %s", e)
        raise Exception(f"Speech recognition service error: {e}")
    except Exception as e:
        logger.error("❌ Unexpected STT error in Step 3: %s", e)
        raise Exception(f"Speech recognition failed: {e}")


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

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_file = tmp.name
            shutil.copyfileobj(file.file, tmp)

        file_size_kb = os.path.getsize(temp_file) / 1024
        logger.info("💾 File saved — temp path: %s | size: %.2f KB", temp_file, file_size_kb)

    except Exception as e:
        logger.error("❌ Error saving file: %s", e)
        raise HTTPException(status_code=500, detail="Error saving the uploaded file.")

    # Validate WAV
    is_valid, message = validate_wav(temp_file)
    if not is_valid:
        os.remove(temp_file)
        raise HTTPException(status_code=400, detail=message)

    try:
        # STT
        original_text, detected_lang = wav_to_text(temp_file)

        # Transliterate to Roman
        romanized_text = transliterate_to_roman(original_text, detected_lang)

        # Translate
        translated_text = translate_to_english(original_text)

        # Classify
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