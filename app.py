from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import tempfile
import speech_recognition as sr
import wave
 
# =========================================================
# Model (will be loaded at FastAPI startup)
# =========================================================
model_name = "sohampal0011/risk-classifier"
model_pipeline = None
 
try:
    from googletrans import Translator
except Exception:
    class Translator:
        def translate(self, text, dest="en"):
            class Result:
                def __init__(self, value):
                    self.text = value
 
            return Result(text)
 
app = FastAPI()
 
translator = Translator()
 
 
@app.on_event("startup")
def load_model():
    """Load tokenizer/model and create a shared pipeline at startup."""
    global model_pipeline
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
 
        model_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )
 
        print("✅ Model pipeline loaded on startup.")
 
    except Exception as e:
        model_pipeline = None
        print("⚠️ Failed to load model on startup:", e)
 
 
@app.get("/health")
async def health_check():
    return {"status": "OK"}
 
 
class AudioResponse(BaseModel):
    original_text: str
    translated_text: str
    risk_level: str
    score: float
 
# -------------------------------
# Validate WAV file
# -------------------------------
def validate_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()
 
            if duration > 30:
                return False, "Audio too long (max 30 seconds)"
 
            return True, "Valid WAV file"
 
    except:
        return False, "Invalid WAV file"
 
 
def wav_to_text(wav_file):
    recognizer = sr.Recognizer()
 
    with sr.AudioFile(wav_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        # Speech was unintelligible
        raise HTTPException(status_code=400, detail="Speech could not be understood")

    except sr.RequestError as e:
        # API was unreachable or unresponsive
        print("Speech recognition request error:", e)
        raise HTTPException(status_code=502, detail="Speech recognition service error")

    except Exception as e:
        print("Unexpected error during speech recognition:", e)
        raise HTTPException(status_code=500, detail="Internal speech recognition error")
 
 
def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        # Fallback: return original text if translation fails
        print("Translation failed, returning original text:", e)
        return text


def ensure_model_loaded():
    """Ensure the Hugging Face pipeline is loaded. Try to lazy-load if missing."""
    global model_pipeline
    if model_pipeline is not None:
        return

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        model_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )

        print("✅ Model pipeline loaded lazily.")

    except Exception as e:
        model_pipeline = None
        print("⚠️ Failed to load model lazily:", e)
 
 
# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/analyze_audio", response_model=AudioResponse)
async def analyze_audio(file: UploadFile = File(...)):
 
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded. Please upload a WAV audio file."
        )
 
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=415,
            detail="Invalid file type. Please upload a WAV audio file only."
        )
 
    allowed_mime_types = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/x-wave",
        "audio/vnd.wav",
        "audio/vnd.wave",
        "application/octet-stream"
    ]
 
    if file.content_type and file.content_type not in allowed_mime_types:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=415,
                detail=f"Invalid audio format. Detected MIME type: {file.content_type}. Only WAV audio files are supported."
            )
 
    # Use a unique temp file per request to handle concurrent uploads safely
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_file = tmp.name
            shutil.copyfileobj(file.file, tmp)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error saving the uploaded file."
        )
 
    is_valid, message = validate_wav(temp_file)
    if not is_valid:
        os.remove(temp_file)
        raise HTTPException(status_code=400, detail=message)
 
    try:
        original_text = wav_to_text(temp_file)
        translated_text = translate_to_english(original_text)
 
        # Ensure model is loaded (attempt lazy load if needed)
        if model_pipeline is None:
            ensure_model_loaded()

        if model_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        result = model_pipeline(translated_text)
        # Validate pipeline output
        if not result or not isinstance(result, list) or "label" not in result[0] or "score" not in result[0]:
            print("Unexpected model output:", result)
            raise HTTPException(status_code=500, detail="Invalid model prediction")
        risk = result[0]["label"]
        score = result[0]["score"]
 
    except HTTPException:
        raise
    except Exception as e:
        print("Error during audio analysis:", e)
        raise HTTPException(status_code=500, detail="Internal processing error")
    finally:
        os.remove(temp_file)  # Always cleaned up, even if an error occurs
 
    return AudioResponse(
        original_text=original_text,
        translated_text=translated_text,
        risk_level=risk,
        score=score*100
    )
