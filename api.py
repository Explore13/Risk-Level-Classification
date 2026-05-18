from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import speech_recognition as sr
from googletrans import Translator
from new_risk_test import predict
import wave

app = FastAPI()

translator = Translator()


class AudioResponse(BaseModel):
    original_text: str
    translated_text: str
    risk_level: str


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

    text = recognizer.recognize_google(audio)
    return text



def translate_to_english(text):
    translated = translator.translate(text, dest='en')
    return translated.text


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

    if file.content_type not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(
            status_code=415,
            detail="Invalid audio format. Only WAV audio files are supported."
        )

    temp_file = "temp.wav"

    
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
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
        risk = predict(translated_text)

    except Exception as e:
        os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

    os.remove(temp_file)

    return AudioResponse(
        original_text=original_text,
        translated_text=translated_text,
        risk_level=risk
    )