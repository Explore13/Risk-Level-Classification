import speech_recognition as sr
from googletrans import Translator
from new_risk_test import predict

print("🚀 Script started...")

# Load translator
translator = Translator()

# -------------------------------
# 1. Audio → Text
# -------------------------------
def wav_to_text(wav_file):
    print("🎧 Processing audio:", wav_file)

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        print("📝 Extracted Text:", text)
        return text

    except Exception as e:
        print("❌ Audio Error:", e)
        return ""


# -------------------------------
# 2. Translate → English (ANY LANGUAGE)
# -------------------------------
def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        print("🌐 Translated to English:", translated.text)
        return translated.text

    except Exception as e:
        print("❌ Translation Error:", e)
        return text


# -------------------------------
# 3. Full Pipeline
# -------------------------------
def process_audio(file_path):
    print("🚀 Starting pipeline...")

    text = wav_to_text(file_path)

    if not text:
        print("⚠️ No text extracted")
        return

    english_text = translate_to_english(text)

    result = predict(english_text)

    print("🚨 Risk Level:", result)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    process_audio(r"D:\Risk_Classification\audio\audio_2.wav")