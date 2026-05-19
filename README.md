# Risk-Level-Classification API Documentation

## Overview

The **Risk-Level-Classification API** is a FastAPI-based service that processes audio files to extract text and classify the content based on risk levels. The service performs the following operations:

1. **Speech Recognition**: Converts WAV audio files to text using Google's speech recognition service
2. **Language Translation**: Translates extracted text to English (if needed)
3. **Risk Classification**: Analyzes the text and classifies it into risk categories using a fine-tuned transformer model from Hugging Face

---

## Base Information

- **Framework**: FastAPI
- **Server**: Uvicorn
- **Python Version**: 3.14+
- **API Version**: 1.0

### Running the Server

```bash
# Start the development server with auto-reload
uvicorn api:app --reload

# Start the production server
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

---

## Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Performs a health check to verify the API is running and operational.

**Request Parameters**: None

**Response**:

```json
{
  "status": "OK"
}
```

**Status Code**: `200 OK`

**Example**:

```bash
curl -X GET "http://localhost:8000/health"
```

---

### 2. Analyze Audio

**Endpoint**: `POST /analyze_audio`

**Description**: Accepts a WAV audio file, converts it to text, translates it to English, and classifies the risk level of the content.

#### Request

**Content-Type**: `multipart/form-data`

**Parameters**:

- `file` (required, File): WAV audio file to analyze
  - **Accepted formats**: `.wav` (audio/wav, audio/x-wav)
  - **Maximum duration**: 30 seconds
  - **Supported codecs**: Standard WAV format

**Example using cURL**:

```bash
curl -X POST "http://localhost:8000/analyze_audio" \
  -H "accept: application/json" \
  -F "file=@path/to/your/file.wav"
```

**Example using Python**:

```python
import requests

url = "http://localhost:8000/analyze_audio"
files = {"file": open("audio.wav", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

**Example using JavaScript/Fetch**:

```javascript
const formData = new FormData();
formData.append("file", audioFile); // audioFile from file input

fetch("http://localhost:8000/analyze_audio", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

#### Response

**Success Response (200 OK)**:

```json
{
  "original_text": "The extracted text from the audio file",
  "translated_text": "The translated text in English (if different from original)",
  "risk_level": "LOW|MEDIUM|HIGH"
}
```

**Response Fields**:

- `original_text` (string): The text extracted from the audio file in its original language
- `translated_text` (string): The text translated to English
- `risk_level` (string): Classification of the content risk level
  - `LOW`: Content poses minimal risk
  - `MEDIUM`: Content poses moderate risk
  - `HIGH`: Content poses significant risk

---

## Error Handling

The API uses standard HTTP status codes and provides descriptive error messages.

### Error Responses

#### 400 Bad Request

Returned when the request is malformed or validation fails.

**Scenarios**:

- No file uploaded
- Audio duration exceeds 30 seconds

**Example Response**:

```json
{
  "detail": "Audio too long (max 30 seconds)"
}
```

#### 415 Unsupported Media Type

Returned when the file format is not supported.

**Scenarios**:

- File is not a `.wav` file
- Content-Type is not `audio/wav` or `audio/x-wav`

**Example Response**:

```json
{
  "detail": "Invalid file type. Please upload a WAV audio file only."
}
```

**Example Response**:

```json
{
  "detail": "Invalid audio format. Only WAV audio files are supported."
}
```

#### 500 Internal Server Error

Returned when server encounters an unexpected error during processing.

**Scenarios**:

- File cannot be saved
- Speech recognition fails
- Model prediction fails
- Temporary file issues

**Example Response**:

```json
{
  "detail": "Error saving the uploaded file."
}
```

---

## Data Models

### AudioResponse

Response model for the `/analyze_audio` endpoint.

```python
{
    "original_text": str,      # Extracted text from audio
    "translated_text": str,    # Text translated to English
    "risk_level": str          # Classification: LOW, MEDIUM, or HIGH
}
```

---

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Upload WAV Audio File                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │  File Validation          │
        │  - Extension: .wav        │
        │  - MIME type check        │
        │  - Duration: ≤ 30 seconds │
        └────────┬──────────────────┘
                 │
                 ▼
      ┌──────────────────────────┐
      │  Speech Recognition      │
      │  (Google SR API)         │
      │  Audio → Text (Original) │
      └────────┬─────────────────┘
               │
               ▼
      ┌──────────────────────────┐
      │  Language Translation    │
      │  → English               │
      └────────┬─────────────────┘
               │
               ▼
      ┌──────────────────────────┐
      │  Risk Classification     │
      │  (Hugging Face Model)    │
      │  Text → Risk Level       │
      └────────┬─────────────────┘
               │
               ▼
        ┌──────────────────────────┐
        │  Return Response         │
        │  {original_text,         │
        │   translated_text,       │
        │   risk_level}            │
        └──────────────────────────┘
```

---

## Technical Details

### Speech Recognition

- **Service**: Google Speech Recognition API (via `SpeechRecognition` library)
- **Input**: WAV audio files
- **Output**: Transcribed text in original language

### Translation

- **Service**: Google Translate (via `googletrans` library, with fallback to passthrough)
- **Target Language**: English
- **Fallback**: If translation service unavailable, returns original text

### Risk Classification

- **Model**: Fine-tuned transformer model from Hugging Face
- **Model ID**: `sohampal0011/risk-classifier`
- **Input**: Text content
- **Output**: Risk classification with confidence score
- **Categories**: LOW, MEDIUM, HIGH

### Dependencies

- **fastapi** (0.136.1+): Web framework
- **uvicorn** (0.47.0+): ASGI server
- **pydantic** (2.13.4+): Data validation
- **SpeechRecognition** (3.16.1+): Audio transcription
- **transformers** (4.57.6+): NLP models
- **torch** (2.12.0+): Deep learning framework
- **numpy** (2.4.5+): Numerical computing
- **requests** (2.34.2+): HTTP library

---

## Usage Examples

### Example 1: Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

**Response**:

```json
{
  "status": "OK"
}
```

---

### Example 2: Analyze Audio File (Python)

```python
import requests
import json

# Prepare the audio file
with open("sample_audio.wav", "rb") as audio_file:
    files = {"file": audio_file}

    # Send request
    response = requests.post(
        "http://localhost:8000/analyze_audio",
        files=files
    )

# Handle response
if response.status_code == 200:
    data = response.json()
    print(f"Original Text: {data['original_text']}")
    print(f"Translated Text: {data['translated_text']}")
    print(f"Risk Level: {data['risk_level']}")
else:
    print(f"Error: {response.status_code}")
    print(f"Details: {response.json()['detail']}")
```

---

### Example 3: Analyze Audio File (JavaScript)

```javascript
async function analyzeAudio(audioFile) {
  const formData = new FormData();
  formData.append("file", audioFile);

  try {
    const response = await fetch("http://localhost:8000/analyze_audio", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail);
    }

    const result = await response.json();
    console.log("Original Text:", result.original_text);
    console.log("Translated Text:", result.translated_text);
    console.log("Risk Level:", result.risk_level);

    return result;
  } catch (error) {
    console.error("Error analyzing audio:", error);
  }
}

// Usage
const audioInput = document.getElementById("audioFile");
analyzeAudio(audioInput.files[0]);
```

---

### Example 4: Batch Processing

```python
import requests
import os
from pathlib import Path

audio_directory = "./audio_files"
results = []

for audio_file in Path(audio_directory).glob("*.wav"):
    with open(audio_file, "rb") as f:
        files = {"file": f}
        response = requests.post(
            "http://localhost:8000/analyze_audio",
            files=files
        )

        if response.status_code == 200:
            results.append({
                "filename": audio_file.name,
                "data": response.json()
            })
        else:
            print(f"Failed to process {audio_file.name}")

# Display results
for result in results:
    print(f"\n{result['filename']}:")
    print(f"  Risk Level: {result['data']['risk_level']}")
```

---

## Constraints & Limitations

| Constraint              | Value             | Details                                 |
| ----------------------- | ----------------- | --------------------------------------- |
| **Max Audio Duration**  | 30 seconds        | Longer files will be rejected           |
| **Supported Format**    | WAV only          | MP3, OGG, FLAC not supported            |
| **Max File Size**       | Depends on server | Generally 50MB+ handled without issue   |
| **Concurrent Requests** | Unlimited         | Limited by server resources             |
| **Response Time**       | 5-30 seconds      | Depends on audio length and content     |
| **Languages**           | All               | Auto-translated to English for analysis |

---

## Performance Considerations

1. **First Request**: May take longer (~10-15 seconds) due to model loading
2. **Subsequent Requests**: Typically 3-5 seconds per request
3. **Audio Length**: Longer audio takes proportionally longer to process
4. **Server Resources**:
   - Requires GPU for optimal performance (falls back to CPU)
   - Minimum 4GB RAM recommended
   - Model size: ~500MB

---

## Security Notes

⚠️ **Current Implementation**:

- No authentication/authorization required
- No rate limiting
- Audio files temporarily stored on disk

**Recommendations for Production**:

1. Implement API key authentication
2. Add rate limiting
3. Use cloud storage for audio files (S3, GCS, etc.)
4. Enable HTTPS/TLS
5. Implement request logging and monitoring
6. Add CORS configuration as needed
7. Validate and sanitize file uploads

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cgi'"

**Solution**: Ensure you're running Python 3.14+ with updated dependencies

```bash
pip install -r requirements.txt
```

### Issue: "No module named 'googletrans'"

**Solution**: Translation falls back to passthrough automatically. This is normal on Python 3.14 with certain httpx versions.

### Issue: "Audio too long (max 30 seconds)"

**Solution**: Reduce audio file duration to 30 seconds or less.

### Issue: "Timeout during speech recognition"

**Solution**: Check internet connection (Google SR API requires connectivity) or try shorter audio file.

### Issue: "CUDA out of memory"

**Solution**: Model automatically falls back to CPU processing. Alternatively:

```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
uvicorn api:app --reload
```

---

## API Status & Monitoring

### Access Interactive Documentation

FastAPI provides automatic API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to:

- View all endpoints
- Test endpoints directly
- See request/response schemas
- Download API specifications

---

## Support & Contribution

For issues, feature requests, or contributions:

- Repository: `Sohampal001/Risk-Level-Classification`
- Report issues through GitHub Issues
- Submit pull requests for improvements

---

## Version History

| Version | Date         | Changes                                                     |
| ------- | ------------ | ----------------------------------------------------------- |
| 1.0     | May 18, 2026 | Initial release with audio analysis and risk classification |
|         |              | - Added `/health` endpoint                                  |
|         |              | - Added `/analyze_audio` endpoint                           |
|         |              | - Python 3.14 compatibility                                 |

---

## License

This project is licensed under the terms specified in the repository.

---

**Last Updated**: May 18, 2026
**Maintained By**: Development Team
