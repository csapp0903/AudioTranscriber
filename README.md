# CloudMusicTranscribe

Audio to MIDI/Sheet Music transcription service (MVP version).

## Architecture

- **FastAPI**: Web API server
- **Redis**: Message broker for Celery
- **Celery**: Async task worker
- **Docker Compose**: Container orchestration

## Quick Start

```bash
# Start all services
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /
```

### Upload Audio
```
POST /upload
Content-Type: multipart/form-data
file: <MP3 file>

Response:
{
    "task_id": "uuid",
    "filename": "example.mp3",
    "message": "File uploaded successfully. Processing started."
}
```

### Check Status
```
GET /status/{task_id}

Response:
{
    "task_id": "uuid",
    "status": "PENDING|PROCESSING|SUCCESS|FAILURE",
    "result": {...}  // when SUCCESS
}
```

### Download Files
```
GET /download/{task_id}                    # List available files
GET /download/{task_id}?file_type=midi     # Download MIDI file
GET /download/{task_id}?file_type=pdf      # Download PDF file
```

## Testing with cURL

```bash
# Upload an MP3 file
curl -X POST -F "file=@test.mp3" http://localhost:8000/upload

# Check task status
curl http://localhost:8000/status/{task_id}

# Download MIDI file
curl -O http://localhost:8000/download/{task_id}?file_type=midi

# Download PDF file
curl -O http://localhost:8000/download/{task_id}?file_type=pdf
```

## Project Structure

```
AudioTranscriber/
├── backend/
│   ├── __init__.py
│   ├── config.py         # Configuration settings
│   ├── celery_app.py     # Celery configuration
│   ├── tasks.py          # Celery tasks (mock processing)
│   └── main.py           # FastAPI application
├── uploads/              # Uploaded audio files
├── outputs/              # Generated MIDI and PDF files
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Development

Stop services:
```bash
docker-compose down
```

View logs:
```bash
docker-compose logs -f web
docker-compose logs -f worker
```
