import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from celery.result import AsyncResult

from backend.celery_app import celery_app
from backend.tasks import process_audio
from backend.config import UPLOAD_DIR, OUTPUT_DIR


app = FastAPI(
    title="CloudMusicTranscribe",
    description="Audio to MIDI/Sheet Music transcription service (MVP)",
    version="0.1.0"
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "CloudMusicTranscribe"}


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an MP3 file for transcription.

    Returns a task_id that can be used to check status and download results.
    """
    # Validate file type
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(
            status_code=400,
            detail="Only MP3 files are supported"
        )

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Create upload directory for this task
    task_upload_dir = Path(UPLOAD_DIR) / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = task_upload_dir / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Submit Celery task
    process_audio.apply_async(
        args=[task_id, str(file_path)],
        task_id=task_id
    )

    return {
        "task_id": task_id,
        "filename": file.filename,
        "message": "File uploaded successfully. Processing started."
    }


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """
    Get the status of a transcription task.

    Possible states: PENDING, PROCESSING, SUCCESS, FAILURE
    """
    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    elif result.status == "PROCESSING":
        response["meta"] = result.info

    return response


@app.get("/download/{task_id}")
def download_files(task_id: str, file_type: Optional[str] = None):
    """
    Download generated files for a completed task.

    Args:
        task_id: The task ID from upload
        file_type: Optional filter - 'midi' or 'pdf'. If not specified, returns file list.
    """
    result = AsyncResult(task_id, app=celery_app)

    if result.status != "SUCCESS":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Current status: {result.status}"
        )

    task_result = result.result
    files = task_result.get("files", {})

    if not file_type:
        # Return list of available files
        return {
            "task_id": task_id,
            "available_files": list(files.keys()),
            "download_urls": {
                k: f"/download/{task_id}?file_type={k}"
                for k in files.keys()
            }
        }

    # Get specific file
    if file_type not in files:
        raise HTTPException(
            status_code=404,
            detail=f"File type '{file_type}' not found. Available: {list(files.keys())}"
        )

    file_path = files[file_type]

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found on server"
        )

    # Determine media type
    media_types = {
        "midi": "audio/midi",
        "pdf": "application/pdf"
    }

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type=media_types.get(file_type, "application/octet-stream")
    )
