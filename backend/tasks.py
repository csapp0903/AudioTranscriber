import os
import time
from pathlib import Path

from backend.celery_app import celery_app
from backend.config import OUTPUT_DIR


@celery_app.task(bind=True)
def process_audio(self, task_id: str, input_file: str):
    """
    Mock audio processing task.

    In production, this would call AI models for transcription.
    For MVP, we simulate processing with sleep and generate empty output files.
    """
    try:
        # Update state to PROCESSING
        self.update_state(state="PROCESSING", meta={"progress": 0})

        # Simulate AI processing time (5 seconds)
        time.sleep(5)

        # Create output directory for this task
        task_output_dir = Path(OUTPUT_DIR) / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Get base filename without extension
        base_name = Path(input_file).stem

        # Generate empty .mid file
        mid_file = task_output_dir / f"{base_name}.mid"
        mid_file.touch()

        # Generate empty .pdf file
        pdf_file = task_output_dir / f"{base_name}.pdf"
        pdf_file.touch()

        # Return result with file paths
        return {
            "status": "SUCCESS",
            "task_id": task_id,
            "files": {
                "midi": str(mid_file),
                "pdf": str(pdf_file)
            }
        }

    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
