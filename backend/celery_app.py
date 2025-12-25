from celery import Celery
from backend.config import REDIS_URL

celery_app = Celery(
    "cloudmusic_transcribe",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_extended=True,
)
