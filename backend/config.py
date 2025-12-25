import os

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# File storage paths
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
