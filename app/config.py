"""FaceService configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

BEDMS_CALLBACK_URL = os.getenv(
    "BEDMS_CALLBACK_URL", "http://localhost:3001/v1/face-recognition/callback"
)
FACE_SERVICE_API_KEY = os.getenv("FACE_SERVICE_API_KEY", "")

INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))

DET_SIZE = int(os.getenv("DET_SIZE", "640"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "50"))
