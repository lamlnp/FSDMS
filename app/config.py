"""FaceService configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(str(BASE_DIR / ".env"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

BEDMS_CALLBACK_URL = os.getenv(
    "BEDMS_CALLBACK_URL", "https://bedms-production.up.railway.app/v1/face-recognition/callback"
)
FACE_SERVICE_API_KEY = os.getenv("FACE_SERVICE_API_KEY", "")


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)

    values = [value.strip() for value in raw.split(",")]
    return [value for value in values if value]


# Browser origins that are allowed to call FaceService directly.
# Default to the deployed FEDMS origin used in the ngrok-backed demo flow.
CORS_ALLOW_ORIGINS = _parse_csv_env(
    "CORS_ALLOW_ORIGINS",
    [
        "https://fedms.vercel.app",
    ],
)

INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))

DET_SIZE = int(os.getenv("DET_SIZE", "640"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "50"))
