"""FaceService configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

# Allow an alternate env profile for local vs remote testing.
# If FACE_SERVICE_ENV_FILE is set, load that file from the FaceService root.
env_file = os.getenv("FACE_SERVICE_ENV_FILE")
if env_file:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = BASE_DIR / env_path
    load_dotenv(str(env_path))
else:
    load_dotenv(str(BASE_DIR / ".env"))

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
