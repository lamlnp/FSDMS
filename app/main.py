"""FaceService — FastAPI application for face detection and embedding.

Endpoints:
    GET  /health          Health check + model status
    POST /detect          Detect faces in an uploaded image (bboxes only)
    POST /register        Detect largest face, return 512D embedding + quality
    POST /cameras/{id}/start   Start camera capture + recognition loop
    POST /cameras/{id}/stop    Stop camera capture
    GET  /cameras/{id}/status  Get camera status
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.camera_manager import camera_manager
from app.config import HOST, PORT
from app.face_engine import face_engine
from app.recognition_loop import start_recognition, stop_recognition
from app.utils import decode_image_bytes, encode_frame_base64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the face engine model on startup, clean up cameras on shutdown."""
    face_engine.load()
    yield
    camera_manager.stop_all()


class CameraStartRequest(BaseModel):
    """Request body for starting a camera."""

    source_type: str = "webcam"  # "webcam" or "rtsp"
    source_url: str = "0"  # webcam index or RTSP URL
    camera_type: str = "checkin"  # "checkin" or "checkout"
    fps_target: int = 5
    recognition_threshold: float = 0.6

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        if v not in ("webcam", "rtsp"):
            raise ValueError('source_type must be "webcam" or "rtsp"')
        return v

    @field_validator("camera_type")
    @classmethod
    def validate_camera_type(cls, v: str) -> str:
        if v not in ("checkin", "checkout"):
            raise ValueError('camera_type must be "checkin" or "checkout"')
        return v

    @field_validator("fps_target")
    @classmethod
    def validate_fps(cls, v: int) -> int:
        if v < 1 or v > 30:
            raise ValueError("fps_target must be between 1 and 30")
        return v

    @field_validator("recognition_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("recognition_threshold must be between 0.0 and 1.0")
        return v


app = FastAPI(
    title="FaceService",
    description="Face detection and embedding service for FPT Dormitory Management System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check. Returns model status."""
    return {
        "status": "ok",
        "model_loaded": face_engine._app is not None,
        "service": "FaceService",
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    """Detect faces in an uploaded image.

    Returns bounding boxes and detection confidence scores (no embeddings).
    Used by the manager UI to preview face detection before registration.
    """
    data = await image.read()
    try:
        img = decode_image_bytes(data)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    detections = face_engine.detect_faces(img)
    return {
        "success": True,
        "faces_count": len(detections),
        "detections": detections,
    }


@app.post("/register")
async def register(image: UploadFile = File(...)):
    """Detect the largest face in the uploaded image and return its 512D embedding.

    Used by BEDMS when a manager registers a student's face.

    Returns:
        embedding: list of 512 floats (L2-normalized ArcFace vector)
        bbox: [x1, y1, x2, y2] of the detected face
        det_score: face detection confidence (0-1)
        quality_score: same as det_score, used for registration quality assessment
        face_crop_base64: JPEG-encoded crop of the detected face
    """
    data = await image.read()
    try:
        img = decode_image_bytes(data)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = face_engine.get_embedding(img)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the image. Ensure the face is clearly visible and well-lit.",
        )

    # Encode face crop as base64 JPEG for storage
    face_crop_b64 = encode_frame_base64(result["face_crop"], quality=90)

    return {
        "success": True,
        "embedding": result["embedding"],
        "bbox": result["bbox"],
        "det_score": result["det_score"],
        "quality_score": result["det_score"],
        "face_crop_base64": face_crop_b64,
    }


# ---------------------------------------------------------------------------
# Camera endpoints
# ---------------------------------------------------------------------------


@app.post("/cameras/{camera_id}/start")
async def camera_start(camera_id: str, body: CameraStartRequest):
    """Register (if new) and start camera capture + recognition loop."""
    info = camera_manager.get_camera(camera_id)
    if info and info.get_status() == "active":
        return {"success": True, "message": "Camera already active", "camera_id": camera_id}

    # Register camera if not yet known
    if not info:
        camera_manager.add_camera(
            camera_id=camera_id,
            source_type=body.source_type,
            source_url=body.source_url,
            camera_type=body.camera_type,
            fps_target=body.fps_target,
            recognition_threshold=body.recognition_threshold,
        )

    try:
        camera_manager.start_camera(camera_id)
        start_recognition(camera_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "message": "Camera started", "camera_id": camera_id}


@app.post("/cameras/{camera_id}/stop")
async def camera_stop(camera_id: str):
    """Stop camera capture and recognition loop."""
    info = camera_manager.get_camera(camera_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    stop_recognition(camera_id)
    camera_manager.stop_camera(camera_id)

    return {"success": True, "message": "Camera stopped", "camera_id": camera_id}


@app.get("/cameras/{camera_id}/status")
async def camera_status(camera_id: str):
    """Get camera runtime status."""
    info = camera_manager.get_camera(camera_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    snap = info.snapshot()
    return {
        "camera_id": info.camera_id,
        "status": snap["status"],
        "camera_type": info.camera_type,
        "source_type": info.source_type,
        "fps_target": info.fps_target,
        "fps_actual": round(snap["fps_actual"], 1),
        "last_frame_at": snap["last_frame_at"],
        "error_message": snap["error_message"],
    }


@app.get("/cameras")
async def cameras_list():
    """List all registered cameras and their status."""
    cameras = camera_manager.get_all_cameras()
    return {
        "cameras": [
            {
                "camera_id": c.camera_id,
                "status": c.get_status(),
                "camera_type": c.camera_type,
                "source_type": c.source_type,
                "fps_actual": round(c.snapshot()["fps_actual"], 1),
            }
            for c in cameras
        ]
    }


# ---------------------------------------------------------------------------
# Run with: python -m app.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
