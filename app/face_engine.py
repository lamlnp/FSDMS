"""Face detection and embedding engine using InsightFace (ArcFace + RetinaFace).

Provides:
- detect_faces: find all faces in an image with bounding boxes
- get_embedding: detect the largest face and return its 512D embedding
- detect_and_embed: detect all faces and return embeddings + bboxes
"""

import logging
import os
import sys

# Register NVIDIA pip-installed CUDA DLL directories so onnxruntime can find them.
# Must happen BEFORE importing onnxruntime / insightface.
if sys.platform == "win32":
    _site_pkgs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "venv", "Lib", "site-packages", "nvidia")
    if os.path.isdir(_site_pkgs):
        _dll_dirs = []
        for _pkg in os.listdir(_site_pkgs):
            _bin = os.path.join(_site_pkgs, _pkg, "bin")
            if os.path.isdir(_bin):
                _dll_dirs.append(os.path.abspath(_bin))
                os.add_dll_directory(os.path.abspath(_bin))
        if _dll_dirs:
            os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from insightface.app import FaceAnalysis

from app.config import DET_SIZE, GPU_DEVICE_ID, INSIGHTFACE_MODEL, MIN_FACE_SIZE

logger = logging.getLogger(__name__)


class FaceEngine:
    """Wraps InsightFace buffalo_l model for face detection + ArcFace embedding."""

    def __init__(self) -> None:
        self._app: FaceAnalysis | None = None

    def load(self) -> None:
        """Download (if needed) and initialize the InsightFace model."""
        logger.info(
            "Loading InsightFace model=%s on device=%d ...",
            INSIGHTFACE_MODEL,
            GPU_DEVICE_ID,
        )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if GPU_DEVICE_ID >= 0
            else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
        self._app.prepare(ctx_id=GPU_DEVICE_ID, det_size=(DET_SIZE, DET_SIZE))
        logger.info("InsightFace model loaded successfully")

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            raise RuntimeError("FaceEngine not loaded. Call load() first.")
        return self._app

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """Detect faces and return bounding boxes + detection scores.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            List of dicts with keys: bbox, det_score.
        """
        faces = self.app.get(image)
        results = []
        for face in faces:
            w = face.bbox[2] - face.bbox[0]
            h = face.bbox[3] - face.bbox[1]
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            results.append(
                {
                    "bbox": face.bbox.tolist(),
                    "det_score": float(face.det_score),
                }
            )
        return results

    def get_embedding(self, image: np.ndarray) -> dict | None:
        """Detect the single largest face and return its 512D embedding.

        Used for face registration (expects one clear face in frame).

        Returns:
            Dict with keys: embedding (list[float]), bbox, det_score, face_crop (ndarray).
            None if no face detected.
        """
        faces = self.app.get(image)
        if not faces:
            return None

        # Filter by minimum face size
        valid = [
            f
            for f in faces
            if (f.bbox[2] - f.bbox[0]) >= MIN_FACE_SIZE
            and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_SIZE
        ]
        if not valid:
            return None

        # Pick the largest face by area
        largest = max(
            valid, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        # Crop face from image for storage
        x1, y1, x2, y2 = [int(c) for c in largest.bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = image[y1:y2, x1:x2]

        return {
            "embedding": largest.embedding.tolist(),
            "bbox": largest.bbox.tolist(),
            "det_score": float(largest.det_score),
            "face_crop": face_crop,
        }

    def detect_and_embed(self, image: np.ndarray) -> list[dict]:
        """Detect all faces and return embeddings + bounding boxes.

        Used in the real-time recognition loop.

        Returns:
            List of dicts with keys: embedding (list[float]), bbox, det_score.
        """
        faces = self.app.get(image)
        results = []
        for face in faces:
            w = face.bbox[2] - face.bbox[0]
            h = face.bbox[3] - face.bbox[1]
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            results.append(
                {
                    "embedding": face.embedding.tolist(),
                    "bbox": face.bbox.tolist(),
                    "det_score": float(face.det_score),
                }
            )
        return results


# Module-level singleton
face_engine = FaceEngine()
