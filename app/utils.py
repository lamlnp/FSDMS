"""Utility functions for image encoding, decoding, and similarity math."""

import base64
import io
import cv2
import numpy as np
from PIL import Image


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG) to a BGR numpy array (OpenCV format)."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def encode_frame_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode a BGR frame to a base64 JPEG string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on a frame. Returns a copy."""
    annotated = frame.copy()
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        score = det.get("det_score", 0)
        label = det.get("label", "")
        confidence = det.get("confidence")

        # Green for matched, yellow for unmatched
        color = (0, 255, 0) if det.get("is_match") else (0, 255, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label text
        if label:
            text = f"{label} ({confidence:.0%})" if confidence else label
        else:
            text = f"{score:.2f}"

        text_y = max(y1 - 10, 20)
        cv2.putText(
            annotated, text, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return annotated


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
