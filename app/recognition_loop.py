"""Recognition loop — captures frames, runs face detection + embedding,
and POSTs results to the BEDMS callback endpoint.

Each active camera gets its own asyncio task that runs at the camera's
configured FPS target.
"""

import asyncio
import logging
import time

import httpx
import numpy as np

from app.camera_manager import CameraInfo, camera_manager
from app.config import BEDMS_CALLBACK_URL, FACE_SERVICE_API_KEY
from app.face_engine import face_engine
from app.utils import draw_detections, encode_frame_base64

logger = logging.getLogger(__name__)

# Active recognition tasks keyed by camera_id
_tasks: dict[str, asyncio.Task] = {}


async def _recognition_worker(camera_id: str) -> None:
    """Main recognition loop for a single camera."""
    info = camera_manager.get_camera(camera_id)
    if not info:
        return

    interval = 1.0 / info.fps_target
    frame_count = 0
    fps_window_start = time.monotonic()

    logger.info(
        "Recognition loop started for camera=%s (target=%d fps)",
        camera_id,
        info.fps_target,
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        while info.get_status() == "active":
            loop_start = time.monotonic()

            frame = camera_manager.get_frame(camera_id)
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # Run face detection + embedding (CPU-bound, run in thread)
            detections = await asyncio.to_thread(
                face_engine.detect_and_embed, frame
            )

            frame_count += 1

            # Update FPS estimate every second
            elapsed_window = time.monotonic() - fps_window_start
            if elapsed_window >= 1.0:
                info.update_stats(frame_count / elapsed_window, time.time())
                frame_count = 0
                fps_window_start = time.monotonic()

            # Build callback payload
            annotated_frame = draw_detections(frame, detections)
            frame_b64 = encode_frame_base64(annotated_frame, quality=70)

            payload = {
                "camera_id": camera_id,
                "camera_type": info.camera_type,
                "timestamp": time.time(),
                "detections": [
                    {
                        "bbox": d["bbox"],
                        "embedding": d["embedding"],
                        "det_score": d["det_score"],
                    }
                    for d in detections
                ],
                "frame_base64": frame_b64,
            }

            # POST to BEDMS callback (retry up to 2 times with backoff)
            for attempt in range(3):
                try:
                    resp = await client.post(
                        BEDMS_CALLBACK_URL,
                        json=payload,
                        headers={"X-API-Key": FACE_SERVICE_API_KEY},
                    )
                    resp.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.HTTPStatusError) as e:
                    if attempt < 2:
                        await asyncio.sleep(0.3 * (attempt + 1))
                    else:
                        logger.warning(
                            "Callback to BEDMS failed after 3 attempts for camera=%s: %s",
                            camera_id,
                            e,
                        )

            # Maintain target FPS
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    logger.info("Recognition loop stopped for camera=%s", camera_id)


def start_recognition(camera_id: str) -> None:
    """Start the recognition loop for a camera as an asyncio task."""
    if camera_id in _tasks and not _tasks[camera_id].done():
        logger.warning("Recognition loop already running for camera=%s", camera_id)
        return

    loop = asyncio.get_event_loop()
    task = loop.create_task(_recognition_worker(camera_id))
    _tasks[camera_id] = task
    logger.info("Recognition task created for camera=%s", camera_id)


def stop_recognition(camera_id: str) -> None:
    """Stop the recognition loop for a camera."""
    task = _tasks.pop(camera_id, None)
    if task and not task.done():
        task.cancel()
        logger.info("Recognition task cancelled for camera=%s", camera_id)
