"""Camera capture manager.

Supports webcam (by index) and RTSP sources through OpenCV VideoCapture.
Handles shared-webcam mode: when multiple camera configs point to the same
physical device (e.g., source_url="0"), the device is opened once and frames
are shared across all consumers.

Migration from webcam to RTSP requires only a config change — no code changes.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Runtime state for a camera source.

    Mutable fields (status, fps_actual, last_frame_at, error_message) are
    read/written from multiple threads.  Use the lock for safe access.
    """

    camera_id: str
    source_type: str  # "webcam" or "rtsp"
    source_url: str  # webcam index as string ("0") or RTSP URL
    camera_type: str  # "checkin" or "checkout"
    fps_target: int = 5
    recognition_threshold: float = 0.6
    status: str = "offline"  # "active", "offline", "error"
    fps_actual: float = 0.0
    last_frame_at: float = 0.0
    error_message: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_stats(self, fps: float, frame_time: float) -> None:
        with self._lock:
            self.fps_actual = fps
            self.last_frame_at = frame_time

    def set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error_message = error

    def get_status(self) -> str:
        with self._lock:
            return self.status

    def snapshot(self) -> dict:
        """Return a thread-safe copy of mutable state."""
        with self._lock:
            return {
                "status": self.status,
                "fps_actual": self.fps_actual,
                "last_frame_at": self.last_frame_at,
                "error_message": self.error_message,
            }


class _SharedCapture:
    """Wraps a single cv2.VideoCapture shared by multiple CameraInfo entries."""

    # Number of consecutive read failures before reporting error
    _FAIL_THRESHOLD = 50  # ~2.5 seconds at 50ms sleep
    # RTSP reconnect settings
    _MAX_RECONNECT_ATTEMPTS = 3
    _RECONNECT_BACKOFF_BASE = 2  # seconds — 2, 4, 8

    def __init__(self, source, on_error=None) -> None:
        self.source = source
        self._on_error = on_error  # callback(error_msg: str)
        self._is_rtsp = isinstance(source, str) and source.startswith("rtsp")
        self.capture = self._open_capture()
        logger.info("VideoCapture opened (source=%s, isOpened=%s)", source, self.capture.isOpened())
        self.ref_count = 0
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.running = False
        self.thread: threading.Thread | None = None

    def _open_capture(self) -> cv2.VideoCapture:
        """Open a VideoCapture with the appropriate backend."""
        if isinstance(self.source, int):
            # DirectShow on Windows for reliable webcam access
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            if self._is_rtsp:
                # Prefer TCP transport for RTSP streams to reduce packet loss
                # and stalls on unstable Wi-Fi / LAN segments.
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            # FFMPEG backend for RTSP / file sources
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self._is_rtsp:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
                read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
                if open_timeout is not None:
                    cap.set(open_timeout, 5000)
                if read_timeout is not None:
                    cap.set(read_timeout, 5000)
        return cap

    def start(self) -> None:
        if self.running:
            return
        if not self.capture.isOpened():
            self.capture = self._open_capture()
            logger.info("VideoCapture re-opened (source=%s, isOpened=%s)", self.source, self.capture.isOpened())
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the source. Returns True on success."""
        for attempt in range(1, self._MAX_RECONNECT_ATTEMPTS + 1):
            backoff = self._RECONNECT_BACKOFF_BASE ** attempt
            logger.warning(
                "Reconnect attempt %d/%d for source=%s in %ds",
                attempt, self._MAX_RECONNECT_ATTEMPTS, self.source, backoff,
            )
            time.sleep(backoff)
            if not self.running:
                return False
            try:
                if self.capture.isOpened():
                    self.capture.release()
                self.capture = self._open_capture()
                if self.capture.isOpened():
                    ret, _ = self.capture.read()
                    if ret:
                        logger.info("Reconnected to source=%s on attempt %d", self.source, attempt)
                        return True
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %s", attempt, e)
        return False

    def _read_loop(self) -> None:
        """Continuously read frames so the buffer stays fresh."""
        # Discard first few frames — webcams on Windows often return
        # garbage/uninitialized buffers before auto-exposure stabilises.
        # RTSP streams don't need this warmup.
        if not self._is_rtsp:
            for _ in range(10):
                if not self.running:
                    return
                self.capture.read()

        consecutive_failures = 0
        error_reported = False

        while self.running:
            ret, frame = self.capture.read()
            if ret:
                consecutive_failures = 0
                if error_reported:
                    error_reported = False
                    logger.info("Camera source=%s recovered", self.source)
                    if self._on_error:
                        self._on_error(None)  # clear error
                with self.lock:
                    self.latest_frame = frame
            else:
                consecutive_failures += 1
                if consecutive_failures >= self._FAIL_THRESHOLD and not error_reported:
                    error_reported = True
                    msg = f"Camera source={self.source} disconnected ({consecutive_failures} consecutive read failures)"
                    logger.error(msg)
                    if self._on_error:
                        self._on_error(msg)

                    # RTSP: attempt auto-reconnect
                    if self._is_rtsp:
                        if self._try_reconnect():
                            consecutive_failures = 0
                            error_reported = False
                            if self._on_error:
                                self._on_error(None)  # clear error
                            continue
                        else:
                            logger.error("All reconnect attempts failed for source=%s, stopping", self.source)
                            break

                time.sleep(0.05)

    def get_frame(self) -> np.ndarray | None:
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.capture.isOpened():
            self.capture.release()
        self.latest_frame = None


class CameraManager:
    """Manages multiple camera sources with shared-device deduplication."""

    def __init__(self) -> None:
        self._cameras: dict[str, CameraInfo] = {}
        self._shared_captures: dict[str | int, _SharedCapture] = {}
        self._lock = threading.Lock()

    def _resolve_source(self, info: CameraInfo) -> str | int:
        """Convert source_url to the right type for OpenCV."""
        if info.source_type == "webcam":
            return int(info.source_url)
        return info.source_url

    def add_camera(
        self,
        camera_id: str,
        source_type: str,
        source_url: str,
        camera_type: str,
        fps_target: int = 5,
        recognition_threshold: float = 0.6,
    ) -> CameraInfo:
        """Register a camera config. Does not start capture yet."""
        info = CameraInfo(
            camera_id=camera_id,
            source_type=source_type,
            source_url=source_url,
            camera_type=camera_type,
            fps_target=fps_target,
            recognition_threshold=recognition_threshold,
        )
        with self._lock:
            self._cameras[camera_id] = info
        return info

    def start_camera(self, camera_id: str) -> CameraInfo:
        """Start capture for a registered camera."""
        with self._lock:
            info = self._cameras.get(camera_id)
            if not info:
                raise ValueError(f"Camera {camera_id} not registered")

            source = self._resolve_source(info)

            # Reuse shared capture if device already opened
            if source not in self._shared_captures:
                def _make_error_cb(src):
                    def _on_error(msg):
                        for cam in self._cameras.values():
                            if self._resolve_source(cam) == src:
                                if msg:
                                    cam.set_status("error", msg)
                                else:
                                    cam.set_status("active")
                    return _on_error

                shared = _SharedCapture(source, on_error=_make_error_cb(source))
                self._shared_captures[source] = shared
            else:
                shared = self._shared_captures[source]

            shared.ref_count += 1
            shared.start()

            info.set_status("active")
            logger.info("Camera %s started (source=%s)", camera_id, source)
            return info

    def stop_camera(self, camera_id: str) -> CameraInfo:
        """Stop capture for a camera. Releases shared device when ref_count hits 0."""
        with self._lock:
            info = self._cameras.get(camera_id)
            if not info:
                raise ValueError(f"Camera {camera_id} not registered")

            source = self._resolve_source(info)
            shared = self._shared_captures.get(source)

            if shared:
                shared.ref_count -= 1
                if shared.ref_count <= 0:
                    shared.stop()
                    del self._shared_captures[source]

            info.set_status("offline")
            info.update_stats(0.0, info.last_frame_at)
            logger.info("Camera %s stopped", camera_id)
            return info

    def get_frame(self, camera_id: str) -> np.ndarray | None:
        """Get the latest frame for a camera."""
        info = self._cameras.get(camera_id)
        if not info:
            return None
        source = self._resolve_source(info)
        shared = self._shared_captures.get(source)
        if not shared:
            return None
        return shared.get_frame()

    def get_camera(self, camera_id: str) -> CameraInfo | None:
        return self._cameras.get(camera_id)

    def get_all_cameras(self) -> list[CameraInfo]:
        return list(self._cameras.values())

    def get_active_cameras(self) -> list[CameraInfo]:
        return [c for c in self._cameras.values() if c.get_status() == "active"]

    def stop_all(self) -> None:
        """Stop all cameras and release all devices."""
        for camera_id in list(self._cameras.keys()):
            try:
                self.stop_camera(camera_id)
            except Exception:
                pass


# Module-level singleton
camera_manager = CameraManager()
