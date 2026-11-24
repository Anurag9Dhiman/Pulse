from __future__ import annotations

from datetime import datetime, timezone

from typing import Any, Callable

from par.sensors.base import CameraSource

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only without opencv installed
    cv2 = None


class USBCamera(CameraSource):
    """Captures frames from a local USB/webcam via OpenCV.

    Unverified in this environment - no camera hardware or opencv-python
    installation is available here. Detection (turning a raw frame into
    named objects) is intentionally left as an injectable hook: Week 3 does
    not include a perception model, only the pipeline a real one plugs into.
    """

    def __init__(
        self,
        device_index: int = 0,
        detector: Callable[[Any], list[dict[str, Any]]] | None = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Install it with `pip install par[camera]`.")
        self._device_index = device_index
        self._capture = cv2.VideoCapture(device_index)
        self._detector = detector or (lambda frame: [])

    def capture(self) -> dict[str, Any]:
        ok, frame = self._capture.read()
        if not ok:
            # Declared, not silent. An unplugged or stalled camera used to
            # return an empty detection list, which the Safety Kernel read
            # as a clear workspace and approved motion through.
            return {
                "detections": [],
                "available": False,
                "detail": f"cv2 VideoCapture.read() failed on device {self._device_index}",
            }
        return {
            "detections": self._detector(frame),
            "available": True,
            "captured_at": datetime.now(timezone.utc),
        }

    def release(self) -> None:
        self._capture.release()
