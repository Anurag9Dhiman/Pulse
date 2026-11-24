from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from par.core.observation import ChannelReport, Observation
from par.core.perception import PERCEPTION


class CameraSource(ABC):
    @abstractmethod
    def capture(self) -> dict[str, Any]:
        """Returns a frame descriptor: at minimum {"detections": [...]}.

        A camera that could not produce a frame reports
        `{"detections": [], "available": False, "detail": "<why>"}` rather
        than an empty list alone. The two are not the same claim, and
        collapsing them is what let a dead camera read as clear space.
        May also carry "captured_at" (a datetime) for freshness checks.
        """


def merge_detections(observation: Observation, camera: CameraSource) -> Observation:
    """Folds a camera's frame into an observation, health included.

    The channel report travels with the detections. A failed capture
    marks perception unavailable instead of contributing nothing
    noticeable: the merged observation would otherwise look exactly like
    one from a working camera that saw an empty room, and the Safety
    Kernel would clear every path through the space the camera was
    supposed to be watching.
    """
    frame = camera.capture()
    available = bool(frame.get("available", True))
    report = ChannelReport(
        available=available,
        captured_at=frame.get("captured_at") or datetime.now(timezone.utc),
        detail=str(frame.get("detail", "")),
    )
    channels = {**observation.channels, PERCEPTION: report}
    return observation.model_copy(
        update={
            "detections": [*observation.detections, *frame.get("detections", [])],
            "channels": channels,
        }
    )
