from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from par.core.observation import Observation


class CameraSource(ABC):
    @abstractmethod
    def capture(self) -> dict[str, Any]:
        """Returns a frame descriptor: at minimum {"detections": [...]}."""


def merge_detections(observation: Observation, camera: CameraSource) -> Observation:
    frame = camera.capture()
    return observation.model_copy(
        update={"detections": [*observation.detections, *frame.get("detections", [])]}
    )
