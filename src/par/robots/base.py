from __future__ import annotations

from abc import ABC, abstractmethod

from par.core.action import Action, ActionResult
from par.core.observation import Observation


class RobotInterface(ABC):
    @abstractmethod
    def get_observation(self) -> Observation: ...

    @abstractmethod
    def execute(self, action: Action) -> ActionResult: ...
