from __future__ import annotations

import logging

from par.telemetry.events import TelemetryEvent


class TelemetryLogger:
    def __init__(self, logger_name: str = "par.telemetry") -> None:
        self._logger = logging.getLogger(logger_name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False

    def log(self, event: TelemetryEvent) -> None:
        self._logger.info(event.model_dump_json())
