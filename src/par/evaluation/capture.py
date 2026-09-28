from __future__ import annotations

from par.telemetry.events import TelemetryEvent


class CapturingTelemetryLogger:
    """Drop-in replacement for TelemetryLogger (duck-typed - Runtime only
    ever calls .log(event)) that keeps events in memory instead of printing
    them. Runtime doesn't retain its own per-step trace, so this is how
    experiments inspect safety decisions, latencies, and rejection reasons
    across a run rather than just the final list[ActionResult]."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def log(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def outcome_counts(self) -> dict[str, int]:
        counts = {"allow": 0, "modify": 0, "deny": 0, "escalate": 0}
        for event in self.events:
            counts[event.safety_decision] = counts.get(event.safety_decision, 0) + 1
        return counts

    def latencies(self) -> list[float]:
        return [event.latency_seconds for event in self.events]
