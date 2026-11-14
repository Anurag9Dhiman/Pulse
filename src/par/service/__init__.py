"""Hosted service boundary around the PAR runtime (requires the `service` extra).

Nothing in par.core / par.robots / par.safety imports this package; it wraps
the synchronous runtime without changing it.

V1 is single-process by construction: TaskRegistry and
RobotConnectionRegistry live in this process's memory, so the service must
run as exactly one worker/replica.
"""
