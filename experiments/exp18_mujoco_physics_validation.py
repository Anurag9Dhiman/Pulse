"""Experiment 18: does the Safety Kernel's collision test agree with physics?

Replaces the earlier Webots trial. Every previous collision result in this
suite was self-referential: the kernel was asked whether an action was safe,
and "safe" was defined by the same geometry the kernel used. Here MuJoCo is
the arbiter. Each proposed motion is (a) judged by the kernel and (b)
executed in a fresh simulation, where the contact solver decides
independently whether the robot actually touched anything.

That yields a confusion matrix against ground truth rather than against the
kernel's own model:

    kernel DENY  + contact    -> correctly prevented
    kernel ALLOW + no contact -> correctly permitted
    kernel ALLOW + contact    -> UNSAFE EXECUTION (the failure that matters)
    kernel DENY  + no contact -> over-cautious (a cost, not a hazard)

Two kernels are scored on identical configurations: the swept test used by
PAR, and the endpoint-only test it replaced. The endpoint-only variant is
what the earlier version of this work shipped and measured as perfect, since
nothing ever checked it against physics.

Soundness requires the kernel's collision_margin to cover the two bodies'
combined radii (0.08 + 0.12 = 0.20m); MARGIN is set above that. A margin
below it would admit contacts no amount of path-sweeping could prevent,
because the kernel reasons about points and MuJoCo simulates volumes.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from par.core.action import Action
from par.core.capability import Capability, RiskLevel
from par.robots.mujoco_robot import MuJoCoRobot
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome

N_TRIALS = 300
SEED = 18
ROBOT_RADIUS = 0.08
OBSTACLE_RADIUS = 0.12
MARGIN = 0.24  # > 0.20 combined radii, with a small buffer
REACH = 1.6


class EndpointOnlyKernel(SafetyKernel):
    """The superseded check: measures only the target, never the path."""

    def _check_collision(self, origin, target, obstacles):
        # Takes the obstacle list the perception contract has already
        # parsed, like the real kernel. The ablated behaviour is unchanged:
        # measure the target alone and ignore the path taken to reach it.
        from par.safety.policy import SafetyDecision

        for obstacle in obstacles:
            if math.dist(target, obstacle.position) < self.profile.collision_margin:
                return SafetyDecision(
                    outcome=PolicyOutcome.DENY, reason=f"collision risk at target {target}"
                )
        return None


def _profile() -> EnvironmentProfile:
    return EnvironmentProfile(
        name="mujoco",
        workspace=Workspace(x=(-REACH, REACH), y=(-REACH, REACH), z=(-1.0, 1.0)),
        max_velocity=10.0,  # generous: this experiment isolates collision, not velocity
        action_timeout_seconds=1.0,
        approval_required=False,
        collision_margin=MARGIN,
    )


def _configurations(rng: random.Random) -> list[dict]:
    configs = []
    while len(configs) < N_TRIALS:
        target = (rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2))
        if math.hypot(*target) < 0.4:
            continue  # too short to be interesting
        # Bias obstacles toward the path so both outcomes are well represented.
        fraction = rng.uniform(0.2, 0.9)
        lateral = rng.uniform(-0.5, 0.5)
        length = math.hypot(*target)
        perpendicular = (-target[1] / length, target[0] / length)
        obstacle = (
            target[0] * fraction + perpendicular[0] * lateral,
            target[1] * fraction + perpendicular[1] * lateral,
        )
        if math.hypot(*obstacle) < ROBOT_RADIUS + OBSTACLE_RADIUS + 0.02:
            continue  # the robot would start already in contact
        configs.append({"target": target, "obstacle": obstacle})
    return configs


def _verdict(kernel: SafetyKernel, robot: MuJoCoRobot, target, capability) -> PolicyOutcome:
    action = Action(
        action_id="a",
        skill_name="move",
        parameters={"x": target[0], "y": target[1], "z": 0.0},
        created_at=datetime.now(timezone.utc),
    )
    return kernel.check(action, robot.get_observation(), capability).outcome


def _score(name: str, kernel_class, configs, capability) -> dict:
    matrix = {"prevented": 0, "permitted": 0, "unsafe_execution": 0, "over_cautious": 0}
    for config in configs:
        target, obstacle = config["target"], config["obstacle"]
        obstacles = [("obstacle", obstacle[0], obstacle[1], OBSTACLE_RADIUS)]

        judged = MuJoCoRobot(obstacles=obstacles, robot_radius=ROBOT_RADIUS)
        outcome = _verdict(kernel_class(_profile()), judged, target, capability)

        # Ground truth: run the motion in its own simulation regardless of
        # the verdict, so denied actions are checked too.
        trial = MuJoCoRobot(obstacles=obstacles, robot_radius=ROBOT_RADIUS)
        trial.execute(
            Action(
                action_id="g",
                skill_name="move",
                parameters={"x": target[0], "y": target[1], "z": 0.0},
                created_at=datetime.now(timezone.utc),
            )
        )
        collided = bool(trial.last_contacts)

        allowed = outcome in (PolicyOutcome.ALLOW, PolicyOutcome.MODIFY)
        if allowed and collided:
            matrix["unsafe_execution"] += 1
        elif allowed:
            matrix["permitted"] += 1
        elif collided:
            matrix["prevented"] += 1
        else:
            matrix["over_cautious"] += 1

    total = sum(matrix.values())
    hazards = matrix["prevented"] + matrix["unsafe_execution"]
    safe = matrix["permitted"] + matrix["over_cautious"]
    return {
        "kernel": name,
        "counts": matrix,
        "n": total,
        "unsafe_execution_rate": matrix["unsafe_execution"] / hazards if hazards else 0.0,
        "interception_rate": matrix["prevented"] / hazards if hazards else 0.0,
        "false_denial_rate": matrix["over_cautious"] / safe if safe else 0.0,
        "n_truly_colliding": hazards,
        "n_truly_clear": safe,
    }


def run() -> dict:
    rng = random.Random(SEED)
    configs = _configurations(rng)
    capability = Capability(
        name="move", description="move", risk=RiskLevel.LOW, env_profiles=["mujoco"]
    )
    swept = _score("swept (PAR)", SafetyKernel, configs, capability)
    endpoint = _score("endpoint-only (superseded)", EndpointOnlyKernel, configs, capability)
    return {
        "experiment": "18_mujoco_physics_validation",
        "simulator": "MuJoCo",
        "n_trials": N_TRIALS,
        "seed": SEED,
        "robot_radius": ROBOT_RADIUS,
        "obstacle_radius": OBSTACLE_RADIUS,
        "collision_margin": MARGIN,
        "results": {"swept": swept, "endpoint_only": endpoint},
        "notes": (
            "Ground truth is MuJoCo's contact solver, computed by executing every "
            "proposed motion in its own simulation regardless of the kernel's verdict. "
            "unsafe_execution_rate is over configurations that genuinely collide; "
            "false_denial_rate is over configurations that genuinely do not."
        ),
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))
    path = Path(__file__).parent / "results" / "exp18_mujoco_physics_validation.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWritten to {path}")
