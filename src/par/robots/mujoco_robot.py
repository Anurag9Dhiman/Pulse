"""A planar mobile robot simulated in MuJoCo.

Implements the same two-method RobotInterface as MockRobot, so the Agent,
Runtime and Safety Kernel above it are the unchanged code paths - what
differs is that an executed action now resolves against a real contact
solver instead of a dictionary update.

That is the point of using it: MuJoCo decides, independently of the Safety
Kernel, whether a motion actually touched anything. Contacts reported here
are physics ground truth, which is what lets Experiment 18 ask whether the
kernel's swept collision test agrees with what the simulator does, rather
than only whether the kernel agrees with its own geometry.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.robots.base import RobotInterface

_SCENE = """
<mujoco model="par_planar">
  <option timestep="{timestep}" gravity="0 0 0"/>
  <default>
    <geom friction="0.4 0.005 0.0001" solref="0.004 1"/>
  </default>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="robot" pos="{start_x} {start_y} {height}">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="{damping}"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="{damping}"/>
      <geom name="robot_geom" type="sphere" size="{robot_radius}" rgba="0.1 0.4 0.8 1" mass="1"/>
    </body>
{obstacles}
  </worldbody>
</mujoco>
"""

_OBSTACLE = (
    '    <body name="{name}" pos="{x} {y} {height}">\n'
    '      <geom name="{name}_geom" type="cylinder" size="{radius} {half_height}"'
    ' rgba="0.8 0.2 0.2 1"/>\n'
    "    </body>"
)


class MuJoCoRobot(RobotInterface):
    """Drives a sphere across a plane toward a commanded target.

    `move` is a closed-loop velocity command stepped through the simulator
    until the robot arrives, stalls, or runs out of simulated time - not a
    teleport. A commanded motion whose straight-line path is blocked
    therefore *fails in physics*, which is what makes a Safety Kernel denial
    checkable against something other than the kernel's own model.
    """

    def __init__(
        self,
        obstacles: list[tuple[str, float, float, float]] | None = None,
        start: tuple[float, float] = (0.0, 0.0),
        robot_radius: float = 0.08,
        speed: float = 0.5,
        timestep: float = 0.004,
        max_seconds: float = 20.0,
        tolerance: float = 0.02,
    ) -> None:
        import mujoco  # imported lazily: par installs without the sim extra

        self._mujoco = mujoco
        self._obstacles = list(obstacles or [])
        self._robot_radius = robot_radius
        self._speed = speed
        self._max_steps = int(max_seconds / timestep)
        self._tolerance = tolerance
        self._holding: str | None = None

        body = "\n".join(
            _OBSTACLE.format(name=name, x=x, y=y, radius=radius, height=0.1, half_height=0.1)
            for name, x, y, radius in self._obstacles
        )
        xml = _SCENE.format(
            timestep=timestep,
            start_x=start[0],
            start_y=start[1],
            height=0.1,
            damping=8.0,
            robot_radius=robot_radius,
            obstacles=body,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.last_contacts: list[str] = []

    # ------------------------------------------------------------ state

    @property
    def position(self) -> tuple[float, float, float]:
        return (float(self.data.qpos[0]), float(self.data.qpos[1]), 0.0)

    def _contacting(self) -> list[str]:
        """Obstacle names currently in contact with the robot geom."""
        touched = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            names = {
                self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, g)
                for g in (contact.geom1, contact.geom2)
            }
            if "robot_geom" not in names:
                continue
            for name in names - {"robot_geom", "floor"}:
                touched.append(name.removesuffix("_geom"))
        return touched

    def get_observation(self) -> Observation:
        x, y, _ = self.position
        return Observation(
            observation_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            source="mujoco",
            robot_state={"position": {"x": x, "y": y, "z": 0.0}, "holding": self._holding},
            detections=[
                {"name": name, "position": {"x": ox, "y": oy, "z": 0.0}, "radius": radius}
                for name, ox, oy, radius in self._obstacles
            ],
            raw={"ncon": int(self.data.ncon)},
        )

    # ---------------------------------------------------------- actions

    def execute(self, action: Action) -> ActionResult:
        handler = getattr(self, f"_do_{action.skill_name}", None)
        if handler is None:
            return self._result(action, False, f"mujoco robot cannot execute '{action.skill_name}'")
        return handler(action)

    def _do_move(self, action: Action) -> ActionResult:
        target = (
            float(action.parameters.get("x", 0.0)),
            float(action.parameters.get("y", 0.0)),
        )
        contacts: set[str] = set()
        for _ in range(self._max_steps):
            x, y, _z = self.position
            dx, dy = target[0] - x, target[1] - y
            distance = math.hypot(dx, dy)
            if distance <= self._tolerance:
                break
            self.data.qvel[0] = self._speed * dx / distance
            self.data.qvel[1] = self._speed * dy / distance
            self._mujoco.mj_step(self.model, self.data)
            contacts.update(self._contacting())

        self.data.qvel[0] = self.data.qvel[1] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

        self.last_contacts = sorted(contacts)
        x, y, _z = self.position
        remaining = math.dist((x, y), target)
        arrived = remaining <= self._tolerance * 2
        if contacts:
            # Physics, not the kernel, is reporting this.
            return self._result(
                action, False, f"collided with {', '.join(sorted(contacts))} en route to {target}"
            )
        if not arrived:
            return self._result(action, False, f"stalled {remaining:.3f}m short of {target}")
        return self._result(action, True, f"moved to ({x:.3f}, {y:.3f})")

    def _do_stop(self, action: Action) -> ActionResult:
        self.data.qvel[0] = self.data.qvel[1] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        return self._result(action, True, "stopped")

    def _do_detect(self, action: Action) -> ActionResult:
        return self._result(action, True, f"detected {len(self._obstacles)} object(s)")

    def _do_pick(self, action: Action) -> ActionResult:
        name = action.parameters.get("object")
        if self._holding is not None:
            return self._result(action, False, f"already holding {self._holding}")
        self._holding = name
        return self._result(action, True, f"picked {name}")

    def _do_place(self, action: Action) -> ActionResult:
        if self._holding is None:
            return self._result(action, False, "not holding anything")
        held, self._holding = self._holding, None
        return self._result(action, True, f"placed {held}")

    def _do_inspect(self, action: Action) -> ActionResult:
        return self._result(action, True, f"inspected {action.parameters.get('target')}")

    def _result(self, action: Action, success: bool, message: str) -> ActionResult:
        return ActionResult(
            action_id=action.action_id,
            success=success,
            message=message,
            completed_at=datetime.now(timezone.utc),
        )

    def state(self) -> dict[str, Any]:
        x, y, _ = self.position
        return {"x": x, "y": y, "contacts": self.last_contacts}
