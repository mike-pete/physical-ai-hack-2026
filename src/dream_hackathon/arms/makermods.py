"""Both arms of a Metal or Maker station through MakerMods' own stack.

- **The followers** are MakerMods' LeRobot followers (github.com/makermods-robotics/lerobot, the commit makermodslab
  pins): `MetalFollower` (Damiao motors) or `MakerFollower` (RobStride motors), one CAN adapter per arm. They clamp
  every target to the vendor's soft limits and ease into the first target slowly (their start-up sync).
- **Start-up and shut-down** follow makermodslab's teleoperation (github.com/makermods-robotics/makermodslab): its
  gripper holding-torque controller and its follower speed cap are installed before connecting; at the end both arms
  return to where they started at makermodslab's gentle rate, then their torque is released.
- **Units**: the followers speak motor degrees, zero at the arm's calibration pose. Joints become radians, nothing
  else changed (the frame Dream's Metal and Maker models read). The gripper maps linearly from its motor angle at
  `gripper_closed_deg` (0) to `gripper_open_deg` (1), the angles `dreamcli setup` records, so recording and
  `dreamcli robot run` map it the same way.

Both arms are brakeless: start a session with them resting where they can go limp, since that is where they return
and are released. `connect(calibrate=False)` never re-zeroes an arm; calibrate it in makermodslab first. A dry run
opens each bus with torque off, as makermodslab's calibration does, and only reads.

@see MakerModsArms
"""

import concurrent.futures
import logging
import math
from typing import Any

import numpy as np

from dream_hackathon import config, ik
from dream_hackathon.arms import Grips, Joints

_log = logging.getLogger(__name__)

## The follower's arm motors, base to wrist, then its gripper motor.
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
GRIPPER = "gripper"


class MakerModsArms:
    """Both Metal or Maker arms (see Arms)."""

    def __init__(self, station: config.StationConfig, *, energize: bool) -> None:
        """Opens both followers.

        @param station The station config (`arm` is `metal` or `maker`).
        @param energize Hold and drive the arms; False opens them with torque off and only reads.
        """
        from makermodslab.arms import MAKER, METAL

        mm = station.makermods
        assert mm is not None
        if station.arm == "metal":
            from lerobot.robots.metal_follower import MetalFollower as Follower
            from lerobot.robots.metal_follower import MetalFollowerConfig as FollowerConfig

            self._family: Any = METAL
        else:
            from lerobot.robots.maker_follower import MakerFollower as Follower
            from lerobot.robots.maker_follower import MakerFollowerConfig as FollowerConfig

            self._family = MAKER
        self.energized = energize
        self._arms = station.arms
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="makermods")
        self._devices: dict[str, Any] = {}
        self._rest_poses: list[tuple[Any, dict[str, float]]] = []
        try:
            for side in ("left", "right"):
                device = Follower(
                    FollowerConfig(
                        port=station.arms[side].port,
                        id=f"dream_hackathon_{side}",
                        can_interface=config.arm_can_interface(station, side),
                    )
                )
                self._open(device)
                self._devices[side] = device
        except BaseException:
            self.close()
            raise
        if energize:
            # Where each arm is now, gripper excluded: the shutdown returns there before letting go.
            for device in self._devices.values():
                self._rest_poses.extend(self._family.capture_rest_poses(device))
        self.start_joints, _ = self.read()

    def read(self) -> tuple[Joints, Grips]:
        degrees = dict(zip(self._devices, self._pool.map(self._read_degrees, self._devices.values()), strict=True))
        joints = {side: np.radians([deg[motor] for motor in MOTORS]) for side, deg in degrees.items()}
        grips = {side: _grip(self._arms[side], deg[GRIPPER]) for side, deg in degrees.items()}
        return joints, grips

    def command(self, joints: Joints, grips: Grips) -> None:
        if not self.energized:
            raise RuntimeError("a dry run never commands the arms")
        sends = []
        for side, device in self._devices.items():
            action = {f"{motor}.pos": math.degrees(value) for motor, value in zip(MOTORS, joints[side], strict=True)}
            action[f"{GRIPPER}.pos"] = _gripper_deg(self._arms[side], grips[side])
            sends.append(self._pool.submit(device.send_action, action))
        for send in sends:
            send.result()

    def close(self) -> None:
        """Returns both arms to where they started (makermodslab's gentle rate), releases torque, disconnects."""
        if self._rest_poses:
            for ok, reason in self._family.return_to_rest(self._rest_poses):
                if not ok:
                    _log.error("an arm did not return to its start pose: %s", reason)
            self._rest_poses = []
        for side, device in self._devices.items():
            for problem in self._family.release_torque(device, f"{side} follower arm"):
                _log.error(problem)
            if self.energized:
                device.disconnect()
            else:
                device.bus.disconnect(False)
        self._devices = {}
        self._pool.shutdown(wait=True)

    def _open(self, device: Any) -> None:
        """Connects one follower the way makermodslab's teleoperation does, or with torque off for a dry run."""
        if not self.energized:
            device.bus.connect(handshake=False)
            device.bus.disable_torque()
            return
        from makermodslab import metal_gripper, torque

        if self._family.supports_gripper_effort_control:
            metal_gripper.install_metal_gripper(device, "")
        self._family.prepare_teleoperation(device)
        self._family.prepare_leader_following(device)
        try:
            device.connect(calibrate=False)
        except BaseException:
            torque.de_energize_can_device(device, "follower arm")
            raise

    def _read_degrees(self, device: Any) -> dict[str, float]:
        """One arm's motor angles in degrees, by motor name."""
        if self.energized:
            observation = device.get_observation()
            return {motor: float(observation[f"{motor}.pos"]) for motor in (*MOTORS, GRIPPER)}
        # Torque off: one fresh read per motor, as makermodslab's calibration reads a limp arm.
        return {motor: float(device.bus.read("Present_Position", motor)) for motor in (*MOTORS, GRIPPER)}


def teleop_ik(station: config.StationConfig) -> dict[str, ik.ArmIK]:
    """Per side, the teleop IK on the arm's URDF, driving its measured tool center point.

    @throws ConfigError An arm's tool center point has not been measured yet.
    """
    mm = station.makermods
    assert mm is not None
    unmeasured = [side for side, arm in station.arms.items() if arm.tcp_xyz_m is None]
    if unmeasured:
        raise config.ConfigError(
            f"'arms.{unmeasured[0]}.tcp_xyz_m' is not set: for the stock gripper, set Dream's own tool point with "
            "`~/dream-env/bin/python scripts/use_dream_tcp.py configs/station.yaml`; for another gripper, measure "
            'its point (docs/hardware_setup.md, "Tool center point")'
        )
    urdf = mm.urdf.read_text()
    teleop = station.teleop
    return {
        side: ik.ArmIK(
            ik.urdf_model(urdf, mm.flange_link, arm.tcp_xyz_m),
            ik.TCP_SITE,
            mm.urdf_joints,
            teleop.max_joint_speed_rad_s,
            teleop.joint_limit_margin_rad,
            joint_signs=mm.joint_signs,
        )
        for side, arm in station.arms.items()
        if arm.tcp_xyz_m is not None
    }


def _grip(arm: config.ArmConfig, motor_deg: float) -> float:
    """The gripper motor's angle as 0 (closed) to 1 (open)."""
    assert arm.gripper_open_deg is not None and arm.gripper_closed_deg is not None
    fraction = (motor_deg - arm.gripper_closed_deg) / (arm.gripper_open_deg - arm.gripper_closed_deg)
    return float(min(1.0, max(0.0, fraction)))


def _gripper_deg(arm: config.ArmConfig, grip: float) -> float:
    """A 0 (closed) to 1 (open) gripper target as the gripper motor's angle."""
    assert arm.gripper_open_deg is not None and arm.gripper_closed_deg is not None
    grip = min(1.0, max(0.0, grip))
    return arm.gripper_closed_deg + grip * (arm.gripper_open_deg - arm.gripper_closed_deg)
