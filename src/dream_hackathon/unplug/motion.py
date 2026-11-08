"""Tool paths solved to joints before the arm moves: straight lines in the arm's base frame, one sample per tick.

The teleop IK (ik.ArmIK) takes one bounded step per call. Here each sample of a line is solved to convergence from
the one before, and a path whose samples miss their pose (out of reach, a wrist flip) is refused with PlanError
before anything moves. The arm then plays the joints back, so its speed along the line is the sample spacing.

@see plan_line(), plan_route(), plan_joint_ramp()
"""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from dream_hackathon import ik

## Solving one sample: calls of ArmIK.step, each this long, until within the tolerances.
_SOLVE_DT_S = 0.1
_SOLVE_ITERATIONS = 60
## A sample is solved when its tool is this close to the line's pose.
POS_TOL_M = 1e-3
ROT_TOL_DEG = 1.0
## Neighbouring samples whose joints differ more than this are a jump (a wrist flip): refused.
MAX_SAMPLE_JUMP_RAD = 0.15


class PlanError(RuntimeError):
    """A path the arm cannot follow: out of reach, out of the workspace, or a jump in its joints."""


@dataclasses.dataclass(frozen=True)
class JointPath:
    """Joints to command, one row per tick; the start (where the arm already is) is not in it.

    Attributes:
    - points: `[N, 6]` radians.
    - label: What the path does, for the logs.
    - max_pos_err_m, max_rot_err_deg: The worst sample's distance from its line pose (0 for a joint ramp).
    """

    points: npt.NDArray[np.float64]
    label: str = ""
    max_pos_err_m: float = 0.0
    max_rot_err_deg: float = 0.0

    @property
    def end(self) -> npt.NDArray[np.float64]:
        return self.points[-1]

    def duration_s(self, dt_s: float) -> float:
        return len(self.points) * dt_s

    def then(self, other: "JointPath", label: str | None = None) -> "JointPath":
        return JointPath(
            np.vstack([self.points, other.points]),
            label or f"{self.label}, {other.label}",
            max(self.max_pos_err_m, other.max_pos_err_m),
            max(self.max_rot_err_deg, other.max_rot_err_deg),
        )


def lifted(pose: npt.NDArray[np.float64], dz_m: float) -> npt.NDArray[np.float64]:
    """The same tool pose `dz_m` higher along the base's z (straight up), turned the same way."""
    out = np.array(pose, dtype=np.float64)
    out[2, 3] += dz_m
    return out


def at_height(pose: npt.NDArray[np.float64], z_m: float) -> npt.NDArray[np.float64]:
    out = np.array(pose, dtype=np.float64)
    out[2, 3] = z_m
    return out


def rotation_angle(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """The angle in radians between two rotation matrices."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, cos))))


def pose_error(
    solver: ik.ArmIK, joints: npt.NDArray[np.float64], pose: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """How far the joints' tool is from a pose: metres, and degrees of turn."""
    actual = solver.fk(joints)
    return (
        float(np.linalg.norm(actual[:3, 3] - pose[:3, 3])),
        math.degrees(rotation_angle(actual[:3, :3], pose[:3, :3])),
    )


def solve_pose(
    solver: ik.ArmIK, pose: npt.NDArray[np.float64], seed: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Joints near `seed` that put the tool at `pose`, with the remaining error (metres, degrees)."""
    joints = np.asarray(seed, dtype=np.float64).copy()
    pos_err, rot_err = pose_error(solver, joints, pose)
    for _ in range(_SOLVE_ITERATIONS):
        if pos_err < POS_TOL_M / 4 and rot_err < ROT_TOL_DEG / 4:
            break
        joints = solver.step(pose, joints, _SOLVE_DT_S)
        pos_err, rot_err = pose_error(solver, joints, pose)
    return joints, pos_err, rot_err


def plan_line(
    solver: ik.ArmIK,
    start_joints: npt.NDArray[np.float64],
    goal: npt.NDArray[np.float64],
    speed_m_s: float,
    dt_s: float,
    *,
    angular_speed_rad_s: float = 0.5,
    label: str = "",
) -> JointPath:
    """A straight line of the tool from where `start_joints` put it to `goal`, turning evenly on the way.

    @param goal `[4, 4]` tool pose in the arm's base frame.
    @param speed_m_s The tool's speed along the line; the turn is also kept under `angular_speed_rad_s`.
    @throws PlanError A sample out of reach or a jump in the joints.
    """
    start = solver.fk(start_joints)
    distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
    turn = _log_rotation(start[:3, :3].T @ goal[:3, :3])
    angle = float(np.linalg.norm(turn))
    samples = max(1, math.ceil(max(distance / (speed_m_s * dt_s), angle / (angular_speed_rad_s * dt_s))))
    points = []
    worst_pos = worst_rot = 0.0
    joints = np.asarray(start_joints, dtype=np.float64)
    for k in range(1, samples + 1):
        t = k / samples
        pose = np.eye(4)
        pose[:3, 3] = start[:3, 3] + t * (goal[:3, 3] - start[:3, 3])
        pose[:3, :3] = start[:3, :3] @ _exp_rotation(t * turn)
        solved, pos_err, rot_err = solve_pose(solver, pose, joints)
        if pos_err > POS_TOL_M or rot_err > ROT_TOL_DEG:
            raise PlanError(
                f"{label or 'path'}: sample {k}/{samples} at {np.round(pose[:3, 3], 3).tolist()} m is out of reach "
                f"({pos_err * 1000:.1f} mm, {rot_err:.1f} deg off)"
            )
        jump = float(np.max(np.abs(solved - joints)))
        if jump > MAX_SAMPLE_JUMP_RAD:
            raise PlanError(f"{label or 'path'}: the joints jump {math.degrees(jump):.0f} deg at sample {k}")
        worst_pos, worst_rot = max(worst_pos, pos_err), max(worst_rot, rot_err)
        points.append(solved)
        joints = solved
    return JointPath(np.array(points), label, worst_pos, worst_rot)


def plan_route(
    solver: ik.ArmIK,
    start_joints: npt.NDArray[np.float64],
    waypoints: Sequence[tuple[npt.NDArray[np.float64], float]],
    dt_s: float,
    *,
    angular_speed_rad_s: float = 0.5,
    label: str = "",
) -> JointPath:
    """Straight lines through tool poses, each `(pose, speed_m_s)`, one after the other."""
    path: JointPath | None = None
    joints = np.asarray(start_joints, dtype=np.float64)
    for i, (pose, speed) in enumerate(waypoints):
        leg = plan_line(
            solver, joints, pose, speed, dt_s, angular_speed_rad_s=angular_speed_rad_s, label=f"{label} leg {i + 1}"
        )
        path = leg if path is None else path.then(leg, label)
        joints = leg.end
    assert path is not None
    return dataclasses.replace(path, label=label)


def plan_travel(
    solver: ik.ArmIK,
    start_joints: npt.NDArray[np.float64],
    goal: npt.NDArray[np.float64],
    travel_z_m: float,
    speed_m_s: float,
    dt_s: float,
    *,
    angular_speed_rad_s: float = 0.5,
    label: str = "",
) -> JointPath:
    """Up to `travel_z_m` (or stays, if already higher), across, then down to `goal`: over the cables, not through."""
    start = solver.fk(start_joints)
    z = max(travel_z_m, start[2, 3], goal[2, 3])
    waypoints = []
    if z - start[2, 3] > 1e-3:
        waypoints.append((at_height(start, z), speed_m_s))
    waypoints.append((at_height(goal, z), speed_m_s))
    if z - goal[2, 3] > 1e-3:
        waypoints.append((np.array(goal), speed_m_s))
    return plan_route(solver, start_joints, waypoints, dt_s, angular_speed_rad_s=angular_speed_rad_s, label=label)


def plan_joint_ramp(
    start: npt.NDArray[np.float64], goal: npt.NDArray[np.float64], speed_rad_s: float, dt_s: float, label: str = ""
) -> JointPath:
    """Every joint straight to its goal together, the farthest at `speed_rad_s`."""
    start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    samples = max(1, math.ceil(float(np.max(np.abs(goal - start))) / (speed_rad_s * dt_s)))
    t = np.arange(1, samples + 1)[:, None] / samples
    return JointPath(start + t * (goal - start), label)


def check_workspace(
    solver: ik.ArmIK, path: JointPath, low_m: npt.NDArray[np.float64], high_m: npt.NDArray[np.float64]
) -> None:
    """@throws PlanError A sample's tool outside the arm's workspace box (the station config's)."""
    for k, joints in enumerate(path.points):
        xyz = solver.fk(joints)[:3, 3]
        if np.any(xyz < low_m) or np.any(xyz > high_m):
            raise PlanError(
                f"{path.label}: sample {k + 1} puts the tool at {np.round(xyz, 3).tolist()} m, outside the workspace "
                f"{np.round(low_m, 2).tolist()} to {np.round(high_m, 2).tolist()}"
            )


def _log_rotation(r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The rotation vector (axis times angle) of a rotation matrix."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Half a turn: the axis from the matrix's diagonal.
        axis = np.sqrt(np.maximum((np.diag(r) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(axis))
        for j in range(3):
            if j != i:
                axis[j] = math.copysign(axis[j], r[i, j] + r[j, i])
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def _exp_rotation(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The rotation matrix of a rotation vector (Rodrigues)."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)
