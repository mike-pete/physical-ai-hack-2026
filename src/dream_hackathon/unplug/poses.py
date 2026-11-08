"""The acting arm's taught poses (configs/unplug_poses.yaml): joints in degrees, as the arm reads them.

A grasp pose is the arm closed on a plug's housing at its base, fingers across its flat faces; its wrist turn is what
the policy keeps all the way down and up. The approach, pull and drop poses are derived from these (motion.py).
Joints, not tool poses, are stored: they mean the same whatever tool center point the station's config names.
"""

import dataclasses
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from dream_hackathon import config

TIPS = ("usb_a", "usb_c")


@dataclasses.dataclass(frozen=True)
class CablePose:
    """One cable.

    Attributes:
    - name: e.g. `cable_1` (numbered in the order they are unplugged).
    - grasp_joints: `[6]` radians: closed on the housing at its base.
    - approach_grip: 0..1, how far open the fingers go down (narrow enough to clear the neighbouring plugs).
    - held_grip: 0..1, what the gripper reads when closed on this housing.
    - tip: The plug in the port, `usb_a` or `usb_c`: its metal tip is what the camera looks for.
    - source: Where the pose came from (an episode and frame, or `taught`).
    """

    name: str
    grasp_joints: npt.NDArray[np.float64]
    approach_grip: float
    held_grip: float
    tip: str = "usb_a"
    source: str = ""


@dataclasses.dataclass(frozen=True)
class UnplugPoses:
    """Every pose the policy needs, for one arm.

    Attributes:
    - side: The acting arm, `left` or `right`.
    - ready_joints: `[6]` radians: over the strip, clear of the cables.
    - drop_joints: `[6]` radians: where an unplugged cable is let go (the tool at table height).
    - cables: In the order they are unplugged.
    """

    side: str
    ready_joints: npt.NDArray[np.float64]
    drop_joints: npt.NDArray[np.float64]
    cables: tuple[CablePose, ...]


def load_poses(path: str | pathlib.Path) -> UnplugPoses:
    """Reads a poses file.

    @throws ConfigError A missing or malformed value; the message names it.
    """
    file = pathlib.Path(path).expanduser()
    if not file.is_file():
        raise config.ConfigError(
            f"{file} does not exist: write it with scripts/mine_unplug_rollouts.py or scripts/teach_unplug_poses.py"
        )
    raw = yaml.safe_load(file.read_text()) or {}
    side = raw.get("side")
    if side not in ("left", "right"):
        raise config.ConfigError(f"{file}: 'side' must be left or right")
    cables = []
    for i, item in enumerate(raw.get("cables") or []):
        where = f"cables[{i}]"
        tip = item.get("tip", "usb_a")
        if tip not in TIPS:
            raise config.ConfigError(f"{file}: {where}.tip must be one of {TIPS}")
        cables.append(
            CablePose(
                name=str(item.get("name") or f"cable_{i + 1}"),
                grasp_joints=_joints(item, "grasp_joints_deg", f"{file}: {where}"),
                approach_grip=_grip(item, "approach_grip", f"{file}: {where}"),
                held_grip=_grip(item, "held_grip", f"{file}: {where}"),
                tip=tip,
                source=str(item.get("source", "")),
            )
        )
    if not cables:
        raise config.ConfigError(f"{file}: no cables")
    for cable in cables:
        if cable.approach_grip <= cable.held_grip:
            raise config.ConfigError(f"{file}: {cable.name}: approach_grip must be wider than held_grip")
    return UnplugPoses(
        side=side,
        ready_joints=_joints(raw, "ready_joints_deg", str(file)),
        drop_joints=_joints(raw, "drop_joints_deg", str(file)),
        cables=tuple(cables),
    )


def save_poses(path: str | pathlib.Path, poses: UnplugPoses, comment: str = "") -> None:
    """Writes a poses file load_poses() reads, with `comment` as a header."""

    def degrees(joints: npt.NDArray[np.float64]) -> list[float]:
        return [round(float(v), 3) for v in np.degrees(joints)]

    body: dict[str, Any] = {
        "side": poses.side,
        "ready_joints_deg": degrees(poses.ready_joints),
        "drop_joints_deg": degrees(poses.drop_joints),
        "cables": [
            {
                "name": c.name,
                "tip": c.tip,
                "grasp_joints_deg": degrees(c.grasp_joints),
                "approach_grip": round(float(c.approach_grip), 3),
                "held_grip": round(float(c.held_grip), 3),
                "source": c.source,
            }
            for c in poses.cables
        ],
    }
    header = "".join(f"# {line}\n" if line else "#\n" for line in comment.splitlines())
    text = header + yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=119)
    pathlib.Path(path).expanduser().write_text(text)


def _joints(raw: dict[str, Any], key: str, where: str) -> npt.NDArray[np.float64]:
    value = raw.get(key)
    try:
        joints = np.radians(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise config.ConfigError(f"{where}: '{key}' must be six numbers") from error
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise config.ConfigError(f"{where}: '{key}' must be six numbers")
    return joints


def _grip(raw: dict[str, Any], key: str, where: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
        raise config.ConfigError(f"{where}: '{key}' must be a number from 0 (closed) to 1 (open)")
    return float(value)
