"""The station config: one YAML file per station (see `configs/`), loaded into frozen dataclasses.

Every value is checked here, so a mistake is reported with the key to fix before anything opens a device.
"""

import dataclasses
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from dream_hackathon import layouts

## What the station templates (`configs/*.yaml`) put where each station's own value belongs.
PLACEHOLDER = "REPLACE"


class ConfigError(ValueError):
    """A station config value is missing or wrong; the message names the key."""


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    """One camera.

    Attributes:
    - role: `overhead`, `wrist_l` or `wrist_r`.
    - kind: `uvc` (any webcam OpenCV opens) or `realsense`.
    - device: The UVC device, e.g. `/dev/v4l/by-id/...` or `0`; None for a RealSense.
    - serial: The RealSense serial number; None for a UVC camera.
    - width, height, fps: The capture mode.
    """

    role: str
    kind: str
    device: str | None
    serial: str | None
    width: int
    height: int
    fps: int


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    """One follower arm.

    Attributes:
    - side: `left` or `right`.
    - port: YAM: the CAN interface (e.g. `can_follower_l`). MakerMods: the adapter (`/dev/serial/by-id/...` for slcan,
      `can0` for socketcan).
    - quest_to_base: 3x3 rotation taking a Quest-frame vector into this arm's base frame.
    - workspace_min_m, workspace_max_m: The box, in this arm's base frame, the tool point is kept inside.
    - gripper_open_deg, gripper_closed_deg: MakerMods only: the gripper motor's angle fully open and closed.
    - tcp_xyz_m: MakerMods only: the tool center point in the flange frame, metres; None until measured.
    """

    side: str
    port: str
    quest_to_base: npt.NDArray[np.float64]
    workspace_min_m: npt.NDArray[np.float64]
    workspace_max_m: npt.NDArray[np.float64]
    gripper_open_deg: float | None
    gripper_closed_deg: float | None
    tcp_xyz_m: tuple[float, float, float] | None


@dataclasses.dataclass(frozen=True)
class TeleopConfig:
    """How the Quest drives the arms.

    Attributes:
    - max_joint_speed_rad_s: No joint target moves faster than this.
    - translation_scale: Hand motion to tool motion; 1.0 moves the tool as far as the hand.
    - joint_limit_margin_rad: IK keeps every joint this far inside its limits.
    """

    max_joint_speed_rad_s: float
    translation_scale: float
    joint_limit_margin_rad: float


@dataclasses.dataclass(frozen=True)
class YamConfig:
    """YAM settings.

    Attributes:
    - gripper: The i2rt gripper type, e.g. `linear_4310`.
    - tcp_site: The site in i2rt's MuJoCo model the Quest drives, e.g. `grasp_site`.
    """

    gripper: str
    tcp_site: str


@dataclasses.dataclass(frozen=True)
class MakerModsConfig:
    """MakerMods (Metal or Maker arm) settings.

    Attributes:
    - can_interface: `slcan` or `socketcan`.
    - urdf: The arm's URDF, for the teleop IK.
    - flange_link: The URDF link the gripper is mounted on.
    - urdf_joints: The six URDF joints, base to wrist, in the follower's motor order.
    - joint_signs: Per joint, +1 or -1: how a follower angle maps onto its URDF joint.
    """

    can_interface: str
    urdf: pathlib.Path
    flange_link: str
    urdf_joints: tuple[str, ...]
    joint_signs: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class StationConfig:
    """A whole station.

    Attributes:
    - arm: `yam`, `metal` or `maker`.
    - family: `yam` or `makermods` (layouts.FAMILIES).
    - task: The task text recorded with each episode when `dhk-collect` gets no `--task`, or None (the collector
      then records collect.DEFAULT_TASK). Optional: Dream's service labels every dataset from its head camera.
    - dataset_dir: The LeRobot v2.1 dataset folder episodes are added to.
    - fps: Recording rate.
    - control_hz: Teleop loop rate, a whole multiple of fps.
    - arms: Per side, its ArmConfig.
    - cameras: The three cameras.
    - record_width, record_height: The size frames are recorded at (16:9).
    - teleop: TeleopConfig.
    - yam: YamConfig on a YAM station, else None.
    - makermods: MakerModsConfig on a MakerMods station, else None.
    """

    arm: str
    family: str
    task: str | None
    dataset_dir: pathlib.Path
    fps: int
    control_hz: int
    arms: dict[str, ArmConfig]
    cameras: tuple[CameraConfig, ...]
    record_width: int
    record_height: int
    teleop: TeleopConfig
    yam: YamConfig | None
    makermods: MakerModsConfig | None


## The config `dreamcli setup` writes, whose `follower_l.home` and `follower_r.home` are Dream-1's home pose.
DREAM_STATION_PATH = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
## The arms whose joints the kit records in Dream's own frame (robots.metal: "the motor angles in radians, sign +1,
## no offset"), so Dream's home can be driven as it is. Dream's Maker joints are turn-corrected, the YAM's are i2rt's.
_DREAM_HOME_ARMS = ("metal",)
_DREAM_SECTIONS = {"left": "follower_l", "right": "follower_r"}


def dream_home(station: StationConfig, path: pathlib.Path = DREAM_STATION_PATH) -> dict[str, npt.NDArray[np.float64]]:
    """Dream-1's home pose for each arm, read from the config `dreamcli setup` wrote, in the kit's joint frame.

    Each Dream arm section is matched to the kit's arm by its port, so a swapped mapping is refused rather than
    sending an arm to the other one's home.

    @param station The station config.
    @param path Dream's station config.
    @return Per side, six joint angles in radians.
    @throws ConfigError There is no usable home: the message says why.
    """
    if station.arm not in _DREAM_HOME_ARMS:
        raise ConfigError(f"Dream's home is used on {', '.join(_DREAM_HOME_ARMS)} arms only, not {station.arm}")
    if not path.is_file():
        raise ConfigError(f"{path} does not exist: `dreamcli setup` writes it, with each arm's home")
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: could not be read: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    if raw.get("body") != station.arm:
        raise ConfigError(f"{path} is for a {raw.get('body')!r} body; this station's arms are {station.arm!r}")
    home = {}
    for side, section in _DREAM_SECTIONS.items():
        arm = raw.get(section)
        if not isinstance(arm, dict):
            raise ConfigError(f"{path} has no '{section}' section")
        port = station.arms[side].port
        if arm.get("can_channel") != port:
            raise ConfigError(
                f"{path}: '{section}.can_channel' is {arm.get('can_channel')!r}, but 'arms.{side}.port' is {port!r}: "
                "the two configs do not name the same arm"
            )
        value = arm.get("home")
        try:
            joints = np.asarray(value if isinstance(value, list) else [], dtype=float)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{path}: '{section}.home' must be six joint angles in radians: {error}") from error
        if joints.shape != (6,) or not np.all(np.isfinite(joints)) or np.any(np.abs(joints) > np.pi):
            raise ConfigError(f"{path}: '{section}.home' must be six joint angles in radians, each within pi")
        home[side] = joints
    return home


def load(path: str | pathlib.Path) -> StationConfig:
    """Reads and checks a station config.

    @param path The YAML file.
    @return The config.
    @throws ConfigError A value is missing or wrong; the message names the key.
    """
    file = pathlib.Path(path)
    if not file.is_file():
        raise ConfigError(
            f"{path} does not exist: copy your arm's template (configs/yam.yaml, metal.yaml or maker.yaml) to it "
            "and fill it in (docs/hardware_setup.md)"
        )
    raw = yaml.safe_load(file.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    placeholder = _placeholder(raw)
    if placeholder:
        raise ConfigError(f"'{placeholder}' still says REPLACE: fill in this station's value")
    arm = _get(raw, "arm", str)
    if arm not in layouts.FAMILIES:
        raise ConfigError(f"'arm' is {arm!r}; expected one of {sorted(layouts.FAMILIES)}")
    family = layouts.FAMILIES[arm]
    fps = _get(raw, "fps", int)
    control_hz = _get(raw, "control_hz", int)
    if fps not in (30, 15, 10, 5):
        raise ConfigError(f"'fps' is {fps}; Dream fine-tuning takes 30, 15, 10 or 5")
    if control_hz % fps:
        raise ConfigError(f"'control_hz' ({control_hz}) must be a whole multiple of 'fps' ({fps})")
    record_width = _get(raw, "record_width", int)
    record_height = _get(raw, "record_height", int)
    if record_width * 9 != record_height * 16:
        raise ConfigError(f"'record_width' x 'record_height' ({record_width}x{record_height}) must be 16:9")
    arms_raw = _get(raw, "arms", dict)
    arms = {side: _arm(side, _get(arms_raw, side, dict, f"arms.{side}"), family) for side in ("left", "right")}
    cameras_raw = _get(raw, "cameras", dict)
    if set(cameras_raw) != set(layouts.CAMERA_ROLES):
        raise ConfigError(f"'cameras' names {sorted(cameras_raw)}; expected exactly {list(layouts.CAMERA_ROLES)}")
    cameras = tuple(_camera(role, _get(cameras_raw, role, dict, f"cameras.{role}")) for role in layouts.CAMERA_ROLES)
    teleop_raw = _get(raw, "teleop", dict)
    teleop = TeleopConfig(
        max_joint_speed_rad_s=_get(teleop_raw, "max_joint_speed_rad_s", float, "teleop.max_joint_speed_rad_s"),
        translation_scale=_get(teleop_raw, "translation_scale", float, "teleop.translation_scale"),
        joint_limit_margin_rad=_get(teleop_raw, "joint_limit_margin_rad", float, "teleop.joint_limit_margin_rad"),
    )
    return StationConfig(
        arm=arm,
        family=family,
        task=_task(raw),
        dataset_dir=pathlib.Path(_get(raw, "dataset_dir", str)).expanduser(),
        fps=fps,
        control_hz=control_hz,
        arms=arms,
        cameras=cameras,
        record_width=record_width,
        record_height=record_height,
        teleop=teleop,
        yam=_yam(_get(raw, "yam", dict)) if family == "yam" else None,
        makermods=_makermods(_get(raw, "makermods", dict)) if family == "makermods" else None,
    )


def _placeholder(raw: Any, where: str = "") -> str | None:
    """The first key, as `a.b.c`, whose value is a template's `REPLACE` placeholder; None when there is none."""
    if isinstance(raw, dict):
        for key, value in raw.items():
            found = _placeholder(value, f"{where}.{key}" if where else str(key))
            if found:
                return found
    elif isinstance(raw, str) and PLACEHOLDER in raw:
        return where
    return None


def _task(raw: dict[str, Any]) -> str | None:
    """The optional `task` text, stripped; None when it is left out or blank."""
    value = raw.get("task")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"'task' is {value!r}; expected text in quotes, or leave the key out")
    return value.strip() or None


def _get(raw: dict[str, Any], key: str, kind: type, where: str | None = None) -> Any:
    """`raw[key]`, which must be a `kind`; ints are accepted where floats are asked."""
    where = where or key
    if key not in raw or raw[key] is None:
        raise ConfigError(f"'{where}' is missing")
    value = raw[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ConfigError(f"'{where}' is {value!r}; expected a {kind.__name__}")
    return value


def _vector(raw: dict[str, Any], key: str, where: str, shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """`raw[key]` as a float array of `shape`."""
    value = np.asarray(_get(raw, key, list, where), dtype=np.float64)
    if value.shape != shape:
        raise ConfigError(f"'{where}' has shape {list(value.shape)}; expected {list(shape)}")
    return value


def _arm(side: str, raw: dict[str, Any], family: str) -> ArmConfig:
    """One arm's section."""
    where = f"arms.{side}"
    rotation = _vector(raw, "quest_to_base", f"{where}.quest_to_base", (3, 3))
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ConfigError(f"'{where}.quest_to_base' must be a rotation matrix (orthonormal, determinant +1)")
    low = _vector(raw, "workspace_min_m", f"{where}.workspace_min_m", (3,))
    high = _vector(raw, "workspace_max_m", f"{where}.workspace_max_m", (3,))
    if np.any(low >= high):
        raise ConfigError(f"'{where}.workspace_min_m' must be below '{where}.workspace_max_m' on every axis")
    open_deg = closed_deg = None
    tcp = None
    if family == "makermods":
        open_deg = _get(raw, "gripper_open_deg", float, f"{where}.gripper_open_deg")
        closed_deg = _get(raw, "gripper_closed_deg", float, f"{where}.gripper_closed_deg")
        if open_deg == closed_deg:
            raise ConfigError(f"'{where}.gripper_open_deg' and 'gripper_closed_deg' must differ")
        if raw.get("tcp_xyz_m") is not None:
            tcp = tuple(_vector(raw, "tcp_xyz_m", f"{where}.tcp_xyz_m", (3,)).tolist())
    return ArmConfig(
        side=side,
        port=str(_get(raw, "port", str, f"{where}.port")),
        quest_to_base=rotation,
        workspace_min_m=low,
        workspace_max_m=high,
        gripper_open_deg=open_deg,
        gripper_closed_deg=closed_deg,
        tcp_xyz_m=tcp,
    )


def _camera(role: str, raw: dict[str, Any]) -> CameraConfig:
    """One camera's section."""
    where = f"cameras.{role}"
    kind = _get(raw, "type", str, f"{where}.type")
    if kind not in ("uvc", "realsense"):
        raise ConfigError(f"'{where}.type' is {kind!r}; expected 'uvc' or 'realsense'")
    device = str(raw["device"]) if raw.get("device") is not None else None
    serial = str(raw["serial"]) if raw.get("serial") is not None else None
    if kind == "uvc" and device is None:
        raise ConfigError(f"'{where}.device' is missing (a UVC camera's /dev/v4l/by-id/... path or index)")
    if kind == "realsense" and serial is None:
        raise ConfigError(f"'{where}.serial' is missing (rs-enumerate-devices lists RealSense serials)")
    return CameraConfig(
        role=role,
        kind=kind,
        device=device,
        serial=serial,
        width=_get(raw, "width", int, f"{where}.width"),
        height=_get(raw, "height", int, f"{where}.height"),
        fps=_get(raw, "fps", int, f"{where}.fps"),
    )


def _yam(raw: dict[str, Any]) -> YamConfig:
    """The `yam` section."""
    return YamConfig(
        gripper=_get(raw, "gripper", str, "yam.gripper"), tcp_site=_get(raw, "tcp_site", str, "yam.tcp_site")
    )


def _makermods(raw: dict[str, Any]) -> MakerModsConfig:
    """The `makermods` section."""
    interface = _get(raw, "can_interface", str, "makermods.can_interface")
    if interface not in ("slcan", "socketcan"):
        raise ConfigError(f"'makermods.can_interface' is {interface!r}; expected 'slcan' or 'socketcan'")
    joints = tuple(str(name) for name in _get(raw, "urdf_joints", list, "makermods.urdf_joints"))
    signs = tuple(float(sign) for sign in _get(raw, "joint_signs", list, "makermods.joint_signs"))
    if len(joints) != layouts.JOINTS or len(signs) != layouts.JOINTS:
        raise ConfigError(f"'makermods.urdf_joints' and 'makermods.joint_signs' must list {layouts.JOINTS} entries")
    if any(sign not in (1.0, -1.0) for sign in signs):
        raise ConfigError("'makermods.joint_signs' entries must be 1 or -1")
    return MakerModsConfig(
        can_interface=interface,
        urdf=pathlib.Path(_get(raw, "urdf", str, "makermods.urdf")).expanduser(),
        flange_link=_get(raw, "flange_link", str, "makermods.flange_link"),
        urdf_joints=joints,
        joint_signs=signs,
    )
