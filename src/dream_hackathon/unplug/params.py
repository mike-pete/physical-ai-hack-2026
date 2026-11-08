"""The unplugging policy's speeds, limits and vision thresholds, from configs/unplug.yaml (every key optional).

Grips are the loop's 0 (closed) to 1 (open) for the jaw stroke: a Metal gripper's 1.0 is 100 mm open, so a USB
housing held across its flat faces reads about 0.06 to 0.09 (layouts.GRIP_SCALE).
"""

import dataclasses
import pathlib
from typing import Any

import yaml

from dream_hackathon import config
from dream_hackathon.unplug.perception import PerceptionParams


@dataclasses.dataclass(frozen=True)
class MotionParams:
    """How the arm moves, and when a step counts as done or failed. Lengths in metres, speeds per second."""

    transit_speed_m_s: float = 0.04  # between the ready pose, the cables and the drop spot
    descend_speed_m_s: float = 0.02  # down onto a plug
    pull_speed_m_s: float = 0.015  # straight up with the plug
    angular_speed_rad_s: float = 0.5  # how fast the tool may turn along a path
    joint_speed_rad_s: float = 0.4  # joint-space ramps (to the ready pose, back to the start)
    approach_up_m: float = 0.06  # the pose above a plug: its grasp pose this much higher
    travel_clearance_m: float = 0.03  # transits run at least this far above the highest approach pose
    pull_m: float = 0.04  # how far the plug is pulled up
    pull_check_at_m: float = 0.015  # once the pull has been commanded this far ...
    pull_check_min_rise_m: float = 0.012  # ... the tool must have risen this much, or the plug is stuck
    min_rise_m: float = 0.03  # a pull counts only if the tool rose this much
    attempts: int = 5  # tries per cable: the first and four retries
    retry_dz_m: float = -0.001  # each retry grasps this much deeper ...
    retry_dz_max_m: float = -0.003  # ... down to this much below the taught grasp
    drop_up_m: float = 0.05  # the pose above the drop spot
    arrive_tol_rad: float = 0.02  # a path's end is reached when every joint is this close ...
    settle_s: float = 0.3  # ... for this long
    max_track_err_rad: float = 0.10  # a joint this far behind its command: blocked
    path_slack_s: float = 3.0  # a path may take this much longer than planned before it times out
    grasp_grip: float = 0.0  # the close command; the gripper's force hold stops it on the housing
    open_grip: float = 1.0  # fully open, at the drop spot
    empty_grip_max: float = 0.03  # closed this far means nothing is between the fingers
    held_margin: float = 0.03  # held while the grip stays above the cable's held_grip less this
    grip_steady: float = 0.01  # the grip counts as settled when it changes less than this ...
    grip_steady_s: float = 0.3  # ... over this long
    grasp_dwell_s: float = 0.8  # wait at least this long after closing before judging the grasp
    release_dwell_s: float = 1.0  # wait this long after opening at the drop spot
    fail_dwell_s: float = 0.8  # wait this long after opening on a failed attempt
    verify_frames: int = 3  # wrist frames judged after a pull ...
    verify_votes: int = 2  # ... of which this many must show the tip
    verify_delay_s: float = 0.3  # let the arm settle before the first frame
    require_vision: bool = True  # False: a pull that rose and still holds counts, whatever the camera says
    on_give_up: str = "skip"  # after the last attempt: "skip" to the next cable, or "abort"
    # Locating the plugs from the viewing pose (perception.py), when a view reference is given:
    probe_m: float = 0.01  # the probe moves that measure how the image shifts with the tool, each way
    locate_settle_s: float = 0.4  # still this long at a pose before its frames are used
    locate_frames: int = 6  # frames judged at each pose ...
    locate_agree: int = 3  # ... of which this many must place the row within locate_agree_px of their median
    locate_agree_px: float = 3.0
    max_correction_m: float = 0.03  # a strip moved farther than this from the taught grasps aborts the run
    require_locate: bool = True  # False: when the row cannot be placed, go on with the taught grasps
    relocate_each_cable: bool = True  # back to the viewing pose before every cable, not only at the start


@dataclasses.dataclass(frozen=True)
class VisionParams:
    """The tip check on the acting arm's wrist frames (BGR, the recording size). Boxes are x0, y0, x1, y1 fractions.

    The defaults are a starting point: tune them on recorded frames with scripts/tune_unplug_vision.py.
    """

    tip_roi: tuple[float, float, float, float] = (0.30, 0.55, 0.70, 1.00)  # just below the fingertips
    housing_roi: tuple[float, float, float, float] = (0.30, 0.25, 0.70, 0.60)  # between the fingers
    metal_s_max: int = 60  # the tip: low saturation ...
    metal_v_min: int = 165  # ... and bright
    orange_h: tuple[int, int] = (5, 25)  # OpenCV hue, 0 to 179
    orange_s_min: int = 120
    orange_v_min: int = 90
    min_tip_area_px: dict[str, int] = dataclasses.field(default_factory=lambda: {"usb_a": 150, "usb_c": 60})
    tip_aspect: tuple[float, float] = (1.2, 6.0)  # the tip blob's long side over its short side
    min_tip_gain_px: int = 100  # metal pixels in the tip box after the pull, less before it
    min_orange_frac: float = 0.15  # the cable's housing still fills this much of the housing box


@dataclasses.dataclass(frozen=True)
class UnplugParams:
    motion: MotionParams = dataclasses.field(default_factory=MotionParams)
    vision: VisionParams = dataclasses.field(default_factory=VisionParams)
    perception: PerceptionParams = dataclasses.field(default_factory=PerceptionParams)

    def scaled(self, speed_scale: float) -> "UnplugParams":
        """The same parameters with every speed times `speed_scale` (at most 1: a run is never made faster)."""
        if not 0.0 < speed_scale <= 1.0:
            raise config.ConfigError(f"the speed scale must be in (0, 1], not {speed_scale}")
        m = self.motion
        slower = dataclasses.replace(
            m,
            transit_speed_m_s=m.transit_speed_m_s * speed_scale,
            descend_speed_m_s=m.descend_speed_m_s * speed_scale,
            pull_speed_m_s=m.pull_speed_m_s * speed_scale,
            angular_speed_rad_s=m.angular_speed_rad_s * speed_scale,
            joint_speed_rad_s=m.joint_speed_rad_s * speed_scale,
        )
        return dataclasses.replace(self, motion=slower)


def load_params(path: str | pathlib.Path | None) -> UnplugParams:
    """The parameters in a YAML file (`motion:` and `vision:` sections), defaults for what it leaves out.

    @throws ConfigError A key that is not a parameter, or a value of the wrong kind.
    """
    if path is None:
        return UnplugParams()
    file = pathlib.Path(path).expanduser()
    if not file.is_file():
        raise config.ConfigError(f"{file} does not exist")
    raw = yaml.safe_load(file.read_text()) or {}
    if not isinstance(raw, dict) or set(raw) - {"motion", "vision", "perception"}:
        raise config.ConfigError(f"{file}: only 'motion', 'vision' and 'perception' sections are read")
    motion = _section(MotionParams, raw.get("motion") or {}, "motion")
    vision = _section(VisionParams, raw.get("vision") or {}, "vision")
    perception = _section(PerceptionParams, raw.get("perception") or {}, "perception")
    if motion.on_give_up not in ("skip", "abort"):
        raise config.ConfigError("motion.on_give_up must be 'skip' or 'abort'")
    if motion.attempts < 1 or not 1 <= motion.verify_votes <= motion.verify_frames:
        raise config.ConfigError("need attempts >= 1 and 1 <= verify_votes <= verify_frames")
    if not 1 <= motion.locate_agree <= motion.locate_frames:
        raise config.ConfigError("need 1 <= locate_agree <= locate_frames")
    return UnplugParams(motion, vision, perception)


def _section(kind: type, raw: dict[str, Any], name: str) -> Any:
    fields = {field.name: field for field in dataclasses.fields(kind)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise config.ConfigError(f"{name}: unknown keys {sorted(unknown)}")
    defaults = kind()
    values = {}
    for key, value in raw.items():
        default = getattr(defaults, key)
        try:
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise TypeError
                values[key] = value
            elif isinstance(default, int | float) and not isinstance(default, bool):
                values[key] = type(default)(value)
            elif isinstance(default, tuple):
                values[key] = tuple(type(item)(v) for item, v in zip(default, value, strict=True))
            elif isinstance(default, dict):
                values[key] = {str(k): int(v) for k, v in dict(value).items()}
            else:
                values[key] = str(value)
        except (TypeError, ValueError) as error:
            raise config.ConfigError(f"{name}.{key}: {value!r} is not like {default!r}") from error
    return dataclasses.replace(defaults, **values)
