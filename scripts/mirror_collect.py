"""Mirror teleoperation with recording: move the left arm by hand, the right arm follows.

Writes a LeRobot v2.1 dataset in the same format as dhk-collect, compatible with
tools/add_eef.py and tools/finetune.py.

Usage (THE RIGHT ARM WILL MOVE — support it before starting):
    uv run python scripts/mirror_collect.py \\
        --config configs/station.yaml \\
        --task "pick up the red cube"

    # The same with the left arm weightless (gravity compensation; Metal arms only, requires pinocchio). Without
    # --gravity the left arm is opened torque-off and you carry its full weight. The dashboard and the console say
    # which one is running, and an episode's start line repeats it.
    uv run python scripts/mirror_collect.py \\
        --config configs/station.yaml \\
        --task "pick up the red cube" \\
        --gravity

Each task gets a dataset folder of its own under --data-root (default ~/hackathon_data), named after the task and the
Pacific time it was started: the task above is saved in a folder like
~/hackathon_data/pick-up-the-red-cube-20261107-094215. Type another task on the dashboard and press "Set Task" to
create the next folder; the episodes after that are saved there. Start without --task to begin from the dashboard.
To record everything into one folder instead, give --dataset ~/hackathon_data/my-task.

Controls (type then Enter, or use the browser dashboard on http://localhost:8080):
    Enter   Start a new episode (idle) / save the current episode (recording)
    d       Discard the current episode
    h       Return follower to home pose
    q       Quit (saves active episode if one is running)

The left arm is opened torque-off (or gravity-compensated with --gravity) — move it freely.
The right arm is opened energized and tracks the left arm in real time.

Star Arm 102 leader (Metal stations): a Star arm leads the right arm instead, and the left Metal arm rests torque off
(support it or lay it down) and is recorded still. docs/star_leader.md is the guide.
Calibrate the Star arm in makermodslab first (Metal arm, "Star Arm 102 leader"); its port is under
/dev/serial/by-id (on the station a CH340: usb-1a86_USB_Serial-if00-port0):
    uv run python scripts/mirror_collect.py --config configs/station.yaml --task "pick up the red cube" \\
        --leader-device star --star-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 --star-id <name> \\
        --leader left --follower right
The right arm follows only once the Star arm is brought within ENGAGE_TOLERANCE_DEG of its pose on every joint
(at start, and again after homing); until then it holds, and episodes cannot start.
"""

import argparse
import dataclasses
import datetime
import hashlib
import http.server
import json
import math
import pathlib
import queue
import re
import sys
import threading
import time
import unicodedata
import zoneinfo
from collections.abc import Callable
from typing import Any, Literal

import cv2
import numpy as np

try:  # The Star Arm 102 leader's driver comes with the makermods extra; only --leader-device star needs it.
    from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderMetalTeleopConfig
except ImportError:
    RebotArm102Leader = RebotArm102LeaderMetalTeleopConfig = None

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
GRIPPER = "gripper"

## The homing ramp's joint speed (rad/s), and how close to home counts as home (radians).
HOME_SPEED = 0.5
HOME_TOLERANCE_RAD = 1e-3
## Homing gives up after this long and mirroring resumes, so a joint that cannot reach home cannot lock the session.
HOME_TIMEOUT_S = 30.0
## The task an episode records when none is given, and the longest task text Dream accepts (as in collect.py).
DEFAULT_TASK = "mirror teleoperation"
TASK_MAX_CHARS = 512
## How the leader arm is held, as the dashboard and the console report it. Without gravity compensation its motors are
## off, so the person carries the arm's weight and its gearbox friction.
LEADER_GRAVITY = "gravity"
LEADER_TORQUE_OFF = "torque_off"
LEADER_GRAVITY_FAILED = "gravity_failed"
## With --leader-device star a Star Arm 102 leads the follower, and the other Metal arm rests torque off and is
## recorded as it is: still, so a model trained on the episodes keeps that arm still too.
LEADER_STAR = "star"
_LEADER_MODE_TEXT = {
    LEADER_GRAVITY: "gravity compensation",
    LEADER_TORQUE_OFF: "torque off, no gravity compensation",
    LEADER_GRAVITY_FAILED: "torque off, --gravity FAILED",
    LEADER_STAR: "follows the Star Arm 102 leader",
}
## A Star leader takes over the follower only once each of its joints is within this many degrees of the follower's,
## so the follower never jumps to wherever the Star arm happens to be (at start, and after homing).
ENGAGE_TOLERANCE_DEG = 15.0
## This many failed Star reads in a row disengage it, so after the outage it must come back through the gate above.
STAR_MAX_READ_FAILURES = 3
## An unreliable single Star read is trusted when its raw angle is this close to the value the servo library held.
STAR_GLITCH_TOLERANCE_DEG = 10.0
## The dashboard reads at most this many bytes of a request body, and waits this long for a client's bytes.
MAX_BODY_BYTES = 4096
REQUEST_TIMEOUT_S = 5
## Each task's dataset goes in a folder of its own under the data root, named after the task and the Pacific time the
## task was started (pick-up-the-cube-20261107-094215). A folder's name is at most FOLDER_MAX_CHARS long, the stamp
## included, so the task's part of it is cut at SLUG_MAX_CHARS.
DEFAULT_DATA_ROOT = "~/hackathon_data"
FOLDER_MAX_CHARS = 50
PACIFIC_ZONE = "America/Los_Angeles"
STAMP_FORMAT = "%Y%m%d-%H%M%S"
STAMP_CHARS = len(datetime.datetime(2000, 1, 1).strftime(STAMP_FORMAT))
SLUG_MAX_CHARS = FOLDER_MAX_CHARS - STAMP_CHARS - 1

_CAMERA_PATH = re.compile(r"^/api/camera/([a-z_]+)\.jpg$")
## The dashboard answers only to the loopback names, on any port (an SSH forward may use another local one): a web page
## that rebinds its own name to 127.0.0.1 then gets nothing.
_LOOPBACK_HOST = re.compile(r"(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


# ------------------------------------------------------------------ arm helpers

def _read_leader_degs(device, gravity: bool) -> dict[str, float]:
    if gravity:
        # MetalLeader: get_action() returns the current joint positions as the teleop command.
        action = device.get_action()
        return {m: float(action[f"{m}.pos"]) for m in (*MOTORS, GRIPPER)}
    return {m: float(device.bus.read("Present_Position", m)) for m in (*MOTORS, GRIPPER)}


def _read_star_single(device, motor: str, servo_id: int) -> float:
    """A Star servo's multi-turn angle from a single read, for a servo the sync read missed; raises if that fails.

    The servo library marks a reading near 0 deg unreliable (its filter takes it for a power-cycle A->0->B glitch)
    and holds the last value, yet the servo did answer: a read that gets no answer raises instead. Probed on the
    station's Star arm, a joint resting at its calibrated zero reads unreliable every tick while its raw angle keeps
    changing. So an unreliable read is trusted when its raw angle is within STAR_GLITCH_TOLERANCE_DEG of the held
    one, which a real glitch (raw jumps to 0 from far away) is not.
    """
    try:
        sample = device.bus.read_angle(servo_id, multi_turn=True)
    except Exception as e:
        raise RuntimeError(f"Star servo {motor} (id={servo_id}) did not answer ({e})") from e
    raw = float(sample.raw_deg)
    if not sample.reliable and abs(raw - float(sample.filtered_deg)) > STAR_GLITCH_TOLERANCE_DEG:
        raise RuntimeError(
            f"Star servo {motor} (id={servo_id}) read {raw:.1f} deg, {sample.filtered_deg:.1f} deg before: a glitch"
        )
    return raw


def _read_star_degs(device) -> dict[str, float]:
    """The Star leader's joints and gripper in the Metal follower's degrees, as its Metal preset maps them.

    The same mapping as `RebotArm102Leader.get_action()` (unwrap each servo's multi-turn angle into the window its
    follower range implies, scale, clamp to the range), except that a servo that did not answer this tick raises here.
    get_action() repeats the last reading instead, and so does `_read_raw_positions()`: the servo bus's sync_monitor
    hands back the last good angle, marked reliable=False, for a servo that missed this cycle. Either would be
    commanded and recorded as if it were fresh.

    A servo the sync read marked unreliable is read again on its own (`_read_star_single`): on the station's Star arm
    that happens whenever a joint rests within about 1 deg of its calibrated zero.
    """
    ids = device.config.joint_ids
    monitors = device.bus.sync_monitor(list(ids.values()))
    ranges, directions = device.config.joint_ranges, device.config.joint_directions
    degs = {}
    for motor in (*MOTORS, GRIPPER):
        monitor = monitors.get(ids[motor])
        if monitor is not None and monitor.reliable:
            angle = monitor.angle_deg
        else:
            angle = _read_star_single(device, motor, ids[motor])
        low, high = ranges[motor]
        direction = directions[motor]
        unwrapped, _turns = device._round_to_valid_range(angle, low / direction, high / direction)
        degs[motor] = max(float(low), min(float(high), unwrapped * direction))
    return degs


def _joint_offset_deg(a_rad: np.ndarray, b_rad: np.ndarray) -> float:
    """The largest difference between two arms' joints (gripper excluded), in degrees."""
    return float(np.degrees(np.max(np.abs(np.asarray(a_rad) - np.asarray(b_rad)))))


def _engaged(command_joints: np.ndarray, follower_joints: np.ndarray) -> bool:
    """Whether a Star leader may take over the follower: every joint within ENGAGE_TOLERANCE_DEG (gripper excluded)."""
    return _joint_offset_deg(command_joints, follower_joints) <= ENGAGE_TOLERANCE_DEG


def _star_read_outcome(failures: int, read_ok: bool, engaged: bool) -> tuple[int, bool]:
    """The (consecutive failures, engaged) after one Star read.

    An engaged Star leader that failed STAR_MAX_READ_FAILURES reads in a row disengages: when its reads come back it
    may be anywhere, and following it at once would jump the follower there.
    """
    failures = 0 if read_ok else failures + 1
    return failures, engaged and failures < STAR_MAX_READ_FAILURES


def _records_frame(leader_ok: bool, follower_ok: bool, command_ok: bool, engaged: bool) -> bool:
    """Whether this tick's frame may be recorded: every read succeeded and the follower is following its command.

    A disengaged Star leader (before the gate, or after repeated read failures) is read but not sent to the follower,
    which holds; recording its pose as the action would label frames with a command the follower never got.
    """
    return leader_ok and follower_ok and command_ok and engaged


def _not_following_label(side: str, offset_deg: float | None) -> str:
    """The dashboard's leader line while the follower (`side`) waits for the Star arm to come to its pose."""
    off = "" if offset_deg is None else f" (off by {offset_deg:.0f}°)"
    return f"{side} arm: not following yet: move the Star arm to the follower's pose{off}"


def _record_maps(
    leader: str,
    follower: str,
    leader_joints: np.ndarray,
    leader_grip: float,
    follower_joints: np.ndarray,
    follower_grip: float,
    command_joints: np.ndarray,
    command_grip: float,
) -> tuple[dict, dict, dict, dict]:
    """A frame's (state joints, state grips, action joints, action grips), each keyed by side.

    The follower's state is what it measured and its action what it was sent (the command). The --leader Metal arm
    records itself as both: with a Metal leader the command is that arm's own pose, so this is the leader-mirrors
    layout; with a Star leader that arm rests, and recording it as it is teaches a model to keep it still.
    """
    state_joints = {leader: leader_joints, follower: follower_joints}
    state_grips = {leader: leader_grip, follower: follower_grip}
    action_joints = {leader: leader_joints, follower: command_joints}
    action_grips = {leader: leader_grip, follower: command_grip}
    return state_joints, state_grips, action_joints, action_grips


def _open_star_leader(port: str, calibration_id: str):
    """Opens the Star leader with its Metal preset; exits when its driver, its servos or its calibration is missing."""
    if RebotArm102Leader is None:
        sys.exit("--leader-device star needs the Star driver: uv sync --extra makermods --extra realsense")
    device = RebotArm102Leader(RebotArm102LeaderMetalTeleopConfig(port=port, id=calibration_id))
    if not device.calibration_fpath.is_file():
        found = sorted(p.stem for p in device.calibration_dir.glob("*.json"))
        sys.exit(
            f"No Star calibration named {calibration_id!r} in {device.calibration_dir} (found: {found or 'none'}). "
            "Calibrate the Star arm in makermodslab (Metal arm, 'Star Arm 102 leader'), "
            "then pass its name as --star-id."
        )
    device.connect(calibrate=False)
    return device


def _read_follower_degs(device) -> dict[str, float]:
    obs = device.get_observation()
    return {m: float(obs[f"{m}.pos"]) for m in (*MOTORS, GRIPPER)}


def _degs_to_joints_grip(degs: dict[str, float], arm_cfg) -> tuple[np.ndarray, float]:
    joints = np.radians([degs[m] for m in MOTORS])
    assert arm_cfg.gripper_open_deg is not None and arm_cfg.gripper_closed_deg is not None
    span = arm_cfg.gripper_open_deg - arm_cfg.gripper_closed_deg
    grip = float(min(1.0, max(0.0, (degs[GRIPPER] - arm_cfg.gripper_closed_deg) / span)))
    return joints, grip


def _gripper_deg(arm_cfg, grip: float) -> float:
    # Allow grip slightly below 0 so the follower can push past the calibrated closed position.
    return arm_cfg.gripper_closed_deg + min(1.0, max(-1.0, grip)) * (arm_cfg.gripper_open_deg - arm_cfg.gripper_closed_deg)


def _command_follower(device, arm_cfg, joints_rad: np.ndarray, grip: float, grip_extra: float = 0.0) -> None:
    action = {f"{m}.pos": math.degrees(v) for m, v in zip(MOTORS, joints_rad)}
    # grip_extra shifts the follower gripper more closed than the leader (closes the gap at the hard stop).
    action[f"{GRIPPER}.pos"] = _gripper_deg(arm_cfg, grip - grip_extra)
    device.send_action(action)


# ------------------------------------------------------------------ loop helpers

def _home_step(
    measured: np.ndarray | None, home: np.ndarray, max_step: float, elapsed_s: float
) -> tuple[Literal["reached", "timeout", "hold", "move"], np.ndarray | None]:
    """One tick of the homing ramp.

    @param measured The follower's joints in radians, or None when they could not be read this tick.
    @param home The joints to return to.
    @param max_step The most any joint may move this tick, in radians.
    @param elapsed_s How long homing has been running.
    @return ("reached", None) once every joint is within HOME_TOLERANCE_RAD of home; ("timeout", None) after
        HOME_TIMEOUT_S; ("hold", None) while the joints are unknown (a ramp from a placeholder pose would command a
        jump); otherwise ("move", the next joint command).
    """
    if measured is not None and float(np.max(np.abs(home - measured))) < HOME_TOLERANCE_RAD:
        return "reached", None
    if elapsed_s > HOME_TIMEOUT_S:
        return "timeout", None
    if measured is None:
        return "hold", None
    return "move", measured + np.clip(home - measured, -max_step, max_step)


def _episode_action(cmd: str, recording: bool) -> Literal["start", "save", "ignore"]:
    """What a start or save command does now.

    Enter and `s` on the keyboard toggle. The dashboard's "start" and "save" never do each other's job, so a double
    click cannot save the episode it just began, or begin the one it just saved.
    """
    if cmd in ("", "s"):
        return "save" if recording else "start"
    if cmd == "start":
        return "ignore" if recording else "start"
    if cmd == "save":
        return "save" if recording else "ignore"
    return "ignore"


def _save_episode(episode) -> int | None:
    """Saves an episode and returns its index, or None when it was too short to keep.

    save() has already discarded a too-short episode; reporting it is enough, since ending the session would leave the
    follower energized in the middle of the loop.
    """
    from dream_hackathon.dataset import DatasetError

    try:
        return episode.save()
    except DatasetError as error:
        print(f"✗ Not saved: {error}")
        return None


def _task_or_default(text: str) -> str:
    """The task text an episode records: `text` without surrounding blanks, or DEFAULT_TASK when nothing is left."""
    return text.strip() or DEFAULT_TASK


def _clean_task(text: str) -> str:
    """A task text from the page: one line (control characters become spaces), trimmed, at most TASK_MAX_CHARS."""
    return _CONTROL_CHARACTERS.sub(" ", text).strip()[:TASK_MAX_CHARS].rstrip()


def _leader_mode(gravity_asked: bool, gravity_active: bool) -> str:
    """How the leader arm is held: LEADER_GRAVITY once compensation runs, else torque off.

    A torque-off arm that was asked for gravity compensation is reported as LEADER_GRAVITY_FAILED, so a silent fallback
    cannot pass for the mode that was wanted.
    """
    if gravity_active:
        return LEADER_GRAVITY
    return LEADER_GRAVITY_FAILED if gravity_asked else LEADER_TORQUE_OFF


def _leader_mode_label(mode: str, side: str) -> str:
    """The words for a leader mode, e.g. "left arm: gravity compensation"."""
    return f"{side} arm: {_LEADER_MODE_TEXT[mode]}"


def _host_allowed(host: str | None) -> bool:
    """True when a request's Host header names the loopback address the dashboard is served on."""
    return host is not None and _LOOPBACK_HOST.fullmatch(host.strip().lower()) is not None


def _task_slug(task: str) -> str:
    """The task's part of its folder name: lowercase letters, digits and hyphens, at most SLUG_MAX_CHARS.

    Accents are dropped (é becomes e) and anything else, slashes and dots included, becomes a hyphen, so the name can
    never point outside the data root. A task with no Latin letter or digit gets `task-` and a short hash of its text,
    so different tasks never share a folder. Two texts with the same slug are the same task.
    """
    ascii_text = unicodedata.normalize("NFKD", task).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:SLUG_MAX_CHARS].strip("-")
    return slug or "task-" + hashlib.sha1(task.encode(), usedforsecurity=False).hexdigest()[:8]


def _pacific_time(moment: datetime.datetime) -> datetime.datetime:
    """`moment` (a timezone-aware time) in Pacific time, daylight saving included.

    @throws zoneinfo.ZoneInfoNotFoundError The station has no time zone data (Ubuntu's `tzdata` package).
    """
    return moment.astimezone(zoneinfo.ZoneInfo(PACIFIC_ZONE))


def _now_pacific() -> datetime.datetime:
    return _pacific_time(datetime.datetime.now(datetime.UTC))


def _task_folder(task: str, started: datetime.datetime) -> str:
    """The name of the dataset folder for a task started at `started` (timezone-aware): the task's slug, then the
    Pacific time it started, e.g. `pick-up-the-cube-20261107-094215`. That is lowercase letters, digits and hyphens, at
    most FOLDER_MAX_CHARS, which is what Dream accepts for a dataset's name.
    """
    return f"{_task_slug(task)}-{_pacific_time(started).strftime(STAMP_FORMAT)}"


class _TaskFolders:
    """The dataset of the current task: one folder per task under `root`, named after it and the time it was started,
    and created when the task is set.

    The task stays in its folder for as long as it is the current one: setting the same task again goes on in that
    folder, and a different task starts a folder of its own (as does the same task in a later session, or after another
    task in between). A `pinned` folder replaces all that: the one folder is used whatever the task says.

    Attributes:
    - writer: The open dataset, or None until a task has been set.
    - error: Why the last task's folder could not be opened, or None. While it is set, `writer` is still the previous
      task's, so the caller must not record.
    """

    def __init__(
        self,
        root: pathlib.Path,
        open_writer: Callable[[pathlib.Path], Any],
        pinned: pathlib.Path | None = None,
        clock: Callable[[], datetime.datetime] = _now_pacific,
    ) -> None:
        """@param open_writer Opens the dataset writer for a folder; may raise DatasetError.
        @param clock The current time, timezone-aware; it stamps a new task's folder name.
        """
        self.root = root
        self.pinned = pinned
        self.writer: Any = None
        self.error: str | None = None
        self._open_writer = open_writer
        self._clock = clock
        self._slug: str | None = None  # the task the open folder was made for

    @property
    def folder(self) -> pathlib.Path | None:
        return None if self.writer is None else self.writer.root

    @property
    def saved_episodes(self) -> int:
        return 0 if self.writer is None else self.writer.total_episodes

    def use(self, task: str) -> bool:
        """Opens, creating it if needed, the folder for `task`, unless the open one is already that task's.

        @return True when the task's folder is open, so recording into it is right.
        """
        from dream_hackathon.dataset import DatasetError

        slug = _task_slug(task)
        if self.writer is not None and (self.writer.root == self.pinned if self.pinned else slug == self._slug):
            self.error = None
            return True
        try:
            path = self.pinned or self.root / _task_folder(task, self._clock())
        except zoneinfo.ZoneInfoNotFoundError:
            self.error = f"this station has no time zone data for {PACIFIC_ZONE} (install tzdata)"
            return False
        try:
            path.mkdir(parents=True, exist_ok=True)
            self.writer = self._open_writer(path)
        except (OSError, DatasetError) as error:
            self.error = f"{path.name}: {error}"
            return False
        self._slug = slug
        self.error = None
        return True


# ------------------------------------------------------------------ input thread

_cmd_q: queue.Queue[str] = queue.Queue()


def _input_loop() -> None:
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            _cmd_q.put(line.strip().lower())
        except Exception:
            break


# ------------------------------------------------------------------ dashboard

@dataclasses.dataclass
class _DashState:
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    recording: bool = False
    homing: bool = False
    frames: int = 0
    total_episodes: int = 0
    task: str = ""
    leader_grip: float = 0.0
    shoulder_pan_deg: float = 0.0
    ## One of the LEADER_* modes and its words; empty until the leader arm is open.
    leader_mode: str = ""
    leader_mode_label: str = ""
    camera_roles: list = dataclasses.field(default_factory=list)
    # True when the task names the dataset folder; `folder` is the open one, `folder_error` why a task's could not be
    # opened, and `task_version` counts the tasks set from the page (the control loop opens the folder of a new one).
    per_task: bool = False
    folder: str = ""
    folder_error: str = ""
    task_version: int = 0


_PAGE = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Mirror Collect</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: monospace; background: #111; color: #eee; margin: 0; padding: 1rem; }
  h1 { margin: 0 0 1rem; font-size: 1.1rem; color: #aaa; letter-spacing: 0.05em; }
  #cameras { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .cam-wrap { display: flex; flex-direction: column; }
  .cam-wrap img { width: 320px; height: 180px; object-fit: cover; background: #222; border: 1px solid #333; display: block; }
  .cam-label { color: #666; font-size: 0.75rem; margin-top: 0.25rem; text-align: center; }
  #status-bar { background: #1a1a1a; border: 1px solid #2a2a2a; padding: 0.6rem 1rem; border-radius: 4px; margin-bottom: 1rem; display: flex; gap: 2rem; align-items: center; flex-wrap: wrap; font-size: 0.9rem; }
  .rec  { color: #f55; font-weight: bold; }
  .homing { color: #fa0; font-weight: bold; }
  .idle { color: #666; }
  #folder { color: #8ab; font-size: 0.8rem; word-break: break-all; margin: -0.5rem 0 1rem; }
  #folder.err { color: #f55; }
  .gravity { color: #8f8; }
  .torque_off { color: #fa0; }
  .gravity_failed { color: #f55; font-weight: bold; }
  .star { color: #8f8; }
  #controls { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
  button { font-family: monospace; font-size: 0.9rem; padding: 0.45rem 1rem; border: 1px solid #444; background: #1e1e1e; color: #ccc; cursor: pointer; border-radius: 3px; }
  button:hover:not(:disabled) { background: #2a2a2a; border-color: #666; }
  button:disabled { opacity: 0.3; cursor: default; }
  #btn-primary { background: #0d2a0d; border-color: #2d6a2d; color: #8f8; }
  #btn-primary:hover:not(:disabled) { background: #133813; }
  #btn-discard { background: #2a0d0d; border-color: #6a2d2d; color: #f88; }
  #btn-discard:hover:not(:disabled) { background: #381313; }
  #task-row { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  #task-input { font-family: monospace; font-size: 0.9rem; flex: 1; background: #1a1a1a; border: 1px solid #3a3a3a; color: #eee; padding: 0.4rem 0.6rem; border-radius: 3px; min-width: 0; }
  #task-input:focus { outline: none; border-color: #555; }
</style>
</head>
<body>
<h1>Mirror Collect Dashboard</h1>
<div id="cameras"></div>
<div id="status-bar">
  <span id="mode" class="idle">&#9675; idle</span>
  <span id="leader">leader arm: opening&#8230;</span>
  <span id="frames">0 frames</span>
  <span id="epcnt">0 episodes</span>
  <span id="pan">pan=&#8212;</span>
  <span id="grip">grip=&#8212;</span>
</div>
<div id="controls">
  <button id="btn-primary" onclick="primaryAction()">Start Episode</button>
  <button id="btn-discard" onclick="doPost('/api/episode/discard')" disabled>Discard</button>
  <button id="btn-home" onclick="doPost('/api/home')">&#8629; Go to Home</button>
</div>
<div id="task-row">
  <input id="task-input" type="text" placeholder="Task description (set before starting episode)">
  <button onclick="setTask()">Set Task</button>
</div>
<div id="folder"></div>
<script>
const HDR = {'X-Mirror-Dashboard': '1'};
let _recording = false, _homing = false, _cameras = [];

function doPost(url, body) {
  const opts = {method: 'POST', headers: {...HDR}};
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  fetch(url, opts).catch(() => {});
}

function primaryAction() {
  if (_recording) doPost('/api/episode/save');
  else doPost('/api/episode/start');
}

function setTask() {
  const t = document.getElementById('task-input').value.trim();
  if (t) doPost('/api/task', {task: t});
}

function updateCameras(cameras) {
  if (JSON.stringify(cameras) === JSON.stringify(_cameras)) return;
  _cameras = cameras;
  const grid = document.getElementById('cameras');
  grid.innerHTML = '';
  cameras.forEach(role => {
    const wrap = document.createElement('div');
    wrap.className = 'cam-wrap';
    const img = document.createElement('img');
    img.id = 'cam-' + role;
    const lbl = document.createElement('div');
    lbl.className = 'cam-label';
    lbl.textContent = role;
    wrap.appendChild(img);
    wrap.appendChild(lbl);
    grid.appendChild(wrap);
  });
}

function refreshCameras() {
  _cameras.forEach(role => {
    const img = document.getElementById('cam-' + role);
    if (img) img.src = '/api/camera/' + role + '.jpg?t=' + Date.now();
  });
}

function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(s => {
    _recording = s.recording;
    _homing = s.homing;
    updateCameras(s.cameras || []);

    const mode = document.getElementById('mode');
    if (s.homing)         { mode.className = 'homing'; mode.innerHTML = '&#8629; HOMING'; }
    else if (s.recording) { mode.className = 'rec';    mode.innerHTML = '&#9679; REC'; }
    else                  { mode.className = 'idle';   mode.innerHTML = '&#9675; idle'; }

    const leader = document.getElementById('leader');
    leader.className = s.leader_mode || '';
    leader.textContent = s.leader_mode_label || 'leader arm: opening\\u2026';

    document.getElementById('frames').textContent = s.frames + ' frames';
    document.getElementById('epcnt').textContent  = s.total_episodes + ' episodes';
    document.getElementById('pan').textContent    = 'pan=' + (s.shoulder_pan_deg != null ? s.shoulder_pan_deg.toFixed(1) + '\\u00b0' : '\\u2014');
    document.getElementById('grip').textContent   = 'grip=' + (s.leader_grip != null ? s.leader_grip.toFixed(2) : '\\u2014');

    const busy = s.recording || s.homing;
    // Without a folder (no task set yet, or its folder could not be opened) an episode has nowhere to go.
    const noFolder = (s.per_task && !s.folder) || !!s.folder_error;
    document.getElementById('btn-primary').disabled  = s.homing || (!s.recording && noFolder);
    document.getElementById('btn-primary').textContent = s.recording ? 'Save Episode' : 'Start Episode';
    document.getElementById('btn-discard').disabled  = !s.recording;
    document.getElementById('btn-home').disabled     = busy;

    const folder = document.getElementById('folder');
    folder.className = s.folder_error ? 'err' : '';
    folder.textContent = s.folder_error ? 'folder: ' + s.folder_error
      : s.folder ? 'folder: ' + s.folder
      : s.per_task ? 'folder: none yet, set a task' : '';

    const inp = document.getElementById('task-input');
    if (!inp.value) {
      inp.placeholder = s.task
        || (s.per_task ? 'Task name: creates its folder, e.g. pick up the red cube' : 'Task description');
    }
  }).catch(() => {});
}

setInterval(pollStatus, 300);
setInterval(refreshCameras, 200);
pollStatus();
</script>
</body>
</html>
"""


def _make_handler(dash_state: _DashState, cmd_q: "queue.Queue[str]", cams: dict) -> type:
    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S  # a client that stalls mid-request cannot hold a thread for ever

        def do_GET(self) -> None:  # noqa: N802
            if not _host_allowed(self.headers.get("Host")):
                self._send(403, "text/plain", b"forbidden")
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", _PAGE.encode())
            elif path == "/api/status":
                with dash_state.lock:
                    snap = {
                        "recording": dash_state.recording,
                        "homing": dash_state.homing,
                        "frames": dash_state.frames,
                        "total_episodes": dash_state.total_episodes,
                        "task": dash_state.task,
                        "leader_grip": dash_state.leader_grip,
                        "shoulder_pan_deg": dash_state.shoulder_pan_deg,
                        "leader_mode": dash_state.leader_mode,
                        "leader_mode_label": dash_state.leader_mode_label,
                        "cameras": list(dash_state.camera_roles),
                        "per_task": dash_state.per_task,
                        "folder": dash_state.folder,
                        "folder_error": dash_state.folder_error,
                    }
                self._send(200, "application/json", json.dumps(snap).encode())
            elif m := _CAMERA_PATH.match(path):
                role = m.group(1)
                if role not in cams:
                    self._send(404, "text/plain", b"no such camera")
                    return
                try:
                    frame = cams[role].latest()
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError("encode failed")
                    self._send(200, "image/jpeg", buf.tobytes(), extra={"Cache-Control": "no-store"})
                except Exception:
                    self._send(503, "text/plain", b"camera unavailable")
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            # An action is taken only from the page itself: a loopback Host, the custom header (which a cross-site
            # form cannot set) and, when the browser sends an Origin, one that names this page's address.
            if (
                not _host_allowed(host)
                or not self.headers.get("X-Mirror-Dashboard")
                or (origin and origin != f"http://{host}")
            ):
                self._send(403, "text/plain", b"forbidden")
                return
            path = self.path.split("?", 1)[0]
            if path == "/api/episode/start":
                cmd_q.put("start")
            elif path == "/api/episode/save":
                cmd_q.put("save")
            elif path == "/api/episode/discard":
                cmd_q.put("d")
            elif path == "/api/home":
                cmd_q.put("h")
            elif path == "/api/task":
                length = self.headers.get("Content-Length", "")
                if not (length.isascii() and length.isdigit()):
                    self._send(400, "text/plain", b"bad request")
                    return
                if int(length) > MAX_BODY_BYTES:
                    self._send(413, "text/plain", b"request too large")
                    return
                try:
                    body = json.loads(self.rfile.read(int(length)))
                    task = _clean_task(str(body.get("task", "")))
                except (ValueError, AttributeError, OSError, RecursionError):
                    self._send(400, "text/plain", b"bad request")
                    return
                with dash_state.lock:
                    refused = dash_state.per_task and not task  # the task names the folder, so it cannot be blank
                    if not refused:
                        dash_state.task = task
                        dash_state.task_version += 1
                if refused:
                    self._send(400, "text/plain", b"a task name is needed")
                    return
            else:
                self._send(404, "text/plain", b"not found")
                return
            self._send(200, "application/json", b'{"ok":true}')

        def _send(self, code: int, ctype: str, body: bytes, *, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass  # silence per-request logs

    return Handler


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror teleoperation + recording. THE RIGHT ARM WILL MOVE.")
    parser.add_argument("--config", default="configs/station.yaml")
    parser.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT,
        help="Where each task's dataset folder goes: the task you set (here or on the dashboard) names its folder "
        f"(default {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--dataset", default="",
        help="Record every episode into this one folder instead, e.g. ~/hackathon_data/my-task; the task is then only "
        "a label.",
    )
    parser.add_argument("--task", default="", help="Task description recorded with every episode (names its folder)")
    parser.add_argument("--leader", default="left", choices=["left", "right"], help="Which arm to move by hand")
    parser.add_argument("--follower", default="right", choices=["left", "right"], help="Which arm tracks the leader")
    parser.add_argument(
        "--gravity", action="store_true",
        help="Gravity compensation / weightless mode on the leader arm (Metal arms only; requires pinocchio).",
    )
    parser.add_argument(
        "--gravity-kd", type=float, default=0.0, metavar="KD",
        help="Velocity damping on the leader arm in gravity mode (default 0 = no resistance when moving).",
    )
    parser.add_argument(
        "--grip-extra", type=float, default=0.15, metavar="EXTRA",
        help="How much extra the follower gripper closes past the leader's calibrated zero (default 0.15).",
    )
    parser.add_argument(
        "--no-gripper-hold", action="store_true",
        help="Skip the metal gripper holding-torque controller (use if the gripper motor is unresponsive).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default 8080).")
    parser.add_argument(
        "--leader-device",
        default="metal",
        choices=["metal", "star"],
        help="metal (default): the --leader Metal arm is moved by hand. star: a Star Arm 102 leads the follower, and "
        "the --leader Metal arm rests torque off and is recorded still.",
    )
    parser.add_argument("--star-port", help="The Star leader's serial port, e.g. /dev/serial/by-id/usb-1a86_...")
    parser.add_argument("--star-id", help="The Star leader's calibration name (as makermodslab saved it)")
    args = parser.parse_args()

    if args.leader == args.follower:
        print("Error: leader and follower must be different arms.")
        return
    star = args.leader_device == "star"
    if star and not (args.star_port and args.star_id):
        parser.error("--leader-device star needs --star-port and --star-id")
    if star and args.gravity:
        parser.error("--gravity is for a Metal leader arm; leave it out with --leader-device star")

    from dream_hackathon import cameras as cam_mod
    from dream_hackathon import config as cfg
    from dream_hackathon import dataset as ds
    from dream_hackathon import layouts

    station = cfg.load(args.config)
    mm = station.makermods
    if mm is None:
        print("This script is for Metal/Maker arm stations only.")
        return

    if station.arm == "metal":
        from lerobot.robots.metal_follower import MetalFollower as Follower
        from lerobot.robots.metal_follower import MetalFollowerConfig as FollowerConfig
        from makermodslab import metal_gripper
        from makermodslab.arms import METAL as FAMILY
    else:
        from lerobot.robots.maker_follower import MakerFollower as Follower
        from lerobot.robots.maker_follower import MakerFollowerConfig as FollowerConfig
        from makermodslab.arms import MAKER as FAMILY
        metal_gripper = None  # type: ignore[assignment]

    # ---- open cameras
    print("Opening cameras ...")
    cams = cam_mod.open_cameras(station.cameras, width=station.record_width, height=station.record_height)
    vk = layouts.video_keys(station.family)
    video_key_map = {"overhead": vk["overhead"], "wrist_l": vk["wrist_l"], "wrist_r": vk["wrist_r"]}

    # ---- dashboard state + server
    task_known = bool(args.dataset) or bool(args.task.strip())  # else the first task comes from the page
    dash_state = _DashState(task=_task_or_default(args.task) if task_known else "", per_task=not args.dataset)
    with dash_state.lock:
        dash_state.camera_roles = list(cams.keys())

    try:
        _server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", args.port),
            _make_handler(dash_state, _cmd_q, cams),
        )
        _server.daemon_threads = True
        threading.Thread(target=_server.serve_forever, name="dashboard", daemon=True).start()
        print(f"dashboard: http://localhost:{args.port}  (remote: ssh -L {args.port}:localhost:{args.port} <user>@<station>)")
    except OSError as e:
        print(f"Warning: could not start dashboard on port {args.port} ({e}). Continuing without it.")

    # ---- open the Star leader first: a missing driver or calibration then exits before any arm is open
    star_dev = None
    if star:
        print(f"Opening the Star Arm 102 leader on {args.star_port} (calibration {args.star_id!r}) ...")
        star_dev = _open_star_leader(args.star_port, args.star_id)
        print("  Star leader: connected")

    # ---- open leader arm (torque off or gravity-compensated)
    leader_cfg = station.arms[args.leader]
    leader_gravity = False
    if args.gravity:
        if station.arm != "metal":
            print("--gravity is only supported for Metal arms; falling back to torque-off.")
        else:
            try:
                from lerobot.teleoperators.metal_leader import MetalLeader, MetalLeaderConfig
                print(f"Opening {args.leader} arm (gravity compensation — weightless mode) ...")
                kd_kwargs: dict = {}
                try:
                    # Reduce velocity damping so the arm moves freely; not all versions expose kd.
                    MetalLeaderConfig(port=leader_cfg.port, id="_probe", can_interface=mm.can_interface, kd=0.0)
                    kd_kwargs = {"kd": args.gravity_kd}
                except TypeError:
                    pass
                leader_dev = MetalLeader(MetalLeaderConfig(
                    port=leader_cfg.port, id="mirror_leader", can_interface=mm.can_interface,
                    hold_kp_on_disconnect=0.0, **kd_kwargs,
                ))
                leader_dev.connect()
                leader_gravity = True
                print(f"  {args.leader}: gravity compensation active — arm is weightless")
            except Exception as e:
                print(f"--gravity failed ({e}); falling back to torque-off.")

    if not leader_gravity:
        role = "resting, recorded still" if star else "leader"
        print(f"Opening {args.leader} arm (torque off — {role}) ...")
        leader_dev = Follower(FollowerConfig(
            port=leader_cfg.port, id="mirror_leader", can_interface=mm.can_interface
        ))
        leader_dev.bus.connect(handshake=False)
        leader_dev.bus.disable_torque()
        print(f"  {args.leader}: connected (torque off — {'leave it resting' if star else 'move it by hand'})")

    leader_mode = LEADER_STAR if star else _leader_mode(args.gravity, leader_gravity)
    leader_label = _leader_mode_label(leader_mode, args.follower if star else args.leader)
    with dash_state.lock:
        dash_state.leader_mode = leader_mode
        dash_state.leader_mode_label = leader_label
    print(f"  {leader_label}")

    # ---- open follower arm (energized — THE ARM WILL MOVE)
    print(f"Opening {args.follower} arm (energized — follower, THE ARM WILL MOVE) ...")
    follower_cfg = station.arms[args.follower]
    follower_dev = Follower(FollowerConfig(
        port=follower_cfg.port, id=f"mirror_follower", can_interface=mm.can_interface
    ))
    if metal_gripper is not None and FAMILY.supports_gripper_effort_control and not args.no_gripper_hold:
        metal_gripper.install_metal_gripper(follower_dev, "")
    FAMILY.prepare_teleoperation(follower_dev)
    FAMILY.prepare_leader_following(follower_dev)
    follower_dev.connect(calibrate=False)
    rest_poses = list(FAMILY.capture_rest_poses(follower_dev))
    # Capture follower start joints (home pose) right after connect.
    _home_degs = _read_follower_degs(follower_dev)
    home_joints = np.radians([_home_degs[m] for m in MOTORS])
    home_grip = 1.0  # open gripper at home
    print(f"  {args.follower}: connected (energized)")

    # ---- move follower to home
    print("Moving follower to home pose ...")
    dt_home = 1.0 / station.control_hz
    while True:
        try:
            curr_degs = _read_follower_degs(follower_dev)
            curr_joints = np.radians([curr_degs[m] for m in MOTORS])
        except Exception:
            break
        delta = home_joints - curr_joints
        dist = float(np.max(np.abs(delta)))
        if dist < 1e-3:
            break
        step = np.clip(delta, -HOME_SPEED * dt_home, HOME_SPEED * dt_home)
        _command_follower(follower_dev, follower_cfg, curr_joints + step, home_grip, args.grip_extra)
        time.sleep(dt_home)
    print("  follower at home.")

    time.sleep(0.5)

    # ---- dataset writer
    state_names = []
    for side in layouts.SIDE_ORDER[station.family]:
        state_names += [f"{side}_joint_{i}" for i in range(layouts.JOINTS)] + [f"{side}_gripper"]

    def open_writer(path: pathlib.Path) -> "ds.DatasetWriter":
        return ds.DatasetWriter(
            root=path,
            fps=station.fps,
            robot_type=station.arm,
            state_names=state_names,
            video_keys=list(video_key_map.values()),
            frame_width=station.record_width,
            frame_height=station.record_height,
        )

    pinned = pathlib.Path(args.dataset).expanduser() if args.dataset else None
    data_root = pathlib.Path(args.data_root).expanduser()
    folders = _TaskFolders(data_root, open_writer, pinned=pinned)
    if task_known:
        if folders.use(args.task):
            print(f"\nDataset: {folders.folder}  (episodes so far: {folders.saved_episodes})")
        else:
            print(f"\n✗ Cannot use the dataset folder ({folders.error}). Set another task on the dashboard.")
    else:
        print(f"\nNo task yet: set one on the dashboard. Its name becomes a folder in {data_root}.")

    # ---- start input thread
    threading.Thread(target=_input_loop, daemon=True).start()

    # ---- control loop
    dt = 1.0 / station.control_hz
    record_every = station.control_hz // station.fps
    episode = None
    tick = 0
    homing_since: float | None = None
    applied_task_version = 0
    # A Metal leader is the follower's command from the first tick. A Star leader takes over only once it is brought
    # to the follower's pose (and again after each homing), so the follower never jumps to where the Star arm is.
    engaged = not star
    offset_deg: float | None = None  # the Star arm's distance from the follower's pose while not engaged
    star_failures = 0  # consecutive failed Star reads

    print("\nReady. Controls (type then Enter, or use the dashboard):")
    print("  Enter — start episode / save episode")
    print("  d     — discard episode")
    print("  h     — go to home pose")
    print("  q     — quit\n")

    try:
        while True:
            t0 = time.monotonic()

            # Snapshot the page's state once per tick (avoid holding the lock during arm I/O)
            with dash_state.lock:
                _homing_now = dash_state.homing
                task_version, task_now = dash_state.task_version, dash_state.task

            # Read arms. A failed read (a motor that stopped answering, say) is never turned into a pose that is
            # commanded or recorded: the follower holds where it is and the frame is skipped.
            leader_ok = follower_ok = True
            try:
                leader_degs = _read_leader_degs(leader_dev, leader_gravity)
                leader_joints, leader_grip = _degs_to_joints_grip(leader_degs, leader_cfg)
            except Exception as e:
                print(f"leader read error: {e}")
                leader_ok = False
                leader_joints = np.zeros(6)
                leader_grip = 0.0

            try:
                follower_degs = _read_follower_degs(follower_dev)
                follower_joints, follower_grip = _degs_to_joints_grip(follower_degs, follower_cfg)
            except Exception as e:
                print(f"follower read error: {e}")
                follower_ok = False
                follower_joints = np.zeros(6)
                follower_grip = 0.0

            # The follower's command: the Metal leader's pose, or the Star leader's (in follower degrees, its gripper
            # in the follower's convention). A failed Star read is never commanded or recorded either.
            if star:
                command_ok = True
                try:
                    command_joints, command_grip = _degs_to_joints_grip(_read_star_degs(star_dev), follower_cfg)
                except Exception as e:
                    print(f"Star leader read error: {e}")
                    command_ok = False
                    command_joints = np.zeros(6)
                    command_grip = 0.0
                was_engaged = engaged
                star_failures, engaged = _star_read_outcome(star_failures, command_ok, engaged)
                if was_engaged and not engaged:
                    print(
                        f"Star leader disengaged after {star_failures} failed reads in a row: bring it back to the "
                        "follower's pose to follow again."
                    )
            else:
                command_ok, command_joints, command_grip = leader_ok, leader_joints, leader_grip

            # A Star leader engages once it is within ENGAGE_TOLERANCE_DEG of the follower (never while homing).
            if not engaged and not _homing_now:
                if command_ok and follower_ok:
                    offset_deg = _joint_offset_deg(command_joints, follower_joints)
                    if _engaged(command_joints, follower_joints):
                        engaged, offset_deg = True, None
                        print(f"Star leader engaged (within {ENGAGE_TOLERANCE_DEG:.0f}°): the follower follows it.")
                    elif tick % (station.control_hz * 2) == 0:
                        print(
                            f"Not following yet: move the Star arm to the follower's pose (off by {offset_deg:.0f}°)"
                        )
                else:
                    offset_deg = None

            # Command follower: home sequence takes priority over mirroring
            if _homing_now:
                if homing_since is None:
                    homing_since = time.monotonic()
                outcome, command = _home_step(
                    follower_joints if follower_ok else None,
                    home_joints,
                    HOME_SPEED * dt,
                    time.monotonic() - homing_since,
                )
                if outcome == "move":
                    try:
                        _command_follower(follower_dev, follower_cfg, command, home_grip, args.grip_extra)
                    except Exception as e:
                        print(f"home step error: {e}")
                elif outcome != "hold":
                    with dash_state.lock:
                        dash_state.homing = False
                    if star:  # the Star arm stayed where it was: bring it to the follower again before it leads
                        engaged = False
                    if outcome == "reached":
                        print("↩ Home reached.")
                    else:
                        print(
                            f"↩ Homing gave up after {HOME_TIMEOUT_S:.0f} s; mirroring resumes where the arm stopped."
                        )
            else:
                homing_since = None
                if command_ok and engaged:
                    try:
                        _command_follower(follower_dev, follower_cfg, command_joints, command_grip, args.grip_extra)
                    except Exception as e:
                        print(f"command error: {e}")

            # Record frame
            recordable = _records_frame(leader_ok, follower_ok, command_ok, engaged)
            if episode is not None and recordable and tick % record_every == 0:
                images = {video_key_map[role]: cams[role].latest() for role in ("overhead", "wrist_l", "wrist_r")}
                measured_joints, measured_grips, joints_map, grips_map = _record_maps(
                    args.leader,
                    args.follower,
                    leader_joints,
                    leader_grip,
                    follower_joints,
                    follower_grip,
                    command_joints,
                    command_grip,
                )
                vec_state = layouts.to_vector(station.arm, measured_joints, measured_grips)
                vec_action = layouts.to_vector(station.arm, joints_map, grips_map)
                episode.add_frame(state=vec_state, action=vec_action, images=images)

            tick += 1

            # A task set on the page: create and open its folder as soon as no episode is recording (an episode in
            # progress keeps the folder it started in).
            if episode is None and task_version != applied_task_version:
                applied_task_version = task_version
                if pinned is None and task_now.strip():
                    if folders.use(task_now):
                        print(f"Dataset: {folders.folder}  (episodes so far: {folders.saved_episodes})")
                    else:
                        print(f"✗ Cannot use the folder for this task: {folders.error}")

            # Update dashboard state
            with dash_state.lock:
                dash_state.recording = episode is not None
                dash_state.frames = episode.frames if episode is not None else 0
                dash_state.total_episodes = folders.saved_episodes
                dash_state.folder = "" if folders.folder is None else str(folders.folder)
                dash_state.folder_error = folders.error or ""
                dash_state.leader_grip = command_grip
                dash_state.shoulder_pan_deg = math.degrees(command_joints[0])
                dash_state.leader_mode_label = (
                    leader_label if engaged else _not_following_label(args.follower, offset_deg)
                )

            # Handle keyboard / dashboard commands
            try:
                cmd = _cmd_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                if cmd in ("", "s", "start", "save"):  # Enter / s toggle; the dashboard asks for start or save
                    action = _episode_action(cmd, recording=episode is not None)
                    with dash_state.lock:
                        _is_homing = dash_state.homing
                        task_set = bool(dash_state.task.strip())
                        task_text = _task_or_default(dash_state.task)
                    if action == "start" and _is_homing:
                        print("(cannot start while homing — wait for home to complete)")
                    elif action == "start" and not engaged:
                        off = "" if offset_deg is None else f", off by {offset_deg:.0f}°"
                        print(f"(cannot start: not following yet — move the Star arm to the follower's pose{off})")
                    elif action == "start" and pinned is None and not task_set:
                        print("(set a task first: its name becomes the folder the episodes are saved in)")
                    elif action == "start" and not folders.use(task_text):  # the task's folder, whatever the loop did
                        print(f"✗ Cannot start: {folders.error}")
                    elif action == "start":
                        episode = folders.writer.start_episode(task_text)
                        print(
                            f"● Recording episode {folders.saved_episodes} in {folders.folder.name} ...  "
                            f"({leader_label})"
                        )
                    elif action == "save":
                        idx = _save_episode(episode)
                        if idx is not None:
                            print(f"✓ Saved episode {idx}  ({folders.saved_episodes} total in {folders.folder.name})")
                        episode = None  # a too-short episode is already discarded by save()
                    else:
                        print("(already recording)" if episode is not None else "(no episode recording)")
                elif cmd == "d":                    # discard
                    if episode is not None:
                        episode.discard()
                        print("✗ Discarded.")
                        episode = None
                    else:
                        print("(no episode recording)")
                elif cmd == "h":                    # go to home
                    if episode is not None:
                        print("(cannot home while recording — save or discard first)")
                    else:
                        with dash_state.lock:
                            dash_state.homing = True
                        print("↩ Returning to home pose ...")
                elif cmd == "q":                    # quit
                    if episode is not None:
                        idx = _save_episode(episode)
                        if idx is not None:
                            print(f"✓ Saved episode {idx} before quitting.")
                    break

            # Print status every ~2s
            if tick % (station.control_hz * 2) == 0:
                status = "● REC" if episode is not None else ("↩ HOME" if _homing_now else "○ idle")
                pan = math.degrees(command_joints[0])
                print(
                    f"{status}  shoulder_pan={pan:+.1f}°  grip={command_grip:.2f}  episodes={folders.saved_episodes}"
                )

            # Sleep to maintain control rate
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        if episode is not None:
            episode.discard()
            print("\nCtrl+C — discarded active episode.")
        print("Stopping.")
    finally:
        print("Returning follower to start pose ...")
        for ok, reason in FAMILY.return_to_rest(rest_poses):
            if not ok:
                print(f"  warning: {reason}")
        for problem in FAMILY.release_torque(follower_dev, f"{args.follower} follower"):
            print(f"  warning: {problem}")
        follower_dev.disconnect()
        if leader_gravity:
            try:
                from makermodslab.maker_rest_pose import stop_gravity_compensation
                stop_gravity_compensation(leader_dev)
            except Exception:
                pass
            leader_dev.disconnect()
        else:
            leader_dev.bus.disconnect(False)
        if star_dev is not None:
            try:
                star_dev.disconnect()
            except Exception as e:
                print(f"  warning: Star leader disconnect: {e}")
        print("Done.")


if __name__ == "__main__":
    main()
