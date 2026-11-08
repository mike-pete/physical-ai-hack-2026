"""Records Quest-teleoperated demonstrations into a LeRobot v2.1 dataset.

```
uv run dhk-collect --config configs/yam.yaml --dataset-dir ~/hackathon_data/cubes
uv run dhk-collect --config configs/yam.yaml --dataset-dir ~/hackathon_data/cubes --task "put the red cube in the bowl"
uv run dhk-collect --config configs/yam.yaml --dry-run     # read only: shows the targets, moves nothing
uv run dhk-collect --config configs/yam.yaml --sim         # YAM: i2rt's simulated arms, blank camera frames
uv run dhk-collect --config configs/station.yaml --policy graph --poses configs/unplug_poses.yaml [--dry-run]
```

`--policy graph` drives one arm with the scripted unplugging policy (unplug/graph.py) instead of the Quest, which it
does not need: Enter in the terminal opens the arms once every other device is ready, and Enter again (or Start on the
dashboard) starts the run; Stop on the dashboard holds the arm where it is. `--record` saves one episode per cable
unplugged. At the end, or on Ctrl+C, the arm first opens its gripper and rises straight up, then returns as below.

Hold a controller's grip to drive that arm, pull its trigger to close its gripper. A starts an episode and, pressed
again, saves it; B discards it; the right joystick click sends both arms back to where they started (not while
recording). Ctrl+C ends the session: an episode still recording is discarded, and the arms return to where they
started before the vendor's shutdown releases them.

Starting: the dashboard comes up first, before any device is opened. The session then waits for the Quest, the
cameras and the arms' links, showing what each needs on the page and in the terminal (devices.py), and opens the
arms only once everything is ready and A is pressed on the right controller (wait_and_open_arms()). During the
session the devices are checked every second, and a problem (a camera unplugged, a CAN link down, the Quest silent)
is shown the same way; a camera reconnects by itself once it is back.

Over SSH: the session serves a browser dashboard on the station's loopback address (dashboard.py says how to open
it). A lost connection or a `kill` ends the session the same safe way as Ctrl+C, with the rest of its output in
LOG_PATH; run it inside tmux to keep it going instead. Only one program drives the arms at a time (station_lock.py).

The loop runs at the config's `control_hz`; every `control_hz / fps`-th tick is recorded: the measured joints and
grippers as `observation.state`, the commanded ones as `action`, and each camera's newest frame.

@see Collector, main()
"""

import argparse
import dataclasses
import os
import pathlib
import re
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from dream_hackathon import (
    arms,
    cameras,
    config,
    dashboard,
    dataset,
    devices,
    ik,
    layouts,
    quest,
    station_lock,
    teleop,
)
from dream_hackathon.unplug import graph as unplug_graph
from dream_hackathon.unplug import motion as unplug_motion
from dream_hackathon.unplug import params as unplug_params
from dream_hackathon.unplug import perception as unplug_perception
from dream_hackathon.unplug import poses as unplug_poses
from dream_hackathon.unplug import verify as unplug_verify

## Homing and the end-of-session return move every joint at most this fast.
HOME_SPEED_RAD_S = 0.5
## Homing is done when every joint is this close to its start.
HOME_TOLERANCE_RAD = 1e-3
## A live arm whose measured joints fall this far behind the command is re-synced and its clutch released.
MAX_TRACKING_ERROR_RAD = 0.35
## Homing gives up (and the arms stop where they are) after this long.
HOME_TIMEOUT_S = 30.0
## How often the status line is printed, and how often the dashboard's state is refreshed.
STATUS_EVERY_S = 1.0
SNAPSHOT_EVERY_S = 0.2
## How long a warning stays on the dashboard.
WARNING_SHOWN_S = 10.0
## Where a session's output goes once its terminal is gone (a lost SSH connection).
LOG_PATH = pathlib.Path("~/.cache/dream_hackathon/collect.log").expanduser()
## The task text an episode records when neither `--task` nor the config names one. A task text is optional for
## the participant, since Dream's service labels every dataset from its head camera, but the LeRobot v2.1 files
## (and Dream's checks of them) need one non-empty text per episode.
DEFAULT_TASK = "do the demonstrated task"
## Dream's limits on a task text: its length, and no control characters (it must be one line).
TASK_MAX_CHARS = 512
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")

## How often the devices are checked, while waiting for them and during a session.
CHECK_EVERY_S = 1.0
## How often the Quest is read while the session waits for A.
WAIT_TICK_S = 0.05
## A opens the arms only once everything has been ready this long: a press in the moment the Quest's data comes back
## is not taken for one.
A_AFTER_READY_S = 1.0

## Frames by camera role, at the recording size.
FrameSource = Callable[[], dict[str, npt.NDArray[np.uint8]]]
## The graph policy's params file, when it exists and --unplug-config names none.
UNPLUG_CONFIG = pathlib.Path("configs/unplug.yaml")
## The graph policy's viewing pose and reference row, when it exists and --view names none.
UNPLUG_VIEW = pathlib.Path("configs/unplug_view.yaml")
## Where the graph policy's tip-check pictures go, one folder per session.
UNPLUG_DEBUG_DIR = pathlib.Path("~/.cache/dream_hackathon/unplug").expanduser()
## The shutdown retreat waits this long for the gripper to open before rising.
RETREAT_OPEN_S = 0.8
## The shutdown retreat stops where it is when a joint falls this far behind.
RETREAT_MAX_LAG_RAD = 0.10
## A Quest that reports nothing: what a scripted policy's session reads instead of a headset.
_IDLE_QUEST = quest.QuestState(
    {"left": np.eye(4), "right": np.eye(4)}, {"left": 0.0, "right": 0.0}, {"left": 0.0, "right": 0.0}, frozenset()
)


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    """What `--policy graph` runs: the poses and parameters, which cables, and whether episodes are recorded."""

    poses: pathlib.Path
    params: pathlib.Path | None = None
    cables: tuple[int, ...] | None = None
    stop_after: str | None = None
    speed_scale: float = 1.0
    record: bool = False
    view: pathlib.Path | None = None
    capture_view: bool = False


class SessionControls:
    """What the dashboard can change in a session: the task text new episodes record, and the dataset's episodes.

    Shared by the control loop and the dashboard's threads. The dataset is guarded by `lock`: the collector holds it
    while an episode records, and a delete takes it without waiting, so a delete is refused while an episode records
    and no episode starts while a delete runs. A delete renumbers the later episodes (Dream wants 0, 1, 2, ...) and
    reloads the writer from the folder, so the next episode is numbered right.

    Attributes:
    - task: The task text the next episode records.
    - writer: The dataset's writer.
    - lock: Held while an episode records, or while episodes are deleted.
    """

    def __init__(self, make_writer: Callable[[], dataset.DatasetWriter], task: str) -> None:
        """@param make_writer Opens the dataset's writer; called again after a delete, to reload it."""
        self._make_writer = make_writer
        self.writer = make_writer()
        self.task = task
        self.lock = threading.Lock()

    def set_task(self, text: str) -> tuple[bool, str]:
        """Sets the task text the next episode records; an empty text goes back to DEFAULT_TASK."""
        text = (text or "").strip()
        if not text:
            self.task = DEFAULT_TASK
            return True, f"task cleared: new episodes record {DEFAULT_TASK!r}"
        if len(text) > TASK_MAX_CHARS:
            return False, f"the task is {len(text)} characters; Dream takes up to {TASK_MAX_CHARS}"
        if _CONTROL_CHARACTER.search(text):
            return False, "the task must be one line of plain text"
        self.task = text
        return True, f"task set: new episodes record {text!r}"

    def delete(self, indices: list[int]) -> tuple[bool, str]:
        """Deletes episodes by number, renumbering the later ones; refused while an episode records."""
        chosen = sorted(set(indices))
        if not chosen:
            return False, "no episodes chosen"
        if not self.lock.acquire(blocking=False):
            return False, "an episode is recording: save it (A) or discard it (B) first"
        try:
            left = dataset.drop_episodes(self.writer.root, set(chosen))
            self.writer = self._make_writer()
        except dataset.DatasetError as error:
            return False, f"nothing deleted: {error}"
        finally:
            self.lock.release()
        names = ", ".join(str(index) for index in chosen)
        return True, f"deleted episode{'s' if len(chosen) > 1 else ''} {names}; {left} left, the later ones renumbered"


@dataclasses.dataclass
class _Command:
    """What the loop last commanded: per side, six joints (radians) and the gripper (0 closed to 1 open)."""

    joints: arms.Joints
    grips: arms.Grips


class Collector:
    """The teleop-and-record loop over injected devices, one tick at a time (tick()).

    Attributes:
    - homing: Whether the arms are on their way back to where they started.
    - episode: The episode being recorded, or None.
    """

    def __init__(
        self,
        station: config.StationConfig,
        task: str,
        robot: arms.Arms,
        headset: "quest.Quest | None",
        frames: FrameSource,
        iks: dict[str, ik.ArmIK],
        writer: dataset.DatasetWriter,
        log: Callable[[str], None] = print,
        controls: SessionControls | None = None,
        policy: unplug_graph.UnplugGraph | None = None,
        record_policy: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Holds the devices; the command starts where the arms are.

        @param station The station config.
        @param task The task text recorded with each episode.
        @param robot Both arms.
        @param headset The Quest (anything with read() returning a quest.QuestState); None with a policy.
        @param frames Returns each camera's newest frame, at the recording size.
        @param iks Per side, the teleop IK.
        @param writer The dataset episodes are saved to.
        @param log Where events are reported.
        @param controls The task and dataset the dashboard can change; None keeps `task` and `writer` as they are.
        @param policy A scripted policy driving its one arm instead of the Quest; the other arm holds still.
        @param record_policy Whether the policy's runs are recorded, one episode per cable.
        @param clock The policy's time, in seconds.
        """
        self._station = station
        self._controls = controls or SessionControls(lambda: writer, task)
        self._robot = robot
        self._headset = headset
        self._policy = policy
        self._record_policy = record_policy
        self._clock = clock
        self._frames = frames
        self._iks = iks
        self._log = log
        self._dt_s = 1.0 / station.control_hz
        self._record_every = station.control_hz // station.fps
        self._video_keys = layouts.video_keys(station.family)
        self._ticks = 0
        joints, grips = robot.read()
        self._command = _Command({s: q.copy() for s, q in joints.items()}, dict(grips))
        self._clutches = {
            side: teleop.Clutch(
                quest_to_base=arm.quest_to_base,
                workspace_min_m=arm.workspace_min_m,
                workspace_max_m=arm.workspace_max_m,
                translation_scale=station.teleop.translation_scale,
            )
            for side, arm in station.arms.items()
        }
        self._warnings: list[tuple[float, str]] = []
        self.homing = False
        self.episode: dataset.Episode | None = None
        if policy is not None:
            side = policy.side
            policy.attach(joints[side], grips[side], robot.start_joints[side], ghost=not robot.energized)

    def tick(self) -> None:
        """One control tick: read, react to the buttons, move the command, send it, record every n-th tick."""
        measured, measured_grips = self._robot.read()
        if self._policy is not None:
            if self.homing:
                self._step_home()
            else:
                self._drive(measured, measured_grips)
        else:
            state = self._headset.read() if self._headset is not None else _IDLE_QUEST
            self._handle_buttons(state.pressed)
            if self.homing:
                self._step_home()
            else:
                self._follow(state, measured)
        if self._robot.energized:
            self._resync_lagging(measured)
            self._robot.command(self._command.joints, self._command.grips)
        if self.episode is not None and self._ticks % self._record_every == 0:
            try:
                images = self._frames()
            except cameras.CameraError as error:
                # An episode missing a camera's frames cannot train; teleop goes on, and the next one can record.
                self._warn(f"{error}: the episode is discarded")
                self.discard()
            else:
                self.episode.add_frame(
                    layouts.to_vector(self._station.arm, measured, measured_grips),
                    layouts.to_vector(self._station.arm, self._command.joints, self._command.grips),
                    {self._video_keys[role]: frame for role, frame in images.items()},
                )
        self._ticks += 1

    def snapshot(self) -> dict[str, Any]:
        """The session's state, for the dashboard and the status line; call it from the loop's thread (it runs FK).

        @return A JSON-ready dict: `mode`, `recording`, `frames`, `seconds`, `episodes`, `task`, `dataset`, `homing`,
            per side under `arms` its `engaged`, `tool_m` and `grip`, the recent `warnings`, and `control_hz`.
        """
        now = time.monotonic()
        self._warnings = [(when, text) for when, text in self._warnings if now - when < WARNING_SHOWN_S]
        per_arm = {}
        policy = self._policy
        for side in ("left", "right"):
            xyz = self._iks[side].fk(self._command.joints[side])[:3, 3]
            if policy is not None:
                engaged = side == policy.side and policy.started and not policy.done
            else:
                engaged = self._clutches[side].engaged
            per_arm[side] = {
                "engaged": engaged,
                "tool_m": [round(float(value), 4) for value in xyz],
                "grip": round(float(self._command.grips[side]), 3),
            }
        frames = self.episode.frames if self.episode is not None else 0
        extra = {"policy": policy.status()} if policy is not None else {}
        return extra | {
            "mode": "live" if self._robot.energized else "dry run",
            "recording": self.episode is not None,
            "frames": frames,
            "seconds": frames / self._station.fps,
            "episodes": self._controls.writer.total_episodes,
            "task": self._controls.task,
            "dataset": str(self._controls.writer.root),
            "homing": self.homing,
            "arms": per_arm,
            "warnings": [text for _, text in self._warnings],
            "control_hz": self._station.control_hz,
        }

    def go_home(self, timeout_s: float = HOME_TIMEOUT_S) -> None:
        """Brings the arms back to where they started at HOME_SPEED_RAD_S, ticking until they arrive or time is up."""
        self.homing = True
        deadline_s = time.monotonic() + timeout_s
        while self.homing:
            if time.monotonic() > deadline_s:
                self._warn("the arms did not reach their start in time: stopping where they are")
                self.homing = False
                return
            started_s = time.perf_counter()
            self._step_home()
            if self._robot.energized:
                measured, _ = self._robot.read()
                self._resync_lagging(measured)
                self._robot.command(self._command.joints, self._command.grips)
            time.sleep(max(0.0, self._dt_s - (time.perf_counter() - started_s)))

    def discard(self) -> None:
        """Drops the episode being recorded, if any."""
        if self.episode is not None:
            try:
                self.episode.discard()
            finally:
                self.episode = None
                self._controls.lock.release()
            self._log("episode discarded")

    def _handle_buttons(self, pressed: frozenset[str]) -> None:
        if "A" in pressed:
            self._toggle_recording()
        if "B" in pressed:
            self.discard()
        if "RJ" in pressed:
            if self.episode is not None:
                self._log("homing is off while recording: save (A) or discard (B) first")
            else:
                self.homing = True
                for clutch in self._clutches.values():
                    clutch.engaged = False

    def _toggle_recording(self) -> None:
        controls = self._controls
        if self.episode is None:
            if not self._robot.energized:
                self._log("a dry run does not record")
                return
            # Held until the episode is saved or discarded, so the dashboard cannot delete episodes under it.
            if not controls.lock.acquire(blocking=False):
                self._log("episodes are being deleted from the dashboard: press A again in a moment")
                return
            try:
                self.episode = controls.writer.start_episode(controls.task)
            except BaseException:
                controls.lock.release()
                raise
            self._log(f"recording episode {controls.writer.total_episodes}: {controls.task!r}")
            return
        episode, self.episode = self.episode, None
        try:
            index = episode.save()
        except dataset.DatasetError as error:
            self._log(f"not saved: {error}")
            return
        finally:
            controls.lock.release()
        self._log(f"saved episode {index} ({episode.frames} frames); {controls.writer.total_episodes} in the dataset")

    def _follow(self, state: "quest.QuestState", measured: arms.Joints) -> None:
        """Moves each arm's command toward its clutch's target; a released clutch holds it.

        In a dry run the arms never move, so a released clutch puts the command where the real arm is: move an arm by
        hand and the status line's tool position follows it, which checks the IK model against the real arm.
        """
        for side, clutch in self._clutches.items():
            solver = self._iks[side]
            joints = self._command.joints[side]
            target = clutch.update(state.grip[side], state.poses[side], solver.fk(joints))
            if target is None:
                if not self._robot.energized:
                    self._command.joints[side] = measured[side].copy()
                continue
            self._command.joints[side] = solver.step(target, joints, self._dt_s)
            self._command.grips[side] = 1.0 - state.trigger[side]

    def _drive(self, measured: arms.Joints, measured_grips: arms.Grips) -> None:
        """Moves the policy's arm's command one tick along its run; the other arm holds (or, in a dry run, follows
        the real arm, as an unclutched one does in teleop). Starts and ends the policy's episodes when recording."""
        policy = self._policy
        assert policy is not None
        side = policy.side
        joints, grip, _ = policy.act(self._clock(), measured[side], measured_grips[side])
        self._command.joints[side] = joints
        self._command.grips[side] = grip
        if not self._robot.energized:
            for other in measured:
                if other != side:
                    self._command.joints[other] = measured[other].copy()
        for kind, details in policy.events():
            if not (self._record_policy and self._robot.energized):
                continue
            if kind == "episode_start" and self.episode is None:
                self._toggle_recording()
            elif kind == "episode_end" and self.episode is not None:
                if details.get("success"):
                    self._toggle_recording()  # saves it
                else:
                    self.discard()

    def retreat(self) -> None:
        """At the end of a policy session: opens the policy arm's gripper, then lifts its tool straight up.

        So the joint-space return to the start (go_home()) begins clear of the strip and the cables. It stops where
        it is if the arm falls behind; a dry run skips it.
        """
        policy = self._policy
        if policy is None or not self._robot.energized:
            return
        side = policy.side
        self._command.grips[side] = 1.0
        measured, _ = self._robot.read()
        self._command.joints[side] = measured[side].copy()
        end_s = time.monotonic() + RETREAT_OPEN_S
        while time.monotonic() < end_s:
            self._robot.command(self._command.joints, self._command.grips)
            time.sleep(self._dt_s)
        path = policy.retreat_path(self._robot.read()[0][side])
        if path is None:
            return
        for joints in path.points:
            started_s = time.perf_counter()
            measured, _ = self._robot.read()
            if float(np.max(np.abs(self._command.joints[side] - measured[side]))) > RETREAT_MAX_LAG_RAD:
                self._warn("the arm cannot rise: stopping the retreat where it is")
                self._command.joints[side] = measured[side].copy()
                self._robot.command(self._command.joints, self._command.grips)
                return
            self._command.joints[side] = joints
            self._robot.command(self._command.joints, self._command.grips)
            time.sleep(max(0.0, self._dt_s - (time.perf_counter() - started_s)))

    def _step_home(self) -> None:
        max_step = HOME_SPEED_RAD_S * self._dt_s
        done = True
        for side, start in self._robot.start_joints.items():
            joints = teleop.step_toward(self._command.joints[side], start, max_step)
            self._command.joints[side] = joints
            done = done and bool(np.all(np.abs(joints - start) <= HOME_TOLERANCE_RAD))
        if done:
            self.homing = False

    def _warn(self, text: str) -> None:
        """Logs a warning and shows it on the dashboard for WARNING_SHOWN_S."""
        self._log(f"warning: {text}")
        self._warnings.append((time.monotonic(), text))

    def _resync_lagging(self, measured: arms.Joints) -> None:
        """Re-syncs an arm that cannot follow (blocked, or stalled) to where it is, and releases its clutch."""
        for side, joints in measured.items():
            error = float(np.max(np.abs(self._command.joints[side] - joints)))
            if error > MAX_TRACKING_ERROR_RAD:
                self._warn(f"{side} arm is {np.degrees(error):.0f} deg behind its target: holding where it is")
                self._command.joints[side] = joints.copy()
                self._clutches[side].engaged = False
                if self._policy is not None and side == self._policy.side:
                    self._policy.stop(f"the {side} arm fell behind")


def run(
    station: config.StationConfig,
    task: str,
    *,
    dry_run: bool,
    sim: bool,
    quest_ip: str | None,
    dashboard_port: int = dashboard.DEFAULT_PORT,
    policy: PolicySpec | None = None,
) -> None:
    """Opens the devices, runs the loop until Ctrl+C (or a lost SSH connection), then shuts everything down safely.

    @param policy A scripted policy to run instead of Quest teleop, or None.
    @throws StationBusyError Another program (a collector, a check, `dreamcli`) holds the station's arms.
    @throws PlanError The policy's run has a path the arm cannot follow (before any device opens).
    """
    iks = arms.open_ik(station)  # first: a missing tool center point stops here, before any device opens
    graph = _make_graph(station, iks, policy) if policy is not None else None

    def make_writer() -> dataset.DatasetWriter:
        return dataset.DatasetWriter(
            station.dataset_dir,
            station.fps,
            station.arm,
            _state_names(station),
            list(layouts.video_keys(station.family).values()),
            station.record_width,
            station.record_height,
        )

    controls = SessionControls(make_writer, task)
    # A simulated YAM drives no hardware, so it takes no lock.
    lock = None if sim else station_lock.acquire("dhk-collect", station.dataset_dir.resolve())
    try:
        _run_locked(
            station,
            controls,
            iks,
            dry_run=dry_run,
            sim=sim,
            quest_ip=quest_ip,
            port=dashboard_port,
            graph=graph,
            record_policy=policy.record if policy is not None else False,
        )
    finally:
        if lock is not None:
            lock.close()


def _make_graph(station: config.StationConfig, iks: dict[str, ik.ArmIK], spec: PolicySpec) -> unplug_graph.UnplugGraph:
    """The graph policy, its whole run planned (and printed) before anything opens."""
    poses = unplug_poses.load_poses(spec.poses)
    params = unplug_params.load_params(spec.params).scaled(spec.speed_scale)
    view = unplug_perception.load_view(spec.view) if spec.view is not None else None
    if view is None:
        _say("graph policy: no view reference: the taught grasps are used as they are, nothing is located")
    arm = station.arms[poses.side]
    try:
        graph = unplug_graph.UnplugGraph(
            poses,
            params,
            iks[poses.side],
            arm.workspace_min_m,
            arm.workspace_max_m,
            1.0 / station.control_hz,
            log=_say,
            cables=spec.cables,
            stop_after=spec.stop_after,
            view=view,
            capture_view=spec.view if spec.capture_view else None,
        )
    except ValueError as error:  # a cable number or step that does not exist (PlanError is not a ValueError)
        raise config.ConfigError(f"graph policy: {error}") from error
    _say(graph.report)
    return graph


def _run_locked(
    station: config.StationConfig,
    controls: SessionControls,
    iks: dict[str, ik.ArmIK],
    *,
    dry_run: bool,
    sim: bool,
    quest_ip: str | None,
    port: int,
    graph: unplug_graph.UnplugGraph | None = None,
    record_policy: bool = False,
) -> None:
    """run() with the station's lock held."""
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _hang_up)
    if os.environ.get("SSH_CONNECTION") and not (os.environ.get("TMUX") or os.environ.get("STY")):
        _say(
            "note: this session runs over SSH outside tmux. If the connection drops it ends, safely (the arms return "
            "and are released); start it inside `tmux new -A -s collect` to keep it running instead."
        )
    found = devices.DeviceSet(station, sim=sim, quest_ip=quest_ip, use_quest=graph is None)
    blank = np.zeros((station.record_height, station.record_width, 3), dtype=np.uint8)
    # A policy session: the first Enter in the terminal opens the arms, later ones start the run (as Start does).
    enter = _EnterKey() if graph is not None else None
    collector: Collector | None = None

    def frame(role: str) -> npt.NDArray[np.uint8]:
        if sim:
            return blank
        camera = found.cameras.get(role)
        if camera is None:
            raise cameras.CameraError(f"camera '{role}' is not open yet")
        return camera.latest()

    def frames() -> dict[str, npt.NDArray[np.uint8]]:
        return {role: frame(role) for role in layouts.CAMERA_ROLES}

    def log(line: str) -> None:
        _say(line)
        if board is not None:
            board.event(line)

    def fix(fix_id: str) -> tuple[bool, str]:
        ok, message = found.apply_fix(fix_id)
        log(f"dashboard: {message}")
        return ok, message

    def set_task(text: str) -> tuple[bool, str]:
        ok, message = controls.set_task(text)
        log(f"dashboard: {message}")
        return ok, message

    def delete(indices: list[int]) -> tuple[bool, str]:
        ok, message = controls.delete(indices)
        log(f"dashboard: {message}")
        return ok, message

    def drive_policy(action: str) -> tuple[bool, str]:
        assert graph is not None
        if action == "stop":
            graph.stop("Stop on the dashboard")
            message = "stopped: the arm holds where it is (Ctrl+C in the terminal ends the session)"
        elif collector is None:
            return False, "the arms are not open yet: press Enter in the terminal first"
        elif graph.started:
            return False, "the run has already started"
        else:
            graph.start()
            message = "started"
        log(f"dashboard: graph {message}")
        return True, message

    board = _open_dashboard(
        station,
        frame,
        port,
        on_fix=fix,
        on_task=set_task,
        on_delete=delete,
        task_of=lambda: controls.task,
        on_policy=drive_policy if graph is not None else None,
    )

    monitor = None
    try:
        reported: dict[str, str | None] = {}
        try:
            robot = wait_and_open_arms(
                station,
                found,
                board,
                log,
                dry_run=dry_run,
                sim=sim,
                reported=reported,
                start_event=enter.pressed if enter is not None else None,
            )
        except KeyboardInterrupt:
            log("session ended before the arms were opened")
            return
        monitor = _Monitor(found, board, log, reported)
        try:
            if graph is not None:
                wrist = {"left": "wrist_l", "right": "wrist_r"}[graph.side]
                debug_dir = UNPLUG_DEBUG_DIR / time.strftime("%Y%m%d-%H%M%S")
                graph.connect(
                    log=log,
                    verifier=unplug_verify.TipVerifier(graph.vision, lambda: frame(wrist), debug_dir),
                    frame_of=frame,
                    debug_dir=debug_dir,
                )
            collector = Collector(
                station,
                controls.task,
                robot,
                found.headset,
                frames,
                iks,
                controls.writer,
                log=log,
                controls=controls,
                policy=graph,
                record_policy=record_policy,
            )
            if graph is not None and enter is not None:
                enter.on_press = graph.start
                log(
                    "graph policy ready: press Enter here (or Start on the dashboard) to start the run; Stop on the "
                    f"dashboard holds the arm; Ctrl+C ends the session. Locating and tip-check pictures: {debug_dir}"
                )
            _loop(collector, 1.0 / station.control_hz, board, quiet=graph is not None)
        finally:
            # From here the shutdown runs to its end: a hang-up, a kill or another Ctrl+C would leave the arms
            # wherever they are. The emergency stop is the way to halt it.
            for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(stop_signal, signal.SIG_IGN)
            try:
                if collector is not None:
                    collector.discard()
                    if graph is not None and robot.energized:
                        graph.stop("the session is ending")
                        log("opening the gripper and lifting the tool clear (the emergency stop halts it)...")
                        collector.retreat()
                    if robot.energized:
                        log("returning the arms to where they started (the emergency stop halts them)...")
                        collector.go_home()
            finally:
                robot.close()
                log("session ended")
    finally:
        if monitor is not None:
            monitor.stop()
        found.close()
        if board is not None:
            board.close()


def wait_and_open_arms(
    station: config.StationConfig,
    found: devices.DeviceSet,
    board: dashboard.Dashboard | None,
    log: Callable[[str], None],
    *,
    dry_run: bool,
    sim: bool,
    reported: dict[str, str | None] | None = None,
    open_arms: Callable[..., arms.Arms] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    start_event: threading.Event | None = None,
) -> arms.Arms:
    """Opens the devices as they become ready, then the arms once everything is ready and A is pressed.

    With `start_event` (a scripted policy's session, no Quest) the event stands for A: it is set by Enter in the
    terminal, counts under the same rules, and one set before everything is ready is dropped, not kept for later.

    Every CHECK_EVERY_S it tries the devices that are not open yet and shows each one's state on the page and, when
    it changes, in the terminal. Nothing opens the arms before A (a simulated YAM needs no A): on a YAM, opening them
    makes them hold their pose and sweep the grippers. A counts only once everything has been ready A_AFTER_READY_S,
    and only as a new press (Quest.read() never reports a button that was down when its data came back). Arms that
    fail to open are reported, and A tries again.

    @param reported Each device's last reported problem, filled in here so the session's monitor goes on from it.
    @return The open arms.
    """
    reported = {} if reported is None else reported
    open_arms = open_arms or arms.open_arms
    phase = ""
    arm_error: str | None = None
    ready = False
    ready_since_s = 0.0
    next_check_s = clock()
    while True:
        if clock() >= next_check_s:
            found.try_open()
            statuses = found.statuses()
            ready = all(status.problem is None for status in statuses)
            if arm_error is not None:
                statuses.append(devices.Status("arms", arm_error))
            _report(statuses, reported, log)
            if board is not None:
                board.devices(statuses)
            next_check_s = clock() + CHECK_EVERY_S
        wanted = ("opening" if sim else "ready") if ready else "waiting"
        if wanted != phase:
            phase = wanted
            if board is not None:
                board.phase("ready_enter" if phase == "ready" and start_event is not None else phase)
            if phase == "waiting":
                waiting_for = [name for name, problem in reported.items() if problem is not None and name != "arms"]
                log(f"waiting for: {', '.join(waiting_for)} (the dashboard says what each needs)")
            elif phase == "ready":
                ready_since_s = clock()
                key = "Enter here" if start_event is not None else "A on the right controller"
                log(f"everything is ready: press {key} to open the arms (Ctrl+C quits)")
        settled = clock() - ready_since_s >= A_AFTER_READY_S
        if start_event is not None:
            go = start_event.is_set()
            if go:
                start_event.clear()
                if not (phase == "ready" and settled):
                    log("Enter: not everything is ready yet, so the arms stay closed")
                    go = False
        else:
            pressed = found.headset.read().pressed if found.headset is not None else frozenset()
            go = phase == "ready" and settled and "A" in pressed
        if phase == "opening" or go:
            if board is not None:
                board.phase("opening")
            log("opening the arms" + ("" if dry_run or sim else ": keep clear of them and of the grippers' jaws"))
            try:
                robot = open_arms(station, energize=not dry_run, sim=sim)
            except Exception as error:  # noqa: BLE001 - shown, and A tries again
                arm_error = f"did not open: {error}; fix it, then press {'Enter' if start_event else 'A'} again"
                log(f"the arms {arm_error}")
                phase, next_check_s = "", clock()
                continue
            if arm_error is not None:
                reported.pop("arms", None)
                if board is not None:
                    board.devices(found.statuses())
            if board is not None:
                board.phase("")
            return robot
        sleep(WAIT_TICK_S)


class _EnterKey:
    """Enter in the terminal, read on a background thread: each press calls `on_press` if set, else sets `pressed`."""

    def __init__(self) -> None:
        self.pressed = threading.Event()
        self.on_press: Callable[[], None] | None = None
        threading.Thread(target=self._run, name="enter-key", daemon=True).start()

    def _run(self) -> None:
        try:
            for _ in sys.stdin:
                if self.on_press is not None:
                    self.on_press()
                else:
                    self.pressed.set()
        except (OSError, ValueError):
            pass  # the terminal is gone: the dashboard's buttons remain


class _Monitor:
    """Checks the devices every CHECK_EVERY_S during a session, on its own thread, and shows what changes."""

    def __init__(
        self,
        found: devices.DeviceSet,
        board: dashboard.Dashboard | None,
        log: Callable[[str], None],
        reported: dict[str, str | None],
        every_s: float = CHECK_EVERY_S,
    ) -> None:
        self._found = found
        self._board = board
        self._log = log
        self._reported = reported
        self._every_s = every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="device-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._every_s):
            try:
                statuses = self._found.statuses()
            except Exception as error:  # noqa: BLE001 - a check that fails must not end the session
                statuses = [devices.Status("devices", f"could not be checked: {error}")]
            _report(statuses, self._reported, self._log)
            if self._board is not None:
                self._board.devices(statuses)


def _report(statuses: list[devices.Status], reported: dict[str, str | None], log: Callable[[str], None]) -> None:
    """Logs each device whose state changed since the last report: its problem, or that it is ready again.

    A problem whose words stay the same but whose numbers change (`no controller data for 15 s`, then 16 s) is the
    same problem: it is logged once, not every second.
    """
    for status in statuses:
        if status.name in reported and _same(reported[status.name], status.problem):
            continue
        if status.problem is not None:
            log(f"{status.name}: {status.problem}")
        elif reported.get(status.name) is not None or status.name not in reported:
            log(f"{status.name}: ready")
        reported[status.name] = status.problem


def _same(problem: str | None, other: str | None) -> bool:
    """Whether two problems are the same but for their numbers (seconds waited, frame counts)."""
    if problem is None or other is None:
        return problem is other
    return re.sub(r"\d+(\.\d+)?", "#", problem) == re.sub(r"\d+(\.\d+)?", "#", other)


def _open_dashboard(
    station: config.StationConfig,
    frame: Callable[[str], npt.NDArray[np.uint8]],
    port: int,
    on_fix: Callable[[str], tuple[bool, str]] | None = None,
    on_task: Callable[[str], tuple[bool, str]] | None = None,
    on_delete: Callable[[list[int]], tuple[bool, str]] | None = None,
    task_of: Callable[[], str] | None = None,
    on_policy: Callable[[str], tuple[bool, str]] | None = None,
) -> dashboard.Dashboard | None:
    """Starts the dashboard and says how to open it; a taken port leaves the session without one."""
    if not port:
        return None
    try:
        board = dashboard.Dashboard(
            port,
            frame,
            layouts.video_keys(station.family),
            station.dataset_dir,
            title=f"{socket.gethostname()}: {station.arm} station",
            on_fix=on_fix,
            on_task=on_task,
            on_delete=on_delete,
            task_of=task_of,
            on_policy=on_policy,
        )
    except OSError as error:
        _say(
            f"dashboard: port {port} is not free ({error}); going on without it. Usually an `ssh -L {port}:...` "
            "that was run on the station itself holds it: type `exit` in that terminal (the forward belongs on "
            "your laptop), or pick another port with --dashboard-port"
        )
        return None
    _say(board.how_to_open())
    return board


def _loop(
    collector: Collector, period_s: float, board: dashboard.Dashboard | None = None, quiet: bool = False
) -> None:
    """Ticks at a steady rate until Ctrl+C; a late tick starts the next one at once rather than catching up.

    @param quiet Skip the teleop instructions (a policy session prints its own).
    """
    if not quiet:
        _say(
            "ready: hold a grip to drive an arm, A to record, B to discard, right joystick click to home, "
            "Ctrl+C to quit"
        )
    next_s = time.perf_counter()
    snapshot_s = status_s = next_s
    ticks = 0
    try:
        while True:
            collector.tick()
            ticks += 1
            now = time.perf_counter()
            if now >= snapshot_s:
                snapshot = collector.snapshot()
                snapshot["loop_hz"] = ticks / max(now - snapshot_s + SNAPSHOT_EVERY_S, 1e-6)
                ticks = 0
                if board is not None:
                    board.publish(snapshot)
                if now >= status_s:
                    _say(_status_line(snapshot))
                    status_s = now + STATUS_EVERY_S
                snapshot_s = now + SNAPSHOT_EVERY_S
            next_s = max(next_s + period_s, now)
            time.sleep(max(0.0, next_s - time.perf_counter()))
    except KeyboardInterrupt:
        pass


def _status_line(snapshot: dict[str, Any]) -> str:
    """The terminal's status line: what is recording, and per arm whether it is driving and where its tool is."""
    parts = [f"REC {snapshot['frames']} frames" if snapshot["recording"] else "idle"]
    if snapshot["homing"]:
        parts.append("homing")
    policy = snapshot.get("policy")
    if policy is not None:
        where = f" {policy['cable']} {policy['cable_number']}/{policy['cables']}" if policy["cable"] else ""
        attempt = f" try {policy['attempt']}/{policy['attempts']}" if policy["cable"] else ""
        parts.append(f"graph {policy['state']}{where}{attempt}")
    for side in ("left", "right"):
        arm = snapshot["arms"][side]
        x, y, z = arm["tool_m"]
        parts.append(f"{side} {'ON ' if arm['engaged'] else 'off'} tool [{x:+.3f} {y:+.3f} {z:+.3f}] m")
    return " | ".join(parts)


def _say(line: str) -> None:
    """Prints a line; once the terminal is gone (a lost SSH connection), writes it to LOG_PATH instead."""
    try:
        print(line, flush=True)
    except OSError:
        _log_to_file()
        print(line, flush=True)


def _log_to_file() -> None:
    """Sends the rest of the session's output to LOG_PATH: writing to a closed terminal would fail the shutdown."""
    if getattr(sys.stdout, "name", None) == str(LOG_PATH):
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(LOG_PATH, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - kept to the end
    print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}: the terminal is gone; the session's output continues here")


def _stop(signum: int, frame: Any) -> None:
    """SIGTERM: end the session the way Ctrl+C does."""
    raise KeyboardInterrupt


def _hang_up(signum: int, frame: Any) -> None:
    """SIGHUP, a lost SSH connection: end the session the way Ctrl+C does, writing the rest to LOG_PATH."""
    _log_to_file()
    print("the SSH connection was lost: ending the session safely")
    raise KeyboardInterrupt


def _state_names(station: config.StationConfig) -> list[str]:
    """One name per value of the recorded vectors, e.g. `right_joint_0` ... `right_gripper`."""
    names = []
    for side in layouts.SIDE_ORDER[station.family]:
        names += [f"{side}_joint_{i}" for i in range(layouts.JOINTS)] + [f"{side}_gripper"]
    return names


def choose_task(*texts: str | None) -> str:
    """The first of `texts` that is not blank, stripped; DEFAULT_TASK when all are."""
    for text in texts:
        if text and text.strip():
            return text.strip()
    return DEFAULT_TASK


def main(argv: list[str] | None = None) -> None:
    """The `dhk-collect` command."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="the station config, e.g. configs/yam.yaml")
    parser.add_argument(
        "--task",
        help="optional: a task text recorded with each episode (default: the config's task, else a neutral text; "
        "Dream labels every dataset from its overhead camera)",
    )
    parser.add_argument("--dataset-dir", help="the dataset folder episodes are added to (default: the config's)")
    parser.add_argument("--dry-run", action="store_true", help="read the arms and the Quest; command nothing")
    parser.add_argument("--sim", action="store_true", help="YAM: i2rt's simulated arms and blank camera frames")
    parser.add_argument("--quest-ip", help="the Quest's address for adb over Wi-Fi (default: USB)")
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=dashboard.DEFAULT_PORT,
        help=f"the browser dashboard's port on the station (0: none; default {dashboard.DEFAULT_PORT})",
    )
    policy = parser.add_argument_group("scripted policy (--policy graph: one arm unplugs USB cables, no Quest)")
    policy.add_argument("--policy", choices=("teleop", "graph"), default="teleop", help="what drives the arms")
    policy.add_argument("--poses", help="the taught poses (scripts/mine_unplug_rollouts.py writes them)")
    policy.add_argument(
        "--unplug-config", help=f"speeds, limits and vision thresholds (default: {UNPLUG_CONFIG} if present)"
    )
    policy.add_argument("--cables", help="which cables, numbered as in the poses file, e.g. 1 or 2,3 (default: all)")
    policy.add_argument(
        "--stop-after", choices=sorted(unplug_graph.STOP_AFTER), help="hold after this step of the first cable"
    )
    policy.add_argument("--speed-scale", type=float, default=1.0, help="slower runs: every speed times this (<= 1)")
    policy.add_argument("--record", action="store_true", help="record one episode per cable unplugged")
    policy.add_argument(
        "--view",
        help=f"the viewing pose and reference row the plugs are located against (default: {UNPLUG_VIEW} if present)",
    )
    policy.add_argument("--no-view", action="store_true", help="locate nothing: use the taught grasps as they are")
    policy.add_argument(
        "--capture-view",
        action="store_true",
        help="only go to the viewing pose and save what the cameras see there into the --view file, as the new "
        "reference (the strip where the taught grasps are right)",
    )
    args = parser.parse_args(argv)
    try:
        station = config.load(args.config)
    except config.ConfigError as error:
        sys.exit(f"config: {error}")
    if args.dataset_dir:
        station = dataclasses.replace(station, dataset_dir=pathlib.Path(args.dataset_dir).expanduser())
    spec = None
    if args.policy == "graph":
        if args.sim:
            sys.exit("--policy graph drives real arms (or a dry run of them): not with --sim")
        if not args.poses:
            sys.exit("--policy graph needs --poses (scripts/mine_unplug_rollouts.py writes them)")
        params_path = args.unplug_config or (UNPLUG_CONFIG if UNPLUG_CONFIG.is_file() else None)
        try:
            cables = tuple(int(n) for n in args.cables.split(",")) if args.cables else None
        except ValueError:
            sys.exit("--cables takes numbers, e.g. 1 or 2,3")
        view_path = None if args.no_view else args.view or (UNPLUG_VIEW if UNPLUG_VIEW.is_file() else None)
        if args.capture_view and view_path is None:
            sys.exit("--capture-view needs a --view file for the viewing pose (scripts/mine_unplug_rollouts.py)")
        spec = PolicySpec(
            pathlib.Path(args.poses),
            pathlib.Path(params_path) if params_path else None,
            cables,
            args.stop_after,
            args.speed_scale,
            args.record,
            pathlib.Path(view_path) if view_path else None,
            args.capture_view,
        )
    try:
        run(
            station,
            choose_task(args.task, station.task),
            dry_run=args.dry_run,
            sim=args.sim,
            quest_ip=args.quest_ip,
            dashboard_port=args.dashboard_port,
            policy=spec,
        )
    except (station_lock.StationBusyError, quest.QuestError, arms.ArmsError) as error:
        sys.exit(str(error))
    except unplug_motion.PlanError as error:
        sys.exit(f"graph policy: {error}")
    except config.ConfigError as error:  # what the arm model needs, e.g. a Metal or Maker arm's tool center point
        sys.exit(f"config: {error}")
    except dataset.DatasetError as error:  # a --dataset-dir recorded another way (rate, cameras, version)
        sys.exit(f"dataset: {error}")


if __name__ == "__main__":
    main()
