"""The unplugging policy: a graph of states, stepped once per control tick, for one arm.

With a view reference (perception.py), a run first finds the plugs: OVERHEAD checks that the overhead camera sees the
strip's plugs, TO_VIEW takes the arm to the viewing pose (from the demonstrations: above and behind the strip's USB
end, the wrist camera looking down at the plugs), PROBE moves the tool a centimetre each way to measure how the image
shifts with it, and LOCATE places the row of plugs against the reference row, turning its offset in the image into
the strip's offset on the table. Every taught grasp is moved by that offset: the grasps are kept relative to the
plugs, not to the table. Before each later cable the arm returns to the viewing pose and locates again.

For each cable, in order: over the strip at travel height, above the plug, down onto it with the fingers part open,
close on the housing at its base (the gripper's force hold stops on it), pull straight up, look for the plug's
metal tip under the fingers (verify.py), carry the cable to the drop spot, let go, and on to the next. A failed
attempt (missed the housing, slipped, stuck, no tip seen) opens, rises above the plug and tries again a little
deeper, up to `attempts` tries; then the cable is skipped (or the run aborts). After the last cable the arm goes back
over the strip to the ready pose, then to where it started.

```
WAIT_START -> OVERHEAD -> TO_VIEW -> PROBE -> LOCATE -> SELECT ...   (with a view reference; RETREAT -> TO_VIEW)
WAIT_START -> TO_READY -> SELECT -> TRANSIT_ABOVE -> DESCEND -> GRASP -> PULL -> VERIFY
                            ^                          ^        |        |        |  ok
                            |                          |        v        v        v
                            |                       RECOVER_UP <- FAIL_ATTEMPT    TRANSIT_DROP -> PLACE -> RELEASE
                            |                                                                              |
                            +------------------------------- RETREAT <-------------------------------------+
SELECT (no cable left) -> HOME -> PARK -> DONE;   anything wrong (a timeout, the arm blocked in transit) -> ABORT
```

Every path is a JointPath (motion.py) planned before the arm moves; the whole nominal run is planned when the graph
is made, so a pose out of reach stops the session before any device opens. Each tick the command advances one sample,
and only while the arm keeps up: a joint more than `max_track_err_rad` behind is a blocked arm (a failed attempt on
the way down or up, an abort elsewhere). ABORT holds the arm where it is, grip unchanged, until Ctrl+C.

In a dry run (`ghost`) the arms are limp and read only: the graph runs against its own command, as if the arm
followed it exactly, with the gripper reading each cable's held grip when it closes, so the whole run and its timing
can be watched on the dashboard without anything moving. A ghost run judges no grasps or tips, and moves no grasp:
it reports what the wrist camera sees from wherever the limp arm is held.

With `capture_view`, the run only goes to the viewing pose, saves what the wrist and overhead cameras see there as
the new reference (the strip must sit where the taught grasps are right), and goes back.

@see UnplugGraph
"""

import collections
import contextlib
import dataclasses
import math
import pathlib
import threading
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from dream_hackathon import cameras, ik
from dream_hackathon.unplug import motion, perception
from dream_hackathon.unplug.params import UnplugParams
from dream_hackathon.unplug.poses import CablePose, UnplugPoses
from dream_hackathon.unplug.verify import TipVerifier

## What `stop_after` takes: after that step succeeds the arm holds there (HOLD) until Ctrl+C.
STOP_AFTER = {
    "ready": "TO_READY",
    "locate": "LOCATE",
    "above": "TRANSIT_ABOVE",
    "descend": "DESCEND",
    "grasp": "GRASP",
    "pull": "PULL",
    "verify": "VERIFY",
    "place": "PLACE",
}
## The states in which the gripper holds a cable (a ghost run's gripper then reads its held grip).
_HOLDING = ("GRASP", "PULL", "VERIFY", "TRANSIT_DROP", "PLACE")
## How far the shutdown retreat lifts the tool, straight up, before the collector's homing.
RETREAT_UP_M = 0.05


## The probe moves, from the viewing pose: none (the base), +x, back, +y, back; frames are judged at the first three.
_PROBES = ((0.0, 0.0, True), (1.0, 0.0, True), (0.0, 0.0, False), (0.0, 1.0, True), (0.0, 0.0, False))


@dataclasses.dataclass
class _Cable:
    pose: CablePose
    taught: npt.NDArray[np.float64]  # [4, 4] tool pose closed on the housing, as taught
    grasp: npt.NDArray[np.float64]  # the same, moved by the located strip offset
    above: npt.NDArray[np.float64]
    result: str = "pending"  # pending, ok, failed
    number: int = 0  # in the poses file, from 1


class UnplugGraph:
    """One arm's unplugging run, stepped by act() once per control tick.

    start() and stop() may be called from another thread (the dashboard's, the terminal's); the rest only from the
    control loop's.

    Attributes:
    - side: The arm it drives.
    - state: The current state's name.
    - attempt: The current cable's try, from 1.
    """

    def __init__(
        self,
        poses: UnplugPoses,
        params: UnplugParams,
        solver: ik.ArmIK,
        workspace_low_m: npt.NDArray[np.float64],
        workspace_high_m: npt.NDArray[np.float64],
        dt_s: float,
        *,
        verifier: TipVerifier | None = None,
        log: Callable[[str], None] = print,
        cables: Sequence[int] | None = None,
        stop_after: str | None = None,
        view: perception.ViewReference | None = None,
        capture_view: pathlib.Path | None = None,
    ) -> None:
        """Plans the whole nominal run from the viewing pose (or, without a view reference, the ready pose).

        @param cables 1-based numbers of the cables to unplug, in the poses file's order; None for all.
        @param stop_after A key of STOP_AFTER, or None.
        @param view Where to look from and what was seen there; None goes by the taught grasps alone.
        @param capture_view Only capture a new reference into this file (needs `view` for its pose).
        @throws PlanError A path out of reach or out of the workspace.
        @throws ValueError A cable number or `stop_after` that does not exist, or a capture without a view.
        """
        if capture_view is not None and view is None:
            raise ValueError("capturing a view needs a view reference for its pose (--view)")
        self.side = poses.side
        self.vision = params.vision
        self._view = view
        self._capture = capture_view
        self._perc = params.perception
        self._frame_of: Callable[[str], npt.NDArray[np.uint8]] | None = None
        self._debug_dir: pathlib.Path | None = None
        self._jacobian: npt.NDArray[np.float64] | None = None  # image shift (px) per tool move (m), [2, 2]
        self._offset_m = np.zeros(2)
        self._located = ""
        self._probe = 0
        self._samples: list[npt.NDArray[np.float64]] = []
        self._probe_rows: list[npt.NDArray[np.float64]] = []
        self._tries = 0
        self._bar_sets: list[list[perception.Bar]] = []
        self._last_frame: npt.NDArray[np.uint8] | None = None
        self._last_fit: perception.RowFit | None = None
        self._locates = 0
        self._poses = poses
        self._p = params.motion
        self._solver = solver
        self._low, self._high = np.asarray(workspace_low_m), np.asarray(workspace_high_m)
        self._dt = dt_s
        self._verifier = verifier
        self._log = log
        if stop_after is not None and stop_after not in STOP_AFTER:
            raise ValueError(f"stop_after must be one of {sorted(STOP_AFTER)}")
        self._stop_after = STOP_AFTER.get(stop_after or "")
        chosen = list(range(1, len(poses.cables) + 1)) if cables is None else list(cables)
        if not chosen or any(not 1 <= n <= len(poses.cables) for n in chosen):
            raise ValueError(f"cables are numbered 1 to {len(poses.cables)}")
        self._cables = []
        for n in chosen:
            pose = poses.cables[n - 1]
            grasp = solver.fk(pose.grasp_joints)
            self._cables.append(
                _Cable(pose, grasp, grasp.copy(), motion.lifted(grasp, self._p.approach_up_m), number=n)
            )
        self._taught_y = [float(solver.fk(c.grasp_joints)[1, 3]) for c in poses.cables]
        self._drop = solver.fk(poses.drop_joints)
        self._drop_above = motion.lifted(self._drop, self._p.drop_up_m)
        self._ready = solver.fk(poses.ready_joints)
        self._travel_z = max(self._ready[2, 3], max(c.above[2, 3] for c in self._cables) + self._p.travel_clearance_m)
        self._cache: dict[str, tuple[npt.NDArray[np.float64], motion.JointPath]] = {}
        self._start_flag = threading.Event()
        self._stop_flag = threading.Event()
        self._stop_reason = ""
        self._events: list[tuple[str, dict[str, Any]]] = []
        self.state = "WAIT_START"
        self.attempt = 1
        self._index = -1  # into self._cables
        self._q = poses.ready_joints.copy()
        self._grip = 1.0
        self._home = poses.ready_joints.copy()
        self._ghost = False
        self._path: motion.JointPath | None = None
        self._k = 0
        self._entered_s = 0.0
        self._settled_since: float | None = None
        self._deadline_s = math.inf
        self._grips: collections.deque[tuple[float, float]] = collections.deque()
        self._z0 = 0.0
        self._dz = 0.0
        self._votes: list[bool] = []
        self._reason = ""
        self._done_logged = False
        self.report = self._plan_nominal()

    # ----- the loop's interface

    def connect(
        self,
        *,
        log: Callable[[str], None] | None = None,
        verifier: TipVerifier | None = None,
        frame_of: Callable[[str], npt.NDArray[np.uint8]] | None = None,
        debug_dir: pathlib.Path | None = None,
    ) -> None:
        """Sets where events are logged, the tip check and the cameras, once the session's devices exist.

        @param frame_of A camera role's newest frame (raising cameras.CameraError when there is none).
        @param debug_dir Where the locating pictures are saved.
        """
        if log is not None:
            self._log = log
        if verifier is not None:
            self._verifier = verifier
        if frame_of is not None:
            self._frame_of = frame_of
        if debug_dir is not None:
            self._debug_dir = debug_dir

    def attach(
        self, joints: npt.NDArray[np.float64], grip: float, home_joints: npt.NDArray[np.float64], *, ghost: bool
    ) -> None:
        """Starts the command where the arm is; `home_joints` is where PARK takes it at the end."""
        self._q = np.asarray(joints, dtype=np.float64).copy()
        self._grip = float(grip)
        self._home = np.asarray(home_joints, dtype=np.float64).copy()
        self._ghost = ghost

    def start(self) -> None:
        """Starts the run (from WAIT_START); thread-safe."""
        self._start_flag.set()

    def stop(self, reason: str) -> None:
        """Aborts the run: the arm holds where it is; thread-safe."""
        self._stop_reason = reason
        self._stop_flag.set()

    @property
    def done(self) -> bool:
        return self.state in ("DONE", "ABORT", "HOLD")

    @property
    def started(self) -> bool:
        return self._start_flag.is_set()

    def events(self) -> list[tuple[str, dict[str, Any]]]:
        """The episode events since the last call: (`episode_start` | `episode_end`, details)."""
        events, self._events = self._events, []
        return events

    def act(
        self, now_s: float, measured: npt.NDArray[np.float64], measured_grip: float
    ) -> tuple[npt.NDArray[np.float64], float, bool]:
        """One tick: the joints and grip to command for this arm, and whether the run has ended."""
        if self._ghost:
            measured = self._q
            measured_grip = self._ghost_grip()
        if self._stop_flag.is_set() and not self.done:
            self._abort(f"stopped: {self._stop_reason}", measured)
        handler = getattr(self, f"_tick_{self.state.lower()}")
        handler(now_s, np.asarray(measured, dtype=np.float64), float(measured_grip))
        if self.done and not self._done_logged:
            self._done_logged = True
            self._log(f"graph: {self.state.lower()}: {self.summary()}")
        return self._q.copy(), self._grip, self.done

    def summary(self) -> str:
        return ", ".join(f"{c.pose.name} {c.result}" for c in self._cables)

    def status(self) -> dict[str, Any]:
        """The run's state for the dashboard and the status line; JSON-ready."""
        cable = self._cables[self._index] if 0 <= self._index < len(self._cables) else None
        target = self._solver.fk(self._path.end)[:3, 3] if self._path is not None else None
        return {
            "state": self.state,
            "cable": cable.pose.name if cable else None,
            "cable_number": self._index + 1 if cable else 0,
            "cables": len(self._cables),
            "attempt": self.attempt,
            "attempts": self._p.attempts,
            "results": {c.pose.name: c.result for c in self._cables},
            "reason": self._reason,
            "target_m": None if target is None else [round(float(v), 4) for v in target],
            "ghost": self._ghost,
            "started": self.started,
            "located": self._located,
        }

    def retreat_path(self, measured: npt.NDArray[np.float64]) -> motion.JointPath | None:
        """For the shutdown: the tool straight up RETREAT_UP_M from where it is (gripper open), or None if unreachable.

        Run before the collector's joint-space homing, so that never sweeps the fingers through the strip.
        """
        try:
            start = self._solver.fk(measured)
            return motion.plan_line(
                self._solver,
                measured,
                motion.lifted(start, RETREAT_UP_M),
                self._p.transit_speed_m_s,
                self._dt,
                label="shutdown retreat",
            )
        except motion.PlanError as error:
            self._log(f"graph: no retreat: {error}")
            return None

    # ----- planning

    def _plan_nominal(self) -> str:
        """Plans every nominal path from the ready pose, cable after cable, keeping them for the run; the report."""
        lines = [f"graph plan for the {self.side} arm (travel height {self._travel_z:.3f} m):"]
        q = self._poses.ready_joints.copy() if self._view is None else self._view.joints.copy()
        if self._view is not None:
            view = self._view_pose()
            lines.append(f"  viewing pose at {np.round(view[:3, 3], 3).tolist()} m; locating moves grasps")
            for dx, dy, _ in _PROBES[1:2] + _PROBES[3:4]:
                self._plan(f"probe:{dx}:{dy}", q, self._build_probe(dx, dy))
        for i, cable in enumerate(self._cables):
            self._index = i
            for key, build in (
                ("above", self._build_above),
                ("descend", self._build_descend),
                ("pull", self._build_pull),
                ("drop", self._build_drop),
                ("place", self._build_place),
                ("retreat", self._build_retreat),
            ):
                path = self._plan(f"{cable.pose.name}:{key}", q, build)
                lines.append(self._describe(f"{cable.pose.name} {key}", path))
                q = path.end
            if self._view is not None and self._p.relocate_each_cable:
                path = self._plan(f"{cable.pose.name}:view", q, self._build_view)
                lines.append(self._describe(f"{cable.pose.name} to view", path))
                q = path.end
        path = self._plan("home", q, self._build_home)
        lines.append(self._describe("back to ready", path))
        self._index = -1
        return "\n".join(lines)

    def _describe(self, name: str, path: motion.JointPath) -> str:
        xyz = self._solver.fk(path.end)[:3, 3]
        return (
            f"  {name:<18} {len(path.points):4d} ticks {path.duration_s(self._dt):5.1f} s  to "
            f"[{xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}] m  worst {path.max_pos_err_m * 1000:.2f} mm "
            f"{path.max_rot_err_deg:.2f} deg"
        )

    def _plan(
        self, key: str, start: npt.NDArray[np.float64], build: Callable[[npt.NDArray[np.float64]], motion.JointPath]
    ) -> motion.JointPath:
        cached = self._cache.get(key)
        if cached is not None and np.allclose(cached[0], start, atol=1e-4):
            return cached[1]
        path = build(start)
        motion.check_workspace(self._solver, path, self._low, self._high)
        self._cache[key] = (np.asarray(start).copy(), path)
        return path

    def _cable(self) -> _Cable:
        return self._cables[self._index]

    def _build_above(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        return motion.plan_travel(
            self._solver,
            q,
            self._cable().above,
            self._travel_z,
            p.transit_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label=f"to above {self._cable().pose.name}",
        )

    def _build_descend(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        return motion.plan_line(
            self._solver,
            q,
            motion.lifted(self._cable().grasp, self._dz),
            p.descend_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label=f"down onto {self._cable().pose.name}",
        )

    def _build_pull(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        return motion.plan_line(
            self._solver,
            q,
            motion.lifted(self._solver.fk(q), p.pull_m),
            p.pull_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label=f"pull {self._cable().pose.name}",
        )

    def _build_drop(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        return motion.plan_travel(
            self._solver,
            q,
            self._drop_above,
            self._travel_z,
            p.transit_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label="to the drop spot",
        )

    def _build_place(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        return motion.plan_line(
            self._solver,
            q,
            self._drop,
            p.descend_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label="down to let go",
        )

    def _build_retreat(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        return motion.plan_line(
            self._solver,
            q,
            self._drop_above,
            p.transit_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label="up from the drop spot",
        )

    def _build_recover(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        return motion.plan_line(
            self._solver,
            q,
            self._cable().above,
            p.transit_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label=f"back above {self._cable().pose.name}",
        )

    def _build_home(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        p = self._p
        to_ready = motion.plan_travel(
            self._solver,
            q,
            self._ready,
            self._travel_z,
            p.transit_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label="back to ready",
        )
        # The line ends at the ready pose; finish on the ready joints themselves (a few hundredths of a degree).
        settle = motion.plan_joint_ramp(to_ready.end, self._poses.ready_joints, p.joint_speed_rad_s, self._dt)
        return to_ready.then(settle, "back to ready")

    def _view_pose(self) -> npt.NDArray[np.float64]:
        assert self._view is not None
        return self._solver.fk(self._view.joints)

    def _build_view(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        """Over the cables to the viewing pose, ending on its joints exactly (the reference was seen from them)."""
        assert self._view is not None
        p = self._p
        to_view = motion.plan_travel(
            self._solver,
            q,
            self._view_pose(),
            self._travel_z,
            p.transit_speed_m_s,
            self._dt,
            angular_speed_rad_s=p.angular_speed_rad_s,
            label="to the viewing pose",
        )
        settle = motion.plan_joint_ramp(to_view.end, self._view.joints, p.joint_speed_rad_s, self._dt)
        return to_view.then(settle, "to the viewing pose")

    def _build_probe(self, dx: float, dy: float) -> Callable[[npt.NDArray[np.float64]], motion.JointPath]:
        def build(q: npt.NDArray[np.float64]) -> motion.JointPath:
            goal = self._view_pose()
            goal[0, 3] += dx * self._p.probe_m
            goal[1, 3] += dy * self._p.probe_m
            path = motion.plan_line(
                self._solver, q, goal, self._p.descend_speed_m_s, self._dt, label=f"probe {dx:+.0f} {dy:+.0f}"
            )
            if dx == 0.0 and dy == 0.0:
                assert self._view is not None  # back on the viewing joints exactly
                path = path.then(motion.plan_joint_ramp(path.end, self._view.joints, 0.2, self._dt), path.label)
            return path

        return build

    # ----- state machinery

    def _hold_if_asked(self, now_s: float) -> bool:
        """At a step's success, before its next move: holds there if `stop_after` names this step."""
        if self._stop_after is None or self.state != self._stop_after:
            return False
        self._log(f"graph: stopping after {self.state.lower()} as asked: holding (Ctrl+C ends the session)")
        self._go("HOLD", now_s)
        return True

    def _go(self, state: str, now_s: float, path: motion.JointPath | None = None, timeout_s: float = math.inf) -> None:
        self.state = state
        self._entered_s = now_s
        self._path = path
        self._k = 0
        self._settled_since = None
        self._grips.clear()
        self._deadline_s = now_s + (
            path.duration_s(self._dt) + self._p.path_slack_s if path is not None else timeout_s
        )

    def _move(self, state: str, now_s: float, key: str, build: Callable[[npt.NDArray[np.float64]], motion.JointPath]):
        """Enters a moving state with its path; a path that cannot be planned aborts."""
        try:
            path = self._plan(key, self._q, build)
        except motion.PlanError as error:
            self._abort(f"cannot plan {key}: {error}", self._q)
            return
        self._go(state, now_s, path)

    def _follow(self, now_s: float, measured: npt.NDArray[np.float64]) -> str:
        """Advances the command one sample: `blocked`, `moving`, `arrived` or `timeout`."""
        assert self._path is not None
        if float(np.max(np.abs(self._q - measured))) > self._p.max_track_err_rad:
            return "blocked"
        if self._k < len(self._path.points):
            self._q = self._path.points[self._k].copy()
            self._k += 1
        if now_s > self._deadline_s:
            return "timeout"
        if self._k < len(self._path.points):
            return "moving"
        if float(np.max(np.abs(self._path.end - measured))) <= self._p.arrive_tol_rad:
            if self._settled_since is None:
                self._settled_since = now_s
            if now_s - self._settled_since >= self._p.settle_s:
                return "arrived"
        else:
            self._settled_since = None
        return "moving"

    def _grip_settled(self, now_s: float, grip: float) -> bool:
        self._grips.append((now_s, grip))
        while self._grips and now_s - self._grips[0][0] > self._p.grip_steady_s:
            self._grips.popleft()
        values = [g for _, g in self._grips]
        # Most of the window, not all of it: at a tick rate that does not divide it, the window holds one read fewer.
        spans = self._grips[-1][0] - self._grips[0][0] >= self._p.grip_steady_s * 0.75
        return spans and max(values) - min(values) < self._p.grip_steady

    def _held(self, grip: float) -> bool:
        cable = self._cable().pose
        return grip > self._p.empty_grip_max and grip >= cable.held_grip - self._p.held_margin

    def _ghost_grip(self) -> float:
        if self.state in _HOLDING and 0 <= self._index < len(self._cables):
            return max(self._grip, self._cable().pose.held_grip)
        return self._grip

    def _abort(self, reason: str, measured: npt.NDArray[np.float64]) -> None:
        self._reason = reason
        self._log(f"graph: ABORT: {reason}; holding where the arm is (Ctrl+C ends the session)")
        self._q = np.asarray(measured, dtype=np.float64).copy()
        if 0 <= self._index < len(self._cables) and self._cable().result == "pending":
            self._events.append(("episode_end", {"cable": self._cable().pose.name, "success": False}))
        self._stop_after = None
        self._go("ABORT", self._entered_s)

    def _fail(self, now_s: float, reason: str) -> None:
        self._reason = reason
        self._log(f"graph: {self._cable().pose.name} try {self.attempt}/{self._p.attempts} failed: {reason}")
        self._grip = self._cable().pose.approach_grip
        self._go("FAIL_ATTEMPT", now_s, timeout_s=self._p.fail_dwell_s)

    # ----- the states

    def _tick_wait_start(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        if self._start_flag.is_set():
            self._log("graph: started")
            if self._view is not None:
                self._go("OVERHEAD", now_s, timeout_s=2.0)
                return
            path = motion.plan_joint_ramp(self._q, self._poses.ready_joints, self._p.joint_speed_rad_s, self._dt)
            self._go("TO_READY", now_s, dataclasses.replace(path, label="to the ready pose"))

    # ----- finding the plugs

    def _grab(self, role: str) -> npt.NDArray[np.uint8] | None:
        if self._frame_of is None:
            return None
        try:
            return self._frame_of(role)
        except cameras.CameraError:
            return None

    def _save_debug(self, tag: str, image: npt.NDArray[np.uint8]) -> None:
        if self._debug_dir is None:
            return
        with contextlib.suppress(OSError):
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self._debug_dir / f"{tag}.jpg"), image)

    def _wrist_role(self) -> str:
        return {"left": "wrist_l", "right": "wrist_r"}[self.side]

    def _tick_overhead(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        """Step 1: the overhead camera finds the strip's plugs (a coarse check: where, and whether it moved)."""
        assert self._view is not None
        frame = self._grab("overhead")
        if frame is None and now_s <= self._deadline_s:
            return
        if frame is None:
            self._log("graph: overhead: no frame; going on to the viewing pose")
        else:
            seen = perception.find_strip_overhead(frame)
            if seen is None:
                self._log("graph: overhead: the strip's plugs are not in view; going to look from the wrist anyway")
            elif self._view.overhead_px is not None:
                du, dv = seen - self._view.overhead_px
                self._log(
                    f"graph: overhead: plugs at {np.round(seen).tolist()} px, {du:+.0f} {dv:+.0f} px from the "
                    "reference"
                )
            else:
                self._log(f"graph: overhead: plugs at {np.round(seen).tolist()} px")
        self._move("TO_VIEW", now_s, "view:first", self._build_first_view)

    def _build_first_view(self, q: npt.NDArray[np.float64]) -> motion.JointPath:
        """From the start (the arm folded at rest): every joint straight to the viewing joints."""
        assert self._view is not None
        path = motion.plan_joint_ramp(q, self._view.joints, self._p.joint_speed_rad_s, self._dt)
        return dataclasses.replace(path, label="to the viewing pose")

    def _tick_to_view(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} on the way to the viewing pose", measured)
        elif result == "arrived":
            self._start_measuring()
            if self._jacobian is None and self._capture is None and not self._ghost:
                self._probe = 0
                self._probe_rows = []
                self._go("PROBE", now_s, timeout_s=30.0)
            else:
                self._go("LOCATE", now_s, timeout_s=self._locate_timeout())

    def _locate_timeout(self) -> float:
        return self._p.locate_settle_s + self._p.locate_frames * 0.5 + 3.0

    def _start_measuring(self) -> None:
        self._samples = []
        self._tries = 0
        self._bar_sets = []
        self._last_frame = None
        self._last_fit = None

    def _measure_row(self, now_s: float) -> str:
        """One frame towards placing the row: `pending`, `done` (self._samples full) or `nothing` (no camera)."""
        assert self._view is not None
        if now_s - self._entered_s < self._p.locate_settle_s:
            return "pending"
        frame = self._grab(self._wrist_role())
        if frame is None:
            return "nothing" if now_s > self._deadline_s else "pending"
        self._tries += 1
        bars = perception.detect_bars(frame, self._perc)
        fit = perception.fit_row(bars, self._view.slots_px, self._perc, self._occupied())
        self._bar_sets.append(bars)
        self._last_frame = frame
        self._last_fit = fit
        if fit is not None:
            self._samples.append(fit.offset_px)
        return "done" if self._tries >= self._p.locate_frames else "pending"

    def _occupied(self) -> list[bool] | None:
        """Per reference slot (by image column), whether its plug may still stand: not one this run took out.

        Which cable stands in which slot follows from the taught grasps' order across the strip and the way the
        probe moves showed the image to turn: without the probe (or with a reference of another size), None.
        """
        assert self._view is not None
        if self._jacobian is None or len(self._view.slots_px) != len(self._poses.cables):
            return None
        order = list(np.argsort(self._taught_y))  # cables by their side-to-side place, smallest y first
        if self._jacobian[0, 1] > 0:  # the image moves right as the tool moves left: the farthest left is leftmost
            order.reverse()
        out = {cable.number: cable.result != "ok" for cable in self._cables}
        return [out.get(int(index) + 1, True) for index in order]

    def _agreed_offset(self) -> npt.NDArray[np.float64] | None:
        """The row's offset most frames agree on, or None."""
        if len(self._samples) < self._p.locate_agree:
            return None
        samples = np.array(self._samples)
        median = np.median(samples, axis=0)
        close = np.linalg.norm(samples - median, axis=1) <= self._p.locate_agree_px
        if int(close.sum()) < self._p.locate_agree:
            return None
        return samples[close].mean(axis=0)

    def _tick_probe(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        """Measures how the row moves in the image when the tool moves: the image Jacobian, no calibration needed."""
        if self._path is not None:
            result = self._follow(now_s, measured)
            if result in ("blocked", "timeout"):
                self._abort(f"{result} during a probe move", measured)
                return
            if result != "arrived":
                return
            self._path, self._entered_s = None, now_s
            self._start_measuring()
        dx, dy, judge = _PROBES[self._probe]
        if judge:
            status = self._measure_row(now_s)
            if status == "pending":
                return
            offset = self._agreed_offset() if status == "done" else None
            if offset is None:
                self._locate_failed(now_s, measured, f"the row could not be placed at probe {self._probe}")
                return
            self._probe_rows.append(offset)
        self._probe += 1
        if self._probe < len(_PROBES):
            ndx, ndy, _ = _PROBES[self._probe]
            try:
                path = self._plan(f"probe:{ndx}:{ndy}", self._q, self._build_probe(ndx, ndy))
            except motion.PlanError as error:
                self._abort(f"cannot plan a probe move: {error}", measured)
                return
            self._path, self._k, self._settled_since = path, 0, None
            self._deadline_s = now_s + path.duration_s(self._dt) + self._p.path_slack_s
            return
        base, moved_x, moved_y = self._probe_rows
        jacobian = np.column_stack([(moved_x - base) / self._p.probe_m, (moved_y - base) / self._p.probe_m])
        smallest = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
        if smallest < 100.0:  # under a pixel per centimetre one way: the image cannot place the strip that way
            self._locate_failed(now_s, measured, f"the probe moves barely moved the image ({smallest:.0f} px/m)")
            return
        self._jacobian = jacobian
        self._log(f"graph: probe: the image moves {np.round(jacobian / 1000, 2).tolist()} px per mm of tool motion")
        self._start_measuring()
        self._go("LOCATE", now_s, timeout_s=self._locate_timeout())

    def _tick_locate(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        """Places the row of plugs and moves every pending grasp by the strip's offset."""
        assert self._view is not None
        status = self._measure_row(now_s)
        if status == "pending":
            return
        if self._capture is not None:
            self._finish_capture(now_s, measured)
            return
        offset = self._agreed_offset() if status == "done" else None
        self._locates += 1
        if self._last_frame is not None:
            bars = self._bar_sets[-1] if self._bar_sets else []
            picture = perception.annotate_row(self._last_frame, bars, self._last_fit, self._view.slots_px)
            self._save_debug(f"locate_{self._locates}", picture)
        if offset is None:
            self._locate_failed(
                now_s,
                measured,
                f"the row could not be placed ({len(self._samples)} of {self._tries} frames placed it, "
                f"{self._p.locate_agree} must agree)",
            )
            return
        fit = self._last_fit
        plugs = "".join("1" if s else "0" for s in fit.seen) if fit is not None else "?"
        if self._ghost or self._jacobian is None:
            self._located = f"row {offset[0]:+.1f} {offset[1]:+.1f} px (dry run: grasps not moved), plugs {plugs}"
            self._log(f"graph: locate: {self._located}")
        else:
            strip = -np.linalg.solve(self._jacobian, offset)
            if float(np.hypot(*strip)) > self._p.max_correction_m:
                self._abort(
                    f"the strip is {np.hypot(*strip) * 1000:.0f} mm from where the grasps were taught (max "
                    f"{self._p.max_correction_m * 1000:.0f} mm): put it back, or teach again",
                    measured,
                )
                return
            self._apply_offset(strip)
            self._located = f"strip {strip[0] * 1000:+.1f} {strip[1] * 1000:+.1f} mm from taught, plugs {plugs}"
            self._log(f"graph: locate: row {offset[0]:+.1f} {offset[1]:+.1f} px: {self._located}")
        if not self._hold_if_asked(now_s):
            self._go("SELECT", now_s)

    def _locate_failed(self, now_s: float, measured: npt.NDArray[np.float64], why: str) -> None:
        if self._p.require_locate and not self._ghost:
            self._abort(f"locating the plugs failed: {why}", measured)
            return
        self._log(f"graph: locate: {why}; going on with the grasps as they are")
        self._located = f"not located ({why})"
        self._go("SELECT", now_s)

    def _apply_offset(self, strip_m: npt.NDArray[np.float64]) -> None:
        """Moves every pending cable's grasp by the strip's offset on the table, and forgets their old paths."""
        self._offset_m = np.asarray(strip_m, dtype=np.float64)
        for cable in self._cables:
            if cable.result != "pending":
                continue
            cable.grasp = cable.taught.copy()
            cable.grasp[0, 3] += self._offset_m[0]
            cable.grasp[1, 3] += self._offset_m[1]
            cable.above = motion.lifted(cable.grasp, self._p.approach_up_m)
            for key in ("above", "descend", "recover"):
                self._cache.pop(f"{cable.pose.name}:{key}", None)

    def _finish_capture(self, now_s: float, measured: npt.NDArray[np.float64]) -> None:
        """Saves what the wrist camera sees from the viewing pose as the new reference row, then goes home."""
        assert self._view is not None and self._capture is not None
        ports = len(self._poses.cables)
        self._bar_sets = [perception.pick_row(bars, ports) for bars in self._bar_sets]
        most = max((len(bars) for bars in self._bar_sets), default=0)
        if most < self._perc.min_bars:
            self._abort(f"capture: the wrist camera sees {most} plugs from the viewing pose", measured)
            return
        rows = np.array([[[b.x, b.foot] for b in bars] for bars in self._bar_sets if len(bars) == most])
        slots = np.median(rows, axis=0)
        overhead = self._grab("overhead")
        reference = perception.ViewReference(
            self._view.joints,
            slots,
            perception.find_strip_overhead(overhead) if overhead is not None else None,
            "captured on the station by dhk-collect --policy graph --capture-view",
        )
        perception.save_view(
            self._capture,
            reference,
            "The viewing pose and what the wrist camera saw there (unplug/perception.py). Captured with the strip\n"
            "where the taught grasps are right.",
        )
        if self._last_frame is not None:
            self._save_debug("capture", perception.annotate_row(self._last_frame, self._bar_sets[-1], None, slots))
        warning = "" if most == ports else f" (not {ports}: check the picture)"
        self._log(f"graph: capture: saved {most} plugs{warning} to {self._capture}; going back")
        self._index = -1
        for cable in self._cables:
            cable.result = "captured"
        self._move("HOME", now_s, "home", self._build_home)

    def _next_cable(self, now_s: float) -> None:
        """After a cable: back to the viewing pose to locate again, or straight on to the next."""
        pending = any(c.result == "pending" for c in self._cables)
        if self._view is not None and self._p.relocate_each_cable and pending:
            name = self._cable().pose.name if 0 <= self._index < len(self._cables) else "start"
            self._move("TO_VIEW", now_s, f"{name}:view", self._build_view)
        else:
            self._go("SELECT", now_s)

    def _tick_to_ready(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} on the way to the ready pose", measured)
        elif result == "arrived" and not self._hold_if_asked(now_s):
            self._go("SELECT", now_s)

    def _tick_select(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        pending = [i for i, c in enumerate(self._cables) if c.result == "pending"]
        if not pending:
            self._index = -1
            self._move("HOME", now_s, "home", self._build_home)
            return
        self._index = pending[0]
        self.attempt, self._dz, self._reason = 1, 0.0, ""
        cable = self._cable()
        self._log(f"graph: {cable.pose.name} ({cable.pose.tip}): going above it")
        self._events.append(("episode_start", {"cable": cable.pose.name}))
        self._grip = cable.pose.approach_grip
        self._move("TRANSIT_ABOVE", now_s, f"{cable.pose.name}:above", self._build_above)

    def _tick_transit_above(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} on the way above {self._cable().pose.name}", measured)
        elif result == "arrived":
            if grip < self._cable().pose.approach_grip - 0.1:
                return  # still opening; a gripper that never opens ends in the timeout above
            if not self._hold_if_asked(now_s):
                self._move("DESCEND", now_s, f"{self._cable().pose.name}:descend", self._build_descend)

    def _tick_descend(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result == "blocked":
            self._fail(now_s, "blocked on the way down (the fingers hit something)")
        elif result == "timeout":
            self._fail(now_s, "did not reach the grasp pose in time")
        elif result == "arrived" and not self._hold_if_asked(now_s):
            self._grip = self._p.grasp_grip
            if self._verifier is not None and not self._ghost:
                self._verifier.take_reference(f"{self._cable().pose.name}_try{self.attempt}")
            self._go("GRASP", now_s, timeout_s=self._p.grasp_dwell_s + 2.0)

    def _tick_grasp(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        settled = self._grip_settled(now_s, grip)
        if now_s - self._entered_s >= self._p.grasp_dwell_s and settled:
            if grip <= self._p.empty_grip_max:
                self._fail(now_s, f"closed on nothing (grip {grip:.3f})")
            elif not self._held(grip):
                self._fail(now_s, f"the grip {grip:.3f} is not the housing's {self._cable().pose.held_grip:.3f}")
            elif not self._hold_if_asked(now_s):
                self._z0 = float(self._solver.fk(measured)[2, 3])
                self._move("PULL", now_s, f"{self._cable().pose.name}:pull", self._build_pull)
        elif now_s > self._deadline_s:
            self._fail(now_s, f"the gripper did not settle (grip {grip:.3f})")

    def _tick_pull(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        cable = self._cable()
        if not self._held(grip):
            self._fail(now_s, f"slipped (grip {grip:.3f})")
            return
        result = self._follow(now_s, measured)
        if result == "blocked":
            self._fail(now_s, "stuck: the arm cannot pull it up")
            return
        commanded = float(self._solver.fk(self._q)[2, 3]) - self._z0
        risen = float(self._solver.fk(measured)[2, 3]) - self._z0
        if commanded >= self._p.pull_check_at_m and risen < self._p.pull_check_min_rise_m:
            self._fail(now_s, f"stuck: risen {risen * 1000:.0f} mm of {commanded * 1000:.0f} mm")
        elif result == "timeout":
            self._fail(now_s, "the pull did not finish in time")
        elif result == "arrived" and not self._hold_if_asked(now_s):
            self._votes = []
            self._go("VERIFY", now_s, timeout_s=self._p.verify_delay_s + 3.0)
            self._log(f"graph: {cable.pose.name} pulled up {risen * 1000:.0f} mm: looking for its tip")

    def _tick_verify(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        p, cable = self._p, self._cable()
        if not self._held(grip):
            self._fail(now_s, f"dropped it (grip {grip:.3f})")
            return
        if now_s - self._entered_s < p.verify_delay_s:
            return
        risen = float(self._solver.fk(measured)[2, 3]) - self._z0
        vision = self._verifier is not None and self._verifier.has_camera and not self._ghost
        if vision and len(self._votes) < p.verify_frames:
            assert self._verifier is not None
            tag = f"{cable.pose.name}_try{self.attempt}_f{len(self._votes)}"
            verdict = self._verifier.check(cable.pose.tip, tag)
            if verdict is not None:
                self._votes.append(verdict.seen)
                self._reason = verdict.reason
            elif now_s > self._deadline_s:
                vision = False  # the camera stopped giving frames: judge without it
                self._log("graph: warning: no wrist frame to check the tip; judging by the rise alone")
            if vision and len(self._votes) < p.verify_frames:
                return
        seen = sum(self._votes) >= p.verify_votes if vision else not p.require_vision or self._ghost
        if not vision and not self._ghost and p.require_vision:
            self._log("graph: no camera for the tip check, and require_vision is on")
        if risen < p.min_rise_m:
            self._fail(now_s, f"rose only {risen * 1000:.0f} mm")
        elif not seen:
            self._fail(now_s, f"no tip seen ({sum(self._votes)}/{len(self._votes)} frames: {self._reason})")
        else:
            how = (
                f"{sum(self._votes)}/{len(self._votes)} frames see the tip"
                if vision
                else f"risen {risen * 1000:.0f} mm and still held; no tip check"
            )
            self._log(f"graph: {cable.pose.name} is out ({how})")
            if not self._hold_if_asked(now_s):
                self._move("TRANSIT_DROP", now_s, f"{cable.pose.name}:drop", self._build_drop)

    def _tick_transit_drop(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} carrying {self._cable().pose.name} to the drop spot", measured)
        elif result == "arrived":
            self._move("PLACE", now_s, f"{self._cable().pose.name}:place", self._build_place)

    def _tick_place(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} lowering {self._cable().pose.name}", measured)
        elif result == "arrived" and not self._hold_if_asked(now_s):
            self._grip = self._p.open_grip
            self._go("RELEASE", now_s, timeout_s=self._p.release_dwell_s + 2.0)

    def _tick_release(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        if now_s - self._entered_s >= self._p.release_dwell_s and grip > 0.6 * self._p.open_grip:
            cable = self._cable()
            cable.result = "ok"
            self._events.append(("episode_end", {"cable": cable.pose.name, "success": True}))
            self._move("RETREAT", now_s, f"{cable.pose.name}:retreat", self._build_retreat)
        elif now_s > self._deadline_s:
            self._abort(f"the gripper did not open at the drop spot (grip {grip:.3f})", measured)

    def _tick_retreat(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} rising from the drop spot", measured)
        elif result == "arrived":
            self._next_cable(now_s)

    def _tick_fail_attempt(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        if now_s - self._entered_s >= self._p.fail_dwell_s:
            # Up from where the arm is (it may have stopped short), not from where it was told to be.
            self._q = measured.copy()
            self.attempt += 1
            self._move("RECOVER_UP", now_s, f"{self._cable().pose.name}:recover", self._build_recover)

    def _tick_recover_up(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} rising after a failed try", measured)
            return
        if result != "arrived":
            return
        cable = self._cable()
        if self.attempt <= self._p.attempts:
            self._dz = max(self._dz + self._p.retry_dz_m, self._p.retry_dz_max_m)
            self._log(f"graph: {cable.pose.name}: try {self.attempt}/{self._p.attempts}, {self._dz * 1000:+.0f} mm")
            self._move("DESCEND", now_s, f"{cable.pose.name}:descend", self._build_descend)
            return
        cable.result = "failed"
        self.attempt = self._p.attempts
        self._events.append(("episode_end", {"cable": cable.pose.name, "success": False}))
        if self._p.on_give_up == "abort":
            self._abort(f"{cable.pose.name}: every try failed ({self._reason})", measured)
        else:
            self._log(f"graph: giving up on {cable.pose.name} after {self._p.attempts} tries; next cable")
            self._next_cable(now_s)

    def _tick_home(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} on the way back to the ready pose", measured)
        elif result == "arrived":
            path = motion.plan_joint_ramp(self._q, self._home, self._p.joint_speed_rad_s, self._dt, "to the start")
            self._go("PARK", now_s, path)

    def _tick_park(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        result = self._follow(now_s, measured)
        if result in ("blocked", "timeout"):
            self._abort(f"{result} on the way back to the start", measured)
        elif result == "arrived":
            self._go("DONE", now_s)

    def _tick_done(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        pass

    def _tick_hold(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        pass

    def _tick_abort(self, now_s: float, measured: npt.NDArray[np.float64], grip: float) -> None:
        pass
