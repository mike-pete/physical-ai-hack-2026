"""The supervisor: watches a live Dream-1 run, stops it when the task is done, and guides it when it is failing.

`dreamcli robot run` never stops itself and Dream has no success detection. This program attaches to a run you
started (Start in Live View, or one `curl`), reads the recorder's growing video, and every few seconds asks the critic
(tools/task_critic.py) whether the task is done. It never opens the arms or the cameras: it calls the station agent's
HTTP API (`dreamcli robot up`, port 8091) and reads files. docs/supervisor.md has the whole picture.

```
# Look first, change nothing: the station agent, the run, the recorder, the pictures
uv run python tools/supervise.py check
# Watch a run and log what it WOULD do. It sends no control call at all.
uv run python tools/supervise.py run --observe-only
# Watch, stop when done, guide and restart when failing
uv run python tools/supervise.py run
# Replay a demonstration into the loop against a fake station: no robot, no arms
uv run python tools/supervise.py rehearse --video path/to/observation.images.scene/episode_000004.mp4
```

What it does, in order of severity:

1. **Done.** `done_probes` probes in a row say done (every vote yes): stop, then home. One probe never ends a run.
2. **Failing.** The attempt passed its time limit, the station reports an error or stalls, or the task shows no
   progress for `stuck_after_s` (a progress detector from the task file: how much the task's objects moved, then a
   model compare to confirm). The first failure guides in place (a rewritten prompt, `POST /prompt`, no homing). A
   second failure inside the guide window, or the time limit after a guide, stops the run, homes the arms and starts
   the next attempt with a new prompt. After `attempts` attempts it stops, goes home and reports.
3. **Anything else** (not done, unsure) is normal in the middle of a task and never a failure by itself.

If the run ends while it is watching (the station goes idle), someone stopped it: it ends too and starts nothing. A
station in `error` is restarted, like any other failure.

The recorder writes video only while a run is going, so the supervisor cannot look at a scene while the arms are
homing. A new attempt is judged from its own pictures 1 s after it starts, while the arms are still near
home: if that look and the next one say done, the attempt ends early.

Safety: it never starts a run unless you pass `--start`; `--observe-only` makes no control call; a crash, Ctrl+C or
SIGTERM sends `POST /stop` first so the arms hold; at most one control call every 3 seconds; and pictures go to Google
(and to Anthropic with `--rewriter claude`) only after it says what it will send and you say yes.
"""

import argparse
import collections
import concurrent.futures
import contextlib
import dataclasses
import datetime
import json
import os
import pathlib
import queue
import select
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import cv2
import numpy as np
import station_io as sio
import task_critic as tc
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TASK = REPO / "configs" / "tasks" / "usb-cables.yaml"
DEFAULT_STATION_FILE = pathlib.Path("~/.dreamlabs/station.yaml")
DEFAULT_ROLLOUTS = pathlib.Path("~/.dreamlabs/rollouts")
DEFAULT_LOGS = pathlib.Path("~/hackathon_data/supervisor")

## At most one control call (start, stop, home, prompt) in this many seconds.
CONTROL_GAP_S = 3.0
## An early look at a new attempt this many seconds after it starts, while the arms are still near home. If the
## recorder has not written its first picture yet, that look is skipped and the next one is after the warm-up.
ARRIVAL_DELAY_S = 1.0
## Guides in place per attempt before a failure means a restart.
MAX_GUIDES = 2
## A status poll that fails this many times in a row ends the supervisor: the station is gone.
MAX_STATUS_FAILURES = 10
## A recording that has not grown for this long is stale, and a probe is skipped.
STALE_AFTER_S = 5.0
## A probe whose votes were not "unsure" at least this often is informative enough to write the next prompt from.
INFORMATIVE_VISIBLE = 0.5
## Seconds to wait for the station to reach a state after a stop, a home or a start.
SETTLE_TIMEOUT_S = 90.0
## Probes in a row that saw nothing (no recording, stale, face, model error) before the log says it is blind.
BLIND_AFTER = 4
## Station states in which nothing should be asked of it yet.
BUSY_STATES = ("starting", "stopping", "homing", "resetting")

DONE, NOT_DONE, UNSURE = tc.DONE, tc.NOT_DONE, tc.UNSURE

KEYS_HELP = "keys: f finish now (stop, home) | n next attempt now | p pause or resume | x abort (stop, arms hold)"
KEYS_HELP_STEP = KEYS_HELP.replace("| x abort", "| s skip this step (stop, home) | x abort")


class SupervisorError(RuntimeError):
    """The supervisor cannot start or has to stop; the message says what to do."""


# --- Time ------------------------------------------------------------------------------------------------------


class Clock(Protocol):
    """The supervisor's time: seconds since it started, a sleep, and the wall clock (UTC)."""

    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def wall(self) -> datetime.datetime: ...


class RealClock:
    """Real time, optionally `speed` times faster (a rehearsal replaying a demonstration quickly)."""

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self._mono0 = time.monotonic()
        self._wall0 = datetime.datetime.now(datetime.UTC)

    def now(self) -> float:
        return (time.monotonic() - self._mono0) * self.speed

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds) / self.speed)

    def wall(self) -> datetime.datetime:
        return self._wall0 + datetime.timedelta(seconds=self.now())


# --- Progress detectors and the stuck watchdog -----------------------------------------------------------------


class ObjectMotion:
    """The cheap progress detector: how much the pixels of one colour changed between two pictures of a view.

    Only that colour counts, so an arm passing through the picture cannot hide a stall. The change is 0 for
    identical masks and 1 for masks with nothing in common (one minus their overlap), after a small dilation so a pixel
    of compression noise does not count.
    """

    def __init__(self, spec: tc.ObjectMotionSpec):
        self.spec = spec
        self._low = np.array(spec.low, np.uint8)
        self._high = np.array(spec.high, np.uint8)
        self._code = cv2.COLOR_BGR2LAB if spec.space == "lab" else cv2.COLOR_BGR2HSV
        self._kernel = np.ones((3, 3), np.uint8)

    def mask(self, view_bgr: np.ndarray) -> np.ndarray:
        found = cv2.inRange(cv2.cvtColor(view_bgr, self._code), self._low, self._high)
        return cv2.dilate(found, self._kernel) > 0

    @staticmethod
    def change(before: np.ndarray, after: np.ndarray) -> float:
        union = int((before | after).sum())
        return 0.0 if union == 0 else 1.0 - int((before & after).sum()) / union


class Compare(Protocol):
    """Answers whether the task moved forward between two pictures."""

    def compare(self, spec: tc.TaskSpec, before: bytes, after: bytes, seconds: float) -> tc.Vote: ...


@dataclasses.dataclass
class StuckReading:
    """What the watchdog made of one probe."""

    stuck: bool
    note: str
    motion: float | None = None  # the object-motion change since the last probe, when the task has a detector
    quiet_s: float = 0.0  # seconds with no progress
    compare: str | None = None  # the model compare's answer, when it was asked: yes, no or unsure
    compare_reason: str = ""


class Watchdog:
    """Decides whether a run is stuck: the task shows no progress for `stuck_after_s` while the arms are commanded.

    Progress is measured by the detectors the task file names, so any task works:

    - With an `object_motion` detector (the task's objects have a distinctive colour), it measures the change between
      probes. Once there has been none for `stuck_after_s`, a model compare is asked about the earlier and the latest
      picture; stuck needs it to say no. A yes or unsure restarts the clock, so a wrong colour range alone can never
      end a good run.
    - With no detector, the model compare is asked on every other probe, and two no answers in a row mean stuck.
    - With neither a detector nor a compare, nothing can be stuck.

    The caller says whether the station is still commanding the arms; the picture cannot tell.
    """

    def __init__(self, spec: tc.TaskSpec, compare: Compare | None):
        self.spec = spec
        self.config = spec.supervisor
        self.compare = compare
        motion = self.config.object_motion
        self.motion = ObjectMotion(motion) if motion else None
        self._view = spec.views[motion.view] if motion else None
        self.picture_name = self.config.compare_view or spec.cameras[0]
        self.reset(0.0)

    def reset(self, now: float) -> None:
        """Start over: after a guide or a restart the clock, the pictures and the last mask are forgotten."""
        self._mask: np.ndarray | None = None
        self._last_progress = now
        self._pictures: collections.deque[tuple[float, bytes]] = collections.deque()
        self._no_in_a_row = 0
        self._probes_since_compare = 0

    def _ask(self, now: float, before: tuple[float, bytes], after: bytes) -> tc.Vote:
        assert self.compare is not None
        try:
            return self.compare.compare(self.spec, before[1], after, max(1.0, now - before[0]))
        except tc.CriticError as error:
            return tc.Vote("unsure", f"the compare failed: {error}")

    def update(
        self,
        now: float,
        *,
        frames: Mapping[str, np.ndarray],
        pictures: Mapping[str, bytes],
        commanded: bool,
    ) -> StuckReading:
        """Takes one probe.

        @param frames The newest whole frame of each camera (OpenCV, BGR).
        @param pictures The probe's newest picture set, as tc.cut_pictures() makes it.
        @param commanded Whether the station is still sending the arms moves (its `chunks` rose since the last probe).
        """
        self._pictures.append((now, pictures[self.picture_name]))
        horizon = 3 * self.config.stuck_after_s + self.config.probe_every_s
        while len(self._pictures) > 1 and now - self._pictures[0][0] > horizon:
            self._pictures.popleft()
        change = None
        if self.motion is not None and self._view is not None:
            mask = self.motion.mask(tc.crop_fractions(frames[self._view.camera], self._view.box))
            change = None if self._mask is None else self.motion.change(self._mask, mask)
            self._mask = mask
        if not commanded:
            self._last_progress, self._no_in_a_row, self._probes_since_compare = now, 0, 0
            return StuckReading(False, "the arms are not being commanded", change)
        if self.motion is not None:
            return self._by_motion(now, change)
        return self._by_compare(now)

    def _by_motion(self, now: float, change: float | None) -> StuckReading:
        assert self.motion is not None
        if change is None or change >= self.motion.spec.min_change:
            self._last_progress = now
            return StuckReading(False, "the task's objects moved", change)
        quiet = now - self._last_progress
        if quiet < self.config.stuck_after_s:
            return StuckReading(False, f"quiet for {quiet:.0f} s", change, quiet)
        if self.compare is None:
            return StuckReading(False, f"quiet for {quiet:.0f} s, unconfirmed: no model compare", change, quiet)
        before = next(((t, p) for t, p in self._pictures if t >= self._last_progress), self._pictures[0])
        vote = self._ask(now, before, self._pictures[-1][1])
        if vote.answer == "no":
            note = f"no progress for {quiet:.0f} s, and the compare agrees"
            return StuckReading(True, note, change, quiet, "no", vote.reason)
        self._last_progress = now  # a yes or unsure restarts the clock
        return StuckReading(False, "the compare did not confirm it", change, quiet, vote.answer, vote.reason)

    def _by_compare(self, now: float) -> StuckReading:
        if self.compare is None:
            return StuckReading(False, "no progress detector and no model compare")
        self._probes_since_compare += 1
        if self._probes_since_compare < 2:
            return StuckReading(False, "waiting for the next probe to compare", quiet_s=self._quiet())
        self._probes_since_compare = 0
        newest = self._pictures[-1]
        vote = self._ask(now, self._pictures[0], newest[1])
        self._no_in_a_row = self._no_in_a_row + 1 if vote.answer == "no" else 0
        self._pictures.clear()
        self._pictures.append(newest)
        if self._no_in_a_row >= 2:
            return StuckReading(True, "the compare said no twice in a row", None, self._quiet(), "no", vote.reason)
        note = "the compare saw progress or is not sure"
        return StuckReading(False, note, None, self._quiet(), vote.answer, vote.reason)

    def _quiet(self) -> float:
        return self._no_in_a_row * 2 * self.config.probe_every_s


# --- The ladder ------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Decision:
    """What the ladder says to do: `wait`, `finish`, `guide`, `restart`, `give_up` or `ended`."""

    kind: str
    reason: str = ""


def _error_key(last_error: Any) -> str | None:
    return json.dumps(last_error, sort_keys=True, default=str) if last_error else None


def _error_text(last_error: Any) -> str:
    return str(last_error.get("message", last_error)) if isinstance(last_error, Mapping) else str(last_error)


class Ladder:
    """The decisions, as pure code: no clock, no network, no arms; only what was observed.

    One attempt is one run of the model. The first failure of an attempt guides in place; a failure inside the guide
    window (or the time limit after a guide) restarts it; after `attempts` attempts it gives up. `step()` is called
    once per tick, and the caller reports back what it did with `note_guided()` and `new_attempt()`.
    """

    def __init__(self, config: tc.SupervisorSpec, *, now: float, stalls: int = 0, last_error: Any = None):
        self.config = config
        self.attempt = 1
        self.rebase(now, stalls, last_error)

    def rebase(self, now: float, stalls: int = 0, last_error: Any = None) -> None:
        """Start the attempt's clocks and baselines over, without counting an attempt (a run someone else started)."""
        self.started = now
        self.deadline = now + self.config.attempt_s
        self.stalls_base = stalls
        self.error_base = _error_key(last_error)
        self.done_streak = 0
        self.guided_at: float | None = None
        self.guides = 0

    def new_attempt(self, now: float, stalls: int = 0, last_error: Any = None) -> None:
        """A new run of the model began."""
        self.attempt += 1
        self.rebase(now, stalls, last_error)

    def note_guided(self, now: float, stalls: int = 0, last_error: Any = None) -> None:
        """A guide was sent (or, in observe-only mode, would have been): open its window and give it a chance."""
        self.guided_at = now
        self.guides += 1
        self.stalls_base = stalls
        self.error_base = _error_key(last_error)
        self.done_streak = 0
        self.deadline = max(self.deadline, now + self.config.guide_window_s)

    def _failure(self, now: float, stalls: int, last_error: Any, stuck: bool) -> tuple[str, str] | None:
        if last_error and _error_key(last_error) != self.error_base:
            return "error", f"the station reports an error: {_error_text(last_error)}"
        if stalls - self.stalls_base > self.config.max_stalls:
            return "stalls", f"stalls rose by {stalls - self.stalls_base} (the limit is {self.config.max_stalls})"
        if now > self.deadline:
            return "time", f"the attempt passed its time limit ({self.config.attempt_s:g} s)"
        if stuck:
            return "stuck", "the task shows no progress"
        return None

    def step(
        self,
        now: float,
        *,
        state: str | None,
        stalls: int = 0,
        last_error: Any = None,
        probe: str | None = None,
        stuck: bool = False,
    ) -> Decision:
        """One tick.

        @param state The station's state (running, idle, error, ...).
        @param probe The decision of a probe that finished since the last tick (done, not_done, unsure), else None.
        @param stuck Whether the watchdog says the task is stuck.
        """
        if probe == DONE:
            self.done_streak += 1
            if self.done_streak >= self.config.done_probes:
                return Decision("finish", f"{self.done_streak} probes in a row said done")
        elif probe is not None:
            self.done_streak = 0
        if state == "idle":
            return Decision("ended", "the run ended: the station is idle (someone stopped it)")
        if state in BUSY_STATES:
            return Decision("wait", f"the station is {state}")
        if state == "error":
            text = _error_text(last_error) if last_error else "no message"
            return self._escalate(f"the station is in error: {text}")
        failure = self._failure(now, stalls, last_error, stuck)
        if failure is None:
            return Decision("wait")
        kind, text = failure
        again = self.guided_at is not None and (kind == "time" or now - self.guided_at <= self.config.guide_window_s)
        if again or self.guides >= MAX_GUIDES:
            return self._escalate(text + (" again, inside the guide window" if again and kind != "time" else ""))
        return Decision("guide", text)

    def _escalate(self, reason: str) -> Decision:
        if self.attempt >= self.config.attempts:
            return Decision("give_up", f"{reason}; that was attempt {self.attempt} of {self.config.attempts}")
        return Decision("restart", reason)


# --- The log ---------------------------------------------------------------------------------------------------


class EventLog:
    """What the supervisor did and saw: a line of JSON per event in `events.jsonl`, the pictures it sent, and a
    short line on the terminal. A log with no folder only prints."""

    def __init__(
        self,
        folder: pathlib.Path | None,
        clock: Clock,
        *,
        echo: Callable[[str], None] | None = print,
        extra: Mapping[str, Any] | None = None,
        parent: "EventLog | None" = None,
    ):
        """@param extra Fields added to every event (a sequence names the step).
        @param parent A log that also gets every event, on its own time axis: a step's log feeds the sequence's."""
        self.folder = folder
        self._clock = clock
        self._t0 = parent._t0 if parent is not None else clock.now()
        self._echo = echo
        self._extra = dict(extra or {})
        self._parent = parent
        self._lock = threading.Lock()
        if folder is not None:
            (folder / "pictures").mkdir(parents=True, exist_ok=True)

    def write(self, event: str, text: str, **fields: Any) -> dict[str, Any]:
        """@param text The line shown on the terminal; `fields` go to the file only."""
        now = self._clock.now()
        record = {
            "t": round(now - self._t0, 1),
            "utc": self._clock.wall().isoformat(timespec="seconds"),
            "event": event,
            "text": text,
            **self._extra,
            **fields,
        }
        self.append(record)
        if self._echo is not None:
            with self._lock:
                self._echo(f"[{record['t']:7.1f}s] {text}")
        if self._parent is not None:
            self._parent.append(record)
        return record

    def append(self, record: Mapping[str, Any]) -> None:
        """Adds an event that was made elsewhere to `events.jsonl`, without printing it."""
        if self.folder is not None:
            with self._lock, open(self.folder / "events.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")

    def save_pictures(self, tag: str, pictures: Mapping[str, bytes]) -> list[str]:
        """Keeps the pictures that were sent (JPEGs); returns their paths relative to the log folder."""
        if self.folder is None:
            return []
        paths = []
        for name, jpeg in pictures.items():
            relative = pathlib.Path("pictures") / f"{tag}-{name}.jpg"
            (self.folder / relative).write_bytes(jpeg)
            paths.append(str(relative))
        return paths

    def request(self, body: Mapping[str, Any]) -> None:
        """Keeps a model request without its pictures (the callback the critic's backends call)."""
        if self.folder is not None:
            with self._lock, open(self.folder / "requests.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(body, default=str) + "\n")

    def report(self, data: Mapping[str, Any], name: str = "report.json") -> None:
        if self.folder is not None:
            (self.folder / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# --- Control calls ---------------------------------------------------------------------------------------------


class StationApi(Protocol):
    """What the supervisor needs from the station agent (sio.StationClient, or sio.LocalClient over a FakeStation)."""

    def status(self) -> dict[str, Any]: ...

    def stop(self) -> dict[str, Any]: ...

    def home(self) -> dict[str, Any]: ...

    def start(self, prompt: str | None = None, checkpoint: str | None = None) -> dict[str, Any]: ...

    def set_prompt(self, prompt: str) -> dict[str, Any]: ...


class Controller:
    """Every call that changes something goes through here, and nothing else: it logs the call, spaces calls at
    least `gap_s` apart, and in observe-only mode makes none and only says what it would have done."""

    def __init__(
        self, client: StationApi, log: EventLog, clock: Clock, *, observe_only: bool, gap_s: float = CONTROL_GAP_S
    ):
        self.client = client
        self.log = log
        self.clock = clock
        self.observe_only = observe_only
        self.gap_s = gap_s
        self._last: float | None = None
        self.calls = 0

    def send(self, name: str, *, force: bool = False, **args: Any) -> dict[str, Any] | None:
        """@param name stop, home, start or prompt.
        @param force Skip the spacing (a stop on the way out must not wait).
        @return The station's reply; None in observe-only mode.
        @throws sio.StationRefusedError The station said 409: the call is valid in another state.
        @throws sio.StationError The station could not be reached or failed.
        """
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
        if self.observe_only:
            self.log.write("would", f"would call {name}({shown})", action=name, **args)
            return None
        if self._last is not None and not force:
            wait = self._last + self.gap_s - self.clock.now()
            if wait > 0:
                self.clock.sleep(wait)
        self._last = self.clock.now()
        self.calls += 1
        try:
            if name == "stop":
                reply = self.client.stop()
            elif name == "home":
                reply = self.client.home()
            elif name == "start":
                reply = self.client.start(args.get("prompt"), args.get("checkpoint"))
            elif name == "prompt":
                reply = self.client.set_prompt(args["prompt"])
            else:
                raise ValueError(f"no control call {name!r}")
        except sio.StationRefusedError as error:
            self.log.write("refused", f"the station refused {name}: {error}", action=name, **args)
            raise
        except sio.StationError as error:
            self.log.write("failed", f"{name} failed: {error}", action=name, **args)
            raise
        self._last = self.clock.now()
        self.log.write("control", f"called {name}({shown}); the station is {reply.get('state')}", action=name, **args)
        return reply


# --- Keys ------------------------------------------------------------------------------------------------------


class KeyReader:
    """Reads single keys from the terminal (in tmux, say) when stdin is one; otherwise it never has a key."""

    def __init__(self, stream: Any = None):
        self._stream = stream if stream is not None else sys.stdin
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved: tuple[int, list[Any]] | None = None

    def __enter__(self) -> "KeyReader":
        try:
            if not self._stream.isatty():
                return self
            import termios  # noqa: PLC0415 (not on every platform)
            import tty  # noqa: PLC0415

            fd = self._stream.fileno()
            self._saved = (fd, termios.tcgetattr(fd))
            tty.setcbreak(fd)  # Ctrl+C still interrupts
        except (OSError, ValueError, ImportError, AttributeError):
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        fd = self._stream.fileno()
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                char = os.read(fd, 1).decode("utf-8", "ignore").lower()
                if char:
                    self._queue.put(char)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._saved is not None:
            import termios  # noqa: PLC0415

            with contextlib.suppress(OSError):
                termios.tcsetattr(self._saved[0], termios.TCSADRAIN, self._saved[1])

    def get(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


# --- The supervisor --------------------------------------------------------------------------------------------


def has_face(spec: tc.TaskSpec, frames: Mapping[str, np.ndarray]) -> bool:
    """Whether a face is in what would be sent: the cameras' crops are checked, as `critic_eval frames` does."""
    for camera in spec.cameras:
        box = spec.crop.get(camera)
        crop = frames[camera] if box is None else tc.crop_fractions(frames[camera], box)
        if tc.count_faces(crop):
            return True
    return False


@dataclasses.dataclass
class Options:
    observe_only: bool = False
    start: bool = False  # start the first attempt when no run is going (needs `prompt`)
    prompt: str | None = None
    wait_s: float = 120.0  # how long to wait for a run to be going
    tick_s: float = 1.0
    settle_s: float = SETTLE_TIMEOUT_S


@dataclasses.dataclass
class ProbeResult:
    """One look at the scene: the critic's verdict on a few pictures and the watchdog's reading."""

    n: int
    at: float
    verdict: tc.Verdict | None = None
    pictures: dict[str, bytes] = dataclasses.field(default_factory=dict)  # the newest picture set
    reading: StuckReading | None = None
    latency_s: float = 0.0
    skipped: str | None = None  # why nothing was decided, when nothing was


class Supervisor:
    """Attaches to a run and watches it until the task is done, the attempts are used up, or someone stops it.

    @see the module docstring, and docs/supervisor.md.
    """

    def __init__(
        self,
        spec: tc.TaskSpec,
        critic: tc.TaskCritic,
        client: StationApi,
        sources: Mapping[str, sio.FrameSource],
        *,
        log: EventLog,
        clock: Clock,
        compare: Compare | None = None,
        options: Options | None = None,
        keys: Callable[[], str | None] = lambda: None,
        pool: concurrent.futures.Executor | None = None,
        controller: "Controller | None" = None,
        allow_skip: bool = False,
    ):
        """@param pool Where probes run (one worker). None runs them inline, which tests use for determinism.
        @param controller The one that makes control calls, when a sequence shares it between its steps so that the
            3 s spacing holds across them; it then logs to this supervisor's log.
        @param allow_skip Whether the `s` key ends this run as skipped (a step of a sequence), else it is ignored.
        """
        self.spec = spec
        self.config = spec.supervisor
        self.critic = critic
        self.client = client
        self.sources = dict(sources)
        self.log = log
        self.clock = clock
        self.options = options or Options()
        self.keys = keys
        self.pool = pool
        if controller is None:
            controller = Controller(client, log, clock, observe_only=self.options.observe_only)
        controller.log = log
        self.ctl = controller
        self.allow_skip = allow_skip
        self.homed: bool | None = None  # whether the arms were sent home and the station accepted it
        self.watchdog = Watchdog(spec, compare)
        self.ladder: Ladder | None = None
        self.history: list[str] = []
        self.run_id: Any = None
        self.paused = False
        self._status: dict[str, Any] = {}
        self._failures = 0
        self._future: concurrent.futures.Future[ProbeResult] | None = None
        self._next_probe_at: float | None = None
        self._chunks_seen: int | None = None
        self._probe_n = 0
        self._blind = 0
        self._last_probe: ProbeResult | None = None
        self._informative: ProbeResult | None = None
        self._run_going = False
        self._guides_total = 0
        self._probes_done = 0
        self._started_at = clock.now()
        self._calls_at_start = controller.calls
        self._prompt_log: list[dict[str, Any]] = []
        self._probe_decisions: list[str] = []
        self._last_verdict: dict[str, Any] | None = None
        self._folders: list[str] = []

    # --- the run ------------------------------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Attaches, watches, and returns the report. A crash, Ctrl+C or SIGTERM stops the run first (not in
        observe-only mode), so a dead supervisor never leaves a run going unwatched."""
        outcome, detail = "error", ""
        try:
            self._attach()
            outcome, detail = self._loop()
        except KeyboardInterrupt:
            outcome, detail = "interrupted", "interrupted (Ctrl+C or SIGTERM)"
            self.log.write("interrupted", "interrupted: leaving the arms holding")
        except SupervisorError as error:
            outcome, detail = "error", str(error)
            self.log.write("error", f"stopping: {error}")
        except Exception as error:  # noqa: BLE001 (a bug must still leave the run stopped)
            outcome, detail = "crashed", f"{type(error).__name__}: {error}"
            self.log.write("crash", f"crashed: {detail}", traceback=traceback.format_exc())
        finally:
            self._shutdown()
        return self._report(outcome, detail)

    def _shutdown(self) -> None:
        if self._future is not None:
            self._future.cancel()
        if self._run_going and not self.options.observe_only:
            try:
                self.ctl.send("stop", force=True)
                self.log.write("stop", "stopped the run so the arms hold where they are")
            except sio.StationError as error:
                self.log.write("error", f"could not stop the run on the way out: {error}")
            self._run_going = False

    def _report(self, outcome: str, detail: str) -> dict[str, Any]:
        report = {
            "outcome": outcome,
            "detail": detail,
            "observe_only": self.options.observe_only,
            "task": self.spec.name,
            "run": self.run_id,
            "attempts": self.ladder.attempt if self.ladder else 0,
            "guides": self._guides_total,
            "prompts": list(self.history),
            "prompt_log": list(self._prompt_log),
            "probes": self._probes_done,
            "control_calls": self.ctl.calls - self._calls_at_start,
            "elapsed_s": round(self.clock.now() - self._started_at, 1),
            "homed": self.homed,
            "done_at_start": self._done_at_start(outcome),
            "last_verdict": self._last_verdict,
            "folders": list(self._folders),
            "log": str(self.log.folder) if self.log.folder else None,
        }
        self.log.write("end", f"{outcome}: {detail}", **report)
        self.log.report(report)
        return report

    def _done_at_start(self, outcome: str) -> bool:
        """Whether the very first probes already said done: the scene looked finished when the step began (or the
        model finished before the first look, which a 15 s warm-up cannot tell apart)."""
        need = self.config.done_probes
        return outcome == "done" and len(self._probe_decisions) >= need and set(self._probe_decisions[:need]) == {DONE}

    def _note_prompt(self, kind: str, prompt: str, source: str | None, reason: str = "") -> None:
        self._prompt_log.append(
            {"kind": kind, "prompt": prompt, "source": source, "reason": reason, "t": round(self.clock.now(), 1)}
        )

    # --- attaching ----------------------------------------------------------------------------------------------

    def _poll(self, *, required: bool = False) -> dict[str, Any] | None:
        """The station's status, or None when it did not answer (an error only after several in a row)."""
        try:
            status = self.client.status()
        except sio.StationError as error:
            self._failures += 1
            self.log.write("status", f"no status ({self._failures}): {error}")
            if self._failures >= MAX_STATUS_FAILURES or (required and self._failures >= 3):
                raise SupervisorError(f"the station agent did not answer {self._failures} times in a row") from None
            return None
        self._failures = 0
        self._status = status
        return status

    def _attach(self) -> None:
        options = self.options
        waited_since = self.clock.now()
        said = False
        started_it = False
        while True:
            status = self._poll(required=True)
            if status is None:  # no answer yet: _poll raises once there have been enough in a row
                self.clock.sleep(options.tick_s)
                continue
            state = status.get("state")
            if state == "running":
                break
            if state in ("idle", "error") and options.start and not started_it and not options.observe_only:
                if not options.prompt:
                    raise SupervisorError("--start needs --prompt, the instruction to start with")
                self.log.write("start", f"no run is going: starting one with {options.prompt!r}")
                try:
                    self.ctl.send("start", prompt=options.prompt, checkpoint=status.get("checkpoint"))
                except sio.StationRefusedError as error:
                    raise SupervisorError(f"the station refused to start: {error}") from None
                started_it = True
                continue
            if self.clock.now() - waited_since > options.wait_s:
                raise SupervisorError(
                    f"no run is going (the station is {state}): press Start in Live View, or pass --start --prompt"
                )
            if not said:
                self.log.write("wait", f"waiting for a run: the station is {state}")
                said = True
            self.clock.sleep(options.tick_s)
        now = self.clock.now()
        elapsed = float(status.get("run_elapsed_s") or 0.0)
        self.run_id = status.get("run")
        self._run_going = True
        if status.get("prompt"):
            self.history.append(str(status["prompt"]))
            self._note_prompt("start", str(status["prompt"]), "run")
        self.ladder = Ladder(
            self.config, now=now - elapsed, stalls=int(status.get("stalls") or 0), last_error=status.get("last_error")
        )
        self._bind(self.clock.wall() - datetime.timedelta(seconds=elapsed))
        self.watchdog.reset(now)
        self._next_probe_at = max(now, self.ladder.started + self.config.warmup_s)
        mode = "OBSERVE ONLY: it makes no control call" if options.observe_only else "it will act"
        self.log.write(
            "attach",
            f"attached to run {self.run_id} ({elapsed:.0f} s in), prompt {status.get('prompt')!r}; {mode}",
            run=self.run_id,
            prompt=status.get("prompt"),
            checkpoint=status.get("checkpoint"),
            observe_only=options.observe_only,
        )
        self.log.write("keys", KEYS_HELP_STEP if self.allow_skip else KEYS_HELP)

    def _bind(self, since: datetime.datetime) -> None:
        for source in self.sources.values():
            source.bind(since)

    # --- the loop -----------------------------------------------------------------------------------------------

    def _loop(self) -> tuple[str, str]:
        while True:
            outcome = self._tick()
            if outcome is not None:
                return outcome
            self.clock.sleep(self.options.tick_s)

    def _tick(self) -> tuple[str, str] | None:
        assert self.ladder is not None
        now = self.clock.now()
        status = self._poll()
        if status is None:
            return None
        state = status.get("state")
        if state == "running":
            self._run_going = True
            if status.get("run") != self.run_id:
                self._adopt(now, status)
        command = self.keys()
        if command == "p":
            self.paused = not self.paused
            self.log.write("pause", "paused: watching only" if self.paused else "resumed")
            if not self.paused:
                self.ladder.rebase(now, int(status.get("stalls") or 0), status.get("last_error"))
                self.watchdog.reset(now)
                self._next_probe_at = now + self.config.probe_every_s
        probe, stuck = self._collect()
        decision = self._decide(now, status, command, probe, stuck)
        if decision.kind == "wait":
            self._maybe_probe(now, status)
            return None
        return self._apply(now, decision)

    def _decide(
        self, now: float, status: dict[str, Any], command: str | None, probe: str | None, stuck: bool
    ) -> Decision:
        assert self.ladder is not None
        state = status.get("state")
        if command == "s" and not self.allow_skip:
            self.log.write("note", "the s key skips a step of a sequence: this is a single task, so it does nothing")
            command = None
        if command in ("f", "n", "x", "s"):
            self.log.write("key", f"operator pressed {command}", key=command)
            kinds = {"x": "abort", "s": "skip", "f": "finish", "n": "restart"}
            return Decision(kinds[command], f"the operator pressed {command}")
        if self.paused:
            return Decision("ended", "the run ended while paused") if state == "idle" else Decision("wait", "paused")
        return self.ladder.step(
            now,
            state=state,
            stalls=int(status.get("stalls") or 0),
            last_error=status.get("last_error"),
            probe=probe,
            stuck=stuck,
        )

    def _adopt(self, now: float, status: dict[str, Any]) -> None:
        """A run that is not the one being watched is going (someone started one): follow it from its own start."""
        assert self.ladder is not None
        self._drain()
        elapsed = float(status.get("run_elapsed_s") or 0.0)
        self.log.write(
            "adopt", f"run {status.get('run')} started outside the supervisor: following it", run=status.get("run")
        )
        self.run_id = status.get("run")
        if status.get("prompt") and status["prompt"] not in self.history:
            self.history.append(str(status["prompt"]))
            self._note_prompt("adopted", str(status["prompt"]), "run")
        self.ladder.rebase(now - elapsed, int(status.get("stalls") or 0), status.get("last_error"))
        self._bind(self.clock.wall() - datetime.timedelta(seconds=elapsed))
        self.watchdog.reset(now)
        self._chunks_seen = None
        self._next_probe_at = max(now, self.ladder.started + self.config.warmup_s)

    # --- probes -------------------------------------------------------------------------------------------------

    def _maybe_probe(self, now: float, status: dict[str, Any]) -> None:
        if self._future is not None or self.paused or status.get("state") != "running":
            return
        if self._next_probe_at is None or now < self._next_probe_at:
            return
        chunks = int(status.get("chunks") or 0)
        commanded = chunks > (self._chunks_seen if self._chunks_seen is not None else 0)
        self._chunks_seen = chunks
        self._probe_n += 1
        self._future = self._submit(self._probe_job, self._probe_n, commanded)

    def _submit(self, fn: Callable[..., ProbeResult], *args: Any) -> "concurrent.futures.Future[ProbeResult]":
        if self.pool is not None:
            return self.pool.submit(fn, *args)
        future: concurrent.futures.Future[ProbeResult] = concurrent.futures.Future()
        future.set_result(fn(*args))
        return future

    def _drain(self) -> None:
        """Waits for a probe in flight and drops its result: its pictures are from before what is about to change."""
        if self._future is not None:
            with contextlib.suppress(Exception):
                self._future.result()
            self._future = None

    def _wanted(self) -> tuple[str, ...]:
        gate = self.critic.gate
        return tuple(gate.wanted_views(self.spec)) if hasattr(gate, "wanted_views") else self.spec.cameras

    def _faces(self, frames: Mapping[str, np.ndarray]) -> bool:
        return has_face(self.spec, frames)

    def _probe_job(self, n: int, commanded: bool, *, watch: bool = True) -> ProbeResult:
        """Reads the newest pictures, asks the critic, and takes the watchdog's reading. Never raises: a probe that
        could not see says why and decides nothing. Runs on the probe worker."""
        real_start = time.monotonic()
        at = self.clock.now()
        try:
            taken = {camera: source.latest(self.config.frames_per_probe) for camera, source in self.sources.items()}
            stale = max((source.stale_s() or 0.0) for source in self.sources.values())
            if stale > STALE_AFTER_S:
                return ProbeResult(n, at, skipped=f"the recording has not grown for {stale:.0f} s")
            count = min(len(frames) for frames in taken.values())
            if count == 0:
                return ProbeResult(n, at, skipped="the recording has no frames yet")
            sets = [{camera: taken[camera][i].bgr for camera in taken} for i in range(count)]
            if self._faces(sets[-1]):
                return ProbeResult(n, at, skipped="a face is in the picture: nothing was sent")
            pictures = [tc.cut_pictures(self.spec, frames) for frames in sets]
            wanted = self._wanted()
            for i, picture_set in enumerate(pictures):
                keep = set(wanted) | ({self.watchdog.picture_name} if i == len(pictures) - 1 else set())
                self.log.save_pictures(f"probe{n:03d}-{i}", {k: v for k, v in picture_set.items() if k in keep})
            verdict = self.critic.evaluate(pictures, (), rewrite=False)
            reading = None
            if watch:
                reading = self.watchdog.update(at, frames=sets[-1], pictures=pictures[-1], commanded=commanded)
            return ProbeResult(n, at, verdict, pictures[-1], reading, time.monotonic() - real_start)
        except sio.SourceError as error:
            return ProbeResult(n, at, skipped=str(error))
        except tc.CriticError as error:
            return ProbeResult(n, at, skipped=f"the critic failed: {error}")
        except Exception as error:  # noqa: BLE001 (a probe that breaks must not stop the supervisor)
            self.log.write("probe_error", f"probe {n} broke: {error!r}", traceback=traceback.format_exc())
            return ProbeResult(n, at, skipped=f"the probe broke: {error!r}")

    def _collect(self) -> tuple[str | None, bool]:
        """Takes a finished probe: returns its decision and whether the watchdog says stuck."""
        if self._future is None or not self._future.done():
            return None, False
        future, self._future = self._future, None
        result = future.result()
        warmed = (self.ladder.started if self.ladder else 0.0) + self.config.warmup_s
        self._next_probe_at = max(warmed, self.clock.now() + self.config.probe_every_s)
        if result.skipped or result.verdict is None:
            self._blind += 1
            level = " (the supervisor cannot see the run)" if self._blind >= BLIND_AFTER else ""
            self.log.write(
                "skip", f"probe {result.n} skipped: {result.skipped}{level}", probe=result.n, reason=result.skipped
            )
            return None, False
        self._blind = 0
        self._probes_done += 1
        self._last_probe = result
        judged = result.verdict.result
        self._probe_decisions.append(judged.status)
        for source in self.sources.values():
            folder = getattr(source, "folder", None)
            if folder is not None and str(folder) not in self._folders:
                self._folders.append(str(folder))
        if judged.visible >= INFORMATIVE_VISIBLE:
            self._informative = result
        votes = judged.votes or []
        yes = sum(v["answer"] == "yes" for v in votes)
        reasons = list(dict.fromkeys(v["reason"] for v in votes if v["answer"] != "yes" and v["reason"]))
        self._last_verdict = {"probe": result.n, "decision": judged.status, "votes": votes, "reasons": reasons[:3]}
        reading = result.reading
        extra = ""
        if reading is not None:
            motion = "n/a" if reading.motion is None else f"{reading.motion:.2f}"
            extra = f" | motion {motion}, quiet {reading.quiet_s:.0f} s: {reading.note}"
        self.log.write(
            "probe",
            f"probe {result.n}: {judged.status} (yes {yes}/{len(votes)}, {result.latency_s:.1f} s){extra}",
            probe=result.n,
            decision=judged.status,
            frames=[f.status for f in result.verdict.frames],
            votes=votes,
            reasons=reasons[:3],
            latency_s=round(result.latency_s, 1),
            reading=dataclasses.asdict(reading) if reading else None,
        )
        return judged.status, bool(reading and reading.stuck)

    # --- acting -------------------------------------------------------------------------------------------------

    def _apply(self, now: float, decision: Decision) -> tuple[str, str] | None:
        kind, reason = decision.kind, decision.reason
        self.log.write("decision", f"decision: {kind}: {reason}", kind=kind, reason=reason)
        if kind == "ended":
            self._run_going = False
            return "ended", reason
        if kind == "abort":
            self._drain()
            return "aborted", reason
        if kind == "skip":
            self._drain()
            self._stop_and_home()
            return "skipped", reason
        if kind in ("finish", "give_up"):
            self._drain()
            self._stop_and_home()
            return ("done" if kind == "finish" else "gave_up"), reason
        if kind == "guide":
            return self._guide(reason)
        if kind == "restart":
            return self._restart(reason)
        raise ValueError(f"no decision {kind!r}")

    def _try(self, name: str, **args: Any) -> bool:
        """A control call whose refusal is logged and survived. Returns whether the station accepted it."""
        try:
            self.ctl.send(name, **args)
        except sio.StationRefusedError:
            return False
        return True

    def _settle(self, states: Sequence[str], *, tolerant: bool = False) -> dict[str, Any]:
        """Waits until the station is in one of `states`."""
        if self.options.observe_only:
            return self._status
        deadline = self.clock.now() + self.options.settle_s
        while True:
            status = self._poll(required=True)
            if status is not None and status.get("state") in states:
                return status
            if self.clock.now() > deadline:
                if tolerant:
                    return self._status
                raise SupervisorError(f"the station did not reach {list(states)} within {self.options.settle_s:g} s")
            self.clock.sleep(1.0)

    def _stop_and_home(self) -> None:
        """Success or giving up: the run stops where it is and both arms go home. A refused home is logged."""
        if self.options.observe_only:
            self.log.write("would", "would stop the run, then home both arms", action="stop_and_home")
            return
        if self._status.get("state") == "running":
            self._try("stop")
        self._settle(("idle", "error"))
        self._run_going = False
        if not self._try("home"):
            self.homed = False
            self.log.write("note", "the station refused to home: the arms hold where they are")
            return
        self.homed = True
        self._settle(("idle", "error"), tolerant=True)

    def _look_now(self) -> ProbeResult | None:
        """A probe taken for a guide when no earlier one was informative; None when the scene cannot be seen."""
        self._probe_n += 1
        result = self._probe_job(self._probe_n, False, watch=False)
        return result if result.verdict is not None else None

    def _choose_prompt(self) -> tuple[str | None, str | None]:
        """The next prompt: the rewriter's, from the last informative look at the scene, else the task file's list."""
        info = self._informative or self._last_probe or self._look_now()
        if info is not None and info.verdict is not None:
            tag = f"guide{self._guides_total + 1:02d}"
            paths = self.log.save_pictures(tag, info.pictures)
            try:
                prompt, source = self.critic.rewrite_prompt(info.verdict.result, info.pictures, self.history)
            except tc.CriticError as error:
                self.log.write("rewrite_failed", f"the rewriter failed: {error}", pictures=paths)
            else:
                if prompt is not None:
                    return prompt, source
        fallback = tc.next_from_matrix(self.spec, self.history)
        return (fallback, "matrix") if fallback is not None else (None, None)

    def _guide(self, reason: str) -> tuple[str, str] | None:
        """The first answer to a failure: a corrected prompt, in place, no homing."""
        assert self.ladder is not None
        self._drain()
        prompt, source = self._choose_prompt()
        if prompt is None:
            self.log.write("note", "every prompt was tried: restarting instead of guiding")
            return self._restart(reason)
        self.log.write(
            "guide",
            f"guide {self.ladder.guides + 1}: {prompt!r} (from {source}); because {reason}",
            prompt=prompt,
            source=source,
        )
        try:
            self.ctl.send("prompt", prompt=prompt)
        except sio.StationRefusedError:
            return None  # the state changed under us: the next tick looks again
        self.history.append(prompt)
        self._note_prompt("guide", prompt, source, reason)
        self._guides_total += 1
        status = self._status
        stamp = self.clock.now()
        self.ladder.note_guided(stamp, int(status.get("stalls") or 0), status.get("last_error"))
        self.watchdog.reset(stamp)
        self._chunks_seen = None
        self._next_probe_at = stamp + self.config.probe_every_s
        return None

    def _restart(self, reason: str) -> tuple[str, str] | None:
        """The second answer: stop, home, and start the next attempt with a new prompt on the same checkpoint."""
        assert self.ladder is not None
        self._drain()
        prompt, source = self._choose_prompt()
        if prompt is None and self.history:
            prompt, source = self.history[0], "original"
        if prompt is None:
            raise SupervisorError("every prompt was tried and the run has no first prompt to repeat")
        self.log.write(
            "restart",
            f"restart: attempt {self.ladder.attempt + 1} of {self.config.attempts} with {prompt!r} (from {source}); "
            f"because {reason}",
            prompt=prompt,
            source=source,
        )
        status = self._status
        if self.options.observe_only:
            self.ctl.send("stop")
            self.ctl.send("home")
            self.ctl.send("start", prompt=prompt, checkpoint=status.get("checkpoint"))
        else:
            if status.get("state") == "running":
                self._try("stop")
            self._settle(("idle", "error"))
            self._run_going = False
            self._try("home")
            self._settle(("idle", "error"), tolerant=True)
            before = self.clock.wall()
            try:
                self.ctl.send("start", prompt=prompt, checkpoint=status.get("checkpoint"))
            except sio.StationRefusedError as error:
                raise SupervisorError(f"the station refused the next attempt (the arms are home): {error}") from None
            started = self._settle(("running",))
            self._run_going = True
            self.run_id = started.get("run")
            self._bind(before)
            status = started
        self.history.append(prompt)
        self._note_prompt("restart", prompt, source, reason)
        stamp = self.clock.now()
        self.ladder.new_attempt(stamp, int(status.get("stalls") or 0), status.get("last_error"))
        self.watchdog.reset(stamp)
        self._chunks_seen = None
        self._next_probe_at = stamp + ARRIVAL_DELAY_S
        return None


# --- Building one ----------------------------------------------------------------------------------------------


def _gemini_reply(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def build_critic(
    args: argparse.Namespace, spec: tc.TaskSpec, log: Callable[[dict[str, Any]], None]
) -> tuple[tc.TaskCritic, tc.GeminiProgressCheck]:
    """The critic and the model compare: real backends (keys from the environment), or stand-ins that send nothing
    for --dry (every answer unsure, and a fixed prompt)."""
    if spec.vote_gate is None:
        raise SupervisorError(f"the task file for {spec.name!r} has no vote_gate section, which the supervisor needs")
    model = args.model or tc.GEMINI_MODEL
    rewriter: tc.Rewriter | None
    if args.dry:
        key = spec.vote_gate.answer_key
        stand_in = _gemini_reply(json.dumps({key: "unsure", "reason": "dry run: nothing was sent"}))
        gate = tc.GeminiVoteGate("dry", post=lambda url, headers, body: stand_in, log=log)
        progress = _gemini_reply(json.dumps({tc.PROGRESS_KEY: "unsure", "reason": "dry run: nothing was sent"}))
        compare = tc.GeminiProgressCheck("dry", post=lambda url, headers, body: progress, log=log)
        written = _gemini_reply("(dry run: nothing was written)")
        if args.rewriter == "gemini":
            rewriter = tc.GeminiRewriter("dry", post=lambda url, headers, body: written, log=log)
        elif args.rewriter == "claude":
            reply = {"content": [{"type": "text", "text": "(dry run: nothing was written)"}]}
            rewriter = tc.ClaudeRewriter("dry", post=lambda url, headers, body: reply, log=log)
        else:
            rewriter = None
        return tc.TaskCritic(spec, gate, rewriter), compare
    gate = tc.GeminiVoteGate(model=model, log=log)
    compare = tc.GeminiProgressCheck(model=model, log=log)
    if args.rewriter == "gemini":
        rewriter = tc.GeminiRewriter(model=model, log=log)
    elif args.rewriter == "claude":
        rewriter = tc.ClaudeRewriter(log=log)
    else:
        rewriter = None
    return tc.TaskCritic(spec, gate, rewriter), compare


def confirm_send(args: argparse.Namespace, spec: tc.TaskSpec, logs: pathlib.Path) -> None:
    """Says what is about to leave this computer and asks; refuses without a person to ask unless --yes."""
    if args.dry:
        print("Dry run: stand-ins answer and nothing is sent.")
        return
    needed = ["GEMINI_API_KEY"] + (["ANTHROPIC_API_KEY"] if args.rewriter == "claude" else [])
    missing = [name for name in needed if not os.environ.get(name)]
    if missing:
        raise SupervisorError(
            f"set {' and '.join(missing)} in the environment first (type it with `read -rs`, not in a file)"
        )
    model = args.model or tc.GEMINI_MODEL
    config = spec.supervisor
    repeats = spec.vote_gate.repeats if spec.vote_gate else 1
    probes = (
        f"{config.frames_per_probe} pictures every {config.probe_every_s:g} s after the first {config.warmup_s:g} s"
    )
    lines = [
        "About to send pictures of the table outside this computer, after cropping them to it:",
        f"  Gemini (generativelanguage.googleapis.com, model {model}): the zoomed view, {probes}, "
        f"{repeats} questions each; key from GEMINI_API_KEY",
        "  Gemini: two pictures, when the task's objects stop moving, to confirm a stall "
        f"(at most one every {config.stuck_after_s:g} s)",
    ]
    if args.rewriter == "gemini":
        lines.append(f"  Gemini: the scene crop and the zoom, to write each new prompt (model {model})")
    elif args.rewriter == "claude":
        lines.append(f"  Claude (api.anthropic.com, model {tc.ANTHROPIC_MODEL}): the same, to write each new prompt")
    lines += [
        "A picture with a face in the crop is never sent.",
        f"Every picture sent is kept under {logs}; the requests without pictures are in requests.jsonl there.",
    ]
    print("\n".join(lines))
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise SupervisorError("no one to ask: rerun with --yes to send the pictures")
    if input("Send? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SupervisorError("not sent")


def cameras_needed(spec: tc.TaskSpec) -> list[str]:
    """Every camera the pictures are cut from: the task's cameras and the ones its zoomed views use."""
    return list(dict.fromkeys([*spec.cameras, *(view.camera for view in spec.views.values())]))


def log_folder(root: pathlib.Path, clock: Clock) -> pathlib.Path:
    stamp = clock.wall().astimezone().strftime("%Y%m%d-%H%M%S")
    return pathlib.Path(root).expanduser() / f"supervise-{stamp}"


def _station_file() -> dict[str, Any] | None:
    try:
        loaded = yaml.safe_load(DEFAULT_STATION_FILE.expanduser().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def rollouts_dir(args: argparse.Namespace) -> pathlib.Path:
    """The recorder's folder: --rollouts, else `rollouts.dir` of the station file, else Dream's default."""
    if args.rollouts:
        return args.rollouts.expanduser()
    folder = ((_station_file() or {}).get("rollouts") or {}).get("dir")
    return pathlib.Path(folder).expanduser() if folder else DEFAULT_ROLLOUTS.expanduser()


def recorder_enabled() -> bool | None:
    """Whether the station file turns the recorder on; None when it cannot be read."""
    station = _station_file()
    return None if station is None else bool((station.get("rollouts") or {}).get("enabled"))


def _raise_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def command_check(args: argparse.Namespace) -> int:
    """Looks at the station and the recorder and writes one picture of what the supervisor would see. Changes nothing
    and sends nothing."""
    spec = tc.load_task(args.task)
    clock = RealClock()
    client = sio.StationClient(args.url, timeout=10.0)
    problems = 0
    print(f"task: {spec.name}; cameras {cameras_needed(spec)}; {spec.supervisor}")
    status: dict[str, Any] = {}
    try:
        status = client.status()
    except sio.StationError as error:
        print(f"PROBLEM  the station agent: {error}")
        problems += 1
    else:
        print(
            f"ok       the station is {status.get('state')}; run {status.get('run')}, "
            f"prompt {status.get('prompt')!r}, checkpoint {status.get('checkpoint')}, chunks {status.get('chunks')}, "
            f"stalls {status.get('stalls')}, camera ages {status.get('cameras')}"
        )
        if status.get("last_error"):
            print(f"note     last_error: {status['last_error']}")
    enabled = recorder_enabled()
    if enabled is False:
        print(f"PROBLEM  rollouts.enabled is not true in {DEFAULT_STATION_FILE}: the recorder writes nothing to watch")
        problems += 1
    elif enabled is None:
        print(f"note     could not read {DEFAULT_STATION_FILE} to check rollouts.enabled")
    folder = rollouts_dir(args)
    print(f"recorder folder: {folder}")
    out = log_folder(args.logs, clock)
    out.mkdir(parents=True, exist_ok=True)
    frames: dict[str, np.ndarray] = {}
    elapsed = float(status.get("run_elapsed_s") or 0.0)
    for camera in cameras_needed(spec):
        source = sio.GrowingVideo(folder, camera)
        source.bind(clock.wall() - datetime.timedelta(seconds=elapsed) if status.get("state") == "running" else None)
        try:
            taken = source.latest(1)
        except sio.SourceError as error:
            print(f"PROBLEM  camera {camera}: {error}")
            problems += 1
            continue
        frames[camera] = taken[0].bgr
        height, width = taken[0].bgr.shape[:2]
        print(f"ok       camera {camera}: {width}x{height}, newest picture at {taken[0].t_s:.1f} s of its video")
    if all(camera in frames for camera in spec.cameras):
        try:
            pictures = tc.cut_pictures(spec, frames)
        except tc.CriticError as error:
            print(f"PROBLEM  cutting the pictures: {error}")
            problems += 1
        else:
            paths = EventLog(out, clock, echo=None).save_pictures("check", pictures)
            faces = any(
                tc.count_faces(frames[c] if spec.crop.get(c) is None else tc.crop_fractions(frames[c], spec.crop[c]))
                for c in spec.cameras
            )
            print(f"ok       the pictures it would send are {[str(out / p) for p in paths]}; faces found: {faces}")
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        print(f"{'ok' if os.environ.get(name) else 'note':8} {name} is {'set' if os.environ.get(name) else 'not set'}")
    print("PROBLEMS: none" if not problems else f"PROBLEMS: {problems}")
    return 1 if problems else 0


def _exit_code(report: Mapping[str, Any]) -> int:
    return {"done": 0, "ended": 0, "aborted": 0, "interrupted": 0, "gave_up": 3}.get(report["outcome"], 1)


def command_run(args: argparse.Namespace) -> int:
    """Attaches to the station's run and supervises it."""
    if args.observe_only and args.start:
        raise SupervisorError("--observe-only makes no control call, so it cannot --start a run")
    spec = tc.load_task(args.task)
    clock = RealClock()
    folder = log_folder(args.logs, clock)
    confirm_send(args, spec, folder)
    log = EventLog(folder, clock)
    critic, compare = build_critic(args, spec, log.request)
    recorder = rollouts_dir(args)
    sources = {camera: sio.GrowingVideo(recorder, camera, clock=clock.now) for camera in cameras_needed(spec)}
    with contextlib.suppress(OSError, AttributeError):
        os.nice(10)  # the arm thread must never wait for this
    signal.signal(signal.SIGTERM, _raise_interrupt)
    options = Options(observe_only=args.observe_only, start=args.start, prompt=args.prompt, wait_s=args.wait_s)
    with KeyReader() as keys, concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        supervisor = Supervisor(
            spec,
            critic,
            sio.StationClient(args.url),
            sources,
            log=log,
            clock=clock,
            compare=compare,
            options=options,
            keys=keys.get,
            pool=pool,
        )
        report = supervisor.run()
    return _exit_code(report)


def start_commanding(station: sio.FakeStation, speed: float) -> threading.Event:
    """Makes a fake run send the arms a chunk every second, as a real one does. Set the returned event to stop it."""
    stop = threading.Event()

    def commanding() -> None:
        while not stop.wait(1.0 / speed):
            if station.state == "running":
                station.add_chunks(1)

    threading.Thread(target=commanding, daemon=True).start()
    return stop


def command_rehearse(args: argparse.Namespace) -> int:
    """Replays recorded video into the supervisor against a fake station: nothing moves, and what the supervisor
    does is logged. With real models it sends the pictures, after asking."""
    spec = tc.load_task(args.task)
    clock = RealClock(args.speed)
    folder = log_folder(args.logs, clock)
    cameras = cameras_needed(spec)
    videos: dict[str, pathlib.Path] = {}
    for item in args.video:
        name, path = item.split("=", 1) if "=" in item else (cameras[0], item)
        videos[name] = pathlib.Path(path).expanduser()
    if sorted(videos) != sorted(cameras):
        raise SupervisorError(f"give one --video for each of {cameras}, as {cameras[0]}=PATH")
    confirm_send(args, spec, folder)
    log = EventLog(folder, clock)
    critic, compare = build_critic(args, spec, log.request)
    prompt = spec.prompts[0].text if spec.prompts else spec.goal
    station = sio.FakeStation(clock=clock.now, prompt=prompt, state="running")
    sources = {camera: sio.ReplayVideo(path, clock=clock.now, start_s=args.start_s) for camera, path in videos.items()}
    stop = start_commanding(station, args.speed)
    options = Options(observe_only=args.observe_only, wait_s=5.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            supervisor = Supervisor(
                spec,
                critic,
                station.client(),
                sources,
                log=log,
                clock=clock,
                compare=compare,
                options=options,
                pool=pool,
            )
            report = supervisor.run()
    finally:
        stop.set()
    print(f"fake station calls: {station.calls}")
    return _exit_code(report)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    top.add_argument("--task", type=pathlib.Path, default=DEFAULT_TASK, help="the task file (default: usb-cables)")
    top.add_argument("--logs", type=pathlib.Path, default=DEFAULT_LOGS, help=f"where logs go (default {DEFAULT_LOGS})")
    commands = top.add_subparsers(dest="command", required=True)

    def models(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--model", help=f"the Gemini model (default {tc.GEMINI_MODEL})")
        sub.add_argument(
            "--rewriter", choices=["gemini", "claude", "none"], default="gemini", help="who writes the next prompt"
        )
        sub.add_argument("--dry", action="store_true", help="stand-ins answer (always unsure); nothing is sent")
        sub.add_argument("--yes", action="store_true", help="do not ask before sending pictures")
        sub.add_argument("--observe-only", action="store_true", help="make no control call; log what it would do")

    check = commands.add_parser("check", help="look at the station and the recorder; changes and sends nothing")
    check.add_argument("--url", default=sio.DEFAULT_URL, help=f"the station agent (default {sio.DEFAULT_URL})")
    check.add_argument("--rollouts", type=pathlib.Path, help="the recorder's folder (default: the station file's)")
    check.set_defaults(handler=command_check)

    run = commands.add_parser("run", help="watch a run and act on what it sees (or only log, with --observe-only)")
    models(run)
    run.add_argument("--url", default=sio.DEFAULT_URL, help=f"the station agent (default {sio.DEFAULT_URL})")
    run.add_argument("--rollouts", type=pathlib.Path, help="the recorder's folder (default: the station file's)")
    run.add_argument(
        "--start", action="store_true", help="start the first attempt if no run is going (needs --prompt)"
    )
    run.add_argument("--prompt", help="the instruction to start with, for --start")
    run.add_argument("--wait-s", type=float, default=120.0, help="seconds to wait for a run to be going (default 120)")
    run.set_defaults(handler=command_run)

    rehearse = commands.add_parser("rehearse", help="replay a recorded video against a fake station; moves nothing")
    models(rehearse)
    rehearse.add_argument("--video", action="append", required=True, help="a recorded video, or CAMERA=PATH for each")
    rehearse.add_argument(
        "--speed", type=float, default=1.0, help="replay speed (default 1; model calls are not sped up)"
    )
    rehearse.add_argument("--start-s", type=float, default=0.0, help="start the replay this far into the video")
    rehearse.set_defaults(handler=command_rehearse)
    return top


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        code = args.handler(args)
    except (SupervisorError, tc.CriticError, tc.TaskFileError) as error:
        sys.exit(f"supervise: {error}")
    sys.exit(code)


if __name__ == "__main__":
    main()
