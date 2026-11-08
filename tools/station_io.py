"""Talking to a Dream station and reading what its recorder writes. Nothing here moves an arm on its own.

- `StationClient` is the station agent's HTTP API: `dreamcli robot up` serves it on 127.0.0.1:8091. It sends only
  what a caller asks it to, and only tools/supervise.py asks.
- `FakeStation` is a local stand-in with the same routes and a small state machine, for tests and rehearsals.
- `GrowingVideo` reads the newest run's video while it is still being written (the recorder flushes every fragment,
  so a reader can decode its tail about a third of a second behind); `ReplayVideo` plays a recorded video as if it
  were live, for rehearsals.

The agent keeps the cameras open and no route returns a picture, so the recorder is the only way a program outside
Dream sees what the cameras see. It writes video only while a run is going: with `rollouts.enabled: true` in the
station file, and not on a dry run.
"""

import dataclasses
import datetime
import http.server
import json
import math
import pathlib
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import av
import numpy as np

DEFAULT_URL = "http://127.0.0.1:8091"

## The recorder's video of one camera, relative to a run's folder (robots.rollout_episode.video_file).
VIDEO_DIR = "videos/chunk-000"


class StationError(RuntimeError):
    """The station could not be reached, or answered with an error."""


class StationRefusedError(StationError):
    """The station refused a request that is valid in another state (HTTP 409): a start while a run exists, a home
    while one is going. Asking again at once gets the same answer."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class StationClient:
    """The station agent's HTTP API.

    @see FakeStation for the routes and what each does.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float = 70.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ):
        """@param timeout Seconds a call may take. The agent itself waits up to 60 s for a stop or a home move."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener

    def _call(self, method: str, route: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if method == "GET" else json.dumps(body or {}).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with self._opener(request, timeout=self.timeout) as reply:
                payload = json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", "replace") if hasattr(error, "read") else ""
            try:
                message = str(json.loads(text).get("error", text))
            except (ValueError, AttributeError):
                message = text[:300]
            if error.code == 409:
                raise StationRefusedError(409, f"{method} {route} refused: {message}") from None
            raise StationError(f"{method} {route} failed (HTTP {error.code}): {message}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise StationError(
                f"cannot reach the station agent at {self.base_url} ({error}); is `dreamcli robot up` running?"
            ) from None
        except ValueError:
            raise StationError(f"{method} {route}: the reply is not JSON") from None
        if not isinstance(payload, dict):
            raise StationError(f"{method} {route}: the reply is not a JSON object")
        return payload

    def status(self) -> dict[str, Any]:
        """The station's state, prompt, checkpoint, run number, chunks, stalls, last_error and camera ages."""
        return self._call("GET", "/status")

    def stop(self) -> dict[str, Any]:
        """Cancels the run and returns once the arms are holding where they are; they stay powered."""
        return self._call("POST", "/stop")

    def home(self) -> dict[str, Any]:
        """Moves both arms to the home pose. Refused while a run is going."""
        return self._call("POST", "/home")

    def start(self, prompt: str | None = None, checkpoint: str | None = None) -> dict[str, Any]:
        """Starts a run from the current pose (no homing first). Refused unless the station is idle or in error."""
        body = {k: v for k, v in (("prompt", prompt), ("checkpoint", checkpoint)) if v is not None}
        return self._call("POST", "/start", body)

    def set_prompt(self, prompt: str) -> dict[str, Any]:
        """Stores the prompt; a running loop starts a fresh episode with it from where the arms are."""
        return self._call("POST", "/prompt", {"prompt": prompt})


# --- A stand-in station ------------------------------------------------------------------------------------------


class FakeStation:
    """A stand-in for the station agent: the same routes, a small state machine, and hooks to script what a run does.

    | Route | What it does |
    | --- | --- |
    | `GET /status` | the status dict |
    | `POST /start` | idle or error to running, with `prompt` and `checkpoint` (kept when left out); 409 otherwise |
    | `POST /stop` | running to idle (the arms hold); a no-op when idle; 409 while homing |
    | `POST /home` | idle to idle, the arms home; 409 while a run exists |
    | `POST /prompt` | stores the prompt, and records it as applied when a run is going; 400 without one |
    | `POST /reset` | error to idle |

    Every POST is kept in `calls`, so a test can assert what was sent, and in what order.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        prompt: str = "",
        checkpoint: str = "dreamlabs/dream-1-beta",
        state: str = "idle",
    ):
        self._clock = clock
        self._lock = threading.RLock()
        self.state = state
        self.prompt = prompt
        self.checkpoint = checkpoint
        self.run = 1 if state == "running" else 0
        self.chunks = 0
        self.stalls = 0
        self.last_error: dict[str, Any] | None = None
        self.camera_age_s: float | None = 0.05
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.applied_prompts: list[str] = []
        self._started_s: float | None = clock() if state == "running" else None
        self._refusals: dict[str, str] = {}
        self._server: http.server.ThreadingHTTPServer | None = None

    # --- scripting ----------------------------------------------------------------------------------------------

    def add_chunks(self, n: int = 1) -> None:
        """The loop got n more chunks from the model: the arms are being commanded."""
        with self._lock:
            self.chunks += n

    def add_stalls(self, n: int = 1) -> None:
        with self._lock:
            self.stalls += n

    def fail(self, message: str = "the model stopped answering") -> None:
        """The run ended with an error, as the agent reports one."""
        with self._lock:
            self.state = "error"
            self.last_error = {"type": "ControlLoopError", "message": message, "code": None, "reason": None}

    def end_run(self) -> None:
        """The run ended by itself (a step limit, say): the station is idle."""
        with self._lock:
            self.state = "idle"

    def refuse(self, route: str, message: str = "scripted refusal") -> None:
        """The next POST to `route` is refused with a 409."""
        with self._lock:
            self._refusals[route] = message

    # --- the routes ---------------------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "prompt": self.prompt or None,
                "checkpoint": self.checkpoint,
                "run": self.run,
                "chunks": self.chunks,
                "stalls": self.stalls,
                "arms": {"left": True, "right": True},
                "cameras": {"scene": self.camera_age_s, "wrist_l": self.camera_age_s, "wrist_r": self.camera_age_s},
                "session_id": f"fake-{self.run}" if self.run else None,
                "last_error": self.last_error,
                "run_elapsed_s": None if self._started_s is None else round(self._clock() - self._started_s, 1),
            }

    def handle(self, route: str, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """One POST: `(http status, reply)`."""
        with self._lock:
            self.calls.append((route, dict(body)))
            if route in self._refusals:
                return 409, {"error": self._refusals.pop(route)}
            if route == "/start":
                if self.state not in ("idle", "error"):
                    return 409, {"error": f"start refused: station is {self.state}"}
                prompt = body.get("prompt") or self.prompt
                checkpoint = body.get("checkpoint") or self.checkpoint
                if not prompt:
                    return 409, {"error": "start refused: no prompt: none given and none stored"}
                self.prompt, self.checkpoint = prompt, checkpoint
                self.state, self.run, self.chunks, self.stalls, self.last_error = "running", self.run + 1, 0, 0, None
                self._started_s = self._clock()
                return 200, self.status()
            if route == "/stop":
                if self.state == "homing":
                    return 409, {"error": "stop refused: a home move is in progress; it ends within seconds"}
                if self.state == "running":
                    self.state = "idle"
                return 200, self.status()
            if route == "/home":
                if self.state in ("running", "starting", "stopping"):
                    return 409, {"error": f"home refused: station is {self.state}"}
                self.last_error = None
                return 200, self.status()
            if route == "/reset":
                if self.state == "error":
                    self.state, self.last_error = "idle", None
                return 200, self.status()
            if route == "/prompt":
                prompt = body.get("prompt") if "prompt" in body else None
                if "prompt" not in body:
                    return 400, {"error": "body must be a JSON object with a 'prompt' key"}
                if not isinstance(prompt, str) or not prompt.strip():
                    return 400, {"error": "prompt must be a non-empty string"}
                self.prompt = prompt
                if self.state == "running":
                    self.applied_prompts.append(prompt)
                return 200, self.status()
            return 404, {"error": f"no such route: POST {route}"}

    # --- serving it ---------------------------------------------------------------------------------------------

    def serve(self, host: str = "127.0.0.1", port: int = 0) -> str:
        """Starts the HTTP server on its own thread and returns its base URL. Call close() when done."""
        station = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _reply(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] == "/status":
                    self._reply(200, station.status())
                else:
                    self._reply(404, {"error": f"no such route: GET {self.path}"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw.strip() else {}
                    if not isinstance(body, dict):
                        raise ValueError("request body must be a JSON object")
                except ValueError as error:
                    self._reply(400, {"error": str(error)})
                    return
                status, payload = station.handle(self.path.split("?", 1)[0], body)
                self._reply(status, payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = http.server.ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://{host}:{self._server.server_address[1]}"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def client(self) -> "LocalClient":
        """A client for this station that calls it directly, with no socket: the same methods and errors as
        StationClient."""
        return LocalClient(self)


class LocalClient:
    """StationClient's methods, answered by a FakeStation in this process."""

    def __init__(self, station: FakeStation):
        self._station = station

    def _post(self, route: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        status, payload = self._station.handle(route, body or {})
        if status == 409:
            raise StationRefusedError(409, f"POST {route} refused: {payload.get('error')}")
        if status >= 400:
            raise StationError(f"POST {route} failed (HTTP {status}): {payload.get('error')}")
        return payload

    def status(self) -> dict[str, Any]:
        return self._station.status()

    def stop(self) -> dict[str, Any]:
        return self._post("/stop")

    def home(self) -> dict[str, Any]:
        return self._post("/home")

    def start(self, prompt: str | None = None, checkpoint: str | None = None) -> dict[str, Any]:
        return self._post(
            "/start", {k: v for k, v in (("prompt", prompt), ("checkpoint", checkpoint)) if v is not None}
        )

    def set_prompt(self, prompt: str) -> dict[str, Any]:
        return self._post("/prompt", {"prompt": prompt})


# --- Where a probe's pictures come from ----------------------------------------------------------------------------


class SourceError(RuntimeError):
    """No picture can be read yet: the run has no recording, or the file has no frames."""


@dataclasses.dataclass(frozen=True)
class Frame:
    """One picture of a camera: `t_s` seconds into its video, `bgr` an OpenCV image."""

    t_s: float
    bgr: np.ndarray


class FrameSource(Protocol):
    """What the supervisor reads pictures from."""

    def bind(self, since: datetime.datetime | None) -> None:
        """Forget the run being read and look for the one started at or after `since` (UTC)."""

    def latest(self, count: int) -> list[Frame]:
        """The newest pictures, `count` of them spread over the last second, oldest first."""

    def stale_s(self) -> float | None:
        """Seconds since the video last grew, or None before the first read."""


def spread_indexes(n: int, count: int) -> list[int]:
    """`count` indexes spread evenly over `n` frames, the last included (fewer when there are fewer frames)."""
    if n <= 0:
        return []
    if count <= 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


def read_tail(path: pathlib.Path, window_frames: int, keep_groups: int = 4) -> tuple[list[Any], int, float]:
    """The last `window_frames` decoded frames of an MP4 that may still be growing.

    Reads the container's packets (cheap), keeps the last few groups of pictures, and decodes only those, so a probe
    costs the same however long the run has lasted. A fragment still being written ends the file early; what was
    complete is used.

    @return `(frames, packets_seen, fps)`: PyAV frames (convert the ones you need with `to_ndarray`), how many
        packets the file held, and its frame rate.
    @throws SourceError The file cannot be opened or holds no frames yet.
    """
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as error:
        raise SourceError(f"{path} cannot be opened yet: {error}") from None
    with container:
        if not container.streams.video:
            raise SourceError(f"{path} has no video stream yet")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30)
        groups: list[list[Any]] = []
        total = 0
        try:
            for packet in container.demux(stream):
                if packet.size == 0:
                    continue
                total += 1
                if packet.is_keyframe or not groups:
                    groups = [*groups[-(keep_groups - 1) :], [packet]]
                else:
                    groups[-1].append(packet)
        except av.error.FFmpegError:
            pass
        if not groups:
            raise SourceError(f"{path} holds no frames yet")
        stream.codec_context.flush_buffers()
        frames: list[Any] = []
        for packet in (p for group in groups for p in group):
            try:
                frames.extend(packet.decode())
            except av.error.FFmpegError:
                continue
        if not frames:
            raise SourceError(f"{path} holds no decodable frames yet")
        return frames[-window_frames:], total, fps


def _parse_time(text: str) -> datetime.datetime | None:
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.UTC)


class GrowingVideo:
    """The newest run's video for one camera, read while the recorder is still writing it.

    A run's folder is `<time>_run<N>` under the rollouts directory, holding `rollout.json` (with `started_at_utc`) and
    `videos/chunk-000/observation.images.<camera>/episode_000000.partial.mp4`, which becomes `episode_000000.mp4`
    when the run ends. A change of prompt mid-run ends the episode and begins a new one in a folder of its own
    (`<time>_run<N>_2`, `_3`, ...), so each read looks for the newest folder again and follows it.
    """

    def __init__(
        self,
        rollouts_dir: pathlib.Path,
        camera: str,
        *,
        window_s: float = 1.0,
        slack_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rollouts_dir = pathlib.Path(rollouts_dir).expanduser()
        self.camera = camera
        self.window_s = window_s
        self.slack_s = slack_s
        self._clock = clock
        self._since: datetime.datetime | None = None
        self._folder: pathlib.Path | None = None
        self._pinned = False
        self._packets = 0
        self._grew_at: float | None = None

    def bind(self, since: datetime.datetime | None) -> None:
        self._since, self._folder, self._pinned, self._packets, self._grew_at = since, None, False, 0, None

    @property
    def folder(self) -> pathlib.Path | None:
        """The run folder being read, once a read has found it."""
        return self._folder

    def pin(self, folder: pathlib.Path) -> None:
        """Reads this run folder from now on, whatever `bind` said (the audit reads a finished run's last frames)."""
        self._folder, self._pinned = pathlib.Path(folder), True
        self._packets, self._grew_at = 0, None

    def find_run(self) -> pathlib.Path | None:
        """The newest run folder started at or after the bound time (minus a little slack), if there is one."""
        best: tuple[datetime.datetime, pathlib.Path] | None = None
        if not self.rollouts_dir.is_dir():
            return None
        for folder in self.rollouts_dir.iterdir():
            info = folder / "rollout.json"
            if not info.is_file():
                continue
            try:
                started = _parse_time(json.loads(info.read_text(encoding="utf-8")).get("started_at_utc", ""))
            except (OSError, ValueError):
                continue
            if started is None:
                continue
            if self._since is not None and started < self._since - datetime.timedelta(seconds=self.slack_s):
                continue
            if best is None or started > best[0]:
                best = (started, folder)
        return best[1] if best else None

    def video_path(self, folder: pathlib.Path) -> pathlib.Path | None:
        base = folder / VIDEO_DIR / f"observation.images.{self.camera}"
        for name in ("episode_000000.partial.mp4", "episode_000000.mp4"):
            if (base / name).is_file():
                return base / name
        return None

    def latest(self, count: int) -> list[Frame]:
        folder = self._folder if self._pinned else (self.find_run() or self._folder)
        if folder is None:
            raise SourceError(
                f"no recording of this run in {self.rollouts_dir}: set rollouts.enabled: true in the station file "
                "and start a real run (a dry run records nothing)"
            )
        path = self.video_path(folder)
        if path is None:
            raise SourceError(f"{folder} has no video of camera {self.camera!r} yet")
        if folder != self._folder:  # a new episode: its recording grows from nothing
            self._folder, self._packets, self._grew_at = folder, 0, None
        window = max(count, math.ceil(self.window_s * 30) + 1)
        frames, packets, fps = read_tail(path, window)
        if packets != self._packets:
            self._packets, self._grew_at = packets, self._clock()
        elif self._grew_at is None:
            self._grew_at = self._clock()
        out = []
        for i in spread_indexes(len(frames), count):
            decoded = frames[i]
            if decoded.pts is not None and decoded.time_base is not None:
                t_s = float(decoded.pts * decoded.time_base)
            else:
                t_s = (packets - len(frames) + i) / fps
            out.append(Frame(t_s, decoded.to_ndarray(format="bgr24")))
        return out

    def stale_s(self) -> float | None:
        return None if self._grew_at is None else self._clock() - self._grew_at


class ReplayVideo:
    """A recorded video played as if it were live, for a rehearsal: `latest()` returns the pictures at the replay's
    current time, which runs from when the replay started (`speed` times faster than real time) and rests on the
    last picture once the video ends."""

    def __init__(
        self,
        path: pathlib.Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        speed: float = 1.0,
        start_s: float = 0.0,
        spacing_s: float = 0.25,
    ):
        self.path = pathlib.Path(path)
        self._clock = clock
        self.speed = speed
        self.start_s = start_s
        self.spacing_s = spacing_s
        self._t0 = clock()
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            self.fps = float(stream.average_rate or 30)
            self.duration_s = float(stream.duration * stream.time_base) if stream.duration else None
            if self.duration_s is None:
                self.duration_s = sum(1 for _ in container.demux(stream)) / self.fps

    def bind(self, since: datetime.datetime | None) -> None:
        """A replay has one video: binding restarts it from its first picture."""
        self._t0 = self._clock()

    def now_s(self) -> float:
        return min(self.duration_s or 0.0, self.start_s + (self._clock() - self._t0) * self.speed)

    @property
    def ended(self) -> bool:
        return self.duration_s is not None and self.now_s() >= self.duration_s - 1.0 / self.fps

    def _frame_at(self, t_s: float) -> Frame:
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            container.seek(int(max(0.0, t_s) / stream.time_base), stream=stream, backward=True)
            best = None
            for decoded in container.decode(stream):
                time_s = float(decoded.pts * stream.time_base)
                if best is None or time_s <= t_s + 0.5 / self.fps:
                    best = Frame(time_s, decoded.to_ndarray(format="bgr24"))
                if time_s >= t_s:
                    break
            if best is None:
                raise SourceError(f"{self.path} has no picture at {t_s:.1f} s")
            return best

    def latest(self, count: int) -> list[Frame]:
        now = self.now_s()
        times = [max(0.0, now - self.spacing_s * (count - 1 - i)) for i in range(count)]
        return [self._frame_at(t) for t in times]

    def stale_s(self) -> float | None:
        return 0.0
