"""The Prompt Lab: a browser dashboard that keeps prompts, runs one on the arms and saves what the cameras saw.

`uv run dhk-lab` serves it on `127.0.0.1:9100` (`ssh -L 9100:localhost:9100 <user>@<station>`); Ctrl+C stops it. It
replaces `dreamcli robot run --task ...`: the page keeps a list of prompts in SQLite, **Run** plays one on the arms
with Dream's run loop, the cameras are shown live, and each run is saved with its prompt and three full-resolution
videos. **Weightless** lets both arms be placed by hand, **Hold here** keeps them where they are, and **Home arms**
moves them slowly to Dream's home.

Two processes share the work, because Dream's drivers and run loop live in `~/dream-env` and this repo's code does not:

- this server (repo environment): the page, the database, the lock on the station and the worker's lifetime;
- the arm worker, `tools/lab_worker.py` (Dream's environment): the one process that owns the arms and the cameras.
  It takes commands as JSON lines on its stdin and answers with events on its stdout; the latest JPEG of each
  camera goes to a folder this server serves.

@see Lab, Worker, make_handler(), main()
"""

import contextlib
import http.server
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from dream_hackathon import lab_db, station_lock

PORT = 9100
ROOT_ENV = "DHK_LAB_DIR"
FAKE_ENV = "DHK_LAB_FAKE"
DEFAULT_ROOT = "~/hackathon_data/lab"
DREAM_PYTHON = "~/dream-env/bin/python"
WORKER_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "tools" / "lab_worker.py"
## How long the worker has to stop a run, move the arms home and switch them off when the server stops.
SHUTDOWN_WAIT_S = 120.0
## The server stops a run this long after the run's own limit, if the worker has not.
STOP_GRACE_S = 15.0
MAX_BODY_BYTES = 16384
REQUEST_TIMEOUT_S = 10

MAX_ALERTS = 8
STATES = ("starting", "idle", "weightless", "holding", "homing", "running", "stopped")
## Where each command may start from. Run asks for a held pose so a run never starts from a limp arm by accident.
ALLOWED = {
    "weightless": ("idle", "holding"),
    "hold": ("weightless",),
    "home": ("idle", "holding", "weightless"),
    "power_off": ("idle", "holding", "weightless"),
    "clear_fault": ("idle", "holding"),
    "run": ("holding",),
    "stop": ("running",),
}
## The state a command puts the page in at once; the worker's own `state` events confirm or correct it.
AFTER = {"weightless": "weightless", "hold": "holding", "home": "homing", "power_off": "idle", "run": "running"}

_LOOPBACK_HOST = re.compile(r"(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?")
_CAMERA_PATH = re.compile(r"/api/camera/([a-z0-9_]{1,40})\.jpg")
_VIDEO_PATH = re.compile(r"/api/runs/(\d+)/video/([a-z0-9_]{1,40})\.mp4")
_PROMPT_PATH = re.compile(r"/api/prompts/(\d+)(?:/(delete|run))?")
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
## Sent with every action by the page; a form on another site cannot add a header like it.
ACTION_HEADER = "X-Lab-Dashboard"


class LabBusy(lab_db.LabError):
    """A command the arms' current state does not allow."""


class WorkerLike(Protocol):
    """What Lab needs of its worker: the real Worker, or a test's."""

    def send(self, command: dict[str, Any]) -> None: ...
    def alive(self) -> bool: ...


class Worker:
    """The arm worker as a child process, spoken to in JSON lines.

    @param command The argv that starts it.
    @param on_event Called, on this class's reader thread, with each event the worker prints.
    @param on_exit Called once, on that thread, when the worker's output ends.
    """

    def __init__(
        self, command: list[str], on_event: Callable[[dict[str, Any]], None], on_exit: Callable[[int | None], None]
    ) -> None:
        self._process = subprocess.Popen(  # noqa: S603 - our own worker script
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._on_event, self._on_exit = on_event, on_exit
        self._write_lock = threading.Lock()
        threading.Thread(target=self._read, name="lab-worker-reader", daemon=True).start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                print(f"worker: {line.rstrip()}")
                continue
            if isinstance(event, dict):
                self._on_event(event)
        self._on_exit(self._process.wait())

    def send(self, command: dict[str, Any]) -> None:
        """Hands the worker one command.

        @throws lab_db.LabError The worker is gone.
        """
        assert self._process.stdin is not None
        with self._write_lock:
            try:
                self._process.stdin.write(json.dumps(command) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                raise lab_db.LabError("the arm worker is not running: restart dhk-lab") from None

    def alive(self) -> bool:
        return self._process.poll() is None

    def stop(self, wait_s: float) -> None:
        """Asks the worker to shut down (it homes the arms and switches them off), and kills it after `wait_s`."""
        if not self.alive():
            return
        with contextlib.suppress(lab_db.LabError):
            self.send({"cmd": "shutdown"})
        try:
            self._process.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            print("the arm worker did not finish in time; stopping it. Check the arms: they may still be powered.")
            self._process.kill()
            self._process.wait()


class Lab:
    """What the page shows and may do: the worker's state, the current run, the prompts and the runs.

    @param store The database.
    @param frames_dir Where the worker writes each camera's latest JPEG.
    """

    def __init__(self, store: lab_db.Store, frames_dir: pathlib.Path) -> None:
        self.store = store
        self.frames_dir = frames_dir
        self.worker: WorkerLike | None = None
        self.lock = threading.RLock()
        self.state = "starting"
        self.message = "the arm worker is starting"
        self.cameras: list[str] = []
        self.run: dict[str, Any] | None = None
        self._run_started = 0.0
        self.angles: dict[str, Any] = {}  # the Motors tab's last reading, per arm
        self.angles_at = 0.0
        self._watch_sent = 0.0
        self._kernel: tuple[float, dict[str, Any]] | None = None
        self.alerts: list[dict[str, Any]] = []  # what went wrong, newest last; the page shows them until dismissed
        self._alerts = 0
        self._timer: threading.Timer | None = None

    # ------------------------------------------------------------------ worker events

    def on_event(self, event: dict[str, Any]) -> None:
        """Applies one event the worker printed."""
        kind = event.get("event")
        with self.lock:
            if kind == "ready":
                cameras = event.get("cameras")
                self.cameras = [c for c in cameras if isinstance(c, str)] if isinstance(cameras, list) else []
                self._set("idle", "ready")
            elif kind == "state" and event.get("state") in STATES:
                self._set(event["state"], str(event.get("message") or ""))
            elif kind == "run_started":
                self._run_started = time.monotonic()
                self._arm_timer()
            elif kind == "run_ended":
                self._end_run(str(event.get("status") or "error"), str(event.get("error") or ""))
            elif kind == "angles" and isinstance(event.get("arms"), dict):
                self.angles, self.angles_at = event["arms"], time.monotonic()
            elif kind in ("error", "log"):
                self.message = str(event.get("message") or "")
                print(f"worker {kind}: {self.message}")
                if kind == "error":
                    self._alert(self.message)

    def on_exit(self, code: int | None) -> None:
        """The worker's output ended: whatever it was doing is over."""
        with self.lock:
            self._end_run("error", "the arm worker stopped")
            self._alert(f"The arm worker stopped (exit {code}). Restart dhk-lab and check the arms.")
            self._set("stopped", f"the arm worker stopped (exit {code}): restart dhk-lab. Check the arms.")

    def _alert(self, text: str) -> None:
        self._alerts += 1
        self.alerts = [
            *self.alerts[-(MAX_ALERTS - 1) :],
            {"id": self._alerts, "text": f"[{time.strftime('%H:%M:%S')}] {text}"},
        ]
        print(f"ALERT: {text}")

    def _set(self, state: str, message: str) -> None:
        self.state, self.message = state, message

    def _end_run(self, status: str, error: str) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.run is not None:
            if status == "error":
                self._alert(
                    f"Run {self.run['id']} failed ({self.run['prompt_text'][:80]!r}): {error or 'unknown error'}"
                )
            self.store.end_run(self.run["id"], status if status in lab_db.RUN_STATUSES else "error", error)
            self.run = None

    def _arm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self.run is None:
            return
        self._timer = threading.Timer(self.run["max_seconds"] + STOP_GRACE_S, self._watchdog, args=(self.run["id"],))
        self._timer.daemon = True
        self._timer.start()

    def _watchdog(self, run_id: int) -> None:
        with self.lock:
            if self.run is not None and self.run["id"] == run_id and self.worker is not None:
                print(f"run {run_id} outlasted its limit; stopping it")
                with contextlib.suppress(lab_db.LabError):
                    self.worker.send({"cmd": "stop"})

    # ------------------------------------------------------------------ commands from the page

    def command(self, name: str, **fields: Any) -> str:
        """Checks that the arms' state allows a command, then hands it to the worker.

        @return A sentence for the page.
        @throws LabBusy The state does not allow it.
        @throws lab_db.LabError The worker is gone, or the request is not acceptable.
        """
        with self.lock:
            if self.worker is None or not self.worker.alive():
                raise lab_db.LabError("the arm worker is not running: restart dhk-lab")
            if self.state not in ALLOWED[name]:
                raise LabBusy(f"{name.replace('_', ' ')} is not possible while the arms are: {self.state}")
            previous = self.state
            if name in AFTER:
                self._set(AFTER[name], "")
            try:
                self.worker.send({"cmd": name, **fields})
            except lab_db.LabError:
                self._set(previous, self.message)
                raise
            return {
                "weightless": "The arms are weightless: place them, then press Hold here.",
                "hold": "Holding the arms where they are.",
                "home": "Moving the arms home, slowly.",
                "power_off": "Parking the arms and switching them off.",
                "clear_fault": "Clearing the gripper fault.",
                "stop": "Stopping: the arms hold where they are.",
                "run": "Running.",
            }[name]

    def run_prompt(self, prompt_id: int, dry_run: bool, max_seconds: object) -> str:
        """Records a run of a saved prompt and starts it.

        @throws LabBusy The arms are not holding a pose.
        @throws lab_db.LabError The prompt is gone, the limit is wrong, or the worker is.
        """
        with self.lock:
            prompt = self.store.prompt(prompt_id)
            limit = lab_db.clean_max_seconds(max_seconds)
            if self.state not in ALLOWED["run"]:
                raise LabBusy(f"a run needs the arms holding a pose (press Hold here first); they are: {self.state}")
            row = self.store.start_run(
                prompt["text"], prompt["checkpoint"], prompt_id=prompt_id, dry_run=dry_run, max_seconds=limit
            )
            self.run = row
            try:
                return self.command(
                    "run",
                    run_id=row["id"],
                    prompt=row["prompt_text"],
                    checkpoint=row["checkpoint"],
                    dry_run=dry_run,
                    max_seconds=limit,
                    out_dir=row["dir"],
                )
            except lab_db.LabError as error:
                self._end_run("error", str(error))
                raise

    def watch_angles(self) -> dict[str, Any]:
        """What the Motors tab polls; it also asks the worker to keep reading for a few seconds more."""
        with self.lock:
            now = time.monotonic()
            if self.worker is not None and self.worker.alive() and now - self._watch_sent > 1.0:
                self._watch_sent = now
                with contextlib.suppress(lab_db.LabError):
                    self.worker.send({"cmd": "watch"})
            age = now - self.angles_at if self.angles_at else None
            return {"arms": self.angles if age is not None and age < 5 else {}, "age_s": age, "state": self.state}

    def kernel_usb_events(self) -> dict[str, Any]:
        """USB resets and disconnects in the kernel log over the last ten minutes (best effort, cached a few s)."""
        now = time.monotonic()
        if self._kernel is not None and now - self._kernel[0] < 5:
            return self._kernel[1]
        try:
            out = subprocess.run(
                ["journalctl", "-k", "--no-pager", "-o", "short-iso", "--since", "10 minutes ago"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            ).stdout
            words = ("reset", "disconnect")
            lines = [line for line in out.splitlines() if " usb " in line and any(w in line for w in words)]
            result: dict[str, Any] = {"count": len(lines), "last": lines[-8:], "error": ""}
        except (OSError, subprocess.SubprocessError) as error:
            result = {"count": 0, "last": [], "error": f"{type(error).__name__}: {error}"}
        self._kernel = (now, result)
        return result

    def status(self) -> dict[str, Any]:
        """What the page polls."""
        with self.lock:
            run = self.run
            elapsed = time.monotonic() - self._run_started if run and self._run_started else 0.0
            return {
                "state": self.state,
                "message": self.message,
                "alert": self.alerts[-1] if self.alerts else None,
                "alerts": list(self.alerts),
                "cameras": list(self.cameras),
                "run": None
                if run is None
                else {
                    "id": run["id"],
                    "prompt": run["prompt_text"],
                    "checkpoint": run["checkpoint"],
                    "dry_run": bool(run["dry_run"]),
                    "max_seconds": run["max_seconds"],
                    "elapsed_s": round(elapsed, 1),
                },
            }


# ---------------------------------------------------------------------- HTTP


def host_allowed(host: str | None) -> bool:
    """True when a request's Host header names the loopback address the page is served on."""
    return host is not None and _LOOPBACK_HOST.fullmatch(host.strip().lower()) is not None


def make_handler(lab: Lab) -> type[http.server.BaseHTTPRequestHandler]:
    """The request handler class for a Lab.

    Every request needs a loopback Host (so a page on another site cannot reach it through the SSH tunnel by name),
    and every POST, which is every action, the custom header and, when the browser sends one, a matching Origin.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S

        def log_message(self, *_: object) -> None:
            pass

        # -------------------------------------------------------------- replies

        def _send(self, code: int, ctype: str, body: bytes, extra: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: object, code: int = 200) -> None:
            self._send(code, "application/json", json.dumps(data).encode())

        def _fail(self, code: int, message: str) -> None:
            self._json({"ok": False, "error": message}, code)

        # -------------------------------------------------------------- GET

        def do_GET(self) -> None:  # noqa: N802
            if not host_allowed(self.headers.get("Host")):
                return self._fail(403, "forbidden")
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    return self._send(200, "text/html; charset=utf-8", PAGE_PATH.read_bytes())
                if path == "/api/status":
                    return self._json(lab.status())
                if path == "/api/can":
                    return self._json({**lab.watch_angles(), "kernel": lab.kernel_usb_events()})
                if path == "/api/angles":
                    return self._json(lab.watch_angles())
                if path == "/api/prompts":
                    return self._json(lab.store.prompts())
                if path == "/api/runs":
                    return self._json(lab.store.runs())
                match = _CAMERA_PATH.fullmatch(path)
                if match:
                    return self._camera(match.group(1))
                match = _VIDEO_PATH.fullmatch(path)
                if match:
                    return self._video(int(match.group(1)), match.group(2))
            except lab_db.LabError as error:
                return self._fail(404, str(error))
            self._fail(404, "not found")

        def _camera(self, role: str) -> None:
            if role not in lab.cameras:
                return self._fail(404, "no such camera")
            try:
                body = (lab.frames_dir / f"{role}.jpg").read_bytes()
            except OSError:
                return self._fail(503, "no frame yet")
            self._send(200, "image/jpeg", body)

        def _video(self, run_id: int, role: str) -> None:
            run = lab.store.run(run_id)
            path = pathlib.Path(run["dir"]) / f"{role}.mp4"
            if not path.is_file():
                return self._fail(404, "no such video")
            size = path.stat().st_size
            start, end = 0, size - 1
            match = _RANGE.fullmatch(self.headers.get("Range", "").strip())
            partial = False
            if match and (match.group(1) or match.group(2)):
                partial = True
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else end
                else:
                    start = max(0, size - int(match.group(2)))
                end = min(end, size - 1)
                if start > end:
                    return self._send(416, "text/plain", b"", {"Content-Range": f"bytes */{size}"})
            with path.open("rb") as file:
                file.seek(start)
                body = file.read(end - start + 1)
            extra = {"Accept-Ranges": "bytes"}
            if partial:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        # -------------------------------------------------------------- POST

        def do_POST(self) -> None:  # noqa: N802
            host = self.headers.get("Host")
            if not host_allowed(host):
                return self._fail(403, "forbidden")
            if not self.headers.get(ACTION_HEADER):
                return self._fail(403, "forbidden")
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{host}":
                return self._fail(403, "forbidden")
            body = self._body()
            if body is None:
                return
            path = self.path.split("?", 1)[0]
            try:
                self._json({"ok": True, "message": self._act(path, body)})
            except LabBusy as error:
                self._fail(409, str(error))
            except lab_db.LabError as error:
                self._fail(400, str(error))

        def _body(self) -> dict[str, Any] | None:
            length = self.headers.get("Content-Length", "0")
            if not length.isascii() or not length.isdigit():
                self._fail(400, "bad Content-Length")
                return None
            if int(length) > MAX_BODY_BYTES:
                self._fail(413, "too large")
                return None
            raw = self.rfile.read(int(length)) if int(length) else b""
            try:
                data = json.loads(raw) if raw else {}
            except ValueError:
                self._fail(400, "the body is not JSON")
                return None
            if not isinstance(data, dict):
                self._fail(400, "the body is not an object")
                return None
            return data

        def _act(self, path: str, body: dict[str, Any]) -> str:
            if path == "/api/prompts":
                lab.store.add_prompt(body.get("text"), body.get("checkpoint"))
                return "Prompt added."
            match = _PROMPT_PATH.fullmatch(path)
            if match:
                prompt_id, action = int(match.group(1)), match.group(2)
                if action == "delete":
                    lab.store.delete_prompt(prompt_id)
                    return "Prompt deleted."
                if action == "run":
                    dry_run = body.get("dry_run", False)
                    if not isinstance(dry_run, bool):
                        raise lab_db.LabError("dry_run is true or false")
                    return lab.run_prompt(prompt_id, dry_run, body.get("max_seconds"))
                lab.store.update_prompt(prompt_id, body.get("text"), body.get("checkpoint"))
                return "Prompt saved."
            commands = {
                "/api/arms/weightless": "weightless",
                "/api/arms/hold": "hold",
                "/api/arms/home": "home",
                "/api/arms/power_off": "power_off",
                "/api/arms/clear_fault": "clear_fault",
                "/api/stop": "stop",
            }
            if path in commands:
                return lab.command(commands[path])
            raise lab_db.LabError("not found")

    return Handler


PAGE_PATH = pathlib.Path(__file__).with_name("lab_page.html")


# ---------------------------------------------------------------------- start and stop


def worker_command(frames_dir: pathlib.Path, log_file: pathlib.Path | None = None) -> list[str]:
    """The argv of the arm worker: Dream's Python on a station, the fake worker when DHK_LAB_FAKE is set."""
    extra = ["--log-file", str(log_file)] if log_file is not None else []
    if os.environ.get(FAKE_ENV):
        return [sys.executable, str(WORKER_SCRIPT), "--fake", "--frames-dir", str(frames_dir), *extra]
    python = pathlib.Path(DREAM_PYTHON).expanduser()
    if not python.is_file():
        raise SystemExit(
            f"{python} is not there: the Dream library's environment is missing (docs/hardware_setup.md, 6)"
        )
    return [str(python), str(WORKER_SCRIPT), "--frames-dir", str(frames_dir), *extra]


def main() -> None:
    """Starts the Prompt Lab; Ctrl+C (or SIGTERM) stops it."""
    fake = bool(os.environ.get(FAKE_ENV))
    root = pathlib.Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT).expanduser()
    lock = None
    if not fake:
        try:
            lock = station_lock.acquire("dhk-lab")
        except station_lock.StationBusyError as error:
            raise SystemExit(str(error)) from None
    store = lab_db.Store(root)
    closed = store.close_unfinished()
    if closed:
        print(f"{closed} run(s) were left unfinished by an earlier stop; they are marked as errors.")
    frames_dir = pathlib.Path(tempfile.mkdtemp(prefix="dhk-lab-frames-"))
    lab = Lab(store, frames_dir)
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(lab))
    except OSError as error:
        shutil.rmtree(frames_dir, ignore_errors=True)
        raise SystemExit(f"could not serve on port {PORT} ({error}): is another dhk-lab running?") from None
    server.daemon_threads = True
    lab.worker = Worker(worker_command(frames_dir, root / "worker.log"), lab.on_event, lab.on_exit)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, name="lab-http", daemon=True).start()
    print(f"Prompt Lab: http://localhost:{PORT}  (from a laptop: ssh -L {PORT}:localhost:{PORT} <user>@<station>)")
    print("Ctrl+C stops it: a run stops, the arms go home slowly and switch off.")
    try:
        while not stop.wait(0.5):
            if isinstance(lab.worker, Worker) and not lab.worker.alive():
                break
    finally:
        print("stopping ...")
        server.shutdown()
        if isinstance(lab.worker, Worker):
            lab.worker.stop(SHUTDOWN_WAIT_S)
        lab.on_exit(None)
        shutil.rmtree(frames_dir, ignore_errors=True)
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
