"""The Prompt Lab's arm worker: the one process that owns the arms and the cameras (`dhk-lab` starts it).

It runs in the Dream library's environment (`~/dream-env/bin/python tools/lab_worker.py --frames-dir DIR`), because
Dream's drivers and run loop live there. `src/dream_hackathon/lab.py` speaks to it in JSON lines: commands on stdin
(`weightless`, `hold`, `home`, `power_off`, `run`, `stop`, `shutdown`), events on stdout (`ready`, `state`,
`run_started`, `run_ended`, `error`, `log`). The latest JPEG of each camera goes to `DIR/<camera>.jpg` for the live
view, and a run's cameras are recorded at full capture size into `<out_dir>/<camera>.mp4`.

What it does that `dreamcli robot run` does not:

- A run starts from where the arms are (the loop blends from the measured pose into the model's first move) instead of
  homing first, and ends with the arms holding there instead of homing and switching off.
- Weightless: both arms' MIT frames drop to no position gain with the gravity torque kept, which is what lets a
  hand place them. It relies on Dream 1.0.37's `MotorArm`/Damiao internals (`robots.motor_arm`, `robots.damiao`), so
  it refuses with a clear message when they differ, and it is for Metal arms only.
- Home is `robots.home.return_to_home` over a longer time than `dreamcli` uses.

`--fake` runs the same protocol on synthetic cameras and no arms, for trying the dashboard on a laptop.

Nothing here runs unless a person presses a button on the dashboard, and the arms are opened only by the first command
that needs them.
"""

import argparse
import asyncio
import contextlib
import json
import os
import pathlib
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

STATION_CONFIG = "~/.dreamlabs/station.yaml"
LIVE_FPS = 10
LIVE_MAX_WIDTH = 640
RECORD_FPS = 30
## Where `shutdown` homes the arms before switching them off, seconds: slow, like the Home button.
SHUTDOWN_HOME_S = 8.0
## Weightless: the joints' velocity gain, N.m s/rad. Position gain is 0, so the arm follows a hand freely; the
## velocity feedforward (the command tracks the measured pose) cancels most of this damping.
WEIGHTLESS_KD = 0.1
WEIGHTLESS_HZ = 50


class Backend(Protocol):
    """What the worker needs of its arms and cameras: the Dream one, or the fake."""

    camera_names: list[str]

    def start(self) -> None: ...
    def frames(self) -> dict[str, np.ndarray]: ...
    def weightless(self) -> None: ...
    def hold(self) -> None: ...
    def home(self, seconds: float) -> bool: ...
    def power_off(self) -> None: ...
    def arms_ok(self) -> bool: ...
    async def run(self, prompt: str, checkpoint: str, dry_run: bool, stop: asyncio.Event) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------- the real thing


class DreamBackend:
    """Dream's own readers, arms and run loop, in this process."""

    def __init__(self) -> None:
        from robots import bodies, config, followers, home, types
        from robots import run as dream_run

        self._bodies, self._config, self._followers = bodies, config, followers
        self._home, self._types, self._run = home, types, dream_run
        path = pathlib.Path(STATION_CONFIG).expanduser()
        if not path.is_file():
            raise RuntimeError(f"{path} is missing: the organizers run 'dreamcli setup' for the station")
        self.app = config.load(path)
        self.body = bodies.for_config(self.app)
        self.readers: list[Any] = []
        self.arms: dict[str, Any] = {}
        self.camera_names: list[str] = []
        self._weightless: _Weightless | None = None
        self._arm_lock = threading.Lock()

    def start(self) -> None:
        """Opens and starts the cameras; the arms stay closed until a command needs them."""
        self.readers = self._run.make_cameras(self.app)
        for reader in self.readers:
            reader.start()
        self.camera_names = [reader.name for reader in self.readers]

    def frames(self) -> dict[str, np.ndarray]:
        out = {}
        for reader in self.readers:
            frame = reader.frame
            if frame is not None:
                out[reader.name] = frame.data
        return out

    def _open_arms(self) -> None:
        with self._arm_lock:
            if self.arms and self.arms_ok():
                return
            self.arms = {}
            for name, arm_config in (("left", self.app.follower_l), ("right", self.app.follower_r)):
                self.arms[name] = self.body.open_arm(arm_config, name)

    def arms_ok(self) -> bool:
        return bool(self.arms) and all(arm.is_alive() for arm in self.arms.values())

    def weightless(self) -> None:
        self._open_arms()
        if self._weightless is None:
            self._weightless = _Weightless(self.arms, self._types)
        self._weightless.start()

    def hold(self) -> None:
        if self._weightless is not None:
            self._weightless.stop()

    def home(self, seconds: float) -> bool:
        self._open_arms()
        self.hold()
        return self._home.return_to_home(self.arms, duration_s=seconds)

    def power_off(self) -> None:
        self.hold()
        failed = []
        for name, arm in self.arms.items():
            try:
                arm.torque_off()
            except Exception as error:
                failed.append(f"{name}: {error}")
        self.arms = {}
        if failed:
            raise RuntimeError("an arm may still be powered, support it: " + "; ".join(failed))

    async def run(self, prompt: str, checkpoint: str, dry_run: bool, stop: asyncio.Event) -> None:
        """One run of the loop on the open arms: no home move before it, none after it, no switch-off."""
        from robots import control, errors

        run, followers = self._run, self._followers
        app = run.with_task(self.app, prompt)
        run.check_tool_point(app, skip=False)
        dream_config = run.policy_config(app, checkpoint)
        if "checkpoint" not in dream_config:
            raise RuntimeError(f"no checkpoint for a {app.body} body: set one on the prompt")
        arm_kinematics = run.make_kinematics(app)
        client = await asyncio.to_thread(run.open_client, dream_config)
        try:
            if not await asyncio.to_thread(client.wait_for_ik, run.TRAJ_WARMUP_S):
                raise RuntimeError("the model's IK did not compile in time; try again")
            if stop.is_set():
                return
            await asyncio.to_thread(self._open_arms)
            arms = self.arms
            if dry_run:
                arms = {
                    name: followers.FollowerArm(name, run.DryRun(arm), home_rad=arm.home_rad)
                    for name, arm in self.arms.items()
                }
            if any(reader.exposure is not None for reader in self.readers):
                await asyncio.sleep(run.EXPOSURE_SETTLE_S)
            for reader in self.readers:
                reader.set_exposure_hold(True)
            loop = control.ControlLoop(
                followers=arms,
                cameras=self.readers,
                client=client,
                task=app.task,
                kinematics=arm_kinematics,
                inference_log_path=app.inference_log_path,
                camera_owner=False,
                recorder=None,
                **run.loop_settings(app),
            )
            try:
                async with loop:
                    loop_task = asyncio.create_task(loop.run(), name="lab-run-loop")
                    stop_task = asyncio.create_task(stop.wait(), name="lab-run-stop")
                    await asyncio.wait({loop_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
                    stop_task.cancel()
                    if loop_task.done():
                        loop_task.result()
                    loop_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await loop_task
            finally:
                for reader in self.readers:
                    with contextlib.suppress(Exception):
                        reader.set_exposure_hold(False)
            if not dry_run:
                for arm in self.arms.values():
                    with contextlib.suppress(errors.RobotsError):
                        arm.hold()
        finally:
            control.run_teardown("close the dream client", client.close)

    def close(self) -> None:
        for reader in self.readers:
            with contextlib.suppress(Exception):
                reader.stop()


class _Weightless:
    """Both arms with no position gain and the gravity torque kept, so a hand can place them.

    Each arm's driver is a `MotorArm` over a Damiao chain whose `_gains` hold each joint's MIT (kp, kd). While this
    runs those are (0, WEIGHTLESS_KD) and a thread keeps every arm's target on its own measured pose, so the
    velocity feedforward follows the arm instead of fighting the hand. stop() puts the gains back and the arms hold
    wherever they were left.
    """

    def __init__(self, arms: dict[str, Any], types: Any) -> None:
        self._arms, self._types = arms, types
        self._saved: dict[str, dict[str, tuple[float, float]]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        for name, arm in arms.items():
            chain = getattr(getattr(arm, "_driver", None), "chain", None)
            if chain is None or not isinstance(getattr(chain, "_gains", None), dict) or not hasattr(chain, "_gravity"):
                raise RuntimeError(
                    f"weightless is not available on this arm ({name}): it needs a Metal arm on Dream's Damiao driver"
                )

    def start(self) -> None:
        if self._thread is not None:
            return
        for name, arm in self._arms.items():
            chain = arm._driver.chain
            self._saved[name] = dict(chain._gains)
            chain._gains.update({joint: (0.0, WEIGHTLESS_KD) for joint in chain._joints})
        self._stop.clear()
        self._thread = threading.Thread(target=self._follow, name="lab-weightless", daemon=True)
        self._thread.start()

    def _follow(self) -> None:
        while not self._stop.wait(1.0 / WEIGHTLESS_HZ):
            for arm in self._arms.values():
                try:
                    state = arm.read()
                    gripper = min(max(state.gripper, 0.0), 1.0)
                    arm.write(self._types.ArmCommand(positions_rad=state.positions_rad, gripper=gripper))
                except Exception as error:
                    print(f"weightless: {error}", file=sys.stderr)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        for name, arm in self._arms.items():
            arm._driver.chain._gains.update(self._saved.get(name, {}))
        for arm in self._arms.values():
            with contextlib.suppress(Exception):
                state = arm.read()
                arm.write(
                    self._types.ArmCommand(
                        positions_rad=state.positions_rad, gripper=min(max(state.gripper, 0.0), 1.0)
                    )
                )
        self._saved = {}


# ---------------------------------------------------------------------- the fake, for a laptop


class FakeBackend:
    """Synthetic cameras and no arms: the same protocol, so the dashboard can be tried anywhere."""

    camera_names = ["overhead", "wrist_l", "wrist_r"]

    def __init__(self) -> None:
        self._open = False
        self._t0 = time.monotonic()

    def start(self) -> None:
        pass

    def frames(self) -> dict[str, np.ndarray]:
        phase = (time.monotonic() - self._t0) * 40
        out = {}
        for i, name in enumerate(self.camera_names):
            frame = np.zeros((360, 640, 3), np.uint8)
            frame[..., i] = 90
            x = int(phase * (i + 1)) % 600
            frame[150:210, x : x + 40] = 255
            out[name] = frame
        return out

    def weightless(self) -> None:
        self._open = True
        time.sleep(0.3)

    def hold(self) -> None:
        time.sleep(0.2)

    def home(self, seconds: float) -> bool:
        self._open = True
        time.sleep(min(seconds, 2.0))
        return True

    def power_off(self) -> None:
        self._open = False

    def arms_ok(self) -> bool:
        return True

    async def run(self, prompt: str, checkpoint: str, dry_run: bool, stop: asyncio.Event) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=8)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------- video


def jpeg(rgb: np.ndarray) -> bytes:
    """A live-view JPEG of one RGB frame, at most LIVE_MAX_WIDTH wide."""
    import cv2

    height, width = rgb.shape[:2]
    if width > LIVE_MAX_WIDTH:
        rgb = cv2.resize(rgb, (LIVE_MAX_WIDTH, round(height * LIVE_MAX_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("could not encode a frame")
    return buffer.tobytes()


class Recorder:
    """A run's cameras into `<out_dir>/<camera>.mp4`, at capture size, one encoder per camera.

    @param out_dir The run's folder.
    @param frames Returns the latest RGB frame per camera.
    """

    def __init__(self, out_dir: pathlib.Path, frames: Callable[[], dict[str, np.ndarray]]) -> None:
        self._dir, self._frames = out_dir, frames
        self._outputs: dict[str, tuple[Any, Any]] = {}
        self._stop = threading.Event()
        self.error: str = ""
        self._thread = threading.Thread(target=self._run, name="lab-recorder", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _encoder(self, name: str, rgb: np.ndarray) -> tuple[Any, Any]:
        import av

        container = av.open(str(self._dir / f"{name}.mp4"), "w")
        stream = container.add_stream("libx264", rate=RECORD_FPS)
        stream.width, stream.height = rgb.shape[1] // 2 * 2, rgb.shape[0] // 2 * 2
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "18"}
        return container, stream

    def _run(self) -> None:
        import av

        tick = 0
        started = time.monotonic()
        try:
            while not self._stop.is_set():
                for name, rgb in self._frames().items():
                    if name not in self._outputs:
                        self._outputs[name] = self._encoder(name, rgb)
                    container, stream = self._outputs[name]
                    frame = av.VideoFrame.from_ndarray(
                        np.ascontiguousarray(rgb[: stream.height, : stream.width]), format="rgb24"
                    )
                    frame.pts = tick
                    for packet in stream.encode(frame):
                        container.mux(packet)
                tick += 1
                wait = started + tick / RECORD_FPS - time.monotonic()
                if wait > 0:
                    self._stop.wait(wait)
        except Exception as error:
            self.error = f"recording failed: {error}"
        finally:
            for container, stream in self._outputs.values():
                with contextlib.suppress(Exception):
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


# ---------------------------------------------------------------------- the worker


class Worker:
    """Reads commands, drives the backend, and prints events."""

    def __init__(self, backend: Backend, frames_dir: pathlib.Path, events: Any) -> None:
        self.backend, self.frames_dir, self._events = backend, frames_dir, events
        self._emit_lock = threading.Lock()
        self.state = "idle"
        self._run_task: asyncio.Task | None = None
        self._stop_run = asyncio.Event()
        self._done = asyncio.Event()

    def emit(self, **event: Any) -> None:
        with self._emit_lock:
            self._events.write(json.dumps(event) + "\n")
            self._events.flush()

    def set_state(self, state: str, message: str = "") -> None:
        self.state = state
        self.emit(event="state", state=state, message=message)

    def _live_loop(self) -> None:
        """Writes each camera's latest frame for the page, LIVE_FPS times a second."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        while not self._done.is_set():
            for name, rgb in self.backend.frames().items():
                try:
                    tmp = self.frames_dir / f".{name}.tmp"
                    tmp.write_bytes(jpeg(rgb))
                    os.replace(tmp, self.frames_dir / f"{name}.jpg")
                except Exception as error:
                    self.emit(event="error", message=f"live view: {error}")
                    time.sleep(2)
            time.sleep(1.0 / LIVE_FPS)

    async def serve(self, commands: asyncio.Queue) -> None:
        await asyncio.to_thread(self.backend.start)
        threading.Thread(target=self._live_loop, name="lab-live", daemon=True).start()
        self.emit(event="ready", cameras=list(self.backend.camera_names))
        while True:
            command = await commands.get()
            if command.get("cmd") == "shutdown":
                await self._shutdown()
                return
            try:
                await self._handle(command)
            except Exception as error:
                self.emit(event="error", message=f"{command.get('cmd')}: {error}")
                self.set_state(self._settled(), f"{command.get('cmd')} failed: {error}")

    def _settled(self) -> str:
        """The state to fall back to after a failure: holding while the arms answer, else idle."""
        return "holding" if self.backend.arms_ok() else "idle"

    async def _handle(self, command: dict[str, Any]) -> None:
        name = command.get("cmd")
        if name == "stop":
            self._stop_run.set()
        elif self.state == "running":
            self.emit(event="log", message=f"{name} ignored: a run is going")
        elif name == "weightless":
            await asyncio.to_thread(self.backend.weightless)
            self.set_state("weightless", "weightless: place the arms, then press Hold here")
        elif name == "hold":
            await asyncio.to_thread(self.backend.hold)
            self.set_state("holding", "holding where the arms are")
        elif name == "home":
            seconds = float(command.get("seconds") or 10.0)
            self.set_state("homing", "moving home")
            ok = await asyncio.to_thread(self.backend.home, seconds)
            self.set_state(self._settled(), "home reached" if ok else "an arm did not reach home: check the log")
        elif name == "power_off":
            await asyncio.to_thread(self.backend.power_off)
            self.set_state("idle", "the arms are off")
        elif name == "run":
            self._stop_run = asyncio.Event()  # before the task starts, so an early Stop is not lost
            self._run_task = asyncio.create_task(self._run(command), name="lab-run")
        else:
            self.emit(event="error", message=f"unknown command {name!r}")

    async def _run(self, command: dict[str, Any]) -> None:
        run_id = command["run_id"]
        out_dir = pathlib.Path(command["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        keys = ("run_id", "prompt", "checkpoint", "dry_run", "max_seconds")
        (out_dir / "run.json").write_text(json.dumps({k: command.get(k) for k in keys}, indent=2))
        stop, timed_out = self._stop_run, []

        def time_is_up() -> None:
            timed_out.append(True)
            stop.set()

        status, error = "finished", ""
        recorder = Recorder(out_dir, self.backend.frames)
        timer = asyncio.get_running_loop().call_later(float(command["max_seconds"]), time_is_up)
        self.set_state("running", "running")
        try:
            recorder.start()
            self.emit(event="run_started", run_id=run_id)
            await self.backend.run(command["prompt"], command["checkpoint"], bool(command["dry_run"]), stop)
            if stop.is_set() and not timed_out:
                status = "stopped"
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            timer.cancel()
            await asyncio.to_thread(recorder.stop)
            if recorder.error and not error:
                error = recorder.error
            self.emit(event="run_ended", run_id=run_id, status=status, error=error)
            self.set_state(self._settled(), "run over: the arms hold where they are")

    async def _shutdown(self) -> None:
        """Stops a run, homes the arms slowly, switches them off."""
        if self._run_task is not None and not self._run_task.done():
            self._stop_run.set()
            with contextlib.suppress(Exception):
                await self._run_task
        try:
            if self.backend.arms_ok():
                self.emit(event="log", message="shutting down: moving the arms home")
                await asyncio.to_thread(self.backend.home, SHUTDOWN_HOME_S)
            await asyncio.to_thread(self.backend.power_off)
        except Exception as error:
            self.emit(event="error", message=f"shutdown: {error}. Check the arms: they may still be powered.")
        finally:
            self._done.set()
            await asyncio.to_thread(self.backend.close)


async def amain(args: argparse.Namespace, events: Any) -> None:
    loop = asyncio.get_running_loop()
    commands: asyncio.Queue = asyncio.Queue()

    def read_stdin() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                continue
            if isinstance(command, dict):
                loop.call_soon_threadsafe(commands.put_nowait, command)
        loop.call_soon_threadsafe(commands.put_nowait, {"cmd": "shutdown"})  # the server is gone

    threading.Thread(target=read_stdin, name="lab-stdin", daemon=True).start()
    loop.add_signal_handler(signal.SIGTERM, lambda: commands.put_nowait({"cmd": "shutdown"}))
    backend: Backend = FakeBackend() if args.fake else DreamBackend()
    worker = Worker(backend, pathlib.Path(args.frames_dir), events)
    await worker.serve(commands)


def main() -> None:
    parser = argparse.ArgumentParser(description="The Prompt Lab's arm worker (started by dhk-lab).")
    parser.add_argument("--frames-dir", required=True, help="where each camera's latest JPEG is written")
    parser.add_argument("--fake", action="store_true", help="synthetic cameras and no arms")
    args = parser.parse_args()
    # stdout carries only events: anything a library prints goes to stderr instead.
    events = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    # Ctrl+C in the server's terminal reaches this process too; shutdown comes from the server's command.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        asyncio.run(amain(args, events))
    except Exception as error:
        events.write(json.dumps({"event": "error", "message": f"the arm worker failed to start: {error}"}) + "\n")
        events.flush()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
