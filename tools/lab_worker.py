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
import logging
import math
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
ENCODER_THREADS = 2
RECORD_FPS = 30
## Where `shutdown` homes the arms before switching them off, seconds: slow, like the Home button.
SHUTDOWN_HOME_S = 8.0
## The Metal arm's gripper motor id on its CAN bus, and Damiao's clear-error command byte (an `FF*7 xx` frame).
## The model's answers have been taking 3 to 9 s on a busy server, and Dream's loop discards any answer later than
## action_timeout_s (3 s) and holds still instead: the arms barely move. A late answer is still used here, up to
## these limits; the end effector's speed cap still bounds every move.
ACTION_TIMEOUT_S = 10.0
MAX_STALL_TICKS = 400  # empty-buffer ticks (20 Hz) tolerated before a run stops: 20 s
WARMUP_CHECKPOINT = "dreamlabs/dream-1-beta"  # any checkpoint of the body: the compile does not depend on it
GRIPPER_CAN_ID = 7
## The motors tab: how often the angles are read while a page watches, and how long a `watch` keeps that going.
ANGLES_HZ = 4
WATCH_S = 3.0
CLEAR_ERROR = 0xFB
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
    def clear_fault(self) -> str: ...
    def angles(self) -> dict[str, Any]: ...
    def health(self) -> str: ...
    def arms_ok(self) -> bool: ...
    async def run(
        self, prompt: str, checkpoint: str, dry_run: bool, stop: asyncio.Event, ready: Callable[[], None]
    ) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------- the real thing


def adapter_info(channel: str) -> dict[str, Any]:
    """What the kernel says about a CAN adapter's serial device: is it there, which USB port, at what speed."""
    real = os.path.realpath(channel)
    tty = os.path.basename(real)
    info: dict[str, Any] = {
        "channel": channel,
        "tty": tty,
        "present": os.path.exists(real),
        "usb_path": "",
        "speed": None,
    }
    with contextlib.suppress(Exception):
        usb_device = os.path.dirname(os.path.realpath(f"/sys/class/tty/{tty}/device"))
        info["usb_path"] = os.path.basename(usb_device)
        info["speed"] = int(pathlib.Path(usb_device, "speed").read_text())
    return info


class DreamBackend:
    """Dream's own readers, arms and run loop, in this process."""

    def __init__(self) -> None:
        from robots import bodies, config, followers, home, types
        from robots import run as dream_run

        self._bodies, self._config, self._followers = bodies, config, followers
        self._home, self._types, self._run = home, types, dream_run
        self.on_problem = lambda text: print(text, file=sys.stderr)  # the worker swaps in an alert
        self._monitor: dict[str, Any] = {}
        self._reported_dead: set[str] = set()
        self._stats: dict[str, dict[str, dict[str, Any]]] = {}  # per adapter, per motor: the Motors/CAN tabs' counts
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
        self._warm_up_ik()

    def _warm_up_ik(self) -> None:
        """Compiles Dream's IK now, while no arm is open.

        The compile is seconds of CPU that starve every other thread, and an open Metal gripper latches a fault after
        500 ms without a command. A run opens its own client, but JAX keeps what this one compiled in the process, so
        the run's compile is quick and the arms are never among the starved.
        """
        client = None
        try:
            app = self._run.with_task(self.app, "warm up")
            client = self._run.open_client(self._run.policy_config(app, WARMUP_CHECKPOINT))
            if not client.wait_for_ik(self._run.TRAJ_WARMUP_S):
                logging.warning("the IK did not compile in time; a run will compile it with the arms open")
        except Exception:
            logging.exception("could not warm up the IK; a run will compile it with the arms open")
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()

    def frames(self) -> dict[str, np.ndarray]:
        out = {}
        for reader in self.readers:
            frame = reader.frame
            if frame is not None:
                out[reader.name] = frame.data
        return out

    def health(self) -> str:
        """Why an open arm stopped, once per arm, or an empty string while all is well."""
        dead = {name for name, arm in list(self.arms.items()) if not arm.is_alive()}
        self._reported_dead &= dead  # an arm that was let go or came back may be reported again
        text = []
        for name in sorted(dead - self._reported_dead):
            self._reported_dead.add(name)
            reason = self.arms[name].why_stopped() or "it stopped answering (no reason given)"
            text.append(f"the {name} arm stopped: {reason}")
        return "; ".join(text)

    def angles(self) -> dict[str, Any]:
        """Every motor's angle, per arm, for the Motors tab.

        With the arms open it reads the states their own threads keep decoding (no bus traffic of ours). Otherwise it
        sends state queries only (no enable, no command, nothing that moves or energizes a motor) over an adapter of
        its own, which `_open_arms` closes before it opens the arms.
        """
        from robots import metal

        out: dict[str, Any] = {}
        for name, arm_config in self._arm_configs().items():
            can = adapter_info(arm_config.can_channel)
            entry: dict[str, Any] = {"joints": [], "source": "", "error": "", "can": can}
            try:
                arm = self.arms.get(name)
                chain = getattr(getattr(arm, "_driver", None), "chain", None) if arm is not None else None
                if chain is not None and arm.is_alive():
                    entry["source"] = "arm open"
                    states = {j.name: chain.bus.motors[j.name].state for j in metal.JOINTS}
                else:
                    entry["source"] = "read only"
                    states = self._raw_states(arm_config.can_channel)
                stats = self._stats.get(arm_config.can_channel, {})
                for joint in metal.JOINTS:
                    state = states.get(joint.name)
                    motor = chain.bus.motors[joint.name] if entry["source"] == "arm open" else None
                    can.setdefault("motors", []).append(
                        {
                            "name": joint.name,
                            "drops": None if motor is None else motor.drops,
                            **({} if motor is not None else stats.get(joint.name, {})),
                        }
                    )
                    entry["joints"].append(
                        {
                            "name": joint.name,
                            "low": joint.limits_deg[0],
                            "high": joint.limits_deg[1],
                            "deg": None if state is None else round(state.position_deg, 2),
                            "status": None if state is None else state.status,
                            "temp_c": None if state is None else state.temp_rotor_c,
                        }
                    )
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                self._close_monitor()
            out[name] = entry
        return out

    def _raw_states(self, channel: str) -> dict[str, Any]:
        from robots import damiao, metal, robstride

        if not self._arm_lock.acquire(blocking=False):  # an open is going on: skip this reading
            return {}
        try:
            bus = self._monitor.get(channel)
            if bus is None:
                bus = self._monitor[channel] = robstride.open_can_bus(channel)
            can = robstride.python_can()
            states = {}
            stats = self._stats.setdefault(channel, {})
            for joint in metal.JOINTS:
                row = stats.setdefault(joint.name, {"queries": 0, "replies": 0, "last_ms": None, "max_ms": 0.0})
                row["queries"] += 1
                sent = time.monotonic()
                while bus.recv(timeout=0) is not None:
                    pass
                bus.send(
                    can.Message(
                        arbitration_id=damiao.PARAM_ID,
                        data=damiao.refresh_payload(joint.motor_id),
                        is_extended_id=False,
                    ),
                    timeout=0.2,
                )
                deadline = time.monotonic() + 0.03
                while time.monotonic() < deadline:
                    got = bus.recv(timeout=0.01)
                    if got is not None and got.arbitration_id == joint.motor_id + damiao.REPLY_OFFSET:
                        states[joint.name] = damiao.decode_state(bytes(got.data), joint.model, time.monotonic())
                        row["replies"] += 1
                        row["last_ms"] = round((time.monotonic() - sent) * 1000, 1)
                        row["max_ms"] = max(row["max_ms"], row["last_ms"])
                        break
            if not states:
                # Nothing answered: the arms are off, or the adapter re-enumerated under a power cycle and this
                # handle is stale. Let it go so the next reading opens a fresh one.
                self._monitor.pop(channel, None)
                with contextlib.suppress(Exception):
                    bus.shutdown()
            return states
        finally:
            self._arm_lock.release()

    def _close_monitor(self) -> None:
        for bus in self._monitor.values():
            with contextlib.suppress(Exception):
                bus.shutdown()
        self._monitor = {}

    def _arm_configs(self) -> dict[str, Any]:
        return {"left": self.app.follower_l, "right": self.app.follower_r}

    def _open_arms(self) -> None:
        """Opens both arms, or none.

        Dream's open energizes an arm, so a pair opened one after the other could leave the first one powered when
        the second refuses (a gripper fault, say), and the next open would refuse that one as "still powered from an
        earlier session". So both are first opened read-only, which energizes nothing and runs every start-up check;
        only when both pass are they opened for real. If a real open still fails afterwards, the arms that did open
        stay in `self.arms`, driven and held, for Park & power off to switch off: they are never dropped powered.
        """
        with self._arm_lock:
            self._close_monitor()  # the Motors tab's own adapter must not hold the port the arms need
            configs = self._arm_configs()
            if self.arms and not any(arm.is_alive() for arm in self.arms.values()):
                # Nothing in hand is driving an arm any more, so there is nothing to park: let the stale ones go.
                # If an arm is in fact still powered, Dream's own open check below refuses it.
                for arm in self.arms.values():
                    with contextlib.suppress(Exception):
                        arm.close()
                self.arms = {}
            if self.arms:
                if set(self.arms) == set(configs) and self.arms_ok():
                    return
                dead = [
                    f"{n}: {a.why_stopped() or 'it stopped answering'}"
                    for n, a in self.arms.items()
                    if not a.is_alive()
                ]
                missing = [n for n in configs if n not in self.arms]
                what = "; ".join(dead + [f"{n}: not open" for n in missing])
                raise RuntimeError(
                    f"the arms are not both running ({what}). Support both arms and press Park & power off; "
                    "switch an arm's power off and on if it asks"
                )
            check = getattr(self.body, "open_arm_read_only", None)
            if check is not None:
                for name, arm_config in configs.items():
                    try:
                        probe = check(arm_config, name)
                    except Exception as error:
                        self.fault_text = str(error)
                        raise
                    with contextlib.suppress(Exception):
                        probe.close()
            for name, arm_config in configs.items():
                try:
                    self.arms[name] = self.body.open_arm(arm_config, name)
                except Exception as error:
                    self.fault_text = str(error)
                    raise

    def clear_fault(self) -> str:
        """Sends Damiao's clear-error frame to the gripper of each arm whose open just failed on one, and reports.

        Only for arms that are not open, and it sends nothing that energizes or moves a motor. The arm's own open then
        decides whether the motor is disabled and fine: this does not skip Dream's check.
        """
        if self.arms:
            raise RuntimeError("the arms are open: use Park & power off first")
        text = getattr(self, "fault_text", "")
        configs = {"left": self.app.follower_l, "right": self.app.follower_r}
        targets = [c for c in configs.values() if c.can_channel in text] or list(configs.values())
        from robots import damiao, robstride

        notes = []
        for arm_config in targets:
            channel, can_id = arm_config.can_channel, GRIPPER_CAN_ID
            bus = robstride.open_can_bus(channel)
            try:
                message = robstride.python_can().Message(
                    arbitration_id=can_id, data=b"\xff" * 7 + bytes([CLEAR_ERROR]), is_extended_id=False
                )
                bus.send(message, timeout=0.2)
                reply = None
                deadline = time.monotonic() + 0.5
                while reply is None and time.monotonic() < deadline:
                    got = bus.recv(timeout=0.05)
                    if got is not None and got.arbitration_id == can_id + damiao.REPLY_OFFSET:
                        reply = bytes(got.data)
                notes.append(f"{channel[-24:]}: " + (f"gripper status {reply[0] >> 4}" if reply else "no reply"))
            finally:
                bus.shutdown()
        self.fault_text = ""
        return "cleared and re-read: " + "; ".join(notes) + ". Now press Weightless to try the arms again."

    def arms_ok(self) -> bool:
        both = set(self.arms) == set(self._arm_configs())
        return both and all(arm.is_alive() for arm in self.arms.values())

    def weightless(self) -> None:
        self._open_arms()
        # Always built for the arms open now: one kept from before would still hold the arms that were let go, and
        # the new ones would stay at their normal stiffness.
        if self._weightless is not None:
            with contextlib.suppress(Exception):
                self._weightless.stop()
        self._weightless = _Weightless(self.arms, self._types, lambda text: self.on_problem(text))
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

    async def run(
        self, prompt: str, checkpoint: str, dry_run: bool, stop: asyncio.Event, ready: Callable[[], None]
    ) -> None:
        """One run of the loop on the open arms: no home move before it, none after it, no switch-off.

        `ready` is called once the model is up and the loop is about to start: the recording begins there, not
        earlier, because opening the model (its IK compiles) starves every other thread of the computer, and the arms'
        own threads must not be among the starved while the encoders run too.
        """
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
            ready()
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
                **{
                    **run.loop_settings(app),
                    "action_timeout_s": max(ACTION_TIMEOUT_S, app.action_timeout_s),
                    "max_stall_ticks": max(MAX_STALL_TICKS, app.max_stall_ticks),
                },
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

    def __init__(self, arms: dict[str, Any], types: Any, on_problem: Any = None) -> None:
        self._arms, self._types = arms, types
        self._on_problem = on_problem or (lambda text: print(text, file=sys.stderr))
        self._reported: set[str] = set()
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
                    text = f"weightless: {error}"
                    if text not in self._reported:  # once per distinct problem, not 50 times a second
                        self._reported.add(text)
                        self._on_problem(text)

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

    def clear_fault(self) -> str:
        return "fake: nothing to clear"

    def health(self) -> str:
        return ""

    def angles(self) -> dict[str, Any]:
        t = time.monotonic() - self._t0
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
        limits = ((-160, 160), (-180, 0), (0, 180), (-123, 81), (-85, 85), (-145, 145), (0, 120))
        arms = {}
        for side, shift in (("left", 0.0), ("right", 1.0)):
            joints = []
            for i, (name, (low, high)) in enumerate(zip(names, limits, strict=True)):
                mid, span = (low + high) / 2, (high - low) / 2
                deg = mid + 0.4 * span * math.sin(t / 3 + i + shift)
                if side == "right" and name == "elbow_flex":
                    deg = -9.1
                joints.append(
                    {"name": name, "low": low, "high": high, "deg": round(deg, 2), "status": 0, "temp_c": 31.0}
                )
            can = {
                "channel": f"/dev/serial/by-id/fake-{side}",
                "tty": "ttyACM0",
                "present": True,
                "usb_path": "5-1.2.1",
                "speed": 12,
                "motors": [
                    {"name": j["name"], "queries": 40, "replies": 40, "last_ms": 1.2, "max_ms": 3.4} for j in joints
                ],
            }
            arms[side] = {"joints": joints, "source": "fake", "error": "", "can": can}
        return arms

    def arms_ok(self) -> bool:
        return True

    async def run(
        self, prompt: str, checkpoint: str, dry_run: bool, stop: asyncio.Event, ready: Callable[[], None]
    ) -> None:
        ready()
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
        self._started = False

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def _encoder(self, name: str, rgb: np.ndarray) -> tuple[Any, Any]:
        import av

        container = av.open(str(self._dir / f"{name}.mp4"), "w")
        stream = container.add_stream("libx264", rate=RECORD_FPS)
        with contextlib.suppress(Exception):
            stream.codec_context.thread_count = ENCODER_THREADS  # three encoders must not take every core
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
        if self._started:
            self._thread.join(timeout=15)


# ---------------------------------------------------------------------- the worker


class Worker:
    """Reads commands, drives the backend, and prints events."""

    def __init__(self, backend: Backend, frames_dir: pathlib.Path, events: Any) -> None:
        self.backend, self.frames_dir, self._events = backend, frames_dir, events
        backend.on_problem = lambda text: self.emit(event="error", message=text)
        self._emit_lock = threading.Lock()
        self.state = "idle"
        self._run_task: asyncio.Task | None = None
        self._stop_run = asyncio.Event()
        self._done = asyncio.Event()
        self._watch_until = 0.0  # while a page looks at the Motors tab, the angles are read until this time

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
        next_angles = next_health = 0.0
        while not self._done.is_set():
            now = time.monotonic()
            if now >= next_health:
                next_health = now + 0.5
                try:
                    problem = self.backend.health()
                except Exception:
                    problem = ""
                if problem:
                    self.emit(event="error", message=problem)
            if now < self._watch_until and now >= next_angles:
                next_angles = now + 1.0 / ANGLES_HZ
                try:
                    self.emit(event="angles", arms=self.backend.angles())
                except Exception as error:
                    self.emit(event="error", message=f"motors: {error}")
                    self._watch_until = 0.0
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
        if name == "watch":
            self._watch_until = time.monotonic() + WATCH_S
        elif name == "stop":
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
        elif name == "clear_fault":
            note = await asyncio.to_thread(self.backend.clear_fault)
            self.set_state(self._settled(), note)
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

        def ready() -> None:
            recorder.start()
            self.emit(event="run_started", run_id=run_id)

        try:
            await self.backend.run(command["prompt"], command["checkpoint"], bool(command["dry_run"]), stop, ready)
            if stop.is_set() and not timed_out:
                status = "stopped"
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            logging.exception("run %s failed", run_id)  # the traceback goes to worker.log; the page shows one line
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
    parser.add_argument("--log-file", help="also write the libraries' log (Dream's arm threads included) here")
    args = parser.parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
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
