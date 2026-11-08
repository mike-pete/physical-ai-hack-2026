"""What each of the station's devices is doing, and the fix when one is not ready.

`dhk-collect` starts its dashboard before it touches a device, then opens the Quest and the cameras as they become
ready (DeviceSet.try_open()), and shows every device's state on the page and in the terminal (DeviceSet.statuses()).
The checks here open nothing: the Quest is read through `adb devices`, the arms' links through `/sys/class/net` (CAN)
or `/dev` (a serial adapter), so they can run every second, during a session too.

Some problems have a fix the dashboard can run, as a button (DeviceSet.apply_fix()):

- A CAN link that is down, as the links are after every reboot: `sudo -n ip link set <link> up type can bitrate
  1000000`, which `scripts/allow_can_bringup.sh` lets the station's account run without a password, and nothing
  else. It never takes a link down, which would cut open arms off from their motors.
- A Quest whose data stopped (its app went behind Quest Home, or its log stream was cut when it slept): restart the
  teleop app in front, on a new stream (quest.Quest.reconnect()).
- A Quest adb cannot use yet (`unauthorized`, `no permissions`, `offline`), before it is open: restart adb, which also
  shows the headset's "Allow USB debugging" prompt again.

@see Status, DeviceSet, apply_fix()
"""

import dataclasses
import os
import pathlib
import shutil
import subprocess
import threading
from collections.abc import Callable
from typing import Any

from dream_hackathon import cameras, config, quest

## Where Linux lists the network links, the CAN ones among them.
SYS_NET = pathlib.Path("/sys/class/net")
## The `flags` bit of a link that has been brought up (IFF_UP).
_IFF_UP = 0x1
## Controller data older than this means the Quest has stopped sending (asleep, or its link is gone).
QUEST_LOST_AFTER_S = 2.0
## A camera with no new frame for this long is shown as a problem (a shorter gap is a hiccup).
CAMERA_LOST_AFTER_S = 1.0
## The arms' CAN bit rate.
CAN_BITRATE = 1_000_000
## The fix id of bringing a CAN link up: this prefix, then the link's name.
CAN_UP_FIX = "can-up:"
## The fix ids of restarting the Quest's app, and of restarting adb.
QUEST_RESTART_FIX = "quest-restart"
ADB_RESTART_FIX = "adb-restart"


@dataclasses.dataclass(frozen=True)
class Status:
    """One device's state.

    Attributes:
    - name: What it is, e.g. `Quest`, `camera wrist_l`, `left arm link`.
    - problem: What is wrong and what to do about it, or None when it is ready.
    - fix: The id of a fix the dashboard can run for it (apply_fix()), or None.
    - fix_label: The fix's button text.
    """

    name: str
    problem: str | None = None
    fix: str | None = None
    fix_label: str = ""

    def to_json(self) -> dict[str, Any]:
        fix = {"id": self.fix, "label": self.fix_label} if self.fix else None
        return {"name": self.name, "ok": self.problem is None, "problem": self.problem or "", "fix": fix}


def can_link_state(port: str, sys_net: pathlib.Path = SYS_NET) -> str:
    """`missing` (no link of that name), `down` (not brought up), `no carrier` (up, but no arm answers) or `up`."""
    link = sys_net / port
    if not link.exists():
        return "missing"
    if not int((link / "flags").read_text().strip(), 16) & _IFF_UP:
        return "down"
    if (link / "operstate").read_text().strip() not in ("up", "unknown"):
        return "no carrier"
    return "up"


def can_link_problem(port: str, sys_net: pathlib.Path = SYS_NET) -> str | None:
    """What keeps an arm's CAN link from being used, with the command that fixes it, or None when it is ready.

    The links come up down after every reboot, and are named by the setup's udev rule (docs/hardware_setup.md).
    """
    state = can_link_state(port, sys_net)
    if state == "missing":
        others = sorted(path.name for path in sys_net.glob("can*")) if sys_net.exists() else []
        return (
            f"there is no CAN link named {port} (this computer has {', '.join(others) or 'none'}): if it lists "
            'can0 and can1, apply the setup\'s CAN names (docs/hardware_setup.md, "YAM arms"), or call the organizers'
        )
    if state == "down":
        return (
            f"CAN link {port} is down, as the links are after every reboot: "
            f"sudo ip link set {port} up type can bitrate {CAN_BITRATE}"
        )
    if state == "no carrier":
        return f"CAN link {port} is up but hears no arm: check that the arm is powered and its CAN cable is plugged in"
    return None


def serial_port_problem(port: str) -> str | None:
    """What keeps a serial CAN adapter (`/dev/serial/...`, `/dev/ttyACM*`) from being used, or None."""
    if not os.path.exists(port):
        return f"{port} is not there: check that the arm's CAN adapter is plugged in (or call the organizers)"
    if not os.access(port, os.R_OK | os.W_OK):
        return f"this account cannot open {port}: sudo usermod -aG dialout $USER, then log in again"
    return None


def arm_link_problem(station: config.StationConfig, side: str, sys_net: pathlib.Path = SYS_NET) -> str | None:
    """What keeps one arm's link (its CAN interface or its serial adapter) from being used, or None."""
    port = station.arms[side].port
    return can_link_problem(port, sys_net) if over_socketcan(station) else serial_port_problem(port)


def over_socketcan(station: config.StationConfig) -> bool:
    """Whether the arms' links are CAN interfaces (a YAM's, or a MakerMods station set to socketcan)."""
    return station.family == "yam" or (
        station.makermods is not None and station.makermods.can_interface == "socketcan"
    )


def apply_fix(
    station: config.StationConfig,
    fix: str,
    sys_net: pathlib.Path = SYS_NET,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[bool, str]:
    """Runs a fix the dashboard offers (Status.fix); the only one brings one of the station's CAN links up.

    It refuses anything else: a link that is not one of this station's arms', or one that is not down right now (it
    never takes a link down, which would cut open arms off from their motors).

    @return Whether it worked, and what happened, in words for the page and the terminal.
    """
    if not fix.startswith(CAN_UP_FIX):
        return False, f"unknown fix {fix!r}"
    port = fix.removeprefix(CAN_UP_FIX)
    if not over_socketcan(station) or port not in {arm.port for arm in station.arms.values()}:
        return False, f"{port!r} is not one of this station's arm links"
    state = can_link_state(port, sys_net)
    if state != "down":
        return state == "up", f"{port} is {state}: nothing to bring up"
    ip = shutil.which("ip") or "/usr/sbin/ip"
    command = ["sudo", "-n", ip, "link", "set", port, "up", "type", "can", "bitrate", str(CAN_BITRATE)]
    try:
        result = run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"could not bring {port} up: {error}"
    if result.returncode == 0:
        return True, f"brought {port} up"
    # sudo says "a password is required"; sudo-rs, Ubuntu 26.04's, "interactive authentication is required".
    if "password" in result.stderr or "authentication" in result.stderr:
        return False, (
            f"could not bring {port} up: this station asks for a password, so run "
            f"`sudo ip link set {port} up type can bitrate {CAN_BITRATE}` in a terminal (organizers: "
            "`bash scripts/allow_can_bringup.sh` makes this button work)"
        )
    return False, f"could not bring {port} up: {result.stderr.strip() or f'exit {result.returncode}'}"


def quest_problem(adb_devices_output: str) -> str | None:
    """What keeps the Quest from being used over USB, from what `adb devices` prints, or None when it is ready."""
    problem = quest.adb_problem(adb_devices_output)
    if problem is not None:
        return problem
    if not quest.usb_devices(adb_devices_output):
        return (
            "not connected: plug the headset into the station with a USB-C data cable, wake it, and check that its "
            "Developer Mode is on"
        )
    return None


class DeviceSet:
    """The Quest and the cameras, opened as they become ready, and every device's state.

    Attributes:
    - headset: The Quest once it is open, else None.
    - cameras: Per role, the cameras open so far.
    """

    def __init__(
        self,
        station: config.StationConfig,
        *,
        sim: bool = False,
        quest_ip: str | None = None,
        use_quest: bool = True,
        open_quest: Callable[[str | None], Any] = quest.Quest,
        open_camera: Callable[[config.CameraConfig, int, int], Any] = cameras.Camera,
        adb_devices: Callable[[], str] = quest.adb_devices,
        headset_state: Callable[[str | None], tuple[bool | None, str | None]] = quest.headset_state,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        sys_net: pathlib.Path = SYS_NET,
    ) -> None:
        """Opens nothing yet.

        @param station The station config.
        @param sim A simulated YAM: no cameras or arm links to check.
        @param quest_ip The Quest's address for adb over Wi-Fi, or None for USB.
        @param use_quest False for a session that needs no Quest (a scripted policy): it is neither opened nor listed.
        @param open_quest, open_camera, adb_devices, headset_state, run, sys_net The real ones by default; tests pass
            fakes.
        """
        self._station = station
        self._sim = sim
        self._quest_ip = quest_ip
        self._use_quest = use_quest
        self._open_quest = open_quest
        self._open_camera = open_camera
        self._adb_devices = adb_devices
        self._headset_state = headset_state
        self._run = run
        self._sys_net = sys_net
        self._quest_lock = threading.Lock()  # opening or restarting the Quest: the waiting loop, or a page's fix
        self.headset: Any = None
        self.cameras: dict[str, Any] = {}
        self._quest_error: str | None = "not checked yet"
        self._camera_errors: dict[str, str] = {}

    def try_open(self) -> None:
        """Opens the Quest and each camera that is not open yet; one that fails keeps its error for statuses().

        It can take a few seconds (a camera waits up to two for its first frame), so call it outside the control loop.
        """
        with self._quest_lock:
            if self.headset is None and self._use_quest:
                problem = None if self._quest_ip else quest_problem(self._adb_devices())
                if problem is None:
                    try:
                        self.headset = self._open_quest(self._quest_ip)
                    except SystemExit as error:  # oculus_reader exits when it finds no headset
                        problem = (
                            f"could not start the Quest's app (exit {error.code}): replug the headset and wake it"
                        )
                    except Exception as error:  # noqa: BLE001 - shown on the page, and tried again
                        problem = str(error) or type(error).__name__
                self._quest_error = problem
        if self._sim:
            return
        for cfg in self._station.cameras:
            if cfg.role in self.cameras:
                continue
            try:
                self.cameras[cfg.role] = self._open_camera(
                    cfg, self._station.record_width, self._station.record_height
                )
            except cameras.CameraError as error:
                self._camera_errors[cfg.role] = f"{error}: check its USB cable"
            else:
                self._camera_errors.pop(cfg.role, None)

    def statuses(self) -> list[Status]:
        """Every device's state now; opens nothing, so it can run every second, during a session too."""
        found = [self._quest_status()] if self._use_quest else []
        if self._sim:
            return found
        for cfg in self._station.cameras:
            camera = self.cameras.get(cfg.role)
            problem = camera.problem(CAMERA_LOST_AFTER_S) if camera is not None else self._camera_errors.get(cfg.role)
            found.append(Status(f"camera {cfg.role}", problem or (None if camera is not None else "not open yet")))
        for side in ("left", "right"):
            port = self._station.arms[side].port
            problem = arm_link_problem(self._station, side, self._sys_net)
            can_up = over_socketcan(self._station) and can_link_state(port, self._sys_net) == "down"
            found.append(
                Status(
                    f"{side} arm link",
                    problem,
                    fix=f"{CAN_UP_FIX}{port}" if can_up else None,
                    fix_label=f"Bring {port} up" if can_up else "",
                )
            )
        return found

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()
        if self.headset is not None:
            self.headset.close()

    def apply_fix(self, fix: str) -> tuple[bool, str]:
        """Runs a fix the dashboard offers (Status.fix), from the page's thread; see the module's list.

        @return Whether it worked, and what happened, in words for the page and the terminal.
        """
        if fix == QUEST_RESTART_FIX:
            with self._quest_lock:
                if self.headset is None:
                    return False, "the Quest is not open yet: nothing to restart"
                try:
                    self.headset.reconnect()
                except SystemExit as error:  # oculus_reader exits when it cannot reach the headset
                    return (
                        False,
                        f"could not restart the Quest app (exit {error.code}): replug the headset and wake it",
                    )
                except Exception as error:  # noqa: BLE001 - shown on the page
                    return False, f"could not restart the Quest app: {error}"
            return True, "restarted the Quest app: put the headset on and hold the controllers in view"
        if fix == ADB_RESTART_FIX:
            with self._quest_lock:
                if self.headset is not None:
                    return False, "the Quest is open: restarting adb would cut it off (use Restart the Quest app)"
                try:
                    for step in ("kill-server", "start-server"):
                        self._run(["adb", step], capture_output=True, text=True, timeout=20)
                except (OSError, subprocess.TimeoutExpired) as error:
                    return False, f"could not restart adb: {error}"
            return True, 'restarted adb: if the headset shows "Allow USB debugging", accept it (tick Always allow)'
        return apply_fix(self._station, fix, self._sys_net, self._run)

    def _quest_status(self) -> Status:
        if self.headset is None:
            adb_says = None if self._quest_ip else quest.usb_devices(self._adb_devices())
            can_restart_adb = bool(adb_says) and all(state != "device" for _, state in adb_says)
            return Status(
                "Quest",
                self._quest_error,
                fix=ADB_RESTART_FIX if can_restart_adb else None,
                fix_label="Restart adb" if can_restart_adb else "",
            )
        if not self._quest_ip:
            problem = quest_problem(self._adb_devices())
            if problem is not None:
                return Status("Quest", f"{problem}; then restart dhk-collect (Ctrl+C, and run it again)")
        age_s = self.headset.data_age_s()
        if age_s <= QUEST_LOST_AFTER_S:
            return Status("Quest")
        waited = "no controller data yet" if age_s == float("inf") else f"no controller data for {age_s:.0f} s"
        awake, front = self._headset_state(self._quest_ip)
        if awake is False:
            problem = (
                f"{waited}: the headset is asleep; put it on (it wakes when worn) and hold the controllers in view"
            )
        elif front and front != quest.APP:
            problem = (
                f"{waited}: the teleop app is not in front (the headset shows {front}): press Restart the Quest app, "
                "then put the headset on"
            )
        else:
            problem = (
                f"{waited}: press Restart the Quest app, then put the headset on and hold the controllers in view"
            )
        return Status("Quest", problem, fix=QUEST_RESTART_FIX, fix_label="Restart the Quest app")
