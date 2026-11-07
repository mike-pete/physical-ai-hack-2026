"""The device checks, the session's wait for its devices, and a camera that reconnects. No hardware needed."""

import pathlib
import time

import numpy as np
import pytest
from conftest import filled_template

from dream_hackathon import cameras, collect, config, devices, quest

ADB_HEADER = "List of devices attached\n"


def _link(sys_net: pathlib.Path, name: str, flags: str = "0x40081", operstate: str = "up") -> None:
    link = sys_net / name
    link.mkdir(parents=True)
    (link / "flags").write_text(flags + "\n")
    (link / "operstate").write_text(operstate + "\n")


# --- the arms' links


def test_a_can_link_that_is_up_is_ready(tmp_path):
    _link(tmp_path, "can_follower_l")
    assert devices.can_link_problem("can_follower_l", tmp_path) is None


def test_a_can_link_left_down_after_a_reboot_gets_its_command(tmp_path):
    _link(tmp_path, "can_follower_l", flags="0x80", operstate="down")
    problem = devices.can_link_problem("can_follower_l", tmp_path)
    assert "sudo ip link set can_follower_l up type can bitrate 1000000" in problem


def test_a_can_link_up_without_carrier_points_at_the_arm(tmp_path):
    _link(tmp_path, "can_follower_r", operstate="down")
    assert "powered" in devices.can_link_problem("can_follower_r", tmp_path)


def test_unnamed_can_links_point_at_the_naming_step(tmp_path):
    _link(tmp_path, "can0", flags="0x80", operstate="down")
    _link(tmp_path, "can1", flags="0x80", operstate="down")
    problem = devices.can_link_problem("can_follower_l", tmp_path)
    assert "can0, can1" in problem and "CAN names" in problem


def test_a_serial_adapter_that_is_not_plugged_in(tmp_path):
    assert "not there" in devices.serial_port_problem(str(tmp_path / "ttyACM0"))
    port = tmp_path / "ttyACM1"
    port.write_text("")
    assert devices.serial_port_problem(str(port)) is None


# --- the Quest


def test_the_quest_states_adb_reports():
    assert "not connected" in devices.quest_problem(ADB_HEADER + "\n")
    assert "Allow USB debugging" in devices.quest_problem(ADB_HEADER + "1WMHH0000000AB\tunauthorized\n")
    assert "udev rule" in devices.quest_problem(ADB_HEADER + "1WMHH0000000AB\tno permissions (missing udev rules?)\n")
    assert devices.quest_problem(ADB_HEADER + "1WMHH0000000AB\tdevice\n") is None


# --- the device set, with fakes


class FakeHeadset:
    """A Quest whose data age and presses the test sets."""

    def __init__(self) -> None:
        self.age_s = 0.0
        self.press: set[str] = set()
        self.closed = False
        self.reconnects = 0

    def reconnect(self) -> None:
        self.reconnects += 1
        self.age_s = float("inf")

    def data_age_s(self) -> float:
        return self.age_s

    def read(self) -> quest.QuestState:
        pressed, self.press = frozenset(self.press), set()
        poses = {"left": np.eye(4), "right": np.eye(4)}
        return quest.QuestState(poses, {"left": 0.0, "right": 0.0}, {"left": 0.0, "right": 0.0}, pressed)

    def close(self) -> None:
        self.closed = True


class FakeCamera:
    def __init__(self, cfg: config.CameraConfig) -> None:
        self.role = cfg.role
        self.lost: str | None = None

    def problem(self, after_s: float) -> str | None:
        return self.lost

    def close(self) -> None:
        pass


class Station:
    """A YAM station's device set over fakes: which cameras fail to open, what adb says, the CAN links."""

    def __init__(self, tmp_path: pathlib.Path) -> None:
        self.config = config.load(filled_template("yam", tmp_path))
        self.sys_net = tmp_path / "net"
        for name in ("can_follower_l", "can_follower_r"):
            _link(self.sys_net, name)
        self.adb = ADB_HEADER + "1WMHH0000000AB\tdevice\n"
        self.broken_cameras: set[str] = set()
        self.headset = FakeHeadset()
        self.headset_state: tuple[bool | None, str | None] = (True, quest.APP)
        self.commands: list[list[str]] = []
        self.found = devices.DeviceSet(
            self.config,
            open_quest=lambda ip: self.headset,
            open_camera=self._open_camera,
            adb_devices=lambda: self.adb,
            headset_state=lambda ip: self.headset_state,
            run=self._run,
            sys_net=self.sys_net,
        )

    def _run(self, command, **kwargs):
        self.commands.append(command)
        return __import__("subprocess").CompletedProcess(command, 0, "", "")

    def _open_camera(self, cfg: config.CameraConfig, width: int, height: int) -> FakeCamera:
        if cfg.role in self.broken_cameras:
            raise cameras.CameraError(f"camera '{cfg.role}' gave no frame")
        return FakeCamera(cfg)

    def problems(self) -> dict[str, str | None]:
        return {status.name: status.problem for status in self.found.statuses()}


def test_device_set_opens_what_is_ready_and_names_what_is_not(tmp_path):
    station = Station(tmp_path)
    station.adb = ADB_HEADER + "\n"
    station.broken_cameras = {"wrist_l"}
    station.found.try_open()
    problems = station.problems()
    assert set(problems) == {
        "Quest",
        "camera overhead",
        "camera wrist_l",
        "camera wrist_r",
        "left arm link",
        "right arm link",
    }
    assert "not connected" in problems["Quest"] and station.found.headset is None
    assert "gave no frame" in problems["camera wrist_l"] and "USB cable" in problems["camera wrist_l"]
    assert problems["camera overhead"] is None and problems["left arm link"] is None

    station.adb = ADB_HEADER + "1WMHH0000000AB\tdevice\n"
    station.broken_cameras.clear()
    station.found.try_open()
    assert all(problem is None for problem in station.problems().values())


def test_device_set_sees_problems_during_a_session(tmp_path):
    station = Station(tmp_path)
    station.found.try_open()
    station.headset.age_s = 5.0
    station.found.cameras["wrist_r"].lost = "no frame for 3 s"
    (station.sys_net / "can_follower_l" / "flags").write_text("0x80\n")
    problems = station.problems()
    assert "put the headset on" in problems["Quest"]
    assert problems["camera wrist_r"] == "no frame for 3 s"
    assert "sudo ip link set can_follower_l up" in problems["left arm link"]


# --- the wait for the devices, then A


class Clock:
    """A fake clock that a fake sleep moves on; each sleep also runs the test's script for that moment."""

    def __init__(self, script) -> None:
        self.now = 0.0
        self.script = script

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.script(self.now)
        if self.now > 60:
            raise AssertionError("the wait never ended")


def test_the_arms_open_only_once_everything_is_ready_and_a_is_pressed(tmp_path):
    station = Station(tmp_path)
    station.broken_cameras = {"wrist_l"}
    opened, logs = [], []

    def script(now: float) -> None:
        if now > 3:
            station.broken_cameras.clear()  # the camera is plugged in
        if now > 8:
            station.headset.press.add("A")

    def open_arms(cfg, *, energize, sim):
        opened.append(clock.now)
        return "the arms"

    clock = Clock(script)
    robot = collect.wait_and_open_arms(
        station.config,
        station.found,
        None,
        logs.append,
        dry_run=True,
        sim=False,
        open_arms=open_arms,
        clock=clock,
        sleep=clock.sleep,
    )
    assert robot == "the arms" and len(opened) == 1 and opened[0] > 8
    assert any(line.startswith("waiting for: camera wrist_l") for line in logs)
    assert any("press A on the right controller" in line for line in logs)
    assert any(line == "camera wrist_l: ready" for line in logs)


def test_arms_that_fail_to_open_are_reported_and_a_tries_again(tmp_path):
    station = Station(tmp_path)
    attempts, logs = [], []

    def script(now: float) -> None:
        if len(attempts) < 2:
            station.headset.press.add("A")

    def open_arms(cfg, *, energize, sim):
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise RuntimeError("motor 3 does not answer")
        return "the arms"

    clock = Clock(script)
    robot = collect.wait_and_open_arms(
        station.config,
        station.found,
        None,
        logs.append,
        dry_run=False,
        sim=False,
        open_arms=open_arms,
        clock=clock,
        sleep=clock.sleep,
    )
    assert robot == "the arms" and len(attempts) == 2
    assert any("did not open: motor 3 does not answer" in line for line in logs)


def test_the_report_logs_only_what_changed():
    reported, logs = {}, []
    statuses = [devices.Status("Quest"), devices.Status("camera overhead", "no frame")]
    collect._report(statuses, reported, logs.append)
    collect._report(statuses, reported, logs.append)
    assert logs == ["Quest: ready", "camera overhead: no frame"]
    collect._report([devices.Status("Quest"), devices.Status("camera overhead")], reported, logs.append)
    assert logs[-1] == "camera overhead: ready" and len(logs) == 3


# --- a camera that comes back


class ScriptedReader:
    """A camera reader that gives frames, then fails as an unplugged one does, until the test plugs it back."""

    plugged = True
    opened = 0

    def __init__(self, cfg) -> None:
        if not ScriptedReader.plugged:
            raise cameras.CameraError("cannot open the camera")
        ScriptedReader.opened += 1

    def __call__(self):
        if not ScriptedReader.plugged:
            raise RuntimeError("Frame didn't arrive within 1000")
        time.sleep(0.005)
        return np.full((720, 1280, 3), 100, np.uint8)

    def release(self) -> None:
        pass


def test_a_camera_reconnects_by_itself_once_it_is_back(monkeypatch):
    monkeypatch.setattr(cameras, "_open_reader", ScriptedReader)
    monkeypatch.setattr(cameras, "REOPEN_EVERY_S", 0.05)
    cfg = config.CameraConfig(
        role="wrist_l", kind="uvc", device="/dev/video0", serial=None, width=1280, height=720, fps=30
    )
    camera = cameras.Camera(cfg, 320, 180)
    try:
        assert camera.latest().shape == (180, 320, 3) and camera.problem(0.5) is None
        ScriptedReader.plugged = False
        time.sleep(0.7)
        with pytest.raises(cameras.CameraError):
            camera.latest()
        assert "USB cable" in camera.problem(0.5)
        ScriptedReader.plugged = True
        time.sleep(0.3)
        assert camera.latest().shape == (180, 320, 3) and camera.problem(0.5) is None
        assert ScriptedReader.opened >= 2
    finally:
        ScriptedReader.plugged = True
        camera.close()


class FakeRealSense:
    """Enough of pyrealsense2 to see whether a camera's pipeline is started."""

    def __init__(self, connected: list[str]) -> None:
        self.connected = connected
        self.started: list[str] = []
        self.camera_info = type("camera_info", (), {"serial_number": "serial"})
        self.stream = type("stream", (), {"color": "color"})
        self.format = type("format", (), {"bgr8": "bgr8"})
        fake = self

        class Device:
            def __init__(self, serial: str) -> None:
                self._serial = serial

            def get_info(self, what):
                return self._serial

        class Context:
            def query_devices(self):
                return [Device(serial) for serial in fake.connected]

        class Config:
            def enable_device(self, serial):
                self.serial = serial

            def enable_stream(self, *args):
                pass

        class Pipeline:
            def start(self, config):
                fake.started.append(config.serial)

        self.context, self.config, self.pipeline = Context, Config, Pipeline


def test_a_realsense_that_is_not_connected_is_never_started(monkeypatch):
    # Starting one that is not connected blocks for 15 s holding the GIL, which would freeze the whole session.
    fake = FakeRealSense(connected=["111"])
    monkeypatch.setitem(__import__("sys").modules, "pyrealsense2", fake)
    cfg = config.CameraConfig(
        role="wrist_r", kind="realsense", device=None, serial="222", width=1280, height=720, fps=30
    )
    with pytest.raises(cameras.CameraError, match="RealSense 222 is not connected"):
        cameras._RealSenseReader(cfg)
    assert fake.started == []
    fake.connected.append("222")
    cameras._RealSenseReader(cfg)
    assert fake.started == ["222"]


# --- the dashboard's fix: bring a CAN link up


class FakeRun:
    """subprocess.run for `sudo -n ip link set ...`: records the command and answers as the test says."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.commands: list[list[str]] = []
        self.returncode, self.stderr = returncode, stderr

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        return __import__("subprocess").CompletedProcess(command, self.returncode, "", self.stderr)


def test_a_down_can_link_offers_its_fix_and_the_fix_brings_it_up(tmp_path):
    station = Station(tmp_path)
    (station.sys_net / "can_follower_r" / "flags").write_text("0x80\n")
    station.found.try_open()
    by_name = {status.name: status for status in station.found.statuses()}
    assert by_name["right arm link"].fix == "can-up:can_follower_r"
    assert by_name["right arm link"].fix_label == "Bring can_follower_r up"
    assert by_name["left arm link"].fix is None  # up already: nothing to offer

    run = FakeRun()
    ok, message = devices.apply_fix(station.config, "can-up:can_follower_r", station.sys_net, run)
    assert ok and message == "brought can_follower_r up"
    assert run.commands[0][:2] == ["sudo", "-n"]
    assert run.commands[0][3:] == ["link", "set", "can_follower_r", "up", "type", "can", "bitrate", "1000000"]


def test_the_fix_refuses_anything_but_bringing_a_down_arm_link_up(tmp_path):
    station = Station(tmp_path)
    run = FakeRun()
    assert devices.apply_fix(station.config, "can-up:eth0", station.sys_net, run) == (
        False,
        "'eth0' is not one of this station's arm links",
    )
    ok, message = devices.apply_fix(station.config, "can-up:can_follower_l", station.sys_net, run)
    assert ok and "nothing to bring up" in message  # already up: never touched
    assert not devices.apply_fix(station.config, "can-down:can_follower_l", station.sys_net, run)[0]
    assert run.commands == []


def test_a_station_without_the_sudo_rule_gets_the_command_to_type(tmp_path):
    station = Station(tmp_path)
    (station.sys_net / "can_follower_l" / "flags").write_text("0x80\n")
    for refusal in ("sudo: a password is required", "sudo: interactive authentication is required"):  # sudo, sudo-rs
        run = FakeRun(returncode=1, stderr=refusal + "\n")
        ok, message = devices.apply_fix(station.config, "can-up:can_follower_l", station.sys_net, run)
        assert not ok and "sudo ip link set can_follower_l up type can bitrate 1000000" in message
        assert "allow_can_bringup.sh" in message


# --- the Quest's fixes


def test_a_silent_quest_says_why_and_offers_a_restart(tmp_path):
    station = Station(tmp_path)
    station.found.try_open()
    station.headset.age_s = 20.0
    station.headset_state = (True, "com.oculus.vrshell")
    quest_status = station.found.statuses()[0]
    assert "the teleop app is not in front (the headset shows com.oculus.vrshell)" in quest_status.problem
    assert (quest_status.fix, quest_status.fix_label) == ("quest-restart", "Restart the Quest app")
    station.headset_state = (False, "com.oculus.vrshell")
    assert "the headset is asleep" in station.found.statuses()[0].problem
    station.headset.age_s = 0.1
    assert station.found.statuses()[0] == devices.Status("Quest")  # sending again: no problem, no button


def test_restarting_the_quest_app_reconnects_the_open_headset(tmp_path):
    station = Station(tmp_path)
    assert station.found.apply_fix("quest-restart") == (False, "the Quest is not open yet: nothing to restart")
    station.found.try_open()
    ok, message = station.found.apply_fix("quest-restart")
    assert ok and "restarted the Quest app" in message and station.headset.reconnects == 1


def test_restart_adb_is_offered_only_before_the_quest_is_open(tmp_path):
    station = Station(tmp_path)
    station.adb = ADB_HEADER + "1WMHH0000000AB\tunauthorized\n"
    station.found.try_open()
    quest_status = station.found.statuses()[0]
    assert (quest_status.fix, quest_status.fix_label) == ("adb-restart", "Restart adb")
    ok, message = station.found.apply_fix("adb-restart")
    assert ok and "Allow USB debugging" in message
    assert station.commands == [["adb", "kill-server"], ["adb", "start-server"]]
    station.adb = ADB_HEADER + "1WMHH0000000AB\tdevice\n"
    station.found.try_open()
    assert not station.found.apply_fix("adb-restart")[0]  # open now: restarting adb would cut it off
    assert len(station.commands) == 2


def test_a_problem_whose_numbers_change_is_logged_once():
    reported, logs = {}, []
    for seconds in (15, 16, 17):
        collect._report(
            [devices.Status("Quest", f"no controller data for {seconds} s: wake it")], reported, logs.append
        )
    assert logs == ["Quest: no controller data for 15 s: wake it"]
    collect._report([devices.Status("Quest", "no controller data for 18 s: asleep")], reported, logs.append)
    assert len(logs) == 2  # the words changed: a new problem


class FakeOculusReader:
    """oculus_reader's reader, counting how many were made and stopped."""

    made: list["FakeOculusReader"] = []

    def __init__(self, ip_address=None) -> None:
        self.running, self.stopped = True, False
        FakeOculusReader.made.append(self)

    def get_transformations_and_buttons(self):
        return {}, {}

    def stop(self) -> None:
        self.running, self.stopped = False, True


def test_reconnect_restarts_the_app_on_a_new_reader(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules, "oculus_reader.reader", type("m", (), {"OculusReader": FakeOculusReader})
    )
    monkeypatch.setattr(quest, "adb_devices", lambda: ADB_HEADER + "1WMHH0000000AB\tdevice\n")
    shell = []
    monkeypatch.setattr(quest, "adb_shell", lambda command, ip=None: shell.append(command) or "")
    FakeOculusReader.made.clear()
    headset = quest.Quest()
    first = FakeOculusReader.made[0]
    headset.reconnect()
    assert len(FakeOculusReader.made) == 2 and first.stopped and not FakeOculusReader.made[1].stopped
    assert shell == [f"am force-stop {quest.APP}"]
    assert headset.data_age_s() == float("inf")  # no data from the new reader yet


def test_an_a_held_when_the_data_comes_back_is_not_a_press(monkeypatch):
    class Reader(FakeOculusReader):
        transforms: dict = {}
        buttons: dict = {}

        def get_transformations_and_buttons(self):
            return Reader.transforms, Reader.buttons

    monkeypatch.setitem(__import__("sys").modules, "oculus_reader.reader", type("m", (), {"OculusReader": Reader}))
    monkeypatch.setattr(quest, "adb_devices", lambda: ADB_HEADER + "1WMHH0000000AB\tdevice\n")
    headset = quest.Quest()
    assert headset.read().pressed == frozenset()  # no data yet
    Reader.transforms, Reader.buttons = {"r": np.eye(4), "l": np.eye(4)}, {"A": True}
    assert headset.read().pressed == frozenset()  # A was down when the data came: held, not pressed
    Reader.transforms, Reader.buttons = {"r": np.eye(4), "l": np.eye(4)}, {"A": False}
    assert headset.read().pressed == frozenset()
    Reader.transforms, Reader.buttons = {"r": np.eye(4), "l": np.eye(4)}, {"A": True}
    assert headset.read().pressed == frozenset({"A"})  # a real press


def test_an_a_in_the_moment_everything_becomes_ready_does_not_open_the_arms(tmp_path):
    station = Station(tmp_path)
    station.broken_cameras = {"wrist_l"}
    opened, logs, ready_at = [], [], []

    def log(line: str) -> None:
        logs.append(line)
        if "everything is ready" in line:
            ready_at.append(clock.now)

    def script(now: float) -> None:
        if now > 3:
            station.broken_cameras.clear()
        if ready_at and now - ready_at[0] < 0.5:
            station.headset.press.add("A")  # pressed as the session becomes ready: ignored
        if ready_at and now - ready_at[0] > 3:
            station.headset.press.add("A")

    def open_arms(cfg, *, energize, sim):
        opened.append(clock.now)
        return "the arms"

    clock = Clock(script)
    collect.wait_and_open_arms(
        station.config,
        station.found,
        None,
        log,
        dry_run=True,
        sim=False,
        open_arms=open_arms,
        clock=clock,
        sleep=clock.sleep,
    )
    assert len(opened) == 1 and opened[0] - ready_at[0] > 3


def test_a_makermods_station_may_mix_a_serial_and_a_socketcan_adapter(tmp_path):
    path = filled_template("metal", tmp_path)
    text = path.read_text().replace("gripper_open_deg: null", "gripper_open_deg: 110.0")
    text = text.replace("gripper_closed_deg: null", "gripper_closed_deg: 0.0")
    text = text.replace("port: /dev/serial/by-id/0\n", "port: can_follower_r\n")
    path.write_text(text)  # the left arm keeps its serial adapter (its port line has a comment); the right is CAN
    station = config.load(path)
    assert station.arms["left"].port == "/dev/serial/by-id/0" and station.arms["right"].port == "can_follower_r"
    assert config.arm_can_interface(station, "left") == "slcan"
    assert config.arm_can_interface(station, "right") == "socketcan"

    sys_net = tmp_path / "net"
    _link(sys_net, "can_follower_r", flags="0x80")  # down
    assert "is not there" in devices.arm_link_problem(station, "left", sys_net)  # checked as a serial adapter
    assert devices.arm_link_problem(station, "right", sys_net) is not None  # checked as a CAN link
    run = FakeRun()
    ok, message = devices.apply_fix(station, "can-up:can_follower_r", sys_net, run)
    assert ok and message == "brought can_follower_r up"
    assert devices.apply_fix(station, "can-up:/dev/serial/by-id/0", sys_net, run)[0] is False  # never a serial port
