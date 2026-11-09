"""The collector loop over fake arms and a fake Quest: teleop, recording, dry run, homing and the lag guard."""

import dataclasses
import json
import pathlib
import re
import sys

import dream
import numpy as np
import pytest
from conftest import FLANGE, JOINT_NAMES, arm_urdf, filled_template

from dream_hackathon import cameras, collect, config, dataset, ik, layouts, quest

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"
START = np.array([0.0, -0.3, 1.2, 0.6, 0.0, 0.0])  # elbow bent: 5 cm forward is in reach
## A Dream-1 home a little away from the start, so a reset and a return to the start end in different places.
HOME = {"left": START + [0.05, -0.05, 0.1, 0.0, 0.05, 0.0], "right": START + [-0.05, 0.05, -0.1, 0.1, 0.0, -0.05]}


class FakeArms:
    """Arms that track every command exactly, unless stuck."""

    def __init__(self, energized: bool = True) -> None:
        self.energized = energized
        self.start_joints = {side: START.copy() for side in ("left", "right")}
        self.joints = {side: START.copy() for side in ("left", "right")}
        self.grips = {"left": 1.0, "right": 1.0}
        self.commands = 0
        self.stuck = False

    def read(self):
        return {side: q.copy() for side, q in self.joints.items()}, dict(self.grips)

    def command(self, joints, grips) -> None:
        self.commands += 1
        if not self.stuck:
            self.joints = {side: q.copy() for side, q in joints.items()}
            self.grips = dict(grips)

    def close(self) -> None:
        pass


class FakeQuest:
    """A Quest whose next reading the test sets."""

    def __init__(self) -> None:
        self.poses = {"left": np.eye(4), "right": np.eye(4)}
        self.grip = {"left": 0.0, "right": 0.0}
        self.trigger = {"left": 0.0, "right": 0.0}
        self.press: set[str] = set()

    def read(self) -> quest.QuestState:
        pressed, self.press = frozenset(self.press), set()
        return quest.QuestState(
            {s: p.copy() for s, p in self.poses.items()}, dict(self.grip), dict(self.trigger), pressed
        )


def _collector(
    tmp_path: pathlib.Path,
    arms: FakeArms,
    headset: FakeQuest,
    logs: list[str],
    home: dict | None = None,
    home_note: str = "no home pose was given",
) -> collect.Collector:
    station = config.load(filled_template("yam", tmp_path))
    # A box large enough that it never clamps the test arm's tool.
    roomy = {
        side: dataclasses.replace(arm, workspace_min_m=np.full(3, -2.0), workspace_max_m=np.full(3, 2.0))
        for side, arm in station.arms.items()
    }
    station = dataclasses.replace(
        station, dataset_dir=tmp_path / "task", record_width=320, record_height=180, arms=roomy
    )
    model = ik.urdf_model(arm_urdf(), FLANGE, (0.0, 0.0, 0.08))
    iks = {side: ik.ArmIK(model, ik.TCP_SITE, JOINT_NAMES, 2.0, 0.02) for side in ("left", "right")}
    keys = list(layouts.video_keys("yam").values())
    writer = dataset.DatasetWriter(station.dataset_dir, 30, "yam", [f"v{i}" for i in range(14)], keys, 320, 180)
    frame = np.zeros((180, 320, 3), np.uint8)
    frames = lambda: {role: frame for role in layouts.CAMERA_ROLES}  # noqa: E731
    return collect.Collector(
        station,
        "pick up the cube",
        arms,
        headset,
        frames,
        iks,
        writer,
        log=logs.append,
        home=home,
        home_note=home_note,
    )


def test_a_held_grip_drives_the_tool_with_the_hand(tmp_path: pathlib.Path) -> None:
    arms, headset = FakeArms(), FakeQuest()
    collector = _collector(tmp_path, arms, headset, [])
    solver = collector._iks["right"]
    before = solver.fk(arms.joints["right"])[:3, 3]
    headset.grip["right"] = 1.0
    collector.tick()  # engages at the hand's pose
    headset.poses["right"][2, 3] = -0.05  # hand 5 cm forward (Quest -z)
    headset.trigger["right"] = 1.0  # trigger pulled: close the gripper
    for _ in range(120):
        collector.tick()
    after = solver.fk(arms.joints["right"])[:3, 3]
    np.testing.assert_allclose(after - before, [0.05, 0.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(arms.joints["left"], START)  # the other arm held still
    assert arms.grips == {"left": 1.0, "right": 0.0}


def test_a_and_a_again_saves_an_episode_dream_accepts(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    headset.press.add("A")
    for _ in range(2 * dataset.min_frames(30)):  # 60 Hz loop, every 2nd tick recorded
        collector.tick()
    headset.press.add("A")
    collector.tick()
    assert collector.episode is None and logs[-1].startswith("saved episode 0")
    assert dream.datasets.validate_dataset(tmp_path / "task", layouts.embodiment("yam")).ok


def test_b_discards_and_a_dry_run_never_records_or_commands(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    headset.press.add("A")
    collector.tick()
    headset.press.add("B")
    collector.tick()
    assert collector.episode is None and logs[-1] == "episode discarded"

    dry = FakeArms(energized=False)
    collector = _collector(tmp_path, dry, headset, logs)
    headset.press.add("A")
    headset.grip["left"] = 1.0
    for _ in range(10):
        collector.tick()
    assert collector.episode is None and dry.commands == 0 and "a dry run does not record" in logs


def test_homing_returns_to_the_start_but_not_while_recording(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    collector._command.joints["left"] = START + 0.1
    headset.press.add("RJ")
    for _ in range(60):
        collector.tick()
    assert not collector.homing
    np.testing.assert_allclose(arms.joints["left"], START, atol=1e-3)
    headset.press.add("A")
    collector.tick()
    headset.press.add("RJ")
    collector.tick()
    assert not collector.homing and "homing is off while recording" in logs[-1]


def test_an_arm_that_cannot_follow_is_resynced_and_released(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    arms.stuck = True
    collector._command.joints["right"] = START + 0.5
    collector.tick()
    np.testing.assert_allclose(collector._command.joints["right"], START)
    assert "right arm is" in logs[-1]


def test_a_dry_run_follows_the_real_arm_until_engaged(tmp_path: pathlib.Path) -> None:
    dry, headset = FakeArms(energized=False), FakeQuest()
    collector = _collector(tmp_path, dry, headset, [])
    dry.joints["left"] = START + 0.2  # moved by hand
    collector.tick()
    np.testing.assert_allclose(collector._command.joints["left"], START + 0.2)
    headset.grip["left"] = 1.0
    collector.tick()
    dry.joints["left"] = START  # moved again, but the clutch now drives the (virtual) command
    collector.tick()
    np.testing.assert_allclose(collector._command.joints["left"], START + 0.2, atol=1e-3)


def test_snapshot_reports_the_session(tmp_path: pathlib.Path) -> None:
    arms_, headset = FakeArms(), FakeQuest()
    collector = _collector(tmp_path, arms_, headset, [])
    headset.press.add("A")
    collector.tick()
    snapshot = collector.snapshot()
    assert snapshot["mode"] == "live" and snapshot["recording"] and snapshot["episodes"] == 0
    assert set(snapshot["arms"]) == {"left", "right"} and len(snapshot["arms"]["left"]["tool_m"]) == 3
    json.dumps(snapshot)  # the dashboard sends it as JSON
    assert collect._status_line(snapshot).startswith("REC")


def test_a_stalled_camera_discards_the_episode_and_teleop_goes_on(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs)

    def stalled():
        raise cameras.CameraError("camera 'wrist_l' stalled")

    collector._frames = stalled
    headset.press.add("A")
    collector.tick()
    assert collector.episode is None
    assert any("stalled" in line for line in logs)
    assert any("stalled" in warning for warning in collector.snapshot()["warnings"])
    collector.tick()  # the loop keeps running


def test_a_hang_up_ends_the_session_like_ctrl_c_and_logs_to_a_file(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(collect, "LOG_PATH", tmp_path / "collect.log")
    stdout, stderr = sys.stdout, sys.stderr
    try:
        with pytest.raises(KeyboardInterrupt):
            collect._hang_up(1, None)
        collect._say("returning the arms")
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = stdout, stderr
    text = (tmp_path / "collect.log").read_text()
    assert "SSH connection was lost" in text and "returning the arms" in text


def test_homing_gives_up_after_its_timeout(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs)
    arms_.stuck = True
    collector._command.joints["left"] = START + 0.1
    arms_.joints["left"] = START + 0.1
    collector.go_home(timeout_s=0.05)
    assert not collector.homing and any("did not reach their start" in line for line in logs)


def test_the_task_falls_back_to_a_neutral_text():
    assert collect.choose_task(" stack the cups ", "from the config") == "stack the cups"
    assert collect.choose_task(None, "from the config") == "from the config"
    assert collect.choose_task("  ", None) == collect.DEFAULT_TASK
    assert collect.DEFAULT_TASK.strip()


def test_a_missing_tool_center_point_is_one_line_not_a_traceback(tmp_path: pathlib.Path) -> None:
    path = filled_template("maker", tmp_path)
    path.write_text(re.sub(r"tcp_xyz_m: .*", "tcp_xyz_m: null", path.read_text()))  # a config without its point
    with pytest.raises(SystemExit) as stopped:
        collect.main(["--config", str(path), "--dataset-dir", str(tmp_path / "data")])
    assert str(stopped.value.code).startswith("config: 'arms.left.tcp_xyz_m' is not set")


def _accepting(collector: collect.Collector) -> collect.SessionControls:
    """The collector's controls, taking requests as they do once a session's arms are open."""
    collector._controls.accepting = True
    return collector._controls


def _record(collector: collect.Collector) -> None:
    """Ticks long enough for an episode Dream accepts (60 Hz loop, every 2nd tick recorded)."""
    for _ in range(2 * dataset.min_frames(30)):
        collector.tick()


def _settle(collector: collect.Collector, ticks: int = 120) -> None:
    for _ in range(ticks):
        collector.tick()


def test_the_dashboard_starts_saves_and_discards_like_a_and_b(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs)
    controls = _accepting(collector)
    assert controls.request("save") == (True, "save: requested")
    collector.tick()
    assert logs[-1] == "no episode is recording: nothing to save"
    controls.request("start")
    collector.tick()
    assert collector.episode is not None and logs[-1].startswith("recording episode 0")
    controls.request("start")
    collector.tick()
    assert logs[-1] == "an episode is already recording: save or discard it first"
    _record(collector)
    controls.request("save")
    collector.tick()
    assert collector.episode is None and logs[-1].startswith("saved episode 0")
    assert controls.writer.total_episodes == 1
    controls.request("start")
    collector.tick()
    controls.request("discard")
    collector.tick()
    assert collector.episode is None and logs[-1] == "episode discarded" and not controls.lock.locked()
    assert controls.writer.total_episodes == 1


def test_a_request_before_the_arms_open_is_refused_not_queued(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs)
    controls = collector._controls
    ok, message = controls.request("start")
    assert not ok and "press A on the right controller first" in message
    controls.accepting = True
    collector.tick()
    assert collector.episode is None  # the refused request was not kept for later
    assert not controls.request("jump")[0]
    assert controls.request("auto-reset-off") == (True, "reset to home after each episode: off")
    assert not controls.auto_reset
    controls.request("start")
    controls.accepting = False  # the session ends: what is queued is dropped
    assert controls.take_requests() == []


def test_a_reset_reaches_dreams_home_with_the_grippers_open(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs, home=HOME)
    controls = _accepting(collector)
    collector._command.grips.update(left=0.0, right=0.2)  # grippers closed
    controls.request("reset")
    collector.tick()
    assert collector.homing and collector.resetting and "resetting the arms" in logs[-1]
    assert collector.snapshot()["resetting"] and collect._status_line(collector.snapshot()).find("resetting") > 0
    _settle(collector)
    assert not collector.homing and not collector.resetting and logs[-1] == "the arms are at Dream-1's home"
    for side in ("left", "right"):
        np.testing.assert_allclose(arms_.joints[side], HOME[side], atol=1e-3)
    assert arms_.grips == {"left": 1.0, "right": 1.0}


def test_a_reset_is_refused_while_recording_in_a_dry_run_and_without_a_home(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs, home=HOME)
    controls = _accepting(collector)
    controls.request("start")
    collector.tick()
    controls.request("reset")
    collector.tick()
    assert not collector.homing and "reset is off while recording" in logs[-1]

    dry = FakeArms(energized=False)
    collector = _collector(tmp_path, dry, headset, logs, home=HOME)
    _accepting(collector).request("reset")
    collector.tick()
    assert not collector.homing and logs[-1] == "no reset: a dry run never moves the arms" and dry.commands == 0
    assert collector.snapshot()["reset_note"] == "a dry run never moves the arms"

    collector = _collector(tmp_path, FakeArms(), headset, logs, home_note="/x/station.yaml does not exist")
    _accepting(collector).request("reset")
    collector.tick()
    assert not collector.homing and logs[-1] == "no reset: reset to home is off: /x/station.yaml does not exist"
    snapshot = collector.snapshot()
    assert not snapshot["can_reset"] and snapshot["reset_note"].endswith("does not exist")
    json.dumps(snapshot)


def test_auto_reset_follows_a_save_and_b_but_not_a_camera_error(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs, home=HOME)
    controls = _accepting(collector)
    assert collector.snapshot()["auto_reset"] is True  # on by default
    headset.press.add("A")
    _record(collector)
    headset.press.add("A")
    collector.tick()
    assert any(line.startswith("saved episode 0") for line in logs) and collector.resetting
    _settle(collector)
    np.testing.assert_allclose(arms_.joints["left"], HOME["left"], atol=1e-3)

    headset.press.add("A")
    collector.tick()
    headset.press.add("B")
    collector.tick()
    # B is the operator's discard: a reset follows (the arms are already home, so it ends in the same tick).
    assert logs[-3:] == ["episode discarded", logs[-2], "the arms are at Dream-1's home"]
    assert logs[-2].startswith("resetting the arms to Dream-1's home")
    _settle(collector)

    def stalled():
        raise cameras.CameraError("camera 'wrist_l' stalled")

    def resets() -> int:
        return sum(line.startswith("resetting the arms") for line in logs)

    before = resets()
    collector._frames = stalled
    headset.press.add("A")
    collector.tick()
    collector.tick()  # frames are taken on every 2nd tick
    assert collector.episode is None and any("stalled" in line for line in logs)
    assert resets() == before and not collector.homing  # discarded by the camera: the arms stay where they are

    controls.request("auto-reset-off")
    collector._frames = lambda: {role: np.zeros((180, 320, 3), np.uint8) for role in layouts.CAMERA_ROLES}
    controls.request("start")
    collector.tick()
    controls.request("discard")
    collector.tick()
    assert collector.episode is None and resets() == before


def test_no_episode_starts_while_the_arms_move_home(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs, home=HOME)
    controls = _accepting(collector)
    controls.request("reset")
    collector.tick()
    headset.press.add("A")
    collector.tick()
    assert collector.episode is None and logs[-1] == "the arms are moving to home: wait until they stop"
    controls.request("start")
    collector.tick()
    assert collector.episode is None and logs[-1] == "the arms are moving to home: wait until they stop"
    _settle(collector)
    headset.press.add("A")
    collector.tick()
    assert collector.episode is not None


def test_the_joystick_and_the_shutdown_still_return_to_the_start(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs, home=HOME)
    controls = _accepting(collector)
    controls.request("reset")
    _settle(collector)
    np.testing.assert_allclose(arms_.joints["right"], HOME["right"], atol=1e-3)
    headset.press.add("RJ")
    collector.tick()
    assert collector.homing and not collector.resetting
    _settle(collector)
    np.testing.assert_allclose(arms_.joints["right"], START, atol=1e-3)

    controls.request("reset")
    collector.tick()
    assert collector.resetting
    collector.go_home(timeout_s=5.0)  # the end of the session, in the middle of a reset: back to the start
    assert not collector.homing and not collector.resetting
    for side in ("left", "right"):
        np.testing.assert_allclose(arms_.joints[side], START, atol=1e-3)
