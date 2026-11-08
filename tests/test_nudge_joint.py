"""nudge_joint: the nudge's limits, that it moves only the chosen joint, and that every stop eases the arms back.

No hardware: a fake pair of arms stands in for the real ones.
"""

import importlib.util
import math
import pathlib
import types
from types import ModuleType

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "nudge_joint.py"
HZ = 30.0
START = {"left": np.full(6, 0.10), "right": np.full(6, -0.20)}
GRIPS = {"left": 0.0, "right": 0.5}


@pytest.fixture(scope="module")
def nj() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nudge_joint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeArms:
    """Both arms: a joint follows its command exactly, unless it is listed as stuck (it then never moves)."""

    def __init__(self, energized: bool = True, stuck: tuple[tuple[str, int], ...] = ()) -> None:
        self.energized = energized
        self.start_joints = {side: joints.copy() for side, joints in START.items()}
        self.measured = {side: joints.copy() for side, joints in START.items()}
        self.stuck = stuck
        self.commands: list[tuple[dict[str, np.ndarray], dict[str, float]]] = []
        self.closed = False

    def read(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        return {side: joints.copy() for side, joints in self.measured.items()}, dict(GRIPS)

    def command(self, joints: dict[str, np.ndarray], grips: dict[str, float]) -> None:
        assert self.energized, "a dry run never commands the arms"
        self.commands.append(({side: j.copy() for side, j in joints.items()}, dict(grips)))
        for side, target in joints.items():
            for joint, value in enumerate(target):
                if (side, joint) not in self.stuck:
                    self.measured[side][joint] = value

    def close(self) -> None:
        self.closed = True


def _targets() -> dict[str, np.ndarray]:
    return {side: joints.copy() for side, joints in START.items()}


def _nudge(nj: ModuleType, robot: FakeArms, side: str, joint: int, degrees: float, seconds: float = 4.0):
    return nj.nudge(robot, _targets(), dict(GRIPS), side, joint, degrees, seconds, HZ, lambda _s: None)


def test_ramp_eases_in_and_out_and_ends_exactly(nj: ModuleType) -> None:
    positions = nj.ramp(0.2, 0.5, 2.0, HZ)
    assert len(positions) == 60
    assert positions[-1] == 0.5
    steps = np.diff([0.2, *positions])
    assert np.all(steps > 0)
    assert steps[0] < steps[len(steps) // 2] > steps[-1]  # slow at both ends, fastest in the middle


def test_shortest_allowed_time_keeps_the_peak_speed_under_the_limit(nj: ModuleType) -> None:
    for degrees in (1.0, 3.0, nj.MAX_DEGREES):
        positions = nj.ramp(0.0, math.radians(degrees), nj.min_seconds(degrees), HZ)
        peak_deg_s = math.degrees(np.max(np.diff([0.0, *positions]))) * HZ
        assert peak_deg_s <= nj.MAX_PEAK_DEG_S * 1.02  # the ticks sample the ease, so a hair over is the rounding


def test_a_nudge_moves_only_the_chosen_joint_and_ends_where_it_started(nj: ModuleType) -> None:
    robot = FakeArms()
    result = _nudge(nj, robot, "left", 4, 3.0)

    assert result.followed
    assert result.moved_deg == pytest.approx(3.0)
    assert result.back_error_deg == pytest.approx(0.0, abs=1e-9)
    for joints, grips in robot.commands:
        assert grips == GRIPS
        np.testing.assert_array_equal(joints["right"], START["right"])
        np.testing.assert_array_equal(np.delete(joints["left"], 4), np.delete(START["left"], 4))
    peak = max(joints["left"][4] for joints, _ in robot.commands)
    assert peak == pytest.approx(START["left"][4] + math.radians(3.0))
    np.testing.assert_array_equal(robot.commands[-1][0]["left"], START["left"])


def test_a_negative_nudge_goes_the_other_way(nj: ModuleType) -> None:
    robot = FakeArms()
    result = _nudge(nj, robot, "right", 0, -3.0)
    assert result.followed
    assert result.moved_deg == pytest.approx(-3.0)
    assert min(joints["right"][0] for joints, _ in robot.commands) == pytest.approx(
        START["right"][0] - math.radians(3)
    )


def test_a_joint_that_does_not_move_is_reported_not_followed(nj: ModuleType) -> None:
    result = _nudge(nj, FakeArms(stuck=(("left", 2),)), "left", 2, 3.0)
    assert not result.followed
    assert result.moved_deg == pytest.approx(0.0)
    assert "DID NOT FOLLOW" in nj.describe(types.SimpleNamespace(family="yam"), result)


def test_a_joint_blocked_by_a_large_nudge_stops_the_run(nj: ModuleType) -> None:
    with pytest.raises(nj.NudgeAbortedError, match="left joint 3"):
        _nudge(nj, FakeArms(stuck=(("left", 2),)), "left", 2, nj.MAX_DEGREES)


def test_a_joint_that_strays_on_its_own_stops_the_run(nj: ModuleType) -> None:
    measured = {"left": START["left"].copy(), "right": START["right"].copy()}
    measured["right"][5] += math.radians(nj.TRACKING_ABORT_DEG + 1)
    with pytest.raises(nj.NudgeAbortedError, match="right joint 6"):
        nj.check_tracking(measured, _targets())


@pytest.mark.parametrize(
    "argv",
    [
        ["--degrees", "11"],  # over the cap
        ["--degrees", "0"],
        ["--degrees", "-11"],
        ["--degrees", "10", "--seconds", "2"],  # too fast for the size
        ["--joint", "7"],
        ["--joint", "0"],
        ["--side", "middle"],
    ],
)
def test_arguments_outside_the_limits_are_refused(nj: ModuleType, argv: list[str]) -> None:
    base = {"--config": "c.yaml", "--side": "left", "--joint": "6"}
    base.update(dict(zip(argv[::2], argv[1::2], strict=True)))
    with pytest.raises(SystemExit):
        nj.parse_args([part for pair in base.items() for part in pair])


def test_the_defaults_are_inside_the_limits(nj: ModuleType) -> None:
    args = nj.parse_args(["--config", "c.yaml", "--side", "both", "--joint", "6", "5"])
    assert args.degrees == 3.0
    assert args.seconds >= nj.min_seconds(args.degrees)
    assert args.joint == [6, 5]


def _station(nj: ModuleType, family: str = "yam") -> types.SimpleNamespace:
    return types.SimpleNamespace(family=family, control_hz=int(HZ))


def test_a_dry_run_commands_nothing_and_shuts_the_arms(nj: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeArms(energized=False)
    monkeypatch.setattr(nj.arms, "open_arms", lambda station, *, energize: robot if not energize else pytest.fail())
    code = nj.run(
        _station(nj), [("left", 5)], 3.0, 4.0, dry_run=True, ask=lambda _p: pytest.fail("a dry run asks nothing")
    )
    assert code == 0
    assert robot.commands == []
    assert robot.closed


def test_a_full_run_asks_before_powering_and_before_every_move(
    nj: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeArms()
    prompts: list[str] = []
    opened: list[bool] = []

    def open_arms(station: object, *, energize: bool) -> FakeArms:
        opened.append(energize)
        assert prompts, "the arms were opened before the person was asked"
        return robot

    monkeypatch.setattr(nj.arms, "open_arms", open_arms)
    code = nj.run(
        _station(nj),
        [("left", 5), ("right", 5)],
        3.0,
        4.0,
        dry_run=False,
        sleep=lambda _s: None,
        ask=lambda prompt: prompts.append(prompt) or "",
    )
    assert code == 0
    assert opened == [True]
    assert len(prompts) == 3  # powering, then each of the two moves
    assert "emergency stop" in prompts[0]
    assert robot.closed


def test_a_joint_that_does_not_follow_makes_the_exit_code_nonzero(
    nj: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeArms(stuck=(("left", 5),))
    monkeypatch.setattr(nj.arms, "open_arms", lambda station, *, energize: robot)
    code = nj.run(_station(nj), [("left", 5)], 3.0, 4.0, dry_run=False, sleep=lambda _s: None, ask=lambda _p: "")
    assert code == 1
    assert robot.closed


def test_ctrl_c_in_the_middle_of_a_move_eases_back_to_the_start_before_shutting_down(
    nj: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeArms()
    monkeypatch.setattr(nj.arms, "open_arms", lambda station, *, energize: robot)
    ticks = 0

    def sleep(_seconds: float) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 40:  # well into the move out
            raise KeyboardInterrupt

    code = nj.run(_station(nj), [("left", 5)], 3.0, 4.0, dry_run=False, sleep=sleep, ask=lambda _p: "")

    assert code == 130
    assert max(abs(j["left"][5] - START["left"][5]) for j, _ in robot.commands) > 0, "the test never moved the joint"
    np.testing.assert_array_equal(robot.commands[-1][0]["left"], START["left"])  # eased back before shutting down
    assert robot.closed


def test_a_stop_on_a_blocked_joint_still_eases_back_and_shuts_down(
    nj: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeArms(stuck=(("right", 1),))
    monkeypatch.setattr(nj.arms, "open_arms", lambda station, *, energize: robot)
    code = nj.run(
        _station(nj), [("right", 1)], nj.MAX_DEGREES, 4.0, dry_run=False, sleep=lambda _s: None, ask=lambda _p: ""
    )
    assert code == 1
    np.testing.assert_array_equal(robot.commands[-1][0]["right"], START["right"])
    assert robot.closed


def test_a_dead_bus_while_easing_back_still_shuts_the_arms_down(
    nj: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeArms()
    monkeypatch.setattr(nj.arms, "open_arms", lambda station, *, energize: robot)
    ticks = 0

    def sleep(_seconds: float) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 20:
            raise KeyboardInterrupt
        if ticks == 25:  # while easing back
            robot.command = lambda *_a: (_ for _ in ()).throw(OSError("bus gone"))  # type: ignore[method-assign]

    nj.run(_station(nj), [("left", 0)], 3.0, 4.0, dry_run=False, sleep=sleep, ask=lambda _p: "")
    assert robot.closed


def test_main_refuses_to_move_the_arms_without_a_terminal(nj: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nj.config, "load", lambda _path: _station(nj))
    monkeypatch.setattr(nj.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(nj.station_lock, "acquire", lambda _holder: pytest.fail("took the lock"))
    monkeypatch.setattr(nj.arms, "open_arms", lambda *a, **k: pytest.fail("opened the arms"))
    with pytest.raises(SystemExit, match="terminal"):
        nj.main(["--config", "c.yaml", "--side", "left", "--joint", "6"])


def test_main_holds_the_station_lock_for_the_run_and_releases_it(
    nj: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(nj.config, "load", lambda _path: _station(nj))
    monkeypatch.setattr(nj.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        nj.station_lock,
        "acquire",
        lambda holder: events.append("lock") or types.SimpleNamespace(close=lambda: events.append("unlock")),
    )
    monkeypatch.setattr(nj, "run", lambda *a, **k: events.append("run") or 0)
    assert nj.main(["--config", "c.yaml", "--side", "left", "--joint", "6"]) == 0
    assert events == ["lock", "run", "unlock"]


def test_main_reports_a_busy_station_without_opening_the_arms(nj: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nj.config, "load", lambda _path: _station(nj))
    monkeypatch.setattr(nj.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))

    def busy(_holder: str) -> None:
        raise nj.station_lock.StationBusyError("dhk-collect holds the arms")

    monkeypatch.setattr(nj.station_lock, "acquire", busy)
    monkeypatch.setattr(nj.arms, "open_arms", lambda *a, **k: pytest.fail("opened the arms"))
    with pytest.raises(SystemExit, match="dhk-collect"):
        nj.main(["--config", "c.yaml", "--side", "left", "--joint", "6"])
