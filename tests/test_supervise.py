"""tools/supervise.py: the stuck watchdog, the ladder's decisions, and whole runs against a fake station with a fake
clock and scripted models. No arm, no camera and no model is touched."""

import dataclasses
import datetime
import importlib.util
import io
import json
import pathlib
import sys

import av
import numpy as np
import pytest
from supervisor_fakes import (
    REWRITES,
    SALMON,
    FakeClock,
    FakeSource,
    ScriptedCompare,
    ScriptedGate,
    ScriptedRewriter,
)

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
TASK_FILE = pathlib.Path(__file__).resolve().parents[1] / "configs" / "tasks" / "usb-cables.yaml"


@pytest.fixture
def sv(monkeypatch: pytest.MonkeyPatch):
    """The tool's module, loaded the way `python tools/supervise.py` does: its folder is on the import path."""
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("supervise_tool", TOOLS / "supervise.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tc(sv):
    return sys.modules["task_critic"]


@pytest.fixture
def sio(sv):
    return sys.modules["station_io"]


@pytest.fixture
def spec(tc):
    return tc.load_task(TASK_FILE)


# --- fakes -----------------------------------------------------------------------------------------------------


class World:
    """A supervisor wired to a fake station, a fake clock, a scripted critic and a fake camera."""

    def __init__(
        self, sv, tc, sio, spec, tmp_path, *, state="running", answer="no", compare=None, orange=False, **opts
    ):
        self.sv, self.tc, self.sio, self.spec = sv, tc, sio, spec
        self.clock = FakeClock()
        self.station = sio.FakeStation(clock=self.clock.now, prompt="p0", state=state)
        self.gate = ScriptedGate(tc, answer)
        self.rewriter = ScriptedRewriter(REWRITES)
        self.source = FakeSource(sio, orange)
        self.log = sv.EventLog(tmp_path / "log", self.clock, echo=None)
        self.critic = tc.TaskCritic(spec, self.gate, self.rewriter)
        self.compare = compare
        self.timed: list = []
        self.keys_at: list = []
        self.clock.hooks.append(self._on_move)
        self.client = self.station.client()
        self.options = sv.Options(**opts)
        self.supervisor = None

    def _on_move(self, t: float) -> None:
        if self.station.state == "running":
            self.station.add_chunks(1)  # the run keeps sending the arms moves
        for item in [i for i in self.timed if t >= i[0]]:
            self.timed.remove(item)
            item[1]()

    def at(self, t: float, action) -> None:
        self.timed.append((t, action))

    def key(self, t: float, key: str) -> None:
        self.keys_at.append((t, key))

    def _keys(self):
        for item in self.keys_at:
            if self.clock.t >= item[0]:
                self.keys_at.remove(item)
                return item[1]
        return None

    def run(self, client=None):
        self.supervisor = self.sv.Supervisor(
            self.spec,
            self.critic,
            client or self.client,
            {"scene": self.source},
            log=self.log,
            clock=self.clock,
            compare=self.compare,
            options=self.options,
            keys=self._keys,
        )
        return self.supervisor.run()

    def events(self, kind=None):
        path = self.log.folder / "events.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        return [r for r in records if kind is None or r["event"] == kind]

    def routes(self):
        return [route for route, _ in self.station.calls]


@pytest.fixture
def world(sv, tc, sio, spec, tmp_path):
    def make(**kwargs):
        return World(sv, tc, sio, spec, tmp_path, **kwargs)

    return make


# --- the progress detector -------------------------------------------------------------------------------------


def test_object_motion_sees_salmon_and_ignores_the_table(sv, spec) -> None:
    motion = sv.ObjectMotion(spec.supervisor.object_motion)
    view = FakeSource.orange_frame()[152:300, 290:440]
    assert motion.mask(view).any()
    assert not motion.mask(FakeSource.grey_frame()[152:300, 290:440]).any()


def test_object_motion_change_is_zero_for_the_same_cables_and_one_for_moved_ones(sv, spec) -> None:
    motion = sv.ObjectMotion(spec.supervisor.object_motion)
    empty = np.zeros((20, 20), bool)
    left, right = empty.copy(), empty.copy()
    left[2:6, 2:6] = True
    right[12:16, 12:16] = True
    assert motion.change(left, left) == 0.0
    assert motion.change(left, right) == 1.0
    assert motion.change(empty, empty) == 0.0  # nothing of that colour in either: no change


# --- the watchdog ----------------------------------------------------------------------------------------------


def watch(sv, tc, spec, *, answer="no", detector=True, compare=True):
    config = dataclasses.replace(spec.supervisor, object_motion=spec.supervisor.object_motion if detector else None)
    fake = ScriptedCompare(tc, answer) if compare else None
    return sv.Watchdog(dataclasses.replace(spec, supervisor=config), fake), fake


def probe(dog, t, *, commanded=True, shift=0):
    """One probe at time t: the cables where they were, or `shift` pixels further right when they moved."""
    frame = FakeSource.grey_frame()
    frame[200:230, 300 + shift : 330 + shift] = SALMON
    return dog.update(t, frames={"scene": frame}, pictures={"strip": b"strip", "scene": b"scene"}, commanded=commanded)


def test_watchdog_does_not_call_a_moving_scene_stuck(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec)
    for i, t in enumerate(range(0, 100, 10)):
        reading = probe(dog, t, shift=50 * (i % 2))  # the cables jump between two places in the strip
        assert not reading.stuck
    assert compare.calls == []  # the cheap detector alone settled it, no model call


def test_watchdog_asks_the_compare_only_after_the_quiet_time(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec)
    assert not probe(dog, 0).stuck  # the first probe has nothing to compare with
    assert not probe(dog, 10).stuck
    assert compare.calls == []
    reading = probe(dog, 20)
    assert reading.stuck  # quiet since 0: twenty seconds, and the compare says no
    assert reading.compare == "no"
    assert len(compare.calls) == 1
    before, after, seconds = compare.calls[0]
    assert (before, after, seconds) == (b"strip", b"strip", 20.0)


def test_watchdog_needs_the_compare_to_agree_and_a_yes_restarts_the_clock(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec, answer="yes")
    probe(dog, 0)
    probe(dog, 10)
    reading = probe(dog, 20)
    assert not reading.stuck
    assert reading.compare == "yes"
    assert not probe(dog, 30).stuck  # the clock restarted at 20
    assert len(compare.calls) == 1
    assert probe(dog, 40).compare == "yes"  # asked again after another quiet stretch


def test_watchdog_treats_unsure_and_a_failed_compare_as_not_stuck(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec, answer="unsure")
    probe(dog, 0)
    probe(dog, 10)
    assert not probe(dog, 20).stuck

    class Failing:
        def compare(self, spec, before, after, seconds):
            raise tc.CriticError("no network")

    dog, _ = watch(sv, tc, spec)
    dog.compare = Failing()
    probe(dog, 0)
    probe(dog, 10)
    reading = probe(dog, 20)
    assert not reading.stuck
    assert "the compare failed" in reading.compare_reason


def test_watchdog_without_a_compare_logs_a_quiet_scene_and_never_acts_on_it(sv, tc, spec) -> None:
    dog, _ = watch(sv, tc, spec, compare=False)
    probe(dog, 0)
    probe(dog, 10)
    reading = probe(dog, 30)
    assert not reading.stuck
    assert "unconfirmed" in reading.note
    assert reading.quiet_s == 30


def test_watchdog_does_not_count_time_while_the_arms_are_not_commanded(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec)
    probe(dog, 0)
    probe(dog, 10)
    reading = probe(dog, 20, commanded=False)
    assert not reading.stuck
    assert "not being commanded" in reading.note
    assert not probe(dog, 30).stuck  # quiet only since 20 again
    assert compare.calls == []


def test_watchdog_reset_forgets_the_quiet_time(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec)
    probe(dog, 0)
    probe(dog, 10)
    dog.reset(15)
    probe(dog, 20)  # the first after the reset: nothing to compare with
    assert not probe(dog, 30).stuck
    assert compare.calls == []


def test_watchdog_without_a_detector_asks_every_other_probe_and_needs_two_noes(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec, detector=False)
    assert not probe(dog, 0).stuck
    assert not probe(dog, 10).stuck  # first compare: no (1 of 2)
    assert len(compare.calls) == 1
    assert not probe(dog, 20).stuck  # waiting
    assert probe(dog, 30).stuck  # second compare: no again
    assert len(compare.calls) == 2


def test_watchdog_without_a_detector_is_cleared_by_one_yes(sv, tc, spec) -> None:
    dog, compare = watch(sv, tc, spec, detector=False)
    probe(dog, 0)
    assert not probe(dog, 10).stuck  # compare: no (1 of 2)
    compare.answer = "yes"
    probe(dog, 20)
    assert not probe(dog, 30).stuck  # compare: yes, which clears the count
    compare.answer = "no"
    probe(dog, 40)
    assert not probe(dog, 50).stuck  # compare: no, but only once since the yes
    probe(dog, 60)
    assert probe(dog, 70).stuck  # and a second no in a row


def test_watchdog_with_neither_detector_nor_compare_cannot_be_stuck(sv, tc, spec) -> None:
    dog, _ = watch(sv, tc, spec, detector=False, compare=False)
    for t in range(0, 200, 10):
        reading = probe(dog, t)
        assert not reading.stuck
    assert "no progress detector" in reading.note


# --- the ladder ------------------------------------------------------------------------------------------------


@pytest.fixture
def ladder(sv, tc):
    return sv.Ladder(tc.SupervisorSpec(), now=0.0)


def test_a_healthy_run_waits(ladder) -> None:
    assert ladder.step(10.0, state="running").kind == "wait"
    assert ladder.step(80.0, state="running", probe="not_done").kind == "wait"


def test_two_done_probes_in_a_row_finish_and_one_does_not(ladder) -> None:
    assert ladder.step(20.0, state="running", probe="done").kind == "wait"
    decision = ladder.step(30.0, state="running", probe="done")
    assert decision.kind == "finish"
    assert "2 probes in a row" in decision.reason


@pytest.mark.parametrize("between", ["not_done", "unsure"])
def test_a_probe_that_is_not_done_breaks_the_done_streak(ladder, between) -> None:
    assert ladder.step(20.0, state="running", probe="done").kind == "wait"
    assert ladder.step(30.0, state="running", probe=between).kind == "wait"
    assert ladder.step(40.0, state="running", probe="done").kind == "wait"
    assert ladder.step(50.0, state="running", probe="done").kind == "finish"


def test_no_probe_in_a_tick_leaves_the_done_streak_alone(ladder) -> None:
    ladder.step(20.0, state="running", probe="done")
    assert ladder.step(21.0, state="running").kind == "wait"
    assert ladder.step(30.0, state="running", probe="done").kind == "finish"


def test_the_first_failure_guides_the_second_in_the_window_restarts(ladder) -> None:
    assert ladder.step(91.0, state="running").kind == "guide"  # past 90 s
    ladder.note_guided(91.0)
    assert ladder.deadline == 136.0
    assert ladder.step(100.0, state="running").kind == "wait"
    decision = ladder.step(110.0, state="running", stuck=True)
    assert decision.kind == "restart"
    assert "inside the guide window" in decision.reason


def test_the_time_limit_after_a_guide_restarts(ladder) -> None:
    ladder.step(91.0, state="running")
    ladder.note_guided(91.0)
    assert ladder.step(137.0, state="running").kind == "restart"


def test_a_failure_after_the_window_guides_again_up_to_the_cap(ladder) -> None:
    ladder.note_guided(20.0)
    assert ladder.step(70.0, state="running", stuck=True).kind == "guide"  # 50 s later: outside the 45 s window
    ladder.note_guided(70.0)
    assert ladder.guides == 2
    assert ladder.step(130.0, state="running", stuck=True).kind == "restart"  # two guides is the cap


def test_stalls_are_counted_from_the_baseline(ladder) -> None:
    assert ladder.step(10.0, state="running", stalls=3).kind == "wait"  # at the limit of 3, not over
    decision = ladder.step(11.0, state="running", stalls=4)
    assert decision.kind == "guide"
    assert "stalls rose by 4" in decision.reason
    ladder.note_guided(11.0, stalls=4)
    assert ladder.step(20.0, state="running", stalls=6).kind == "wait"


def test_an_error_that_was_there_at_the_start_is_not_a_failure_but_a_new_one_is(sv, tc) -> None:
    old = {"message": "an old failure"}
    ladder = sv.Ladder(tc.SupervisorSpec(), now=0.0, last_error=old)
    assert ladder.step(5.0, state="running", last_error=old).kind == "wait"
    decision = ladder.step(6.0, state="running", last_error={"message": "the model stopped answering"})
    assert decision.kind == "guide"
    assert "the model stopped answering" in decision.reason


def test_an_idle_station_ends_the_watch_and_starts_nothing(ladder) -> None:
    assert ladder.step(30.0, state="idle").kind == "ended"


def test_a_station_in_error_restarts_without_a_guide_and_the_last_attempt_gives_up(sv, tc) -> None:
    ladder = sv.Ladder(tc.SupervisorSpec(attempts=2), now=0.0)
    decision = ladder.step(30.0, state="error", last_error={"message": "boom"})
    assert (decision.kind, "boom" in decision.reason) == ("restart", True)
    ladder.new_attempt(40.0)
    assert ladder.attempt == 2
    assert ladder.guided_at is None
    assert ladder.deadline == 130.0
    decision = ladder.step(50.0, state="error", last_error={"message": "boom"})
    assert decision.kind == "give_up"
    assert "attempt 2 of 2" in decision.reason


@pytest.mark.parametrize("state", ["starting", "stopping", "homing", "resetting"])
def test_nothing_is_decided_while_the_station_is_changing_state(ladder, state) -> None:
    assert ladder.step(500.0, state=state, stuck=True).kind == "wait"


def test_rebase_starts_clocks_over_without_counting_an_attempt(ladder) -> None:
    ladder.note_guided(50.0)
    ladder.rebase(200.0, stalls=7)
    assert (ladder.attempt, ladder.guides, ladder.guided_at, ladder.deadline) == (1, 0, None, 290.0)
    assert ladder.step(210.0, state="running", stalls=9).kind == "wait"


# --- the pieces around the loop --------------------------------------------------------------------------------


def test_the_event_log_keeps_events_and_pictures_and_prints_a_line(sv, tmp_path) -> None:
    clock = FakeClock()
    lines: list[str] = []
    log = sv.EventLog(tmp_path / "run", clock, echo=lines.append)
    clock.t = 12.34
    log.write("probe", "probe 1: done", probe=1, votes=[{"answer": "yes"}])
    paths = log.save_pictures("probe001-0", {"strip": b"jpeg"})
    log.request({"contents": []})
    log.report({"outcome": "done"})
    record = json.loads((tmp_path / "run/events.jsonl").read_text())
    assert (record["event"], record["t"], record["probe"]) == ("probe", 12.3, 1)
    assert record["utc"].startswith("2026-11-08T12:00:12")
    assert lines == ["[   12.3s] probe 1: done"]
    assert paths == ["pictures/probe001-0-strip.jpg"]
    assert (tmp_path / "run" / paths[0]).read_bytes() == b"jpeg"
    assert json.loads((tmp_path / "run/report.json").read_text()) == {"outcome": "done"}
    assert (tmp_path / "run/requests.jsonl").exists()


def test_a_log_without_a_folder_only_prints(sv) -> None:
    lines: list[str] = []
    log = sv.EventLog(None, FakeClock(), echo=lines.append)
    log.write("x", "hello")
    assert log.save_pictures("t", {"a": b"1"}) == []
    log.request({})
    log.report({})
    assert lines == ["[    0.0s] hello"]


def test_the_controller_spaces_calls_and_forces_a_stop(sv, sio, tmp_path) -> None:
    clock = FakeClock()
    station = sio.FakeStation(clock=clock.now, prompt="p")
    ctl = sv.Controller(station.client(), sv.EventLog(tmp_path, clock, echo=None), clock, observe_only=False)
    ctl.send("start", prompt="go")
    assert clock.t == 0.0  # the first call goes at once
    ctl.send("prompt", prompt="again")
    assert clock.t == 3.0  # the second waits for the gap
    ctl.send("stop", force=True)
    assert clock.t == 3.0  # a forced stop does not
    assert [r for r, _ in station.calls] == ["/start", "/prompt", "/stop"]
    assert ctl.calls == 3


def test_the_controller_in_observe_only_mode_calls_nothing(sv, sio, tmp_path) -> None:
    clock = FakeClock()
    station = sio.FakeStation(clock=clock.now, prompt="p")
    log = sv.EventLog(tmp_path, clock, echo=None)
    ctl = sv.Controller(station.client(), log, clock, observe_only=True)
    assert ctl.send("start", prompt="go") is None
    assert ctl.send("stop", force=True) is None
    assert station.calls == []
    assert ctl.calls == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["would", "would"]
    assert "would call start(prompt='go')" in events[0]["text"]


def test_the_controller_logs_a_refusal_and_lets_the_caller_decide(sv, sio, tmp_path) -> None:
    clock = FakeClock()
    station = sio.FakeStation(clock=clock.now, prompt="p")
    ctl = sv.Controller(station.client(), sv.EventLog(tmp_path, clock, echo=None), clock, observe_only=False)
    station.refuse("/start", "not now")
    with pytest.raises(sio.StationRefusedError, match="not now"):
        ctl.send("start", prompt="go")
    assert "refused" in (tmp_path / "events.jsonl").read_text()


def test_the_key_reader_has_no_keys_without_a_terminal(sv) -> None:
    with sv.KeyReader(io.StringIO()) as keys:
        assert keys.get() is None


# --- whole runs ------------------------------------------------------------------------------------------------


def test_two_done_probes_stop_the_run_and_home_the_arms(world) -> None:
    w = world(answer="yes")
    report = w.run()
    assert report["outcome"] == "done"
    assert w.routes() == ["/stop", "/home"]
    assert w.station.state == "idle"
    assert report["attempts"] == 1
    assert report["control_calls"] == 2
    decision = [e for e in w.events("decision") if e["kind"] == "finish"][0]
    assert decision["t"] > 25  # the first probe is at 15 s and the second 10 s after it: never after one


def test_a_done_probe_after_the_first_and_then_not_done_does_not_stop(world) -> None:
    w = world(answer=lambda: "yes" if w.gate.asked <= 3 else "no")  # only the first probe's three pictures say yes
    w.key(60.0, "x")
    report = w.run()
    assert report["outcome"] == "aborted"
    assert "/home" not in w.routes()


def test_the_time_limit_guides_in_place_then_restarts_and_then_gives_up(world) -> None:
    w = world(answer="no")
    report = w.run()
    assert report["outcome"] == "gave_up"
    assert w.routes() == [
        "/prompt", "/stop", "/home", "/start",
        "/prompt", "/stop", "/home", "/start",
        "/prompt", "/stop", "/home",
    ]  # fmt: skip
    assert report["attempts"] == 3
    assert report["guides"] == 3
    assert report["prompts"][0] == "p0"
    assert report["prompts"][1:] == REWRITES[:5]  # each prompt new, in the order the rewriter wrote them
    assert w.station.applied_prompts[0] == REWRITES[0]
    assert w.station.state == "idle"
    starts = [body for route, body in w.station.calls if route == "/start"]
    assert [s["prompt"] for s in starts] == [REWRITES[1], REWRITES[3]]
    assert all(s["checkpoint"] == "dreamlabs/dream-1-beta" for s in starts)


def test_control_calls_are_at_least_three_seconds_apart(world) -> None:
    w = world(answer="no")
    w.run()
    times = [e["t"] for e in w.events("control")]
    assert len(times) >= 8
    assert all(b - a >= 3.0 for a, b in zip(times, times[1:], strict=False))


def test_a_new_attempt_is_looked_at_one_second_after_it_starts(world, sv) -> None:
    w = world(answer="no")
    w.run()
    restart = w.events("restart")[0]
    started = [e for e in w.events("control") if e["action"] == "start"][0]
    probes = [e for e in w.events("probe") if e["t"] > started["t"]]
    assert sv.ARRIVAL_DELAY_S == 1.0
    assert probes[0]["t"] - started["t"] <= 4  # one second in, plus a tick to collect it: not the warm-up's 15 s
    assert restart["t"] < started["t"]


def test_a_station_error_restarts_without_a_guide_and_without_a_stop(world) -> None:
    w = world(answer=lambda: "yes" if w.clock.t > 40 else "no")
    w.at(20.0, lambda: w.station.fail("the model stopped answering"))
    report = w.run()
    assert report["outcome"] == "done"
    assert w.routes() == ["/home", "/start", "/stop", "/home"]
    assert report["attempts"] == 2
    assert w.events("decision")[0]["kind"] == "restart"


def test_a_burst_of_stalls_guides_in_place(world) -> None:
    w = world(answer="no")
    w.at(20.0, lambda: w.station.add_stalls(5))
    w.key(40.0, "x")
    report = w.run()
    assert report["outcome"] == "aborted"
    assert w.routes() == ["/prompt", "/stop"]
    assert "stalls rose by 5" in w.events("guide")[0]["text"]


def test_a_stall_the_compare_confirms_guides_and_a_second_restarts(world, tc) -> None:
    w = world(answer="no", compare=ScriptedCompare(tc, "no"), orange=True)  # the cables never move
    w.key(95.0, "x")  # after the restart, before the new attempt is stuck for long enough to be guided
    report = w.run()
    assert report["outcome"] == "aborted"
    assert w.routes() == ["/prompt", "/stop", "/home", "/start", "/stop"]
    assert "no progress" in w.events("guide")[0]["text"]
    assert "inside the guide window" in w.events("restart")[0]["text"]
    assert len(w.compare.calls) >= 2


def test_a_stall_the_compare_does_not_confirm_is_only_logged(world, tc) -> None:
    w = world(answer="no", compare=ScriptedCompare(tc, "yes"), orange=True)
    w.key(80.0, "x")
    report = w.run()
    assert report["outcome"] == "aborted"
    assert w.routes() == ["/stop"]  # only the abort key's
    assert w.compare.calls  # it was asked, and it kept saying the task moved


def test_observe_only_makes_no_control_call_at_all(world) -> None:
    w = world(answer="no", observe_only=True)
    report = w.run()
    assert w.station.calls == []
    assert report["control_calls"] == 0
    assert report["outcome"] == "gave_up"
    assert report["attempts"] == 3
    would = [e["action"] for e in w.events("would")]
    assert would.count("prompt") == 3
    assert would.count("start") == 2
    assert would[-1] == "stop_and_home"
    assert w.station.state == "running"  # nothing was stopped
    assert not w.events("control")


def test_observe_only_does_not_stop_the_run_on_the_way_out_either(world) -> None:
    w = world(answer="no", observe_only=True)
    w.key(30.0, "x")
    report = w.run()
    assert report["outcome"] == "aborted"
    assert w.station.calls == []


def test_a_run_that_ends_while_watched_is_not_restarted(world) -> None:
    w = world(answer="no")
    w.at(30.0, w.station.end_run)
    report = w.run()
    assert report["outcome"] == "ended"
    assert w.station.calls == []
    assert "idle" in report["detail"]


def test_an_interrupt_stops_the_run_so_the_arms_hold(world) -> None:
    w = world(answer="no")

    def interrupt() -> None:
        raise KeyboardInterrupt

    w.at(30.0, interrupt)
    report = w.run()
    assert report["outcome"] == "interrupted"
    assert w.routes() == ["/stop"]
    assert w.station.state == "idle"


def test_a_crash_stops_the_run_too(world) -> None:
    w = world(answer="no")

    def broken() -> str:
        raise RuntimeError("a bug")

    w.supervisor = None
    w._keys = broken
    report = w.run()
    assert report["outcome"] == "crashed"
    assert "a bug" in report["detail"]
    assert w.routes() == ["/stop"]
    assert w.events("crash")[0]["traceback"]


def test_a_crash_in_observe_only_mode_stops_nothing(world) -> None:
    w = world(answer="no", observe_only=True)
    w._keys = lambda: 1 / 0
    report = w.run()
    assert report["outcome"] == "crashed"
    assert w.station.calls == []


def test_a_refused_prompt_is_logged_and_asked_again_no_sooner_than_three_seconds(world) -> None:
    w = world(answer="no")
    w.station.refuse("/prompt", "busy")
    w.at(20.0, lambda: w.station.add_stalls(5))
    w.key(60.0, "x")
    w.run()
    refused = w.events("refused")
    assert len(refused) == 1
    assert w.routes().count("/prompt") == 2  # the refused call and the one that went through
    prompt = [e for e in w.events("control") if e["action"] == "prompt"][0]
    assert prompt["t"] - refused[0]["t"] >= 3.0


def test_stale_frames_skip_the_probe_and_decide_nothing(world) -> None:
    w = world(answer="yes")
    w.source.stale = 12.0
    w.at(60.0, lambda: setattr(w.source, "stale", 0.0))
    report = w.run()
    skips = w.events("skip")
    assert skips
    assert "has not grown for 12 s" in skips[0]["text"]
    assert report["outcome"] == "done"
    assert w.events("decision")[0]["t"] > 60  # nothing was decided from the old recording


def test_a_face_in_the_picture_means_nothing_is_sent(world, tc, monkeypatch) -> None:
    w = world(answer="yes")
    monkeypatch.setattr(tc, "count_faces", lambda bgr: 1)
    w.key(40.0, "x")
    w.run()
    assert w.gate.asked == 0
    assert "a face is in the picture" in w.events("skip")[0]["text"]


def test_a_failed_model_call_is_logged_and_never_taken_as_an_answer(world, tc) -> None:
    w = world(answer="yes")

    def fail():
        raise tc.CriticError("429 from the model")

    w.gate.answer = fail
    w.key(40.0, "x")
    report = w.run()
    assert report["outcome"] == "aborted"
    assert "the critic failed: 429 from the model" in w.events("skip")[0]["text"]
    assert report["probes"] == 0


def test_a_guide_without_any_look_at_the_scene_uses_the_task_files_prompts(world, tc) -> None:
    w = world(answer="no")
    w.source.stale = 99.0  # the recording never grows: no probe ever sees anything
    w.at(20.0, lambda: w.station.add_stalls(5))
    w.key(40.0, "x")
    w.run()
    guide = w.events("guide")[0]
    assert guide["source"] == "matrix"
    assert guide["prompt"] == w.spec.prompts[0].text
    assert w.rewriter.calls == []


def test_a_rewrite_the_guards_reject_falls_back_to_the_task_files_prompts(world) -> None:
    w = world(answer="no")
    w.rewriter.texts = []  # the next rewrite is 300 characters: too long
    w.at(20.0, lambda: w.station.add_stalls(5))
    w.key(40.0, "x")
    w.run()
    assert w.events("guide")[0]["source"] == "matrix"


def test_the_operator_can_finish_restart_pause_and_abort(world) -> None:
    w = world(answer="no")
    w.key(30.0, "f")
    assert w.run()["outcome"] == "done"
    assert w.routes() == ["/stop", "/home"]

    w = world(answer="no")
    w.key(30.0, "n")
    w.key(80.0, "x")
    report = w.run()
    assert w.routes() == ["/stop", "/home", "/start", "/stop"]
    assert report["attempts"] == 2
    assert "the operator pressed n" in w.events("restart")[0]["text"]

    w = world(answer="no")
    w.key(30.0, "x")
    assert w.run()["outcome"] == "aborted"
    assert w.routes() == ["/stop"]  # the arms hold; they are not sent home


def test_pausing_stops_probes_and_decisions_and_resuming_gives_a_fresh_time_limit(world) -> None:
    w = world(answer="no")
    w.key(20.0, "p")
    w.key(150.0, "p")  # resume at 150 s: well past the first attempt's time limit
    w.key(160.0, "x")
    w.run()
    assert not [e for e in w.events("probe") if 20 < e["t"] < 150]
    assert w.routes() == ["/stop"]  # no guide: the time limit started over at the resume


def test_a_run_started_by_someone_else_is_followed(world) -> None:
    w = world(answer="no")

    def restart_outside() -> None:
        w.station.handle("/stop", {})
        w.station.handle("/start", {"prompt": "outside"})

    w.at(30.0, restart_outside)
    w.key(50.0, "x")
    report = w.run()
    assert w.events("adopt")
    assert "outside" in report["prompts"]
    assert len(w.source.bound) == 2  # the recording is looked up again for the new run
    assert report["attempts"] == 1


def test_it_waits_for_a_run_then_gives_up_without_one(world) -> None:
    w = world(state="idle", wait_s=5.0)
    report = w.run()
    assert report["outcome"] == "error"
    assert "no run is going" in report["detail"]
    assert w.station.calls == []


def test_it_attaches_to_a_run_that_starts_while_it_waits(world) -> None:
    w = world(state="idle", wait_s=30.0)
    w.at(4.0, lambda: w.station.handle("/start", {"prompt": "from live view"}))
    w.key(30.0, "x")
    report = w.run()
    assert report["outcome"] == "aborted"
    assert report["prompts"] == ["from live view"]


def test_start_begins_a_run_only_when_asked_and_given_a_prompt(world) -> None:
    w = world(state="idle", start=True, prompt="unplug the cables")
    w.key(30.0, "x")
    w.run()
    assert w.routes() == ["/start", "/stop"]
    assert w.station.calls[0][1]["prompt"] == "unplug the cables"

    w = world(state="idle", start=True)
    assert "needs --prompt" in w.run()["detail"]
    assert w.station.calls == []


def test_a_station_that_does_not_answer_ends_the_watch(world, sio) -> None:
    class Dead:
        def status(self):
            raise sio.StationError("connection refused")

    w = world()
    report = w.run(client=Dead())
    assert report["outcome"] == "error"
    assert "did not answer 3 times" in report["detail"]


def test_a_missing_status_during_a_run_is_survived_until_ten_in_a_row(world, sio) -> None:
    w = world(answer="no")
    real = w.client.status
    fail_from, fail_to = 20.0, 25.0

    class Flaky:
        def __getattr__(self, name):
            return getattr(w.client, name)

        def status(self):
            if fail_from <= w.clock.t < fail_to:
                raise sio.StationError("timed out")
            return real()

    w.key(60.0, "x")
    report = w.run(client=Flaky())
    assert report["outcome"] == "aborted"  # five misses in a row were fine
    assert len(w.events("status")) == 5


def test_the_report_and_the_pictures_sent_are_kept(world) -> None:
    w = world(answer="yes")
    w.run()
    folder = w.log.folder
    report = json.loads((folder / "report.json").read_text())
    assert report["outcome"] == "done"
    assert report["prompts"] == ["p0"]
    sent = sorted(p.name for p in (folder / "pictures").iterdir())
    assert sent == [f"probe00{n}-{i}-strip.jpg" for n in (1, 2) for i in range(3)]  # only what the gate was sent
    assert all((folder / "pictures" / name).stat().st_size > 0 for name in sent)


# --- building one ----------------------------------------------------------------------------------------------


def args_for(sv, *argv):
    return sv.parser().parse_args(list(argv))


def test_dry_critic_sends_nothing_and_is_always_unsure(sv, tc, spec) -> None:
    args = args_for(sv, "run", "--dry")
    critic, compare = sv.build_critic(args, spec, lambda body: None)
    votes = critic.gate.ask_votes(spec, {"strip": b"jpeg"})
    assert [v.answer for v in votes] == ["unsure", "unsure"]
    assert compare.compare(spec, b"a", b"b", 10.0).answer == "unsure"
    assert critic.rewriter.rewrite(spec, {"strip": b"jpeg"}, [], None) == "(dry run: nothing was written)"


def test_confirm_send_needs_the_key_in_the_environment(sv, spec, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(sv.SupervisorError, match="GEMINI_API_KEY"):
        sv.confirm_send(args_for(sv, "run"), spec, tmp_path)


def test_confirm_send_says_what_leaves_and_refuses_without_a_person(sv, spec, monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(sv.SupervisorError, match="no one to ask"):
        sv.confirm_send(args_for(sv, "run"), spec, tmp_path)
    said = capsys.readouterr().out
    assert "generativelanguage.googleapis.com" in said
    assert "A picture with a face in the crop is never sent" in said
    assert "not-a-real-key" not in said
    sv.confirm_send(args_for(sv, "run", "--yes"), spec, tmp_path)  # --yes goes ahead


def test_confirm_send_asks_about_claude_only_for_the_claude_rewriter(sv, spec, monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(sv.SupervisorError, match="ANTHROPIC_API_KEY"):
        sv.confirm_send(args_for(sv, "run", "--rewriter", "claude"), spec, tmp_path)
    sv.confirm_send(args_for(sv, "run", "--rewriter", "none", "--yes"), spec, tmp_path)
    assert "api.anthropic.com" not in capsys.readouterr().out


def test_confirm_send_in_a_dry_run_needs_no_key(sv, spec, monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    sv.confirm_send(args_for(sv, "run", "--dry"), spec, tmp_path)
    assert "nothing is sent" in capsys.readouterr().out


def test_observe_only_cannot_start_a_run(sv) -> None:
    with pytest.raises(sv.SupervisorError, match="cannot --start"):
        sv.command_run(args_for(sv, "run", "--observe-only", "--start", "--dry"))


def test_cameras_needed_include_the_ones_the_views_use(sv, spec) -> None:
    assert sv.cameras_needed(spec) == ["scene"]


def test_run_ends_cleanly_when_no_run_is_going(sv, sio, tmp_path, monkeypatch, capsys) -> None:
    """The whole command, wired as in use, against a fake station over HTTP that has no run."""
    monkeypatch.setattr(sv.signal, "signal", lambda *args: None)
    station = sio.FakeStation(prompt="p")
    url = station.serve()
    try:
        with pytest.raises(SystemExit) as caught:
            sv.main(
                ["--logs", str(tmp_path), "run", "--dry", "--url", url, "--wait-s", "0.5", "--rollouts", str(tmp_path)]
            )
    finally:
        station.close()
    assert caught.value.code == 1
    folder = next(tmp_path.glob("supervise-*"))
    assert json.loads((folder / "report.json").read_text())["outcome"] == "error"
    assert station.calls == []


# --- the commands that look --------------------------------------------------------------------------------------


def write_video(path: pathlib.Path, seconds: int = 4, fps: int = 10, size=(320, 180)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), 60 + i, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_check_reports_the_station_the_recorder_and_the_pictures(sv, sio, tmp_path, monkeypatch, capsys) -> None:
    rollouts = tmp_path / "rollouts"
    folder = rollouts / "20261108-120000_run1"
    folder.mkdir(parents=True)
    now = datetime.datetime.now(datetime.UTC)
    (folder / "rollout.json").write_text(json.dumps({"status": "recording", "started_at_utc": now.isoformat()}))
    write_video(folder / "videos/chunk-000/observation.images.scene/episode_000000.mp4")
    station_file = tmp_path / "station.yaml"
    station_file.write_text("rollouts:\n  enabled: true\n")
    monkeypatch.setattr(sv, "DEFAULT_STATION_FILE", station_file)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    station = sio.FakeStation(prompt="p", state="running")
    url = station.serve()
    try:
        with pytest.raises(SystemExit) as caught:
            sv.main(["--logs", str(tmp_path / "logs"), "check", "--url", url, "--rollouts", str(rollouts)])
    finally:
        station.close()
    said = capsys.readouterr().out
    assert caught.value.code == 0, said
    assert "the station is running" in said
    assert "camera scene: 320x180" in said
    assert "PROBLEMS: none" in said
    assert "GEMINI_API_KEY is set" in said
    assert "not-a-real-key" not in said
    assert station.calls == []  # it only looked
    pictures = sorted(p.name for p in (tmp_path / "logs").glob("supervise-*/pictures/*"))
    assert pictures == ["check-scene.jpg", "check-strip.jpg"]


def test_check_names_what_is_wrong(sv, tmp_path, monkeypatch, capsys) -> None:
    station_file = tmp_path / "station.yaml"
    station_file.write_text("rollouts:\n  enabled: false\n")
    monkeypatch.setattr(sv, "DEFAULT_STATION_FILE", station_file)
    with pytest.raises(SystemExit) as caught:
        sv.main(
            ["--logs", str(tmp_path), "check", "--url", "http://127.0.0.1:1", "--rollouts", str(tmp_path / "none")]
        )
    said = capsys.readouterr().out
    assert caught.value.code == 1
    assert "PROBLEM  the station agent" in said
    assert "rollouts.enabled is not true" in said
    assert "PROBLEM  camera scene" in said


def test_rehearse_replays_a_video_against_a_fake_station_and_moves_nothing(sv, tmp_path, capsys) -> None:
    video = tmp_path / "episode.mp4"
    write_video(video, seconds=6)
    with pytest.raises(SystemExit) as caught:
        sv.main(["--logs", str(tmp_path / "logs"), "rehearse", "--dry", "--speed", "600", "--video", str(video)])
    said = capsys.readouterr().out
    assert caught.value.code == 3, said  # a dry run is always unsure, so the attempts run out
    folder = next((tmp_path / "logs").glob("supervise-*"))
    report = json.loads((folder / "report.json").read_text())
    assert (report["outcome"], report["attempts"], report["guides"]) == ("gave_up", 3, 3)
    prompts = report["prompts"]
    assert prompts[1] == "(dry run: nothing was written)"
    assert len(set(prompts)) == len(prompts)  # a repeated rewrite fell back to the task file's prompts


def test_rehearse_needs_a_video_for_every_camera(sv, tmp_path) -> None:
    with pytest.raises(SystemExit, match="give one --video for each of"):
        sv.main(["--logs", str(tmp_path), "rehearse", "--dry", "--video", "other=/nowhere.mp4"])
