"""tools/sequence.py: the sequence file, the order of the steps and what happens between them, the audit, and whole
sequences against a fake station with a fake clock and scripted models. No arm, no camera and no model is touched."""

import importlib.util
import json
import pathlib
import sys

import av
import numpy as np
import pytest
import yaml
from supervisor_fakes import REWRITES, FakeClock, FakeSource, ScriptedGate, ScriptedRewriter

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
TASK_FILE = pathlib.Path(__file__).resolve().parents[1] / "configs" / "tasks" / "usb-cables.yaml"
NAMES = ("unplug-cables", "stack-blocks", "golf-balls")
PROMPTS = {
    "unplug-cables": "Unplug the cables.",
    "stack-blocks": "Stack the blocks.",
    "golf-balls": "Put the golf balls in the box.",
}


@pytest.fixture
def sq(monkeypatch: pytest.MonkeyPatch):
    """The tool's module, loaded the way `python tools/sequence.py` does: its folder is on the import path."""
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("sequence_tool", TOOLS / "sequence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sv(sq):
    return sys.modules["supervise"]


@pytest.fixture
def tc(sq):
    return sys.modules["task_critic"]


@pytest.fixture
def sio(sq):
    return sys.modules["station_io"]


# --- the sequence file -----------------------------------------------------------------------------------------


def write_tasks(folder: pathlib.Path, names=NAMES, edit=None) -> dict[str, pathlib.Path]:
    """One task file per name: a copy of the cables task under another name."""
    paths = {}
    for name in names:
        text = TASK_FILE.read_text().replace("name: usb-cables", f"name: {name}", 1)
        if edit:
            text = edit(name, text)
        paths[name] = folder / f"{name}.yaml"
        paths[name].write_text(text)
    return paths


def write_sequence(folder: pathlib.Path, names=NAMES, tasks=None, top=None, **step_extra) -> pathlib.Path:
    tasks = tasks or write_tasks(folder, names)
    steps = []
    for name in names:
        steps.append(
            {
                "name": name,
                "task": str(tasks[name]),
                "prompt": PROMPTS.get(name, "Do the task."),
                **step_extra.get(name, {}),
            }
        )
    body = {"checkpoint": "dreamlabs/dream-1-beta", "sequence": steps, **(top or {})}
    path = folder / "sequence.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def test_a_good_file_loads_with_its_defaults(sq, tmp_path) -> None:
    plan = sq.load_plan(write_sequence(tmp_path))
    assert [s.name for s in plan.steps] == list(NAMES)
    assert plan.steps[0].prompt == "Unplug the cables."
    assert (plan.checkpoint, plan.between_steps, plan.on_give_up, plan.audit) == (
        "dreamlabs/dream-1-beta", "home", "continue", "final"
    )  # fmt: skip
    steps = sq.load_steps(plan)
    assert [s.spec.name for s in steps] == list(NAMES)


def test_the_overrides_replace_the_task_files_limits_for_this_sequence_only(sq, tmp_path) -> None:
    path = write_sequence(tmp_path, **{"stack-blocks": {"attempt_s": 40, "max_attempts": 2}})
    steps = sq.load_steps(sq.load_plan(path))
    config = steps[1].spec.supervisor
    assert (config.attempt_s, config.attempts) == (40.0, 2)
    assert steps[0].spec.supervisor.attempts == 3 and steps[0].spec.supervisor.attempt_s == 90.0
    assert sq.worst_case_s(steps) == 3 * 90 + 2 * 40 + 3 * 90


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["sequence"].append(dict(d["sequence"][0])), "step names must be unique"),
        (lambda d: d["sequence"][0].update(prompt="  "), "sequence\\[0\\].prompt is missing or empty"),
        (lambda d: d["sequence"][0].update(prompt="<your prompt>"), "placeholder prompt"),
        (lambda d: d["sequence"][0].update(prompt="TODO"), "placeholder prompt"),
        (lambda d: d["sequence"][0].update(name="Unplug Cables"), "lowercase letters, digits and hyphens"),
        (lambda d: d["sequence"][0].update(bogus=1), "unknown keys \\['bogus'\\]"),
        (lambda d: d["sequence"][0].update(attempt_s=0), "attempt_s must be a number of at least 1"),
        (lambda d: d["sequence"][0].update(max_attempts=0), "max_attempts must be a whole number of at least 1"),
        (lambda d: d["sequence"][0].update(max_attempts=True), "max_attempts must be a whole number"),
        (lambda d: d.update(on_give_up="retry"), "on_give_up must be one of"),
        (lambda d: d.update(audit="always"), "audit must be one of"),
        (lambda d: d.update(between_steps="pause"), "between_steps must be one of"),
        (lambda d: d.update(bogus=1), "unknown keys \\['bogus'\\]"),
        (lambda d: d.update(checkpoint=""), "checkpoint must be a non-empty string"),
        (lambda d: d.update(sequence=[]), "sequence must be a list with at least one step"),
        (lambda d: d["sequence"].__setitem__(0, "unplug"), "must be a mapping"),
        (lambda d: d["sequence"][0].pop("task"), "sequence\\[0\\].task is missing"),
    ],
)
def test_a_bad_sequence_file_is_refused_naming_the_problem(sq, tmp_path, edit, message) -> None:
    path = write_sequence(tmp_path)
    body = yaml.safe_load(path.read_text())
    edit(body)
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(sq.SequenceError, match=message):
        sq.load_plan(path)


def test_a_missing_or_broken_file_is_refused(sq, tmp_path) -> None:
    with pytest.raises(sq.SequenceError, match="cannot read"):
        sq.load_plan(tmp_path / "nothing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("sequence: [unclosed")
    with pytest.raises(sq.SequenceError, match="not valid YAML"):
        sq.load_plan(broken)
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(sq.SequenceError, match="must be a mapping"):
        sq.load_plan(tmp_path / "list.yaml")


def cut(text: str, start: str, end: str) -> str:
    return text[: text.index(start)] + text[text.index(end) :]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda name, text: cut(text, "vote_gate:\n", "# The supervisor"), "no vote_gate section"),
        (
            lambda name, text: cut(text, "supervisor:\n  attempts", "# A yes/no answer reads as yes"),
            "no supervisor: block",
        ),
        (lambda name, text: text.replace("cables: 4", "cables: 0"), "cables must be at least 1"),
    ],
)
def test_a_task_file_that_will_not_do_is_refused_with_the_step_named(sq, tmp_path, edit, message) -> None:
    tasks = write_tasks(tmp_path, edit=lambda name, text: edit(name, text) if name == "stack-blocks" else text)
    plan = sq.load_plan(write_sequence(tmp_path, tasks=tasks))
    with pytest.raises(sq.SequenceError, match=f"step 'stack-blocks'.*{message}"):
        sq.load_steps(plan)


def test_a_missing_task_file_is_refused(sq, tmp_path) -> None:
    tasks = write_tasks(tmp_path)
    tasks["golf-balls"] = tmp_path / "not-written-yet.yaml"
    plan = sq.load_plan(write_sequence(tmp_path, tasks=tasks))
    with pytest.raises(sq.SequenceError, match="step 'golf-balls'.*does not exist"):
        sq.load_steps(plan)


def test_task_paths_are_taken_from_the_repository_folder(sq, tmp_path) -> None:
    path = write_sequence(tmp_path, tasks={n: pathlib.Path("configs/tasks/usb-cables.yaml") for n in NAMES})
    plan = sq.load_plan(path)
    assert plan.steps[0].task == sq.REPO / "configs/tasks/usb-cables.yaml"
    assert sq.load_steps(plan)[0].spec.name == "usb-cables"


def test_the_shipped_template_is_refused_until_it_is_filled_in(sq) -> None:
    plan_error = None
    try:
        sq.load_plan(sq.DEFAULT_SEQUENCE)
    except sq.SequenceError as error:
        plan_error = str(error)
    assert plan_error is not None and "placeholder prompt" in plan_error


def test_views_of_different_steps_that_overlap_are_reported(sq, tmp_path) -> None:
    steps = sq.load_steps(sq.load_plan(write_sequence(tmp_path)))
    notes = sq.view_overlaps(steps)
    assert len(notes) == 3  # the same strip in each of the three copies
    assert "unplug-cables.strip and stack-blocks.strip overlap on camera scene" in notes[0]

    def move(name, text):
        if name != "golf-balls":
            return text
        return text.replace("box: [0.4531, 0.42, 0.6875, 0.8334]", "box: [0.30, 0.45, 0.40, 0.60]")

    steps = sq.load_steps(sq.load_plan(write_sequence(tmp_path, tasks=write_tasks(tmp_path, edit=move))))
    assert len(sq.view_overlaps(steps)) == 1  # only the first two still share a strip


def test_check_prints_each_step_and_the_worst_case(sq, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        sq.main(["check", str(write_sequence(tmp_path))])
    said = capsys.readouterr().out
    assert caught.value.code == 0
    assert "ok  step 2 stack-blocks: task stack-blocks" in said
    assert "worst case: 810 s of run time" in said
    assert "WARNING  the views unplug-cables.strip and stack-blocks.strip overlap" in said


def test_check_refuses_a_bad_file_with_a_message_and_no_traceback(sq, tmp_path) -> None:
    with pytest.raises(SystemExit, match="sequence: .*cannot read"):
        sq.main(["check", str(tmp_path / "missing.yaml")])


# --- whole sequences -------------------------------------------------------------------------------------------


class SeqWorld:
    """A sequence runner wired to a fake station, a fake clock, scripted critics and fake cameras."""

    def __init__(
        self,
        sq,
        sv,
        tc,
        sio,
        tmp_path,
        *,
        state="idle",
        prompt="",
        answer="yes",
        plan_top=None,
        steps_extra=None,
        **opts,
    ):
        self.sq, self.sv, self.tc, self.sio = sq, sv, tc, sio
        self.clock = FakeClock()
        self.station = sio.FakeStation(clock=self.clock.now, prompt=prompt, state=state)
        self.plan = sq.load_plan(write_sequence(tmp_path, top=plan_top, **(steps_extra or {})))
        self.steps = sq.load_steps(self.plan)
        self.gates = {n: ScriptedGate(tc, answer) for n in NAMES}
        self.rewriters = {n: ScriptedRewriter(REWRITES) for n in NAMES}
        self.sources = {n: FakeSource(sio) for n in NAMES}
        self.audit_source = FakeSource(sio)
        self.log = sv.EventLog(tmp_path / "sequence", self.clock, echo=None)
        self.client = self.station.client()
        self.timed: list = []
        self.keys_at: list = []
        self.watchers: dict = {}
        self.manual: list = []
        self.clock.hooks.append(self._on_move)
        real_write = self.log.write

        def write(event, text, **fields):
            record = real_write(event, text, **fields)
            for action in self.watchers.get(event, []):
                action()
            return record

        self.log.write = write
        self.options = sq.SeqOptions(**opts)
        self.runner = None

    def critic_factory(self, spec):
        return self.tc.TaskCritic(spec, self.gates[spec.name], self.rewriters[spec.name]), None

    def _on_move(self, t: float) -> None:
        if self.station.state == "running":
            self.station.add_chunks(1)
        for item in [i for i in self.timed if t >= i[0]]:
            self.timed.remove(item)
            item[1]()

    def at(self, t: float, action) -> None:
        self.timed.append((t, action))

    def key(self, t: float, key: str) -> None:
        self.keys_at.append((t, key))

    def when(self, event: str, action) -> None:
        self.watchers.setdefault(event, []).append(action)

    def _keys(self):
        for item in self.keys_at:
            if self.clock.t >= item[0]:
                self.keys_at.remove(item)
                return item[1]
        return None

    def person(self, route: str, body=None) -> None:
        """Something a person does by hand: it reaches the station, and is not a call of the supervisor's."""
        self.manual.append((route, dict(body or {})))
        self.station.handle(route, dict(body or {}))

    def run(self, **overrides):
        kwargs = {
            "log": self.log,
            "clock": self.clock,
            "options": self.options,
            "critic_factory": self.critic_factory,
            "source_factory": lambda loaded: {"scene": self.sources[loaded.step.name]},
            "audit_source_factory": lambda loaded, last: {"scene": self.audit_source},
            "keys": self._keys,
            "echo": None,
        }
        kwargs.update(overrides)
        self.runner = self.sq.SequenceRunner(self.plan, self.steps, self.client, **kwargs)
        return self.runner.run()

    def events(self, kind=None):
        records = [json.loads(line) for line in (self.log.folder / "events.jsonl").read_text().splitlines()]
        return [r for r in records if kind is None or r["event"] == kind]

    def routes(self):
        return [route for route, _ in self.station.calls]


@pytest.fixture
def world(sq, sv, tc, sio, tmp_path):
    def make(**kwargs):
        return SeqWorld(sq, sv, tc, sio, tmp_path, **kwargs)

    return make


GIVE_UP = ["/prompt", "/stop", "/home", "/start", "/prompt", "/stop", "/home", "/start", "/prompt", "/stop", "/home"]


def test_all_three_steps_reach_done_with_a_home_after_each(world) -> None:
    w = world(start=True)
    report = w.run()
    assert report["outcome"] == "completed"
    assert [s["outcome"] for s in report["steps"]] == ["done", "done", "done"]
    assert w.routes() == ["/start", "/stop", "/home"] * 3
    starts = [body for route, body in w.station.calls if route == "/start"]
    assert [s["prompt"] for s in starts] == [PROMPTS[n] for n in NAMES]
    assert {s["checkpoint"] for s in starts} == {"dreamlabs/dream-1-beta"}
    assert w.station.state == "idle"
    assert report["control_calls"] == 9
    assert w.sq.exit_code(report) == 0


def test_a_step_that_gives_up_is_marked_and_the_next_step_still_starts(world) -> None:
    w = world(start=True)
    w.gates["unplug-cables"].answer = "no"
    report = w.run()
    assert [s["outcome"] for s in report["steps"]] == ["gave_up", "done", "done"]
    assert w.routes() == ["/start", *GIVE_UP, "/start", "/stop", "/home", "/start", "/stop", "/home"]
    assert report["steps"][0]["attempts"] == 3
    assert report["outcome"] == "completed"
    assert w.sq.exit_code(report) == 3


def test_on_give_up_stop_ends_the_sequence_after_the_arms_are_home(world) -> None:
    w = world(start=True, plan_top={"on_give_up": "stop"})
    w.gates["unplug-cables"].answer = "no"
    report = w.run()
    assert report["outcome"] == "stopped"
    assert report["not_run"] == ["stack-blocks", "golf-balls"]
    assert w.routes() == ["/start", *GIVE_UP]
    assert w.routes()[-1] == "/home"
    assert all(s["audit"] is None for s in report["steps"])  # an unfinished sequence is not audited
    assert w.sq.exit_code(report) == 3


def test_the_next_step_never_starts_before_the_station_is_idle_again(world) -> None:
    w = world(start=True)
    seen = []
    real = w.station.handle

    def spy(route, body):
        seen.append((route, w.station.state))
        return real(route, body)

    w.station.handle = spy
    w.run()
    starts = [state for route, state in seen if route == "/start"]
    assert starts == ["idle", "idle", "idle"]
    homes = [i for i, (route, _) in enumerate(seen) if route == "/home"]
    starts_at = [i for i, (route, _) in enumerate(seen) if route == "/start"]
    assert all(
        home < start for home, start in zip(homes, starts_at[1:], strict=False)
    )  # a home before each next start


def test_control_calls_across_steps_are_three_seconds_apart(world) -> None:
    w = world(start=True)
    w.run()
    times = [e["t"] for e in w.events("control")]
    assert len(times) == 9
    assert all(b - a >= 3.0 for a, b in zip(times, times[1:], strict=False))


def test_a_refused_start_stops_the_sequence_once_and_sends_nothing_else(world) -> None:
    w = world(start=True)
    w.station.refuse("/start", "busy")
    report = w.run()
    assert report["outcome"] == "error"
    assert "refused to start step unplug-cables" in report["detail"]
    assert w.routes() == ["/start"]
    assert len(w.events("refused")) == 1
    assert report["steps"] == []


def test_a_refused_start_of_a_later_step_leaves_the_earlier_step_done_and_home(world) -> None:
    w = world(start=True)
    w.at(20.0, lambda: w.station.refuse("/start", "busy"))
    report = w.run()
    assert report["outcome"] == "error"
    assert [s["outcome"] for s in report["steps"]] == ["done"]
    assert w.routes() == ["/start", "/stop", "/home", "/start"]
    assert report["not_run"] == ["stack-blocks", "golf-balls"]


def test_ctrl_c_in_a_step_stops_the_run_and_does_not_send_the_arms_home(world) -> None:
    w = world(start=True)

    def interrupt() -> None:
        raise KeyboardInterrupt

    w.at(30.0, interrupt)
    report = w.run()
    assert report["outcome"] == "interrupted"
    assert w.routes() == ["/start", "/stop"]
    assert report["steps"][-1]["name"] == "unplug-cables"
    assert "step unplug-cables" in report["detail"]
    assert w.station.state == "idle"


def test_ctrl_c_or_a_crash_between_steps_stops_the_run_that_was_just_started(world) -> None:
    for error, outcome in ((KeyboardInterrupt(), "interrupted"), (RuntimeError("a bug"), "crashed")):
        w = world(start=True)
        real = w.critic_factory

        def factory(spec, real=real, error=error, w=w):
            if spec.name == "stack-blocks":
                raise error
            return real(spec)

        w.critic_factory = factory
        report = w.run()
        assert report["outcome"] == outcome
        assert w.routes() == ["/start", "/stop", "/home", "/start", "/stop"]  # the second run was stopped, not homed
        assert w.station.state == "idle"


def test_observe_only_makes_no_control_call_across_the_whole_sequence(world) -> None:
    w = world(state="running", prompt=PROMPTS["unplug-cables"], observe_only=True)
    steps_after = iter(["stack-blocks", "golf-balls"])

    def hand_start() -> None:
        name = next(steps_after)
        w.at(
            w.clock.t + 2,
            lambda: [w.person("/stop"), w.person("/home"), w.person("/start", {"prompt": PROMPTS[name]})],
        )

    w.when("wait", hand_start)
    report = w.run()
    assert w.station.calls == [(r, b) for r, b in w.manual]  # only the person's
    assert report["control_calls"] == 0
    assert report["outcome"] == "completed"
    assert [s["outcome"] for s in report["steps"]] == ["done", "done", "done"]
    would = [(e["action"], e.get("prompt")) for e in w.events("would")]
    assert ("start", PROMPTS["stack-blocks"]) in would and ("start", PROMPTS["golf-balls"]) in would
    assert [a for a, _ in would].count("stop_and_home") == 3
    assert not w.events("control")


def test_observe_only_waits_for_a_run_the_person_starts_and_gives_up_without_one(world) -> None:
    w = world(state="running", prompt=PROMPTS["unplug-cables"], observe_only=True, wait_s=10.0)
    report = w.run()
    assert report["outcome"] == "error"
    assert "no run was started for step stack-blocks" in report["detail"]
    assert w.station.calls == []


def test_the_audit_reads_every_task_from_the_last_frames_and_reports_it(world) -> None:
    w = world(start=True)
    report = w.run()
    audits = {s["name"]: s["audit"] for s in report["steps"]}
    assert {n: a["result"] for n, a in audits.items()} == {n: "still_done" for n in NAMES}
    assert not any(a["regression"] for a in audits.values())
    pictures = sorted(p.name for p in (w.log.folder / "pictures").glob("audit-*"))
    assert len(pictures) == 9  # three pictures of the zoom for each of the three tasks
    assert w.station.calls[-1][0] == "/home"  # the audit moved nothing


def test_an_arm_over_a_region_reads_unsure_in_the_audit_not_undone(world) -> None:
    w = world(start=True)
    w.when("audit", lambda: setattr(w.gates["stack-blocks"], "answer", "unsure"))
    report = w.run()
    audits = {s["name"]: s["audit"]["result"] for s in report["steps"]}
    assert audits == {"unplug-cables": "still_done", "stack-blocks": "unsure", "golf-balls": "still_done"}
    assert not report["steps"][1]["audit"]["regression"]


def test_a_task_that_reads_undone_after_being_done_is_a_regression(world) -> None:
    w = world(start=True)
    w.when("audit", lambda: setattr(w.gates["unplug-cables"], "answer", "no"))
    report = w.run()
    first = report["steps"][0]
    assert (first["outcome"], first["audit"]["result"], first["audit"]["regression"]) == ("done", "undone", True)
    assert first["audit"]["reasons"] == ["scripted no"]
    assert "regression" in w.sq.render_summary(report)


def test_the_audit_can_be_turned_off(world) -> None:
    w = world(start=True, plan_top={"audit": "off"})
    report = w.run()
    assert all(s["audit"] is None for s in report["steps"])
    assert not w.events("audit")


def test_an_audit_with_a_face_in_the_picture_sends_nothing(world, tc, monkeypatch) -> None:
    w = world(start=True)
    w.when("audit", lambda: monkeypatch.setattr(tc, "count_faces", lambda bgr: 1))
    report = w.run()
    audit = report["steps"][0]["audit"]
    assert audit["result"] == "unavailable" and "a face is in the picture" in audit["reason"]


def test_the_operator_can_skip_a_step_abort_the_sequence_or_finish_a_step(world) -> None:
    w = world(start=True)
    w.gates["unplug-cables"].answer = "no"
    w.key(30.0, "s")
    report = w.run()
    assert [s["outcome"] for s in report["steps"]] == ["skipped", "done", "done"]
    assert w.routes() == ["/start", "/stop", "/home", "/start", "/stop", "/home", "/start", "/stop", "/home"]

    w = world(start=True)
    w.gates["unplug-cables"].answer = "no"  # still going at 30 s
    w.key(30.0, "x")
    report = w.run()
    assert report["outcome"] == "aborted"
    assert w.routes() == ["/start", "/stop", "/home"]  # stopped, and the arms go home
    assert report["not_run"] == ["stack-blocks", "golf-balls"]
    assert report["steps"][0]["outcome"] == "aborted"

    w = world(start=True)
    w.key(28.0, "x")  # step 1 is done by 27 s: the key is read between the steps
    report = w.run()
    assert report["outcome"] == "aborted" and "between steps" in report["detail"]
    assert [s["outcome"] for s in report["steps"]] == ["done"]
    assert w.routes() == ["/start", "/stop", "/home", "/home"]  # the step's home, then the abort's
    assert report["not_run"] == ["stack-blocks", "golf-balls"]

    w = world(start=True)
    w.gates["unplug-cables"].answer = "no"
    w.key(30.0, "q")  # the plan page calls it q; the supervisor calls it f
    report = w.run()
    assert report["steps"][0]["outcome"] == "done"
    assert "operator pressed f" in report["steps"][0]["detail"]


def test_skip_does_nothing_in_a_single_task_run(sv, tc, sio, tmp_path) -> None:
    from supervisor_fakes import FakeClock as Clock  # noqa: PLC0415 (the same fakes, for a plain supervisor)

    clock = Clock()
    station = sio.FakeStation(clock=clock.now, prompt="p", state="running")
    spec = tc.load_task(TASK_FILE)
    keys = iter(["s", None, None, "x"])
    supervisor = sv.Supervisor(
        spec,
        tc.TaskCritic(spec, ScriptedGate(tc, "no"), ScriptedRewriter(REWRITES)),
        station.client(),
        {"scene": FakeSource(sio)},
        log=sv.EventLog(tmp_path, clock, echo=None),
        clock=clock,
        keys=lambda: next(keys, None),
    )
    assert supervisor.run()["outcome"] == "aborted"
    assert "does nothing" in (tmp_path / "events.jsonl").read_text()


def test_an_operator_stop_ends_the_sequence_and_it_starts_nothing(world) -> None:
    w = world(start=True)
    w.gates["unplug-cables"].answer = "no"
    w.at(30.0, w.station.end_run)
    report = w.run()
    assert report["outcome"] == "ended"
    assert w.routes() == ["/start"]
    assert "step unplug-cables" in report["detail"]


def test_a_station_that_is_not_idle_after_home_stops_the_sequence_without_forcing_anything(world) -> None:
    w = world(start=True, settle_s=5.0, home_wait_s=5.0)
    real = w.station.handle

    def stuck_homing(route, body):
        reply = real(route, body)
        if route == "/home":
            w.station.state = "homing"
        return reply

    w.station.handle = stuck_homing
    report = w.run()
    assert report["outcome"] == "error"
    assert "not idle after step unplug-cables" in report["detail"]
    assert w.routes() == ["/start", "/stop", "/home"]  # no second start, no reset
    assert w.station.state == "homing"


def test_a_station_in_error_after_a_step_stops_the_sequence(world) -> None:
    w = world(start=True)
    real = w.station.handle

    def failing_home(route, body):
        reply = real(route, body)
        if route == "/home":
            w.station.fail("a motor fault")
        return reply

    w.station.handle = failing_home
    report = w.run()
    assert report["outcome"] == "error"
    assert "in error after step unplug-cables" in report["detail"]
    assert "/reset" not in w.routes() and w.routes().count("/start") == 1


def test_a_home_refused_inside_a_step_is_sent_once_more(world) -> None:
    w = world(start=True)
    w.at(20.0, lambda: w.station.refuse("/home", "busy"))
    report = w.run()
    assert report["outcome"] == "completed"
    assert w.routes()[:5] == ["/start", "/stop", "/home", "/home", "/start"]
    assert report["steps"][0]["homed"] is False


def test_a_recording_that_never_appears_stops_the_run_and_the_sequence(world, sio) -> None:
    w = world(start=True, recording_wait_s=5.0)

    class NoRecording(FakeSource):
        def latest(self, count):
            raise sio.SourceError("no recording of this run")

    w.sources["unplug-cables"] = NoRecording(sio)
    report = w.run()
    assert report["outcome"] == "error"
    assert "no recording of step unplug-cables appeared in 5 s" in report["detail"]
    assert w.routes() == ["/start", "/stop"]


def test_without_start_the_first_step_attaches_to_a_run_the_person_started(world) -> None:
    w = world(state="running", prompt="a different prompt")
    report = w.run()
    assert report["outcome"] == "completed"
    assert w.routes() == ["/stop", "/home", "/start", "/stop", "/home", "/start", "/stop", "/home"]
    assert "not the sequence file's prompt" in report["steps"][0]["note"]
    assert report["steps"][1]["note"] == ""


def test_a_station_that_is_not_idle_cannot_start_a_step(world) -> None:
    w = world(state="running", start=True, prompt="p")
    report = w.run()
    assert report["outcome"] == "error"
    assert "the station is running, not idle" in report["detail"]
    assert w.station.calls == [("/stop", {})]  # only the stop on the way out


def test_a_step_can_be_given_fewer_attempts_for_this_sequence(world) -> None:
    w = world(start=True, steps_extra={"unplug-cables": {"attempt_s": 20, "max_attempts": 1}})
    w.gates["unplug-cables"].answer = "no"
    report = w.run()
    first = report["steps"][0]
    assert (first["outcome"], first["attempts"]) == ("gave_up", 1)
    assert first["elapsed_s"] < 100


def test_done_at_start_is_flagged_only_when_the_first_probes_already_said_done(world) -> None:
    w = world(start=True)
    report = w.run()
    assert [s["done_at_start"] for s in report["steps"]] == [True, True, True]
    assert "done (done at start)" in w.sq.render_summary(report)

    w = world(start=True)
    w.gates["unplug-cables"].answer = lambda: "yes" if w.clock.t > 40 else "no"
    report = w.run()
    assert report["steps"][0]["outcome"] == "done" and not report["steps"][0]["done_at_start"]


# --- what it leaves behind -------------------------------------------------------------------------------------


def test_every_step_has_a_folder_and_one_log_carries_the_step_on_each_line(world) -> None:
    w = world(start=True)
    w.run()
    folder = w.log.folder
    assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == [
        "01-unplug-cables", "02-stack-blocks", "03-golf-balls", "pictures"
    ]  # fmt: skip
    assert json.loads((folder / "01-unplug-cables/report.json").read_text())["outcome"] == "done"
    assert (folder / "02-stack-blocks/pictures/probe001-0-strip.jpg").exists()
    events = w.events()
    probes = [e for e in events if e["event"] == "probe"]
    assert {e["step"] for e in probes} == set(NAMES)
    assert [e["step"] for e in probes][:2] == ["unplug-cables"] * 2
    assert all(b["t"] >= a["t"] for a, b in zip(events, events[1:], strict=False) if "step" in a and "step" in b)
    own = [json.loads(line) for line in (folder / "01-unplug-cables/events.jsonl").read_text().splitlines()]
    assert own and all(e["step"] == "unplug-cables" for e in own)


def test_the_sequence_report_and_the_combined_table_are_kept(world) -> None:
    w = world(start=True)
    w.gates["unplug-cables"].answer = "no"
    w.run()
    report = json.loads((w.log.folder / "sequence.json").read_text())
    assert report["outcome"] == "completed" and report["checkpoint"] == "dreamlabs/dream-1-beta"
    first = report["steps"][0]
    assert [p["kind"] for p in first["prompt_log"]] == ["start", "guide", "restart", "guide", "restart", "guide"]
    assert first["prompt_log"][1]["prompt"] == REWRITES[0] and first["prompt_log"][1]["source"] == "gemini"
    assert first["last_verdict"]["decision"] == "not_done"
    assert first["folders"] == [] and first["homed"] is True
    table = (w.log.folder / "summary.txt").read_text()
    lines = table.splitlines()
    assert lines[0].split() == ["step", "outcome", "attempts", "seconds", "prompts", "last", "probe", "audit"]
    assert lines[2].split()[:3] == ["unplug-cables", "gave_up", "3"]
    assert lines[3].split()[:2] == ["stack-blocks", "done"] and "still_done" in lines[3]
    assert lines[-1].startswith("sequence completed: 3 steps ran")


def test_a_sequence_that_ends_early_lists_the_steps_that_did_not_run(world) -> None:
    w = world(start=True)
    w.key(30.0, "x")
    report = w.run()
    table = (w.log.folder / "summary.txt").read_text()
    assert "stack-blocks" in table and "not run" in table
    assert report["not_run"] == ["stack-blocks", "golf-balls"]


# --- the commands ----------------------------------------------------------------------------------------------


def args_for(sq, *argv):
    return sq.parser().parse_args(list(argv))


def test_observe_only_cannot_start_a_run(sq, tmp_path) -> None:
    path = write_sequence(tmp_path)
    with pytest.raises(sq.SequenceError, match="cannot --start"):
        sq.command_run(args_for(sq, "run", str(path), "--observe-only", "--start", "--dry"))


def test_confirm_send_names_every_step_and_the_audit_and_needs_the_key(sq, tmp_path, monkeypatch, capsys) -> None:
    steps = sq.load_steps(sq.load_plan(write_sequence(tmp_path)))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(sq.SequenceError, match="GEMINI_API_KEY"):
        sq.confirm_send(args_for(sq, "run"), steps, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(""))
    with pytest.raises(sq.SequenceError, match="no one to ask"):
        sq.confirm_send(args_for(sq, "run"), steps, tmp_path)
    said = capsys.readouterr().out
    assert all(f"step {n}:" in said for n in NAMES)
    assert "for the audit" in said and "A picture with a face in the crop is never sent" in said
    assert "not-a-real-key" not in said
    sq.confirm_send(args_for(sq, "run", "--yes"), steps, tmp_path)
    sq.confirm_send(args_for(sq, "run", "--dry"), steps, tmp_path)


def test_run_ends_cleanly_when_no_run_is_going(sq, sio, sv, tmp_path, monkeypatch) -> None:
    """The whole command, wired as in use, against a fake station over HTTP that has no run."""
    monkeypatch.setattr(sv.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sq.signal, "signal", lambda *args: None)
    station = sio.FakeStation(prompt="p")
    url = station.serve()
    try:
        with pytest.raises(SystemExit) as caught:
            sq.main(
                ["--logs", str(tmp_path / "logs"), "run", str(write_sequence(tmp_path)), "--dry", "--url", url]
                + ["--wait-s", "0.5", "--rollouts", str(tmp_path)]
            )
    finally:
        station.close()
    assert caught.value.code == 1
    folder = next((tmp_path / "logs").glob("sequence-*"))
    report = json.loads((folder / "sequence.json").read_text())
    assert report["outcome"] == "error" and report["steps"][0]["outcome"] == "error"
    assert station.calls == []


def write_video(path: pathlib.Path, seconds: int = 4, fps: int = 10, size=(320, 180)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), 60 + i, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_rehearse_replays_a_video_for_each_step_against_a_fake_station(sq, tmp_path, capsys) -> None:
    path = write_sequence(tmp_path)
    videos = []
    for name in NAMES:
        write_video(tmp_path / f"{name}.mp4", seconds=6)
        videos += ["--video", f"{name}={tmp_path / f'{name}.mp4'}"]
    with pytest.raises(SystemExit) as caught:
        sq.main(["--logs", str(tmp_path / "logs"), "rehearse", str(path), "--dry", "--speed", "900", *videos])
    said = capsys.readouterr().out
    assert caught.value.code == 3, said  # a dry run is always unsure, so every step uses its attempts
    folder = next((tmp_path / "logs").glob("sequence-*"))
    report = json.loads((folder / "sequence.json").read_text())
    assert [s["outcome"] for s in report["steps"]] == ["gave_up"] * 3
    assert (
        report["outcome"] == "completed" and report["control_calls"] == 3 * 12
    )  # a start, then guide, restart, guide, restart, guide, stop, home
    assert all(s["audit"]["result"] == "unsure" for s in report["steps"])  # the stand-ins are unsure about everything
    assert "fake station calls" in said and "stack-blocks" in said


def test_rehearse_needs_a_video_for_each_step_and_no_observe_only(sq, tmp_path) -> None:
    path = write_sequence(tmp_path)
    with pytest.raises(sq.SequenceError, match="give one --video STEP=PATH for each of"):
        sq.command_rehearse(args_for(sq, "rehearse", str(path), "--dry", "--video", "unplug-cables=/nowhere.mp4"))
    with pytest.raises(sq.SequenceError, match="leave out --observe-only"):
        sq.command_rehearse(args_for(sq, "rehearse", str(path), "--dry", "--observe-only", "--video", "x=y"))
