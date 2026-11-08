"""The sequence layer: runs the supervisor on several tasks in a row on one table, the arms going home after each.

A sequence file (`configs/sequences/*.yaml`) lists the steps in order: each names a task file and the prompt to start
the base model with. For every step the layer starts the run, hands it to the supervisor (tools/supervise.py) until the
task is done or the attempts run out, makes sure the arms are home and the station idle, and only then starts the next
step. After the last step it can look once more at all the tasks from the last frames and report whether the earlier
ones still look done. It calls the station agent through the supervisor and nothing else; it never opens the arms or
the cameras. docs/supervisor.md has the whole picture.

```
# Validate the file and every task file it names. No network, no station.
uv run python tools/sequence.py check configs/sequences/tabletop.yaml
# Replay one recorded demonstration per step against a fake station: no robot, no arms
uv run python tools/sequence.py rehearse configs/sequences/tabletop.yaml --dry --video unplug-cables=a.mp4 ...
# Watch a sequence the person runs by hand, and log what it WOULD do. It sends no control call at all.
uv run python tools/sequence.py run configs/sequences/tabletop.yaml --observe-only
# Run the sequence: it stops, homes and starts runs. A person runs this, with a hand on the emergency stop.
uv run python tools/sequence.py run configs/sequences/tabletop.yaml --start
```

The rules:

- One task at a time. Step N+1 starts only after step N finished and the station reports idle again.
- A step finishes done (two probes in a row where every vote says yes) or gave up (its attempts used). Either way the
  arms go home. `on_give_up: continue` runs the next step anyway; `stop` ends the sequence after the home.
- A run stopped by someone else (the station goes idle while it watches) ends the sequence: it does not fight them.
- A 409 from the station, a station that is not idle after home, or a recording that does not appear stops the
  sequence with the arms where they are. Nothing is forced, retried blindly or reset.
- A crash, Ctrl+C or SIGTERM stops the run (not in observe-only mode) and exits without sending the arms home.
- The final audit only reports. It never moves anything.
"""

import argparse
import concurrent.futures
import contextlib
import dataclasses
import os
import pathlib
import re
import signal
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import station_io as sio
import supervise as sv
import task_critic as tc
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = REPO / "configs" / "sequences" / "tabletop.yaml"

TOP_KEYS = {"checkpoint", "sequence", "between_steps", "on_give_up", "audit"}
STEP_KEYS = {"name", "task", "prompt", "attempt_s", "max_attempts"}
NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
PLACEHOLDER = re.compile(r"<[^>]*>|\b(?:todo|tbd|fixme)\b", re.IGNORECASE)

## Seconds to wait for the station to be idle after a step's home.
HOME_WAIT_S = 90.0
## Seconds to wait for the recorder to start writing the new run's video.
RECORDING_WAIT_S = 30.0
## Two views of different steps that share more than this fraction of the smaller one are reported.
OVERLAP_WARN = 0.05


class SequenceError(sv.SupervisorError):
    """The sequence file or a task file it names is wrong, or the sequence cannot start; the message says what."""


class SequenceStop(Exception):  # noqa: N818 (a way out of the loop, not an error)
    """The sequence has to end now: `outcome` is how, the message why."""

    def __init__(self, outcome: str, detail: str):
        super().__init__(detail)
        self.outcome = outcome


# --- The sequence file -----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Step:
    name: str
    task: pathlib.Path
    prompt: str
    attempt_s: float | None = None
    max_attempts: int | None = None


@dataclasses.dataclass(frozen=True)
class Plan:
    path: pathlib.Path
    steps: tuple[Step, ...]
    checkpoint: str | None = None
    between_steps: str = "home"
    on_give_up: str = "continue"
    audit: str = "final"


@dataclasses.dataclass(frozen=True)
class LoadedStep:
    """A step with its task file read, the step's overrides already applied to the task's supervisor limits."""

    step: Step
    spec: tc.TaskSpec


def _choice(raw: Mapping[str, Any], key: str, default: str, allowed: Sequence[str]) -> str:
    value = raw.get(key, default)
    if value not in allowed:
        raise SequenceError(f"sequence file: {key} must be one of {list(allowed)}, got {value!r}")
    return value


def _step(raw: Any, index: int) -> Step:
    where = f"sequence[{index}]"
    if not isinstance(raw, dict):
        raise SequenceError(f"sequence file: {where} must be a mapping with name, task and prompt")
    unknown = sorted(set(raw) - STEP_KEYS)
    if unknown:
        raise SequenceError(f"sequence file: {where} has unknown keys {unknown}; known keys are {sorted(STEP_KEYS)}")
    for key in ("name", "task", "prompt"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise SequenceError(f"sequence file: {where}.{key} is missing or empty")
    name = raw["name"].strip()
    if not NAME.match(name):
        raise SequenceError(
            f"sequence file: {where}.name {name!r} must be lowercase letters, digits and hyphens (it names a folder)"
        )
    prompt = " ".join(raw["prompt"].split())
    if PLACEHOLDER.search(prompt):
        raise SequenceError(f"sequence file: step {name!r} still has a placeholder prompt: {prompt!r}")
    attempt_s = raw.get("attempt_s")
    if attempt_s is not None and (
        not isinstance(attempt_s, int | float) or isinstance(attempt_s, bool) or attempt_s < 1
    ):
        raise SequenceError(
            f"sequence file: step {name!r} attempt_s must be a number of at least 1, got {attempt_s!r}"
        )
    attempts = raw.get("max_attempts")
    if attempts is not None and (not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1):
        raise SequenceError(
            f"sequence file: step {name!r} max_attempts must be a whole number of at least 1, got {attempts!r}"
        )
    task = pathlib.Path(raw["task"]).expanduser()
    task = task if task.is_absolute() else REPO / task
    return Step(name, task, prompt, None if attempt_s is None else float(attempt_s), attempts)


def load_plan(path: str | os.PathLike[str]) -> Plan:
    """Reads and checks a sequence file. The task files it names are read by load_steps().

    @throws SequenceError Something is missing or wrong; the message names it.
    """
    path = pathlib.Path(path).expanduser()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SequenceError(f"cannot read the sequence file {path}: {error}") from None
    except yaml.YAMLError as error:
        raise SequenceError(f"{path} is not valid YAML: {error}") from None
    if not isinstance(raw, dict):
        raise SequenceError(f"sequence file {path}: the top level must be a mapping")
    unknown = sorted(set(raw) - TOP_KEYS)
    if unknown:
        raise SequenceError(f"sequence file: unknown keys {unknown}; known keys are {sorted(TOP_KEYS)}")
    entries = raw.get("sequence")
    if not isinstance(entries, list) or not entries:
        raise SequenceError("sequence file: sequence must be a list with at least one step")
    steps = tuple(_step(entry, i) for i, entry in enumerate(entries))
    names = [s.name for s in steps]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SequenceError(f"sequence file: step names must be unique; repeated: {duplicates}")
    checkpoint = raw.get("checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise SequenceError("sequence file: checkpoint must be a non-empty string")
    return Plan(
        path=path,
        steps=steps,
        checkpoint=checkpoint.strip() if checkpoint else None,
        between_steps=_choice(raw, "between_steps", "home", ["home"]),
        on_give_up=_choice(raw, "on_give_up", "continue", ["continue", "stop"]),
        audit=_choice(raw, "audit", "final", ["final", "off"]),
    )


def load_steps(plan: Plan) -> list[LoadedStep]:
    """Reads every task file of the plan and applies each step's overrides.

    @throws SequenceError A task file is missing, does not load, has no vote gate, or has no `supervisor:` block.
    """
    loaded = []
    for step in plan.steps:
        where = f"step {step.name!r} ({step.task})"
        if not step.task.is_file():
            raise SequenceError(f"{where}: the task file does not exist")
        try:
            spec = tc.load_task(step.task)
            raw = yaml.safe_load(step.task.read_text(encoding="utf-8"))
        except (tc.TaskFileError, tc.CriticError, yaml.YAMLError) as error:
            raise SequenceError(f"{where}: {error}") from None
        if spec.vote_gate is None:
            raise SequenceError(f"{where}: the task file has no vote_gate section, which the supervisor needs")
        if not isinstance(raw.get("supervisor"), dict):
            raise SequenceError(f"{where}: the task file has no supervisor: block with the limits for this task")
        changes: dict[str, Any] = {}
        if step.attempt_s is not None:
            changes["attempt_s"] = step.attempt_s
        if step.max_attempts is not None:
            changes["attempts"] = step.max_attempts
        if changes:
            spec = dataclasses.replace(spec, supervisor=dataclasses.replace(spec.supervisor, **changes))
        loaded.append(LoadedStep(step, spec))
    return loaded


def view_overlaps(steps: Sequence[LoadedStep]) -> list[str]:
    """Warnings about views of different steps on the same camera that overlap: an object of one task could then
    show in the other's zoom or colour mask."""
    notes = []
    for i, first in enumerate(steps):
        for second in steps[i + 1 :]:
            for a_name, a in first.spec.views.items():
                for b_name, b in second.spec.views.items():
                    if a.camera != b.camera:
                        continue
                    width = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
                    height = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
                    if width <= 0 or height <= 0:
                        continue
                    smaller = min(
                        (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]), (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])
                    )
                    share = width * height / smaller
                    if share > OVERLAP_WARN:
                        notes.append(
                            f"the views {first.step.name}.{a_name} and {second.step.name}.{b_name} overlap on camera "
                            f"{a.camera} ({share:.0%} of the smaller one): objects of one task may show in the other's"
                        )
    return notes


def worst_case_s(steps: Sequence[LoadedStep]) -> float:
    """Seconds of run time if every step uses every attempt, before guide windows and the transitions."""
    return sum(s.spec.supervisor.attempts * s.spec.supervisor.attempt_s for s in steps)


def command_check(args: argparse.Namespace) -> int:
    """Validates the file and every task file it names. Needs no network and no station."""
    plan = load_plan(args.file)
    steps = load_steps(plan)
    checkpoint = plan.checkpoint or "(the station's)"
    print(f"{plan.path}: checkpoint {checkpoint}, on_give_up {plan.on_give_up}, audit {plan.audit}")
    for n, loaded in enumerate(steps, 1):
        spec, config = loaded.spec, loaded.spec.supervisor
        print(
            f"ok  step {n} {loaded.step.name}: task {spec.name} ({loaded.step.task.name}), "
            f"cameras {sv.cameras_needed(spec)}, {config.attempts} attempts of {config.attempt_s:g} s, "
            f"prompt {loaded.step.prompt!r}"
        )
    print(f"worst case: {worst_case_s(steps):g} s of run time, before guide windows and the transitions")
    notes = view_overlaps(steps)
    for note in notes:
        print(f"WARNING  {note}")
    cameras = {tuple(sv.cameras_needed(s.spec)) for s in steps}
    if len(cameras) > 1:
        print(f"note     the steps do not all use the same cameras: {sorted(cameras)}")
    print("the sequence file is valid" + (f" ({len(notes)} warnings)" if notes else ""))
    return 0


# --- A step's result -------------------------------------------------------------------------------------------


@dataclasses.dataclass
class StepResult:
    """What a step hands back."""

    name: str
    task: str
    prompt: str
    outcome: str = "not_run"
    detail: str = ""
    attempts: int = 0
    guides: int = 0
    prompt_log: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    elapsed_s: float = 0.0
    last_verdict: dict[str, Any] | None = None
    done_at_start: bool = False
    homed: bool | None = None
    control_calls: int = 0
    run: Any = None
    folders: list[str] = dataclasses.field(default_factory=list)
    log: str | None = None
    note: str = ""
    audit: dict[str, Any] | None = None


class KeyBus:
    """The operator's keys, shared by the layer and the step in progress. `q` is `f`, as the plan page names it."""

    def __init__(self, source: Callable[[], str | None]):
        self._source = source

    def get(self) -> str | None:
        key = self._source()
        return "f" if key == "q" else key


@dataclasses.dataclass
class SeqOptions:
    observe_only: bool = False
    start: bool = False  # start the first step too (else attach to a run the person started)
    wait_s: float = 120.0  # how long to wait for a run to attach to, or for the person to start the next step
    tick_s: float = 1.0
    settle_s: float = sv.SETTLE_TIMEOUT_S
    home_wait_s: float = HOME_WAIT_S
    recording_wait_s: float = RECORDING_WAIT_S


CriticFactory = Callable[[tc.TaskSpec], tuple[tc.TaskCritic, sv.Compare | None]]
SourceFactory = Callable[[LoadedStep], Mapping[str, sio.FrameSource]]
AuditSourceFactory = Callable[[LoadedStep, StepResult], Mapping[str, sio.FrameSource]]


# --- The runner ------------------------------------------------------------------------------------------------


class SequenceRunner:
    """Runs the steps one after another through the supervisor, and owns what happens between them."""

    def __init__(
        self,
        plan: Plan,
        steps: Sequence[LoadedStep],
        client: sv.StationApi,
        *,
        log: sv.EventLog,
        clock: sv.Clock,
        options: SeqOptions,
        critic_factory: CriticFactory,
        source_factory: SourceFactory,
        audit_source_factory: AuditSourceFactory | None = None,
        keys: Callable[[], str | None] = lambda: None,
        pool: concurrent.futures.Executor | None = None,
        echo: Callable[[str], None] | None = print,
    ):
        self.plan = plan
        self.steps = list(steps)
        self.client = client
        self.log = log
        self.clock = clock
        self.options = options
        self.critic_factory = critic_factory
        self.source_factory = source_factory
        self.audit_source_factory = audit_source_factory
        self.bus = KeyBus(keys)
        self.pool = pool
        self._echo = echo
        self.ctl = sv.Controller(client, log, clock, observe_only=options.observe_only)
        self._failures = 0
        self._aborting = False
        self._status: dict[str, Any] = {}
        self._started_at = clock.now()

    # --- the sequence -------------------------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Runs the sequence and returns its report. A crash, Ctrl+C or SIGTERM stops the run (not in observe-only
        mode) and does not send the arms home."""
        results: list[StepResult] = []
        outcome, detail = "error", ""
        try:
            self.log.write(
                "sequence",
                f"sequence {self.plan.path.name}: {len(self.steps)} steps "
                f"({', '.join(s.step.name for s in self.steps)}); "
                + ("OBSERVE ONLY: it makes no control call" if self.options.observe_only else "it will act"),
                steps=[s.step.name for s in self.steps],
                observe_only=self.options.observe_only,
            )
            outcome, detail = self._steps(results)
            if outcome == "completed" and self.plan.audit == "final":
                self._audit(results)
        except KeyboardInterrupt:
            outcome, detail = "interrupted", "interrupted (Ctrl+C or SIGTERM)"
            self.log.write("interrupted", "interrupted: stopping the run, the arms are not sent home")
        except SequenceStop as stop:
            outcome, detail = stop.outcome, str(stop)
            self.log.write("stop", f"the sequence stops ({stop.outcome}): {stop}")
        except sv.SupervisorError as error:
            outcome, detail = "error", str(error)
            self.log.write("error", f"stopping: {error}")
        except Exception as error:  # noqa: BLE001 (a bug must still leave the run stopped)
            outcome, detail = "crashed", f"{type(error).__name__}: {error}"
            self.log.write("crash", f"crashed: {detail}", traceback=traceback.format_exc())
        finally:
            self._shutdown()
        return self._report(outcome, detail, results)

    def _steps(self, results: list[StepResult]) -> tuple[str, str]:
        for index, loaded in enumerate(self.steps):
            result = self._run_step(index, loaded, results)
            results.append(result)
            ended = self._after_step(loaded, result)
            if ended is not None:
                return ended
        return "completed", f"{len(results)} steps ran"

    def _after_step(self, loaded: LoadedStep, result: StepResult) -> tuple[str, str] | None:
        """What the outcome of a step means for the sequence: None to go on to the next step."""
        name, outcome = loaded.step.name, result.outcome
        self.log.write("step_end", f"step {name} ended: {outcome}: {result.detail}", outcome=outcome)
        if outcome in ("done", "gave_up", "skipped"):
            self._transition(loaded, result)
            if outcome == "gave_up" and self.plan.on_give_up == "stop":
                return "stopped", f"step {name} gave up and on_give_up is stop"
            return None
        if outcome == "aborted":
            self._abort_sequence()
            return "aborted", f"the operator aborted during step {name}"
        if outcome == "ended":
            if self.options.observe_only:
                return None  # the person stopped it by hand, as they will start the next one
            return "ended", f"step {name}: {result.detail}"
        return outcome, f"step {name}: {result.detail}"

    def _shutdown(self) -> None:
        if self.options.observe_only:
            return
        try:
            state = self.client.status().get("state")
            if state in ("running", "starting"):
                self.ctl.send("stop", force=True)
                self.log.write("stop", "stopped the run so the arms hold where they are")
        except sio.StationError as error:
            self.log.write("error", f"could not stop the run on the way out: {error}")

    # --- a step -------------------------------------------------------------------------------------------------

    def _poll(self) -> dict[str, Any]:
        """The station's status; an error after three misses in a row."""
        try:
            self._status = self.client.status()
        except sio.StationError as error:
            self._failures += 1
            self.log.write("status", f"no status ({self._failures}): {error}")
            if self._failures >= 3:
                raise SequenceStop(
                    "error", f"the station agent did not answer {self._failures} times in a row"
                ) from None
            return self._status
        self._failures = 0
        return self._status

    def _check_abort(self) -> None:
        """The operator's x between steps: the run is stopped, the arms go home, and the sequence ends."""
        if self._aborting:
            return
        if self.bus.get() == "x":
            self.log.write("key", "operator pressed x", key="x")
            self._abort_sequence()
            raise SequenceStop("aborted", "the operator pressed x between steps")

    def _wait(self, what: str, until: Callable[[dict[str, Any]], bool], seconds: float, fail: str) -> dict[str, Any]:
        """Polls the status until `until` holds; raises SequenceStop(error, fail) after `seconds`."""
        deadline = self.clock.now() + seconds
        while True:
            self._check_abort()
            status = self._poll()
            if status and until(status):
                return status
            if self.clock.now() > deadline:
                raise SequenceStop("error", f"{fail} (waited {seconds:g} s for {what})")
            self.clock.sleep(self.options.tick_s)

    def _wait_idle(self, name: str) -> dict[str, Any]:
        def idle(status: dict[str, Any]) -> bool:
            if status.get("state") == "error":
                raise SequenceStop(
                    "error",
                    f"the station is in error after step {name} ({status.get('last_error')}): the arms stay "
                    "where they are; clear it by hand, nothing was forced or reset",
                )
            return status.get("state") == "idle"

        return self._wait(
            "the station to be idle",
            idle,
            self.options.home_wait_s,
            f"the station is not idle after step {name}: the arms stay where they are",
        )

    def _transition(self, loaded: LoadedStep, result: StepResult) -> None:
        """After a step: the arms are home and the station idle before anything else happens."""
        name = loaded.step.name
        if self.options.observe_only:
            return
        self._wait_idle(name)
        if result.homed is False:
            self.log.write("note", f"step {name}: home was refused inside the step: sending it again")
            try:
                self.ctl.send("home")
            except sio.StationRefusedError as error:
                raise SequenceStop("error", f"the station refused to home after step {name}: {error}") from None
            self._wait_idle(name)
        self.log.write("transition", f"step {name}: the station is idle and the arms are home", step_done=name)

    def _abort_sequence(self) -> None:
        """The operator's x: the run is stopped, and the arms go home. Observe-only mode only logs it."""
        if self.options.observe_only:
            self.log.write("would", "would send both arms home", action="home")
            return
        self._aborting = True
        try:
            self._poll()
            if self._status.get("state") in ("running", "starting"):
                self.ctl.send("stop", force=True)
            self._wait_idle("the abort")
            self.ctl.send("home")
            self.log.write("note", "aborted: both arms were sent home")
        except (sio.StationError, SequenceStop) as error:
            self.log.write("error", f"could not finish the abort: {error}")
        finally:
            self._aborting = False

    def _start_step(self, loaded: LoadedStep, sources: Mapping[str, sio.FrameSource], previous_run: Any) -> None:
        """Starts the step's run: idle, one start, a new run number, and a recording to read."""
        step = loaded.step
        status = self._poll()
        if status.get("state") != "idle":
            raise SequenceStop(
                "error", f"cannot start step {step.name}: the station is {status.get('state')}, not idle"
            )
        before = self.clock.wall()
        for source in sources.values():
            source.bind(before)
        try:
            self.ctl.send("start", prompt=step.prompt, checkpoint=self.plan.checkpoint or status.get("checkpoint"))
        except sio.StationRefusedError as error:
            raise SequenceStop("error", f"the station refused to start step {step.name}: {error}") from None
        self._wait(
            "the run to start",
            lambda s: s.get("state") == "running" and s.get("run") != previous_run,
            self.options.settle_s,
            f"the new run of step {step.name} did not start",
        )
        self._wait_recording(step, sources)

    def _wait_recording(self, step: Step, sources: Mapping[str, sio.FrameSource]) -> None:
        deadline = self.clock.now() + self.options.recording_wait_s
        problem = ""
        while True:
            problem = ""
            for source in sources.values():
                try:
                    source.latest(1)
                except sio.SourceError as error:
                    problem = str(error)
            if not problem:
                return
            if self.clock.now() > deadline:
                if not self.options.observe_only:
                    with contextlib.suppress(sio.StationError):
                        self.ctl.send("stop", force=True)
                raise SequenceStop(
                    "error",
                    f"no recording of step {step.name} appeared in {self.options.recording_wait_s:g} s: {problem}",
                )
            self._check_abort()
            self.clock.sleep(self.options.tick_s)

    def _wait_hand_start(self, loaded: LoadedStep, previous_run: Any) -> None:
        """Observe-only: the person stops and starts runs by hand, so wait for the next step's run to appear."""
        step = loaded.step
        self.ctl.send("start", prompt=step.prompt, checkpoint=self.plan.checkpoint)  # only logs what it would do
        self.log.write("wait", f"waiting for you to start step {step.name} by hand (a new run)")
        self._wait(
            "a new run to be started by hand",
            lambda s: s.get("state") == "running" and s.get("run") != previous_run,
            self.options.wait_s,
            f"no run was started for step {step.name}",
        )

    def _run_step(self, index: int, loaded: LoadedStep, results: Sequence[StepResult]) -> StepResult:
        step, spec = loaded.step, loaded.spec
        previous_run = results[-1].run if results else None
        options = self.options
        self.log.write(
            "step_start", f"step {index + 1} of {len(self.steps)}: {step.name} ({spec.name})", step=step.name
        )
        folder = self.log.folder / f"{index + 1:02d}-{step.name}" if self.log.folder else None
        echo = (lambda line, n=step.name: self._echo(f"[{n}] {line}")) if self._echo else None
        step_log = sv.EventLog(folder, self.clock, echo=echo, extra={"step": step.name}, parent=self.log)
        sources = self.source_factory(loaded)
        started_here = index > 0 or options.start
        if options.observe_only:
            started_here = False
            if index > 0:
                self._wait_hand_start(loaded, previous_run)
        elif started_here:
            self._start_step(loaded, sources, previous_run)
        critic, compare = self.critic_factory(spec)
        supervisor = sv.Supervisor(
            spec,
            critic,
            self.client,
            sources,
            log=step_log,
            clock=self.clock,
            compare=compare,
            options=sv.Options(
                observe_only=options.observe_only,
                wait_s=options.wait_s,
                tick_s=options.tick_s,
                settle_s=options.settle_s,
            ),
            keys=self.bus.get,
            pool=self.pool,
            controller=self.ctl,
            allow_skip=True,
        )
        report = supervisor.run()
        self.ctl.log = self.log
        prompt_log = report.get("prompt_log") or []
        result = StepResult(
            name=step.name,
            task=spec.name,
            prompt=step.prompt,
            outcome=report["outcome"],
            detail=report["detail"],
            attempts=report["attempts"],
            guides=report["guides"],
            prompt_log=prompt_log,
            elapsed_s=report["elapsed_s"],
            last_verdict=report.get("last_verdict"),
            done_at_start=bool(report.get("done_at_start")),
            homed=report.get("homed"),
            control_calls=report["control_calls"],
            run=report.get("run"),
            folders=report.get("folders") or [],
            log=report.get("log"),
        )
        if not started_here and prompt_log and prompt_log[0]["prompt"] != step.prompt:
            result.note = f"the run was started with {prompt_log[0]['prompt']!r}, not the sequence file's prompt"
            self.log.write("note", f"step {step.name}: {result.note}")
        if result.done_at_start:
            self.log.write("note", f"step {step.name}: done at start (the first probes already said done)")
        return result

    # --- the audit ----------------------------------------------------------------------------------------------

    def _audit(self, results: Sequence[StepResult]) -> None:
        """After the last step: every task is judged again from the last frames of the last run. Report only."""
        last = results[-1]
        self.log.write("audit", "final audit: judging every task from the last frames of the last run")
        for loaded, result in zip(self.steps, results, strict=True):
            result.audit = self._audit_one(loaded, result, last)
            self.log.write(
                "audit",
                f"audit {loaded.step.name}: {result.audit['result']}"
                + (" (a regression: the step was done)" if result.audit.get("regression") else ""),
                step_audited=loaded.step.name,
                audit=result.audit,
            )

    def _audit_one(self, loaded: LoadedStep, result: StepResult, last: StepResult) -> dict[str, Any]:
        spec = loaded.spec
        if self.audit_source_factory is None:
            return {"result": "unavailable", "reason": "no way to read the last run's frames"}
        try:
            sources = self.audit_source_factory(loaded, last)
            taken = {camera: source.latest(spec.supervisor.frames_per_probe) for camera, source in sources.items()}
            count = min(len(frames) for frames in taken.values())
            if count == 0:
                return {"result": "unavailable", "reason": "the last run has no frames"}
            sets = [{camera: taken[camera][i].bgr for camera in taken} for i in range(count)]
            if sv.has_face(spec, sets[-1]):
                return {"result": "unavailable", "reason": "a face is in the picture: nothing was sent"}
            pictures = [tc.cut_pictures(spec, frames) for frames in sets]
            critic, _ = self.critic_factory(spec)
            gate = critic.gate
            wanted = tuple(gate.wanted_views(spec)) if hasattr(gate, "wanted_views") else spec.cameras
            for i, picture_set in enumerate(pictures):
                self.log.save_pictures(
                    f"audit-{loaded.step.name}-{i}", {k: v for k, v in picture_set.items() if k in wanted}
                )
            verdict = critic.evaluate(pictures, (), rewrite=False)
        except (sio.SourceError, tc.CriticError) as error:
            return {"result": "unavailable", "reason": str(error)}
        judged = verdict.result
        votes = judged.votes or []
        reasons = list(dict.fromkeys(v["reason"] for v in votes if v["answer"] != "yes" and v["reason"]))
        name = {tc.DONE: "still_done", tc.NOT_DONE: "undone", tc.UNSURE: "unsure"}[judged.status]
        return {
            "result": name,
            "regression": name == "undone" and result.outcome == "done",
            "votes": votes,
            "reasons": reasons[:3],
        }

    # --- the report ---------------------------------------------------------------------------------------------

    def _report(self, outcome: str, detail: str, results: Sequence[StepResult]) -> dict[str, Any]:
        ran = {r.name for r in results}
        rows = [dataclasses.asdict(r) for r in results]
        report = {
            "outcome": outcome,
            "detail": detail,
            "observe_only": self.options.observe_only,
            "sequence": str(self.plan.path),
            "checkpoint": self.plan.checkpoint,
            "on_give_up": self.plan.on_give_up,
            "steps": rows,
            "not_run": [s.step.name for s in self.steps if s.step.name not in ran],
            "control_calls": self.ctl.calls,
            "elapsed_s": round(self.clock.now() - self._started_at, 1),
            "log": str(self.log.folder) if self.log.folder else None,
        }
        table = render_summary(report)
        self.log.write("end", f"{outcome}: {detail}", **{k: v for k, v in report.items() if k != "steps"})
        self.log.report(report, "sequence.json")
        if self.log.folder is not None:
            (self.log.folder / "summary.txt").write_text(table + "\n", encoding="utf-8")
        if self._echo is not None:
            self._echo("\n" + table)
        return report


def render_summary(report: Mapping[str, Any]) -> str:
    """The combined table: step, outcome, attempts, seconds, prompts, the last probe, the audit."""
    header = ["step", "outcome", "attempts", "seconds", "prompts", "last probe", "audit"]
    rows = [header]
    for step in report["steps"]:
        verdict = step.get("last_verdict") or {}
        votes = verdict.get("votes") or []
        probe = (
            f"{verdict['decision']} ({sum(v['answer'] == 'yes' for v in votes)}/{len(votes)} yes)" if verdict else "-"
        )
        audit = step.get("audit") or {}
        label = audit.get("result", "-")
        if audit.get("regression"):
            label += " (regression)"
        outcome = step["outcome"] + (" (done at start)" if step.get("done_at_start") else "")
        rows.append(
            [
                step["name"],
                outcome,
                str(step["attempts"]),
                f"{step['elapsed_s']:.0f}",
                str(len(step["prompt_log"])),
                probe,
                label,
            ]
        )
    for name in report.get("not_run", []):
        rows.append([name, "not run", "-", "-", "-", "-", "-"])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append(f"sequence {report['outcome']}: {report['detail']} ({report['elapsed_s']:.0f} s)")
    return "\n".join(lines)


# --- Commands --------------------------------------------------------------------------------------------------


def confirm_send(args: argparse.Namespace, steps: Sequence[LoadedStep], logs: pathlib.Path) -> None:
    """Says what is about to leave this computer and asks; refuses without a person to ask unless --yes."""
    if args.dry:
        print("Dry run: stand-ins answer and nothing is sent.")
        return
    needed = ["GEMINI_API_KEY"] + (["ANTHROPIC_API_KEY"] if args.rewriter == "claude" else [])
    missing = [name for name in needed if not os.environ.get(name)]
    if missing:
        raise SequenceError(
            f"set {' and '.join(missing)} in the environment first (type it with `read -rs`, not in a file)"
        )
    model = args.model or tc.GEMINI_MODEL
    lines = [f"About to send pictures of the table outside this computer, cropped to it, to Gemini ({model}):"]
    for loaded in steps:
        spec, config = loaded.spec, loaded.spec.supervisor
        views = [spec.vote_gate.view if spec.vote_gate else "?"]
        lines.append(
            f"  step {loaded.step.name}: the zoom {views[0]!r} of camera {spec.views[views[0]].camera!r}, "
            f"{config.frames_per_probe} pictures every {config.probe_every_s:g} s after the first "
            f"{config.warmup_s:g} s, {spec.vote_gate.repeats if spec.vote_gate else 1} questions each; "
            "two pictures to confirm a stall"
        )
    lines.append("  after the last step: the same zoom of every task, from the last frames, for the audit")
    if args.rewriter == "gemini":
        lines.append("  Gemini: the scene crop and the zoom, to write each new prompt")
    elif args.rewriter == "claude":
        lines.append(f"  Claude (api.anthropic.com, model {tc.ANTHROPIC_MODEL}): the same, to write each new prompt")
    lines += [
        "The key comes from GEMINI_API_KEY. A picture with a face in the crop is never sent.",
        f"Every picture sent is kept under {logs}; the requests without pictures are in requests.jsonl there.",
    ]
    print("\n".join(lines))
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise SequenceError("no one to ask: rerun with --yes to send the pictures")
    if input("Send? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SequenceError("not sent")


def folder_for(root: pathlib.Path, clock: sv.Clock) -> pathlib.Path:
    stamp = clock.wall().astimezone().strftime("%Y%m%d-%H%M%S")
    return pathlib.Path(root).expanduser() / f"sequence-{stamp}"


def exit_code(report: Mapping[str, Any]) -> int:
    """0 when every step ran and none gave up, or the operator ended it; 3 when a step gave up; 1 on an error."""
    outcome = report["outcome"]
    if outcome in ("error", "crashed"):
        return 1
    if outcome == "stopped" or any(s["outcome"] == "gave_up" for s in report["steps"]):
        return 3
    return 0


def command_run(args: argparse.Namespace) -> int:
    """Runs the sequence against the station."""
    if args.observe_only and args.start:
        raise SequenceError("--observe-only makes no control call, so it cannot --start a run")
    plan = load_plan(args.file)
    steps = load_steps(plan)
    clock = sv.RealClock()
    folder = folder_for(args.logs, clock)
    confirm_send(args, steps, folder)
    log = sv.EventLog(folder, clock)
    recorder = sv.rollouts_dir(args)

    def critic_factory(spec: tc.TaskSpec) -> tuple[tc.TaskCritic, sv.Compare | None]:
        return sv.build_critic(args, spec, log.request)

    def source_factory(loaded: LoadedStep) -> dict[str, sio.FrameSource]:
        return {c: sio.GrowingVideo(recorder, c, clock=clock.now) for c in sv.cameras_needed(loaded.spec)}

    def audit_source_factory(loaded: LoadedStep, last: StepResult) -> dict[str, sio.FrameSource]:
        if not last.folders:
            raise sio.SourceError(f"step {last.name} read no recording, so there is no last run to look at")
        sources = {c: sio.GrowingVideo(recorder, c, clock=clock.now) for c in sv.cameras_needed(loaded.spec)}
        for source in sources.values():
            source.pin(pathlib.Path(last.folders[-1]))
        return sources

    with contextlib.suppress(OSError, AttributeError):
        os.nice(10)  # the arm thread must never wait for this
    signal.signal(signal.SIGTERM, sv._raise_interrupt)
    options = SeqOptions(observe_only=args.observe_only, start=args.start, wait_s=args.wait_s)
    with sv.KeyReader() as keys, concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        runner = SequenceRunner(
            plan,
            steps,
            sio.StationClient(args.url),
            log=log,
            clock=clock,
            options=options,
            critic_factory=critic_factory,
            source_factory=source_factory,
            audit_source_factory=audit_source_factory,
            keys=keys.get,
            pool=pool,
        )
        report = runner.run()
    return exit_code(report)


def command_rehearse(args: argparse.Namespace) -> int:
    """Replays one recorded video per step against a fake station: nothing moves. With real models it sends the
    replayed pictures, after asking."""
    if args.observe_only:
        raise SequenceError("a rehearsal has no person to start runs by hand: leave out --observe-only")
    plan = load_plan(args.file)
    steps = load_steps(plan)
    names = [s.step.name for s in steps]
    videos: dict[str, pathlib.Path] = {}
    for item in args.video:
        name, _, path = item.partition("=")
        videos[name] = pathlib.Path(path).expanduser()
    if sorted(videos) != sorted(names):
        raise SequenceError(f"give one --video STEP=PATH for each of {names}")
    for loaded in steps:
        if len(sv.cameras_needed(loaded.spec)) != 1:
            raise SequenceError(f"step {loaded.step.name} uses more than one camera: a rehearsal replays one video")
    clock = sv.RealClock(args.speed)
    folder = folder_for(args.logs, clock)
    confirm_send(args, steps, folder)
    log = sv.EventLog(folder, clock)
    station = sio.FakeStation(clock=clock.now, checkpoint=plan.checkpoint or "dreamlabs/dream-1-beta")
    stop = sv.start_commanding(station, args.speed)

    def critic_factory(spec: tc.TaskSpec) -> tuple[tc.TaskCritic, sv.Compare | None]:
        return sv.build_critic(args, spec, log.request)

    def source_factory(loaded: LoadedStep) -> dict[str, sio.FrameSource]:
        camera = sv.cameras_needed(loaded.spec)[0]
        return {camera: sio.ReplayVideo(videos[loaded.step.name], clock=clock.now)}

    def audit_source_factory(loaded: LoadedStep, last: StepResult) -> dict[str, sio.FrameSource]:
        camera = sv.cameras_needed(loaded.spec)[0]
        return {camera: sio.ReplayVideo(videos[last.name], clock=clock.now, start_s=1e9)}  # rests on its last picture

    options = SeqOptions(start=True, wait_s=5.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            runner = SequenceRunner(
                plan,
                steps,
                station.client(),
                log=log,
                clock=clock,
                options=options,
                critic_factory=critic_factory,
                source_factory=source_factory,
                audit_source_factory=audit_source_factory,
                pool=pool,
            )
            report = runner.run()
    finally:
        stop.set()
    print(f"fake station calls: {[route for route, _ in station.calls]}")
    return exit_code(report)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    top.add_argument(
        "--logs", type=pathlib.Path, default=sv.DEFAULT_LOGS, help=f"where logs go (default {sv.DEFAULT_LOGS})"
    )
    commands = top.add_subparsers(dest="command", required=True)

    def file_argument(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("file", type=pathlib.Path, nargs="?", default=DEFAULT_SEQUENCE, help="the sequence file")

    def models(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--model", help=f"the Gemini model (default {tc.GEMINI_MODEL})")
        sub.add_argument(
            "--rewriter", choices=["gemini", "claude", "none"], default="gemini", help="who writes the next prompt"
        )
        sub.add_argument("--dry", action="store_true", help="stand-ins answer (always unsure); nothing is sent")
        sub.add_argument("--yes", action="store_true", help="do not ask before sending pictures")
        sub.add_argument("--observe-only", action="store_true", help="make no control call; log what it would do")

    check = commands.add_parser("check", help="validate the sequence file and its task files; needs no network")
    file_argument(check)
    check.set_defaults(handler=command_check)

    run = commands.add_parser("run", help="run the sequence (or only log, with --observe-only)")
    file_argument(run)
    models(run)
    run.add_argument("--url", default=sio.DEFAULT_URL, help=f"the station agent (default {sio.DEFAULT_URL})")
    run.add_argument("--rollouts", type=pathlib.Path, help="the recorder's folder (default: the station file's)")
    run.add_argument("--start", action="store_true", help="start the first step too; else attach to a run you started")
    run.add_argument(
        "--wait-s", type=float, default=120.0, help="seconds to wait for a run to attach to (default 120)"
    )
    run.set_defaults(handler=command_run)

    rehearse = commands.add_parser("rehearse", help="replay one recorded video per step against a fake station")
    file_argument(rehearse)
    models(rehearse)
    rehearse.add_argument("--video", action="append", required=True, help="STEP=PATH: a recorded video for each step")
    rehearse.add_argument(
        "--speed", type=float, default=1.0, help="replay speed (default 1; model calls are not sped up)"
    )
    rehearse.set_defaults(handler=command_rehearse)
    return top


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        code = args.handler(args)
    except (sv.SupervisorError, tc.CriticError, tc.TaskFileError) as error:
        sys.exit(f"sequence: {error}")
    sys.exit(code)


if __name__ == "__main__":
    main()
