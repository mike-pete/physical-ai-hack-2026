import asyncio
import importlib.util
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools" / "lab_worker.py"


@pytest.fixture(scope="module")
def lw():
    spec = importlib.util.spec_from_file_location("lab_worker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sink:
    def __init__(self):
        self.events = []

    def write(self, text):
        self.events.append(json.loads(text))

    def flush(self):
        pass

    def kinds(self):
        return [e["event"] for e in self.events]

    def states(self):
        return [e["state"] for e in self.events if e["event"] == "state"]


def make(lw, tmp_path):
    sink = Sink()
    return lw.Worker(lw.FakeBackend(), tmp_path / "frames", sink), sink


def run_command(tmp_path, seconds=1):
    out = tmp_path / "run"
    return {
        "cmd": "run",
        "run_id": 7,
        "prompt": "open it",
        "checkpoint": "c",
        "dry_run": True,
        "max_seconds": seconds,
        "out_dir": str(out),
    }


def test_weightless_hold_and_home_walk_the_states(lw, tmp_path):
    async def go():
        worker, sink = make(lw, tmp_path)
        await worker._handle({"cmd": "weightless"})
        await worker._handle({"cmd": "hold"})
        await worker._handle({"cmd": "home", "seconds": 0.1})
        return sink.states()

    assert asyncio.run(go()) == ["weightless", "holding", "homing", "holding"]


def test_a_run_that_hits_its_limit_finishes_and_leaves_the_arms_holding(lw, tmp_path):
    pytest.importorskip("av")

    async def go():
        worker, sink = make(lw, tmp_path)
        await worker._handle(run_command(tmp_path, seconds=1))
        await worker._run_task
        return sink

    sink = asyncio.run(go())
    ended = next(e for e in sink.events if e["event"] == "run_ended")
    assert ended["status"] == "finished" and ended["run_id"] == 7 and not ended["error"]
    assert sink.states() == ["running", "holding"]
    assert sink.kinds().index("run_started") < sink.kinds().index("run_ended")
    out = tmp_path / "run"
    assert json.loads((out / "run.json").read_text())["prompt"] == "open it"
    for camera in ("overhead", "wrist_l", "wrist_r"):
        assert (out / f"{camera}.mp4").stat().st_size > 0


def test_stop_ends_a_run_as_stopped(lw, tmp_path):
    pytest.importorskip("av")

    async def go():
        worker, sink = make(lw, tmp_path)
        await worker._handle(run_command(tmp_path, seconds=60))
        await asyncio.sleep(0.4)
        await worker._handle({"cmd": "stop"})
        await worker._run_task
        return sink

    sink = asyncio.run(go())
    assert next(e for e in sink.events if e["event"] == "run_ended")["status"] == "stopped"


def test_commands_are_ignored_while_a_run_goes(lw, tmp_path):
    async def go():
        worker, sink = make(lw, tmp_path)
        await worker._handle(run_command(tmp_path, seconds=60))
        await asyncio.sleep(0.2)
        await worker._handle({"cmd": "home"})
        worker._stop_run.set()
        await worker._run_task
        return sink

    sink = asyncio.run(go())
    assert "homing" not in sink.states() and "log" in sink.kinds()


def test_a_backend_failure_in_a_run_is_reported_as_an_error(lw, tmp_path):
    class Broken(lw.FakeBackend):
        async def run(self, prompt, checkpoint, dry_run, stop):
            raise RuntimeError("the model refused")

    async def go():
        sink = Sink()
        worker = lw.Worker(Broken(), tmp_path / "frames", sink)
        await worker._handle(run_command(tmp_path))
        await worker._run_task
        return sink

    sink = asyncio.run(go())
    ended = next(e for e in sink.events if e["event"] == "run_ended")
    assert ended["status"] == "error" and "the model refused" in ended["error"]


def test_shutdown_homes_then_powers_off(lw, tmp_path):
    calls = []

    class Spy(lw.FakeBackend):
        def home(self, seconds):
            calls.append(("home", seconds))
            return True

        def power_off(self):
            calls.append(("power_off",))

        def close(self):
            calls.append(("close",))

    async def go():
        worker = lw.Worker(Spy(), tmp_path / "frames", Sink())
        await worker._shutdown()

    asyncio.run(go())
    assert calls == [("home", lw.SHUTDOWN_HOME_S), ("power_off",), ("close",)]


def test_serve_announces_the_cameras_and_writes_live_frames(lw, tmp_path):
    pytest.importorskip("cv2")

    async def go():
        sink = Sink()
        worker = lw.Worker(lw.FakeBackend(), tmp_path / "frames", sink)
        commands = asyncio.Queue()
        task = asyncio.create_task(worker.serve(commands))
        await asyncio.sleep(0.6)
        commands.put_nowait({"cmd": "shutdown"})
        await task
        return sink

    sink = asyncio.run(go())
    ready = sink.events[0]
    assert ready["event"] == "ready" and ready["cameras"] == ["overhead", "wrist_l", "wrist_r"]
    assert (tmp_path / "frames" / "overhead.jpg").read_bytes()[:2] == b"\xff\xd8"


def test_clear_fault_runs_and_the_state_settles(lw, tmp_path):
    async def go():
        worker, sink = make(lw, tmp_path)
        await worker._handle({"cmd": "clear_fault"})
        return sink

    sink = asyncio.run(go())
    assert sink.states() == ["idle"] and "nothing to clear" in sink.events[-1]["message"]
