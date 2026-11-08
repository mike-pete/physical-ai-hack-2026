"""mirror_collect's dashboard: who may talk to it, what a click does, and how homing and saving fail safely."""

import dataclasses
import http.server
import importlib.util
import json
import pathlib
import queue
import socket
import threading
from collections.abc import Iterator
from types import ModuleType

import numpy as np
import pytest

from dream_hackathon import dataset, layouts

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mirror_collect.py"
HOME = np.zeros(6)


@pytest.fixture(scope="module")
def mc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mirror_collect", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Camera:
    def latest(self) -> np.ndarray:
        return np.full((36, 64, 3), 128, np.uint8)


@dataclasses.dataclass
class _Server:
    port: int
    state: object
    commands: "queue.Queue[str]"


@pytest.fixture
def server(mc: ModuleType) -> Iterator[_Server]:
    state = mc._DashState(task="pick it up")
    commands: queue.Queue[str] = queue.Queue()
    handler = mc._make_handler(state, commands, {"overhead": _Camera()})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield _Server(httpd.server_address[1], state, commands)
    httpd.shutdown()
    httpd.server_close()


def _raw(port: int, method: str, path: str, headers: dict[str, str], body: bytes = b"") -> tuple[int, bytes]:
    """Sends one request with exactly these headers (http.client would add its own Host and Content-Length)."""
    head = "\r\n".join([f"{method} {path} HTTP/1.0", *(f"{k}: {v}" for k, v in headers.items())])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(head.encode() + b"\r\n\r\n" + body)
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    status_line, _, payload = data.partition(b"\r\n\r\n")
    return int(status_line.split(b" ", 2)[1]), payload


def _ok(port: int) -> dict[str, str]:
    return {"Host": f"127.0.0.1:{port}", "X-Mirror-Dashboard": "1"}


def _post_task(server: _Server, body: bytes, **headers: str) -> tuple[int, bytes]:
    sent = {**_ok(server.port), "Content-Length": str(len(body)), **headers}
    return _raw(server.port, "POST", "/api/task", sent, body)


# ------------------------------------------------------------------ who may talk to the dashboard


@pytest.mark.parametrize("host", ["localhost", "localhost:8090", "LOCALHOST:9000", "127.0.0.1:8090", "[::1]:8090"])
def test_loopback_hosts_are_allowed(mc: ModuleType, host: str) -> None:
    assert mc._host_allowed(host)


@pytest.mark.parametrize(
    "host",
    [None, "", "evil.example", "evil.example:8090", "localhost.evil.example", "127.0.0.1.evil.example:80",
     "localhost:abc", "localhost:123456", "[::1", "0.0.0.0:8090", "localhost:8090/x"],
)  # fmt: skip
def test_other_hosts_are_refused(mc: ModuleType, host: str | None) -> None:
    assert not mc._host_allowed(host)


def test_a_rebound_host_gets_nothing(server: _Server) -> None:
    rebound = {"Host": "evil.example:8080", "X-Mirror-Dashboard": "1"}
    for method, path in (
        ("GET", "/"),
        ("GET", "/api/status"),
        ("GET", "/api/camera/overhead.jpg"),
        ("POST", "/api/home"),
        ("POST", "/api/episode/start"),
    ):
        assert _raw(server.port, method, path, rebound)[0] == 403, (method, path)
    assert server.commands.empty()


def test_the_page_and_the_cameras_load_for_the_loopback_host(server: _Server) -> None:
    assert _raw(server.port, "GET", "/", {"Host": f"localhost:{server.port}"})[0] == 200
    assert _raw(server.port, "GET", "/api/camera/overhead.jpg", {"Host": f"[::1]:{server.port}"})[0] == 200


def test_a_cross_origin_post_is_refused(server: _Server) -> None:
    assert _raw(server.port, "POST", "/api/home", {**_ok(server.port), "Origin": "http://evil.example"})[0] == 403
    assert server.commands.empty()
    same_origin = {**_ok(server.port), "Origin": f"http://127.0.0.1:{server.port}"}
    assert _raw(server.port, "POST", "/api/home", same_origin)[0] == 200


def test_a_post_without_the_dashboard_header_is_refused(server: _Server) -> None:
    assert _raw(server.port, "POST", "/api/home", {"Host": f"127.0.0.1:{server.port}"})[0] == 403
    assert server.commands.empty()


# ------------------------------------------------------------------ what a click does


def test_start_and_save_are_separate_commands(server: _Server) -> None:
    for path in ("/api/episode/start", "/api/episode/save", "/api/episode/discard", "/api/home"):
        assert _raw(server.port, "POST", path, _ok(server.port))[0] == 200
    assert [server.commands.get_nowait() for _ in range(4)] == ["start", "save", "d", "h"]


@pytest.mark.parametrize(
    ("cmd", "recording", "action"),
    [
        ("", False, "start"),  # Enter and `s` on the keyboard toggle
        ("", True, "save"),
        ("s", False, "start"),
        ("s", True, "save"),
        ("start", False, "start"),
        ("start", True, "ignore"),  # a second click on Start must not save the episode it just began
        ("save", True, "save"),
        ("save", False, "ignore"),  # a second click on Save must not begin a new one
    ],
)
def test_what_a_start_or_save_command_does(mc: ModuleType, cmd: str, recording: bool, action: str) -> None:
    assert mc._episode_action(cmd, recording) == action


# ------------------------------------------------------------------ how the leader arm is held


@pytest.mark.parametrize(
    ("asked", "active", "mode"),
    [
        (False, False, "torque_off"),  # no gravity compensation: the person carries the arm
        (True, True, "gravity"),
        (True, False, "gravity_failed"),  # asked for, but it fell back to torque off: must not look like torque off
    ],
)
def test_the_leader_mode_tells_a_failed_gravity_from_none_asked(
    mc: ModuleType, asked: bool, active: bool, mode: str
) -> None:
    assert mc._leader_mode(asked, active) == mode


def test_every_leader_mode_has_words_naming_the_arm(mc: ModuleType) -> None:
    labels = {mode: mc._leader_mode_label(mode, "left") for mode in ("gravity", "torque_off", "gravity_failed")}
    assert labels["gravity"] == "left arm: gravity compensation"
    assert all(label.startswith("left arm: ") for label in labels.values())
    assert len(set(labels.values())) == 3


def test_the_status_reports_how_the_leader_is_held(server: _Server, mc: ModuleType) -> None:
    status, payload = _raw(server.port, "GET", "/api/status", {"Host": f"127.0.0.1:{server.port}"})
    assert status == 200 and json.loads(payload)["leader_mode"] == ""  # the leader is not open yet
    with server.state.lock:
        server.state.leader_mode = mc.LEADER_TORQUE_OFF
        server.state.leader_mode_label = mc._leader_mode_label(mc.LEADER_TORQUE_OFF, "left")
    snap = json.loads(_raw(server.port, "GET", "/api/status", {"Host": f"127.0.0.1:{server.port}"})[1])
    assert snap["leader_mode"] == "torque_off"
    assert snap["leader_mode_label"] == "left arm: torque off, no gravity compensation"


# ------------------------------------------------------------------ the task text


def test_the_task_is_cleaned(server: _Server) -> None:
    status, _ = _post_task(server, json.dumps({"task": "  pick\nup\tthe cube  "}).encode())
    assert status == 200 and server.state.task == "pick up the cube"


def test_a_task_is_cut_to_dreams_limit(server: _Server, mc: ModuleType) -> None:
    assert _post_task(server, json.dumps({"task": "x" * 600}).encode())[0] == 200
    assert server.state.task == "x" * mc.TASK_MAX_CHARS


def test_a_blank_task_falls_back_to_the_default(mc: ModuleType) -> None:
    assert mc._task_or_default("   ") == mc.DEFAULT_TASK
    assert mc._task_or_default("") == mc.DEFAULT_TASK
    assert mc._task_or_default("  go  ") == "go"


@pytest.mark.parametrize(("length", "status"), [("99999999", 413), ("abc", 400), ("-5", 400), ("5.0", 400), ("", 400)])
def test_a_bad_length_is_refused_without_reading_the_body(server: _Server, length: str, status: int) -> None:
    headers = {**_ok(server.port), "Content-Length": length}
    assert _raw(server.port, "POST", "/api/task", headers)[0] == status
    assert server.state.task == "pick it up"


def test_a_missing_length_is_refused(server: _Server) -> None:
    assert _raw(server.port, "POST", "/api/task", _ok(server.port))[0] == 400


@pytest.mark.parametrize("body", [b"{nope", b"[1]", b"\xff\xfe", b""])
def test_a_body_that_is_not_a_json_object_is_a_400(server: _Server, body: bytes) -> None:
    assert _post_task(server, body)[0] == 400
    assert server.state.task == "pick it up"


# ------------------------------------------------------------------ homing and saving fail safely


def test_homing_ends_when_every_joint_is_at_home(mc: ModuleType) -> None:
    assert mc._home_step(HOME + 5e-4, HOME, 0.1, 0.0) == ("reached", None)


def test_homing_moves_in_capped_steps(mc: ModuleType) -> None:
    outcome, command = mc._home_step(np.array([1.0, -1.0, 0.05, 0, 0, 0]), HOME, 0.1, 1.0)
    assert outcome == "move"
    assert np.allclose(command, [0.9, -0.9, 0.0, 0, 0, 0])


def test_homing_gives_up_after_the_timeout(mc: ModuleType) -> None:
    far = np.full(6, 1.0)
    assert mc._home_step(far, HOME, 0.1, mc.HOME_TIMEOUT_S - 1)[0] == "move"
    assert mc._home_step(far, HOME, 0.1, mc.HOME_TIMEOUT_S + 1) == ("timeout", None)


def test_homing_holds_while_the_follower_cannot_be_read(mc: ModuleType) -> None:
    """A failed read must not be replaced by a placeholder pose: the arm would be ramped from there."""
    assert mc._home_step(None, HOME, 0.1, 1.0) == ("hold", None)
    assert mc._home_step(None, HOME, 0.1, mc.HOME_TIMEOUT_S + 1) == ("timeout", None)


class _Episode:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def save(self) -> int:
        if self._error:
            raise self._error
        return 4


def test_a_saved_episode_returns_its_index(mc: ModuleType) -> None:
    assert mc._save_episode(_Episode()) == 4


def test_a_short_episode_is_reported_and_the_session_goes_on(mc: ModuleType, capsys: pytest.CaptureFixture) -> None:
    assert mc._save_episode(_Episode(dataset.DatasetError("episode too short: 3 frames"))) is None
    assert "episode too short" in capsys.readouterr().out


def test_any_other_save_error_still_stops_the_session(mc: ModuleType) -> None:
    with pytest.raises(OSError, match="disk full"):
        mc._save_episode(_Episode(OSError("disk full")))


# ------------------------------------------------------------------ one folder per task


@pytest.mark.parametrize(
    ("task", "folder"),
    [
        ("pick up the red cube", "pick-up-the-red-cube"),
        ("  Pick Up: the RED cube!!  ", "pick-up-the-red-cube"),
        ("Recoge el cubo, ñandú café", "recoge-el-cubo-nandu-cafe"),
        ("../../etc/passwd", "etc-passwd"),
        ("a/b\\c", "a-b-c"),
        ("CUBE_2 v3", "cube-2-v3"),
    ],
)
def test_a_task_names_its_folder(mc: ModuleType, task: str, folder: str) -> None:
    assert mc._task_folder(task) == folder


def test_a_long_task_gives_a_short_folder_without_a_trailing_hyphen(mc: ModuleType) -> None:
    folder = mc._task_folder("word " * 40)
    assert len(folder) <= mc.FOLDER_MAX_CHARS and not folder.endswith("-") and folder.startswith("word-word")


def test_a_task_without_latin_letters_still_gets_a_folder_of_its_own(mc: ModuleType) -> None:
    assert mc._task_folder("日本語") != mc._task_folder("中文")
    assert mc._task_folder("日本語") == mc._task_folder("日本語")
    assert mc._task_folder("日本語").startswith("task-") and mc._task_folder("!!!").startswith("task-")


@pytest.mark.parametrize("task", ["..", ".", "/", "a/../../b", "~", "$(rm -rf x)", "name\x00x", "", "   "])
def test_a_folder_name_can_never_leave_the_root(mc: ModuleType, tmp_path: pathlib.Path, task: str) -> None:
    path = tmp_path / mc._task_folder(task)
    assert path.parent == tmp_path and path.name not in ("", ".", "..")


class _FakeWriter:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.total_episodes = 0


def _folders(mc: ModuleType, root: pathlib.Path, *, pinned: pathlib.Path | None = None):
    """A _TaskFolders over fake writers, with the list of folders it opened and the folder names it should refuse."""
    opened: list[pathlib.Path] = []
    refused: set[str] = set()

    def open_writer(path: pathlib.Path) -> _FakeWriter:
        if path.name in refused:
            raise dataset.DatasetError(f"{path} holds a dataset recorded another way")
        opened.append(path)
        return _FakeWriter(path)

    return mc._TaskFolders(root, open_writer, pinned=pinned), opened, refused


def test_setting_a_task_creates_its_folder(mc: ModuleType, tmp_path: pathlib.Path) -> None:
    folders, opened, _ = _folders(mc, tmp_path / "hackathon_data")  # the root does not exist yet
    assert folders.folder is None
    assert folders.use("Pick up the cube") is True
    assert (tmp_path / "hackathon_data" / "pick-up-the-cube").is_dir()
    assert folders.folder == tmp_path / "hackathon_data" / "pick-up-the-cube" and folders.error is None


def test_the_same_task_keeps_the_open_dataset(mc: ModuleType, tmp_path: pathlib.Path) -> None:
    folders, opened, _ = _folders(mc, tmp_path)
    folders.use("Pick up the cube")
    writer = folders.writer
    assert folders.use("pick up the CUBE!") is True
    assert folders.writer is writer and len(opened) == 1


def test_a_new_task_gets_a_new_folder_and_the_old_one_stays(mc: ModuleType, tmp_path: pathlib.Path) -> None:
    folders, _, _ = _folders(mc, tmp_path)
    folders.use("stack the blocks")
    folders.use("pour the water")
    assert (tmp_path / "stack-the-blocks").is_dir() and (tmp_path / "pour-the-water").is_dir()
    assert folders.folder == tmp_path / "pour-the-water"


def test_a_folder_that_cannot_be_opened_is_reported(mc: ModuleType, tmp_path: pathlib.Path) -> None:
    folders, _, refused = _folders(mc, tmp_path)
    folders.use("stack the blocks")
    refused.add("pour-the-water")
    assert folders.use("pour the water") is False
    assert "recorded another way" in folders.error
    assert folders.folder == tmp_path / "stack-the-blocks"  # still the old one: the caller must not record into it
    refused.clear()
    assert folders.use("pour the water") is True and folders.error is None
    assert folders.folder == tmp_path / "pour-the-water"


def test_a_root_that_is_a_file_is_reported(mc: ModuleType, tmp_path: pathlib.Path) -> None:
    (tmp_path / "hackathon_data").write_text("not a folder")
    folders, _, _ = _folders(mc, tmp_path / "hackathon_data")
    assert folders.use("stack the blocks") is False
    assert folders.error and folders.folder is None


def test_a_pinned_folder_ignores_the_task(mc: ModuleType, tmp_path: pathlib.Path) -> None:
    pinned = tmp_path / "my-task"
    folders, opened, _ = _folders(mc, tmp_path, pinned=pinned)
    assert folders.use("anything") is True and folders.use("something else") is True
    assert opened == [pinned] and not (tmp_path / "anything").exists()


def test_a_new_task_asks_the_loop_for_its_folder(server: _Server) -> None:
    before = server.state.task_version
    assert _post_task(server, b'{"task": "stack the blocks"}')[0] == 200
    assert server.state.task == "stack the blocks" and server.state.task_version == before + 1


def test_a_blank_task_is_refused_when_the_task_names_the_folder(server: _Server) -> None:
    server.state.per_task = True
    assert _post_task(server, b'{"task": "   "}')[0] == 400
    assert server.state.task == "pick it up" and server.state.task_version == 0


def test_a_blank_task_still_resets_a_pinned_session(server: _Server) -> None:
    assert _post_task(server, b'{"task": "   "}')[0] == 200 and server.state.task == ""


def test_the_status_shows_the_folder(server: _Server) -> None:
    server.state.per_task = True
    server.state.folder = "/data/stack-the-blocks"
    status, body = _raw(server.port, "GET", "/api/status", {"Host": f"127.0.0.1:{server.port}"})
    snap = json.loads(body)
    assert status == 200 and snap["per_task"] is True and snap["folder"] == "/data/stack-the-blocks"
    assert snap["folder_error"] == ""


def test_the_page_has_a_place_for_the_folder(mc: ModuleType) -> None:
    assert 'id="folder"' in mc._PAGE


def test_episodes_are_saved_in_the_folder_of_their_task(mc: ModuleType, tmp_path: pathlib.Path) -> None:
    """With the real dataset writer: each task's episodes land in its own folder, which later episodes continue."""
    keys = list(layouts.video_keys("makermods").values())
    names = [f"v{i}" for i in range(layouts.WIDTH)]
    folders = mc._TaskFolders(tmp_path, lambda path: dataset.DatasetWriter(path, 30, "metal", names, keys, 320, 180))
    image = np.zeros((180, 320, 3), np.uint8)

    def record(task: str) -> int:
        assert folders.use(task)
        episode = folders.writer.start_episode(task)
        for _ in range(60):
            episode.add_frame(
                np.zeros(layouts.WIDTH, np.float32), np.zeros(layouts.WIDTH, np.float32), dict.fromkeys(keys, image)
            )
        return episode.save()

    assert record("stack the blocks") == 0
    assert record("pour the water") == 0  # a new folder starts at episode 0
    assert record("Stack the blocks!") == 1  # the first task's folder goes on from where it stopped
    for name, episodes in (("stack-the-blocks", 2), ("pour-the-water", 1)):
        info = json.loads((tmp_path / name / "meta" / "info.json").read_text())
        assert info["total_episodes"] == episodes, name
    assert (tmp_path / "stack-the-blocks" / "data" / "chunk-000" / "episode_000001.parquet").exists()
    tasks = (tmp_path / "stack-the-blocks" / "meta" / "tasks.jsonl").read_text().splitlines()
    assert {json.loads(line)["task"] for line in tasks} == {"stack the blocks", "Stack the blocks!"}


# ------------------------------------------------------------------ homing both arms


def _traj(mc: ModuleType, follower, leader, peak: float = 1.0, leader_home=HOME):
    return mc._HomeTrajectory((follower, leader), (HOME, leader_home), peak)


def test_the_trajectory_starts_and_ends_at_rest_and_is_sync_free_of_jumps(mc: ModuleType) -> None:
    start = np.array([1.0, -0.5, 0.25, 0, 0, 0])
    traj = _traj(mc, start, start)
    assert np.allclose(traj.targets(0.0)[0], start)
    assert np.allclose(traj.targets(traj.duration_s)[0], HOME)
    assert np.allclose(traj.targets(traj.duration_s * 10)[0], HOME)
    tiny = 1e-3 * traj.duration_s
    assert np.max(np.abs(traj.targets(tiny)[0] - start)) < 1e-6  # eases in: barely moved after a thousandth of it
    assert np.max(np.abs(traj.targets(traj.duration_s - tiny)[0] - HOME)) < 1e-6  # and eases out


def test_every_joint_arrives_together_on_a_straight_path(mc: ModuleType) -> None:
    start = np.array([1.0, -0.5, 0.25, 0, 0, 0])
    traj = _traj(mc, start, start)
    for fraction in (0.2, 0.5, 0.8):
        q = traj.targets(fraction * traj.duration_s)[0]
        blend = (start[0] - q[0]) / start[0]
        assert np.allclose(start * (1 - blend), q)  # the same share of the way for every joint


def test_the_peak_speed_is_the_cap(mc: ModuleType) -> None:
    start = np.array([2.0, -1.0, 0, 0, 0, 0])
    traj = _traj(mc, start, start, peak=0.75)
    times = np.linspace(0.0, traj.duration_s, 2001)
    path = np.array([traj.targets(t)[0] for t in times])
    speed = np.max(np.abs(np.diff(path, axis=0)), axis=1) / np.diff(times)
    assert speed.max() == pytest.approx(0.75, rel=0.01)


def test_both_arms_share_one_duration_set_by_the_farther(mc: ModuleType) -> None:
    near, far = np.full(6, 0.1), np.full(6, 1.0)
    traj = _traj(mc, near, far)
    assert traj.duration_s == pytest.approx(1.875 * 1.0 / 1.0)
    assert np.allclose(traj.targets(traj.duration_s / 2)[0], near / 2)  # the near arm eases over the same time


def test_an_arm_at_home_has_no_duration(mc: ModuleType) -> None:
    assert _traj(mc, HOME, HOME).duration_s == 0.0


def test_both_arms_are_commanded_along_the_trajectory(mc: ModuleType) -> None:
    start = np.full(6, 1.0)
    traj = _traj(mc, start, -start)
    t = traj.duration_s / 2
    outcome, follower, leader = mc._home_both_step(start / 2, HOME, -start / 2, HOME, traj, t)
    assert outcome == "move"
    assert np.allclose(follower, start / 2)  # halfway, and the arm is there
    assert np.allclose(leader, -start / 2)


def test_a_command_never_runs_further_ahead_than_its_cap(mc: ModuleType) -> None:
    start = np.full(6, 1.0)
    traj = _traj(mc, start, start)
    stalled = start  # the arm has not moved while the trajectory has run on
    _, follower, leader = mc._home_both_step(stalled, HOME, stalled, HOME, traj, traj.duration_s)
    assert np.allclose(follower, stalled - mc.HOME_MAX_LEAD_RAD)
    assert np.allclose(leader, stalled - mc.LEADER_HOME_MAX_LEAD_RAD)


def test_an_arm_that_is_home_is_held_there_while_the_other_arrives(mc: ModuleType) -> None:
    far = np.full(6, 1.0)
    traj = _traj(mc, HOME, far)
    outcome, follower, leader = mc._home_both_step(HOME, HOME, far, HOME, traj, 0.0)
    assert outcome == "move"
    assert np.array_equal(follower, HOME)
    assert np.allclose(leader, far)  # the trajectory has not started: the leader is where it is


def test_homing_is_reached_only_when_both_arms_are_home(mc: ModuleType) -> None:
    traj = _traj(mc, np.full(6, 1.0), np.full(6, 1.0))
    assert mc._home_both_step(HOME, HOME, HOME + 0.025, HOME, traj, 1.0)[0] == "reached"
    assert mc._home_both_step(HOME, HOME, np.full(6, 0.5), HOME, traj, 1.0)[0] == "move"


def test_a_leader_without_a_home_is_left_alone(mc: ModuleType) -> None:
    far = np.full(6, 1.0)
    traj = mc._HomeTrajectory((far, np.full(6, 5.0)), (HOME, None), 1.0)
    outcome, follower, leader = mc._home_both_step(far, HOME, np.full(6, 5.0), None, traj, 0.0)
    assert outcome == "move"
    assert follower is not None and leader is None


def test_either_arm_running_out_of_time_ends_homing(mc: ModuleType) -> None:
    far = np.full(6, 1.0)
    traj = _traj(mc, HOME, far)
    assert mc._home_both_step(HOME, HOME, far, HOME, traj, mc.HOME_TIMEOUT_S + 1)[0] == "timeout"


def test_homing_holds_while_neither_arm_can_be_read(mc: ModuleType) -> None:
    traj = _traj(mc, np.full(6, 1.0), np.full(6, 1.0))
    assert mc._home_both_step(None, HOME, None, HOME, traj, 1.0)[0] == "hold"


def test_homing_frames_drive_the_joints_and_keep_gravity_and_the_gripper(mc: ModuleType) -> None:
    commands = {"shoulder_pan": (0.0, 0.0, 10.0, 0.0, 1.5), "gripper": (0.0, 0.0, 30.0, 0.0, 0.2)}
    frames = mc._homing_frames(commands, {"shoulder_pan": 45.0})
    assert frames["shoulder_pan"] == (*mc.LEADER_HOME_GAINS["shoulder_pan"], 45.0, 0.0, 1.5)
    assert frames["gripper"] == commands["gripper"]
    assert mc._homing_frames(commands, None) is commands


def test_the_leader_homing_wrapper_rewrites_frames_only_while_a_target_is_set(mc: ModuleType) -> None:
    sent = []

    class _Bus:
        def sync_write_metal(self, commands: dict) -> None:
            sent.append(commands)

    class _Config:
        use_velocity_feedforward = True

    class _Device:
        bus = _Bus()
        config = _Config()

    device = _Device()
    homing = mc._LeaderHoming(device)
    commands = {"shoulder_pan": (0.0, 0.0, 10.0, 0.0, 1.5)}
    device.bus.sync_write_metal(commands)
    homing.set_target(np.radians([45.0, 0, 0, 0, 0, 0]))
    assert device.config.use_velocity_feedforward is False  # weightless mode's friction assist is off while homing
    device.bus.sync_write_metal(commands)
    homing.set_target(None)
    assert device.config.use_velocity_feedforward is True
    device.bus.sync_write_metal(commands)
    assert sent[0] == commands and sent[2] == commands
    assert sent[1]["shoulder_pan"][:3] == (*mc.LEADER_HOME_GAINS["shoulder_pan"], pytest.approx(45.0))


def test_the_leader_counts_as_home_within_its_looser_bound(mc: ModuleType) -> None:
    traj = _traj(mc, np.full(6, 1.0), np.full(6, 1.0))
    assert mc._home_both_step(HOME, HOME, HOME + 0.025, HOME, traj, 1.0)[0] == "reached"
    assert mc._home_both_step(HOME + 0.04, HOME, HOME, HOME, traj, 1.0)[0] == "move"
