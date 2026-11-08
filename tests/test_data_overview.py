"""The data overview: what `scan()` finds in a folder of datasets, the page it renders, and `dhk-dataset overview`."""

import http.client
import json
import pathlib
import threading
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import write_episodes

from dream_hackathon import data_overview, dataset_cli

ARM = 7


def make(
    root: pathlib.Path, episodes: int = 3, frames: int = 60, tasks: tuple[str, ...] = ("pick up the cube",)
) -> None:
    """A dataset whose channels are named per arm, as the bimanual datasets name them."""
    write_episodes(root, "yam", episodes=episodes, frames=frames, tasks=list(tasks))
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    names = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(ARM)]
    info["features"]["action"]["names"] = info["features"]["observation.state"]["names"] = names
    info_path.write_text(json.dumps(info))


def edit_episode(root: pathlib.Path, episode: int, change: Callable[[np.ndarray, np.ndarray], None]) -> None:
    """Rewrites one episode's action and state arrays in place with `change(action, state)`."""
    path = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path)
    action, state = (np.stack(table[name].to_pylist()) for name in ("action", "observation.state"))
    change(action, state)
    for name, array in (("action", action), ("observation.state", state)):
        table = table.set_column(
            table.schema.get_field_index(name), name, pa.array(array.tolist(), table.schema.field(name).type)
        )
    pq.write_table(table, path)


def notes(row: data_overview.DatasetRow, level: str) -> list[str]:
    return [note.text for note in row.notes if note.level == level]


def test_span() -> None:
    assert data_overview.span([3, 1, 2, 5, 5]) == "1–3, 5"
    assert data_overview.span([]) == ""


def test_a_clean_dataset_counts_and_has_nothing_to_flag(tmp_path: pathlib.Path) -> None:
    make(tmp_path / "cubes", episodes=3, frames=60, tasks=("pick up the cube", "stack them"))
    (tmp_path / "scratch").mkdir()  # a folder that is not a dataset is listed, not scanned
    overview = data_overview.scan(tmp_path)
    (row,) = overview.datasets
    assert overview.skipped == ["scratch"]
    assert (row.name, row.robot_type, row.fps, len(row.episodes), row.frames) == ("cubes", "yam", 30, 3, 180)
    assert row.tasks == ["pick up the cube", "stack them"]
    assert row.size > row.video_size > 0 and row.modified is not None
    assert row.status == "ok" and row.notes == []
    assert overview.episodes == 3 and overview.seconds == 6.0 and overview.overlaps == []


def test_files_that_disagree_with_the_metadata_are_problems(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "cubes"
    make(root, episodes=3)
    next((root / "videos").rglob("episode_000001.mp4")).unlink()
    info = json.loads((root / "meta" / "info.json").read_text())
    info["total_episodes"] = 5
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (row,) = data_overview.scan(tmp_path).datasets
    problems = notes(row, "problem")
    assert row.status == "problem"
    assert any("is missing in episode 1" in text for text in problems)
    assert any("counts 5 episodes" in text for text in problems)


def test_an_unreadable_data_file_is_a_problem_not_a_crash(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "cubes"
    make(root, episodes=2)
    (root / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"not a parquet file")
    (row,) = data_overview.scan(tmp_path).datasets
    assert any("cannot be read" in text and "episode 0" in text for text in notes(row, "problem"))


def test_a_still_arm_and_a_mirror_recording_are_noticed(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "cubes"
    make(root, episodes=3)
    edit_episode(root, 0, lambda action, state: (action[:, :ARM].fill(0.5), state[:, :ARM].fill(0.5)))
    edit_episode(root, 1, lambda action, state: action[:, ARM:].__setitem__(slice(None), action[:, :ARM]))
    (row,) = data_overview.scan(tmp_path).datasets
    assert [episode.still for episode in row.episodes] == [("left",), (), ()]
    assert [episode.mirrored for episode in row.episodes] == [False, True, False]
    assert any("left arm does not move" in text and "episode 0" in text for text in notes(row, "check"))
    assert any("identical" in text and "episode 1" in text for text in notes(row, "note"))
    assert row.status == "check"


def test_an_episode_shorter_than_dream_trains_on_is_a_check(tmp_path: pathlib.Path) -> None:
    # The writer refuses a too-short episode, so record a normal one and shorten what episodes.jsonl says.
    root = tmp_path / "tiny"
    make(root, episodes=1, frames=60)
    episodes = root / "meta" / "episodes.jsonl"
    episodes.write_text(episodes.read_text().replace('"length": 60', '"length": 10'))
    (row,) = data_overview.scan(tmp_path).datasets
    assert any("shorter than" in text for text in notes(row, "check"))


def test_the_same_recording_in_two_datasets_is_found_once_and_counted(tmp_path: pathlib.Path) -> None:
    make(tmp_path / "a", episodes=3)
    make(tmp_path / "b", episodes=3)  # the same seed: the same actions
    make(tmp_path / "c", episodes=2, frames=70)
    overview = data_overview.scan(tmp_path)
    (overlap,) = overview.overlaps
    assert (overlap.a, overlap.a_episodes, overlap.b, overlap.b_episodes) == ("a", [0, 1, 2], "b", [0, 1, 2])
    assert overlap.kind == "same recording" and overlap.size > 0
    assert 0 < overview.duplicate_bytes <= overlap.size * 1.01
    a, b, c = overview.datasets
    assert any("Same recordings as b" in text for text in notes(a, "note"))
    assert any("Same recordings as a" in text for text in notes(b, "note"))
    assert not any("Same recordings" in text for text in notes(c, "note"))


def test_an_eef_copy_is_built_from_its_source_or_out_of_date(tmp_path: pathlib.Path) -> None:
    make(tmp_path / "task", episodes=3)
    make(tmp_path / "task-eef", episodes=3)
    make(tmp_path / "other", episodes=3)
    make(tmp_path / "other-eef", episodes=2)
    rows = {row.name: row for row in data_overview.scan(tmp_path).datasets}
    assert any("Built from task" in text for text in notes(rows["task-eef"], "note"))
    assert any("Out of date" in text and "--replace" in text for text in notes(rows["other-eef"], "check"))


def test_the_page_escapes_what_the_data_says(tmp_path: pathlib.Path) -> None:
    make(tmp_path / "cubes", episodes=2, tasks=("<script>alert(1)</script>",))
    page = data_overview.render_html(data_overview.scan(tmp_path))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "cubes" in page and "Episode table" in page


def test_the_page_for_an_empty_folder_says_so(tmp_path: pathlib.Path) -> None:
    page = data_overview.render_html(data_overview.scan(tmp_path))
    assert "No datasets found" in page


def test_the_cli_writes_the_page(tmp_path: pathlib.Path, capsys) -> None:
    make(tmp_path / "data" / "cubes", episodes=2)
    out = tmp_path / "out" / "page.html"
    dataset_cli.main(["overview", str(tmp_path / "data"), "--out", str(out)])
    assert "1 datasets, 2 episodes" in capsys.readouterr().out
    assert "Hackathon data" in out.read_text()


def test_the_server_answers_get_on_loopback_and_refuses_other_hosts(tmp_path: pathlib.Path) -> None:
    make(tmp_path / "cubes", episodes=2)
    server = data_overview.make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for host, status in ((f"localhost:{server.server_port}", 200), ("evil.example", 403)):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            connection.request("GET", "/", headers={"Host": host})
            response = connection.getresponse()
            body = response.read().decode()
            assert response.status == status
            assert ("cubes" in body) == (status == 200)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request("POST", "/")
        assert connection.getresponse().status == 501
    finally:
        server.shutdown()
        server.server_close()
