"""The data page's actions: what they do to the folders, and that they refuse while a recorder writes into one."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import dream
import pytest
from conftest import write_episodes

from dream_hackathon import data_edit, dataset, layouts, station_lock


@pytest.fixture(autouse=True)
def isolated(tmp_path: pathlib.Path, monkeypatch) -> None:
    """No real station lock, and the edit log inside the test's folder."""
    monkeypatch.setattr(station_lock, "LOCK_PATH", tmp_path / "arms.lock")
    monkeypatch.setattr(data_edit, "LOG_PATH", tmp_path / "edits.log")


def check(folder: pathlib.Path) -> None:
    report = dream.datasets.validate_dataset(folder, layouts.embodiment("yam"))
    assert report.ok, report.error


def make(tmp_path: pathlib.Path, **datasets: list[str]) -> pathlib.Path:
    """A data root with one dataset per keyword: its tasks, a 55-frame episode for each entry, in turn."""
    root = tmp_path / "data"
    for name, tasks in datasets.items():
        write_episodes(root / name, "yam", episodes=len(tasks), frames=55, tasks=tasks)
    return root


def tasks_in(folder: pathlib.Path) -> list[str]:
    return ["; ".join(e.tasks) for e in dataset.list_episodes(folder)]


def test_deleting_episodes_renumbers_and_says_what_is_left(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, cubes=["a", "b", "c", "d"])
    message = data_edit.delete_episodes(root, "cubes", [1, 2])
    assert message == "Deleted 2 episodes of cubes; 2 left, numbered 0 to 1."
    assert tasks_in(root / "cubes") == ["a", "d"]
    check(root / "cubes")
    assert "delete-episodes" in (tmp_path / "edits.log").read_text()


def test_deleting_needs_real_episodes_and_a_real_dataset(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, cubes=["a", "b"])
    for episodes in ([], "1", [True], [-1], None):
        with pytest.raises(data_edit.EditError, match="at least one"):
            data_edit.delete_episodes(root, "cubes", episodes)
    with pytest.raises(data_edit.EditError, match="no episode 9"):
        data_edit.delete_episodes(root, "cubes", [9])
    for name in ("nope", "../data", "", ".", "cubes/meta"):
        with pytest.raises(data_edit.EditError, match="no dataset"):
            data_edit.delete_episodes(root, name, [0])
    assert len(dataset.list_episodes(root / "cubes")) == 2


def test_renaming_a_task_reaches_every_dataset_that_has_it(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, one=["a", "b"], two=["a", "a"], three=["c"])
    message = data_edit.rename_task(root, "a", "lift the cube")
    assert message == "Renamed the task to “lift the cube” in 2 datasets (3 episodes)."
    assert tasks_in(root / "one") == ["lift the cube", "b"]
    assert tasks_in(root / "two") == ["lift the cube"] * 2
    assert tasks_in(root / "three") == ["c"]
    for name in ("one", "two", "three"):
        check(root / name)


def test_renaming_a_task_that_is_nowhere_or_to_nothing_is_refused(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, one=["a"])
    with pytest.raises(data_edit.EditError, match="no dataset has episodes of"):
        data_edit.rename_task(root, "zzz", "b")
    with pytest.raises(data_edit.EditError, match="needs a text"):
        data_edit.rename_task(root, "a", "  ")


def test_renaming_some_episodes_of_one_dataset_only(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, one=["a", "a", "a"], two=["a"])
    data_edit.rename_episodes_task(root, "one", [0, 2], "b")
    assert tasks_in(root / "one") == ["b", "a", "b"]
    assert tasks_in(root / "two") == ["a"]
    check(root / "one")


def test_deleting_a_task_removes_its_episodes_everywhere_and_the_row(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, one=["a", "b", "a", "c"], two=["b"], three=["c", "c"])
    message = data_edit.delete_task(root, "a")
    assert message == "Deleted the task “a” and 2 episodes in 1 dataset."
    assert tasks_in(root / "one") == ["b", "c"]
    rows = [json.loads(line) for line in (root / "one" / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert rows == [{"task_index": 0, "task": "b"}, {"task_index": 1, "task": "c"}]
    for name in ("one", "two", "three"):
        check(root / name)
    # A task listed in a dataset but used by none of its episodes is forgotten too.
    message = data_edit.delete_task(root, "c")
    assert message == "Deleted the task “c” and 3 episodes in 2 datasets."
    assert tasks_in(root / "one") == ["b"] and tasks_in(root / "three") == []


def test_deleting_a_task_from_one_dataset_leaves_the_others(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, one=["a", "b"], two=["a", "b"])
    data_edit.delete_task(root, "a", "one")
    assert tasks_in(root / "one") == ["b"] and tasks_in(root / "two") == ["a", "b"]
    check(root / "one")
    with pytest.raises(data_edit.EditError, match="no dataset has the task"):
        data_edit.delete_task(root, "a", "one")


def test_an_unused_task_is_forgotten_without_touching_an_episode(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, one=["a", "b"])
    dataset.drop_episodes(root / "one", {0})  # a is listed and unused now
    message = data_edit.delete_task(root, "a", "one")
    assert message == "Deleted the task “a”."
    assert tasks_in(root / "one") == ["b"]
    check(root / "one")


def test_renaming_a_folder_and_its_eef_copy(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, cubes=["a"], other=["a"])
    shutil.copytree(root / "cubes", root / "cubes-eef")
    message = data_edit.rename_dataset(root, "cubes", "blocks", with_eef=True)
    assert message == "Renamed the folder cubes to blocks and cubes-eef to blocks-eef."
    assert sorted(p.name for p in root.iterdir()) == ["blocks", "blocks-eef", "other"]
    check(root / "blocks")
    data_edit.rename_dataset(root, "blocks", "cubes", with_eef=False)  # the copy keeps its name
    assert sorted(p.name for p in root.iterdir()) == ["blocks-eef", "cubes", "other"]


@pytest.mark.parametrize("bad", ["", "my task", "../x", "a/b", ".hidden", "-x", "x" * 101, "café"])
def test_a_bad_folder_name_is_refused(tmp_path: pathlib.Path, bad: str) -> None:
    root = make(tmp_path, cubes=["a"])
    with pytest.raises(data_edit.EditError, match="use letters"):
        data_edit.rename_dataset(root, "cubes", bad)
    assert [p.name for p in root.iterdir()] == ["cubes"]


def test_a_taken_or_unchanged_folder_name_is_refused_and_nothing_moves(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, cubes=["a"], other=["a"])
    shutil.copytree(root / "cubes", root / "cubes-eef")
    (root / "scratch").mkdir()
    with pytest.raises(data_edit.EditError, match="already exists"):
        data_edit.rename_dataset(root, "cubes", "other")
    with pytest.raises(data_edit.EditError, match="already exists"):
        data_edit.rename_dataset(root, "cubes", "scratch")
    with pytest.raises(data_edit.EditError, match="name already"):
        data_edit.rename_dataset(root, "cubes", "cubes")
    (root / "new-eef").mkdir()  # the copy cannot take its new name, so neither folder moves
    with pytest.raises(data_edit.EditError, match="new-eef already exists"):
        data_edit.rename_dataset(root, "cubes", "new", with_eef=True)
    assert sorted(p.name for p in root.iterdir()) == ["cubes", "cubes-eef", "new-eef", "other", "scratch"]


ACTIONS = {
    "delete episodes": lambda root: data_edit.delete_episodes(root, "busy", [0]),
    "rename episodes": lambda root: data_edit.rename_episodes_task(root, "busy", [0], "z"),
    "rename task": lambda root: data_edit.rename_task(root, "a", "z"),
    "delete task": lambda root: data_edit.delete_task(root, "a"),
    "delete task in one": lambda root: data_edit.delete_task(root, "a", "busy"),
    "rename folder": lambda root: data_edit.rename_dataset(root, "busy", "z"),
}


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_every_action_is_refused_while_a_recorder_writes_into_a_folder_it_would_touch(
    tmp_path: pathlib.Path, monkeypatch, action: str
) -> None:
    root = make(tmp_path, busy=["a", "a"], quiet=["a", "b"])
    before = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(data_edit, "recordings", lambda: {(root / "busy").resolve(): "mirror_collect.py (pid 7)"})
    with pytest.raises(data_edit.EditError, match="busy is being recorded into by mirror_collect.py \\(pid 7\\)"):
        ACTIONS[action](root)
    assert {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == before


def test_a_quiet_dataset_can_still_be_edited_while_another_records(tmp_path: pathlib.Path, monkeypatch) -> None:
    root = make(tmp_path, busy=["a"], quiet=["a", "b"])
    monkeypatch.setattr(data_edit, "recordings", lambda: {(root / "busy").resolve(): "x (pid 1)"})
    data_edit.delete_episodes(root, "quiet", [1])
    assert tasks_in(root / "quiet") == ["a"]


def test_the_station_lock_names_the_folder_a_collector_records_into(tmp_path: pathlib.Path, monkeypatch) -> None:
    folder = tmp_path / "data" / "cubes"
    monkeypatch.setattr(
        station_lock, "holder", lambda path=None: {"pid": 5, "holder": "dhk-collect", "dataset": str(folder)}
    )
    assert data_edit.recordings()[folder.resolve()] == "dhk-collect (pid 5)"


@pytest.fixture
def fake_recorder(tmp_path: pathlib.Path):
    """Starts a program whose arguments look like a mirror_collect.py run, and stops it after the test."""
    started: list[subprocess.Popen] = []

    def start(*arguments: str) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", "scripts/mirror_collect.py", *arguments],
            cwd=tmp_path,
        )
        started.append(process)
        time.sleep(0.3)
        return process

    yield start
    for process in started:
        process.kill()
        process.wait()


def test_a_running_mirror_collect_is_found_by_its_dataset_argument(tmp_path: pathlib.Path, fake_recorder) -> None:
    root = make(tmp_path, cubes=["a"])
    process = fake_recorder("--dataset", "data/cubes", "--task", "a")  # relative to the program's own folder
    assert data_edit.recordings()[(root / "cubes").resolve()] == f"mirror_collect.py (pid {process.pid})"
    with pytest.raises(data_edit.EditError, match="cubes is being recorded into by mirror_collect.py"):
        data_edit.delete_episodes(root, "cubes", [0])


def test_a_mirror_collect_given_a_data_root_writes_into_its_newest_dataset(
    tmp_path: pathlib.Path, fake_recorder
) -> None:
    root = make(tmp_path, older=["a"], newer=["a"])
    stamp = time.time()
    for name, shift in (("older", -100), ("newer", 0)):
        for file in (root / name / "meta").iterdir():
            os.utime(file, (stamp + shift, stamp + shift))
    fake_recorder("--data-root=data", "--task", "a")
    busy = data_edit.recordings()
    assert (root / "newer").resolve() in busy and (root / "older").resolve() not in busy


def test_text_that_only_mentions_a_recorder_is_not_one(tmp_path: pathlib.Path) -> None:
    root = make(tmp_path, cubes=["a"])
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            f"uv run python scripts/mirror_collect.py --dataset {root / 'cubes'}",
        ]
    )
    try:
        time.sleep(0.3)
        assert (root / "cubes").resolve() not in data_edit.recordings()
    finally:
        process.kill()
        process.wait()
