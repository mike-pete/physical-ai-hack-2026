"""Renaming and forgetting a task: what the files say afterwards agrees with itself and passes Dream's local checks."""

import json
import pathlib

import dream
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import write_episodes

from dream_hackathon import dataset, layouts


def check(root: pathlib.Path) -> None:
    report = dream.datasets.validate_dataset(root, layouts.embodiment("yam"))
    assert report.ok, report.error


def tasks_of(root: pathlib.Path) -> list[tuple[int, str]]:
    rows = [json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()]
    return [(row["task_index"], row["task"]) for row in rows]


def episode_tasks(root: pathlib.Path) -> list[tuple[str, ...]]:
    return [episode.tasks for episode in dataset.list_episodes(root)]


def frame_indices(root: pathlib.Path, episode: int) -> set[int]:
    table = pq.read_table(root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
    return set(table["task_index"].to_numpy().tolist())


def stats_index(root: pathlib.Path, episode: int) -> float:
    rows = [json.loads(line) for line in (root / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    return rows[episode]["stats"]["task_index"]["mean"][0]


def total_tasks(root: pathlib.Path) -> int:
    return json.loads((root / "meta" / "info.json").read_text())["total_tasks"]


def test_renaming_every_episode_of_a_task_renames_its_row_in_place(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=3, frames=55, tasks=["pick up the cube"])
    dataset.retask_episodes(root, {0, 1, 2}, "  lift the cube ")
    assert tasks_of(root) == [(0, "lift the cube")]
    assert episode_tasks(root) == [("lift the cube",)] * 3
    assert total_tasks(root) == 1
    check(root)


def test_renaming_some_episodes_splits_off_a_new_task_with_its_own_index(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=3, frames=55, tasks=["pick up the cube"])
    dataset.retask_episodes(root, {1}, "pick up the red cube")
    assert tasks_of(root) == [(0, "pick up the cube"), (1, "pick up the red cube")]
    assert episode_tasks(root) == [("pick up the cube",), ("pick up the red cube",), ("pick up the cube",)]
    assert [frame_indices(root, e) for e in range(3)] == [{0}, {1}, {0}]
    assert [stats_index(root, e) for e in range(3)] == [0.0, 1.0, 0.0]
    assert total_tasks(root) == 2
    check(root)


def test_renaming_to_an_existing_task_joins_it_and_leaves_the_old_row_unused(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=4, frames=55, tasks=["pick up the cube", "stack the cups"])
    dataset.retask_episodes(root, {1, 3}, "pick up the cube")
    assert episode_tasks(root) == [("pick up the cube",)] * 4
    assert [frame_indices(root, e) for e in range(4)] == [{0}] * 4
    assert tasks_of(root) == [(0, "pick up the cube"), (1, "stack the cups")]  # listed, unused, as after a drop
    check(root)


def test_forgetting_a_task_nothing_uses_renumbers_the_ones_left(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=4, frames=55, tasks=["a", "b", "c"])
    dataset.drop_episodes(root, {0, 3})  # leaves episodes of b and c; a is unused at index 0
    assert tasks_of(root) == [(0, "a"), (1, "b"), (2, "c")]
    dataset.forget_task(root, "a")
    assert tasks_of(root) == [(0, "b"), (1, "c")]
    assert episode_tasks(root) == [("b",), ("c",)]
    assert [frame_indices(root, e) for e in range(2)] == [{0}, {1}]
    assert [stats_index(root, e) for e in range(2)] == [0.0, 1.0]
    assert total_tasks(root) == 2
    check(root)


def test_forgetting_the_last_task_in_the_list_changes_no_episode(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=55, tasks=["a", "b"])
    dataset.drop_episodes(root, {1})
    dataset.forget_task(root, "b")
    assert tasks_of(root) == [(0, "a")]
    assert frame_indices(root, 0) == {0}
    check(root)


def test_a_task_in_use_is_not_forgotten_and_an_unlisted_one_is_a_no_op(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=55, tasks=["a"])
    with pytest.raises(dataset.DatasetError, match="still use"):
        dataset.forget_task(root, "a")
    dataset.forget_task(root, "never listed")
    assert tasks_of(root) == [(0, "a")]


def test_retasking_refuses_what_it_cannot_do_and_changes_nothing(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=55, tasks=["a"])
    before = (root / "meta" / "tasks.jsonl").read_text()
    with pytest.raises(dataset.DatasetError, match="no episode 5"):
        dataset.retask_episodes(root, {0, 5}, "b")
    with pytest.raises(dataset.DatasetError, match="needs a text"):
        dataset.retask_episodes(root, {0}, "   ")
    assert (root / "meta" / "tasks.jsonl").read_text() == before
    # An episode with two tasks cannot be given one.
    lines = (root / "meta" / "episodes.jsonl").read_text().replace('["a"]', '["a", "x"]', 1)
    (root / "meta" / "episodes.jsonl").write_text(lines)
    with pytest.raises(dataset.DatasetError, match="2 tasks"):
        dataset.retask_episodes(root, {0}, "b")


def test_a_stale_task_index_in_an_episode_is_put_right_when_it_is_renamed(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=55, tasks=["a"])
    path = root / "data" / "chunk-000" / "episode_000001.parquet"
    table = pq.read_table(path)
    stale = table.set_column(
        table.schema.get_field_index("task_index"), "task_index", pa.array(np.full(table.num_rows, 7, dtype=np.int64))
    )
    pq.write_table(stale, path)
    dataset.retask_episodes(root, {0, 1}, "b")
    assert [frame_indices(root, e) for e in range(2)] == [{0}, {0}]
    check(root)
