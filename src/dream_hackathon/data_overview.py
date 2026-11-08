"""A page about every LeRobot v2.1 dataset under one folder (`~/hackathon_data`): what is there, its videos, and the
edits that tidy it.

```
uv run dhk-dataset overview --port 9000 --config configs/station.yaml
```

The station has no screen: forward the port with `ssh -L 9000:localhost:9000 <user>@<station>` and open
http://localhost:9000. The server listens on this computer only.

The page lists every task and, under it, the dataset folders that hold its episodes, one block per episode to one time
scale; click a block or a folder to watch that episode's three cameras. Per dataset it shows the robot, cameras, size,
when it was last written, and what looks wrong or is worth knowing (see `scan()`); across datasets it shows episodes
that are stored more than once. Its actions change recorded data: rename a task (in every dataset that has it, or for
some episodes), delete episodes, delete a task with its episodes, and rename a dataset folder. Each one is done by
`data_edit`, refuses while a recorder writes into a folder it would touch, and says what it did.

An action is taken only from the page itself (a custom header, an `Origin` that names this address and a `Host` that
is loopback), so no other website open in the same browser can send one through the SSH tunnel.

@see scan(), overview_json(), serve(), data_edit
"""

import collections
import contextlib
import dataclasses
import datetime
import hashlib
import http.server
import itertools
import json
import os
import pathlib
import re
import time
import urllib.parse
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dream_hackathon import data_edit, dataset

DEFAULT_ROOT = "~/hackathon_data"
## `dhk-collect`'s dashboard has 8080.
DEFAULT_PORT = 8081
## An arm whose action and state never move more than this over an episode is reported as not moving.
_STILL = 1e-6
## Task colours are the first seven categorical slots; later tasks share the grey "other".
_TASK_SLOTS = 7
_LEVELS = ("problem", "check", "note")


@dataclasses.dataclass
class Note:
    """One thing to know about a dataset.

    Attributes:
    - level: `"problem"` (the files disagree: fix before training), `"check"` (probably unintended) or `"note"`.
    - text: What it is, in a sentence.
    """

    level: str
    text: str


@dataclasses.dataclass
class EpisodeRow:
    """One episode of a dataset.

    Attributes:
    - index: The episode number.
    - frames: The length `episodes.jsonl` states.
    - task: The task text (several tasks are joined by `"; "`).
    - size: Bytes of the episode's parquet and videos.
    - modified: When its parquet was last written (epoch seconds), `None` when the file is missing.
    - action_hash: A fingerprint of its whole action array, `None` when the parquet cannot be read.
    - side_hashes: One fingerprint per arm (`"left"`, `"right"`).
    - still: The arms (`"left"`, `"right"`) that never move in it.
    - mirrored: Both arms' actions are identical.
    - missing: The cameras (`"scene"`, ...) whose video file for this episode is not there.
    """

    index: int
    frames: int
    task: str
    size: int
    modified: float | None
    action_hash: str | None = None
    side_hashes: dict[str, str] = dataclasses.field(default_factory=dict)
    still: tuple[str, ...] = ()
    mirrored: bool = False
    missing: tuple[str, ...] = ()


@dataclasses.dataclass
class DatasetRow:
    """One dataset folder.

    Attributes:
    - name: The folder's name.
    - path: The folder.
    - robot_type: As `info.json` records it.
    - fps: Frames per second.
    - cameras: Each camera's video key, width and height.
    - episodes: Its episodes, in order.
    - size: Bytes of everything in the folder.
    - video_size: Bytes of `videos/`.
    - modified: When the newest episode was written (epoch seconds).
    - notes: What looks wrong or is worth knowing.
    - listed_tasks: The tasks `meta/tasks.jsonl` lists, used or not.
    - chunk_size, video_path: How `info.json` places an episode's files.
    """

    name: str
    path: pathlib.Path
    robot_type: str
    fps: int
    cameras: list[tuple[str, int, int]]
    episodes: list[EpisodeRow]
    size: int
    video_size: int
    modified: float | None
    notes: list[Note]
    listed_tasks: list[str] = dataclasses.field(default_factory=list)
    chunk_size: int = dataset.CHUNKS_SIZE
    video_path: str = dataset.VIDEO_PATH

    @property
    def frames(self) -> int:
        """Frames over all episodes."""
        return sum(episode.frames for episode in self.episodes)

    @property
    def seconds(self) -> float:
        """Recorded time."""
        return self.frames / self.fps if self.fps else 0.0

    @property
    def tasks(self) -> list[str]:
        """The tasks its episodes use, in order of first use."""
        return list(dict.fromkeys(episode.task for episode in self.episodes))

    @property
    def status(self) -> str:
        """`"problem"` or `"check"` when a note says so, else `"ok"`."""
        levels = {note.level for note in self.notes}
        return "problem" if "problem" in levels else "check" if "check" in levels else "ok"


@dataclasses.dataclass
class Overlap:
    """Episodes of two datasets that carry the same recording (or the same arm's actions).

    Attributes:
    - a, b: The two datasets' names.
    - a_episodes, b_episodes: Their episode numbers that match.
    - kind: `"same recording"` (identical actions) or `"same <arm> arm"` (one arm's actions only).
    - size: Bytes of the smaller side for `"same recording"`; 0 for a partial match, whose videos can differ.
    """

    a: str
    a_episodes: list[int]
    b: str
    b_episodes: list[int]
    kind: str
    size: int


@dataclasses.dataclass
class Overview:
    """Everything the page shows.

    Attributes:
    - root: The scanned folder.
    - scanned_at: When (epoch seconds).
    - datasets: One per dataset folder, by name.
    - overlaps: Episodes shared between datasets.
    - duplicate_bytes: Bytes that the repeated recordings take beyond one copy of each.
    - skipped: Folders without `meta/info.json`, not scanned.
    """

    root: pathlib.Path
    scanned_at: float
    datasets: list[DatasetRow]
    overlaps: list[Overlap]
    duplicate_bytes: int
    skipped: list[str]

    @property
    def episodes(self) -> int:
        """Episodes over all datasets."""
        return sum(len(row.episodes) for row in self.datasets)

    @property
    def seconds(self) -> float:
        """Recorded time over all datasets."""
        return sum(row.seconds for row in self.datasets)

    @property
    def size(self) -> int:
        """Bytes over all datasets."""
        return sum(row.size for row in self.datasets)


def span(numbers: Iterable[int]) -> str:
    """Numbers as ranges: `[1, 2, 3, 5]` is `"1–3, 5"`."""
    parts = []
    for _, group in itertools.groupby(enumerate(sorted(set(numbers))), key=lambda pair: pair[1] - pair[0]):
        run = [number for _, number in group]
        parts.append(str(run[0]) if len(run) == 1 else f"{run[0]}–{run[-1]}")
    return ", ".join(parts)


def _where(indices: list[int], total: int) -> str:
    if len(indices) == total:
        return f"all {total} episodes" if total > 1 else "its only episode"
    return f"episode{'s' if len(indices) > 1 else ''} {span(indices)}"


def _folder_size(path: pathlib.Path) -> int:
    return sum(
        os.path.getsize(os.path.join(directory, name))
        for directory, _, names in os.walk(path)
        for name in names
        if os.path.isfile(os.path.join(directory, name))
    )


def _read_jsonl(path: pathlib.Path, notes: list[Note]) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        notes.append(Note("problem", f"meta/{path.name} cannot be read ({error})"))
        return []


def _arm_channels(names: object) -> dict[str, list[int]]:
    """Channel indices per arm from names such as `left_joint_0`; empty unless every name has an arm prefix."""
    if not isinstance(names, list) or not all(isinstance(name, str) and "_" in name for name in names):
        return {}
    arms: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        arms.setdefault(name.split("_", 1)[0], []).append(index)
    return arms if len(arms) >= 2 else {}


def _matrix(table: pa.Table, column: str) -> npt.NDArray[np.float32]:
    """A list column as a rows x width float32 array."""
    chunked = table[column]
    rows = len(chunked)
    if rows == 0:
        return np.zeros((0, 0), np.float32)
    return chunked.combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(rows, -1)


def _fingerprint(array: npt.NDArray[np.float32]) -> str:
    return hashlib.blake2b(np.ascontiguousarray(array).tobytes(), digest_size=8).hexdigest()


def _analyse(
    parquet: pathlib.Path, arms: dict[str, list[int]], state_arms: dict[str, list[int]], names: object
) -> tuple[int, str, dict[str, str], tuple[str, ...], bool]:
    """Reads one episode's parquet: its row count, the fingerprints of its actions, which arms stay still, mirrored."""
    table = pq.read_table(parquet, columns=["action", "observation.state"])
    action, state = _matrix(table, "action"), _matrix(table, "observation.state")
    whole = _fingerprint(action)
    if len(action) == 0:
        return table.num_rows, whole, {}, (), False
    sides = {arm: _fingerprint(action[:, channels]) for arm, channels in arms.items()}
    still = tuple(
        arm
        for arm, channels in arms.items()
        if np.ptp(action[:, channels], axis=0).max() <= _STILL
        and (arm not in state_arms or np.ptp(state[:, state_arms[arm]], axis=0).max() <= _STILL)
    )
    mirrored = False
    if len(arms) == 2 and isinstance(names, list):
        first, second = arms.values()
        suffix = [[names[i].split("_", 1)[1] for i in channels] for channels in (first, second)]
        mirrored = suffix[0] == suffix[1] and bool(np.array_equal(action[:, first], action[:, second]))
    return table.num_rows, whole, sides, still, mirrored


def _listed_tasks(path: pathlib.Path) -> list[str]:
    """The tasks `meta/tasks.jsonl` lists; none when it cannot be read (the notes already say so)."""
    try:
        return [str(row.get("task")) for row in dataset.task_rows(path)]
    except (OSError, ValueError, AttributeError):
        return []


def _scan_dataset(path: pathlib.Path) -> DatasetRow:
    """One dataset folder: its numbers, and the notes that its files agreeing (or not) with each other produce."""
    meta = path / "meta"
    notes: list[Note] = []
    try:
        info = json.loads((meta / "info.json").read_text())
        fps, features = int(info["fps"]), info["features"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        notes.append(Note("problem", f"meta/info.json cannot be read ({error!r})"))
        return DatasetRow(path.name, path, "?", 0, [], [], _folder_size(path), 0, None, notes)
    video_keys = sorted(key for key, feature in features.items() if feature.get("dtype") == "video")
    cameras = [(key, int(features[key]["shape"][1]), int(features[key]["shape"][0])) for key in video_keys]
    names = features.get("action", {}).get("names")
    arms = _arm_channels(names)
    state_arms = _arm_channels(features.get("observation.state", {}).get("names"))
    chunk_size = int(info.get("chunks_size", dataset.CHUNKS_SIZE))
    data_path = info.get("data_path", dataset.DATA_PATH)
    video_path = info.get("video_path", dataset.VIDEO_PATH)

    episodes: list[EpisodeRow] = []
    problems: dict[str, list[int]] = collections.defaultdict(list)
    for entry in _read_jsonl(meta / "episodes.jsonl", notes):
        index, frames = int(entry["episode_index"]), int(entry["length"])
        chunk = index // chunk_size
        parquet = path / data_path.format(episode_chunk=chunk, episode_index=index)
        videos = {
            key: path / video_path.format(episode_chunk=chunk, video_key=key, episode_index=index)
            for key in video_keys
        }
        row = EpisodeRow(index, frames, "; ".join(entry.get("tasks") or []), 0, None)
        row.size = sum(file.stat().st_size for file in [parquet, *videos.values()] if file.exists())
        absent = []
        for key, video in videos.items():
            if not video.exists():
                camera = key.removeprefix("observation.images.")
                absent.append(camera)
                problems[f"video {camera} is missing"].append(index)
        row.missing = tuple(absent)
        if not parquet.exists():
            problems["data file is missing"].append(index)
        else:
            row.modified = parquet.stat().st_mtime
            try:
                rows, row.action_hash, row.side_hashes, row.still, row.mirrored = _analyse(
                    parquet, arms, state_arms, names
                )
            except (OSError, ValueError, KeyError, pa.ArrowException) as error:
                problems[f"data file cannot be read ({type(error).__name__})"].append(index)
            else:
                if rows != frames:
                    problems[f"data file holds {rows} rows, episodes.jsonl says {frames}"].append(index)
        episodes.append(row)
    for message, indices in problems.items():
        notes.append(Note("problem", f"{message.capitalize()} in {_where(indices, len(episodes))}"))
    _consistency_notes(path, info, episodes, notes)
    _motion_notes(path, episodes, arms, fps, notes)
    modified = max((row.modified for row in episodes if row.modified), default=None)
    return DatasetRow(
        name=path.name,
        path=path,
        robot_type=str(info.get("robot_type", "?")),
        fps=fps,
        cameras=cameras,
        episodes=episodes,
        size=_folder_size(path),
        video_size=_folder_size(path / "videos") if (path / "videos").is_dir() else 0,
        modified=modified,
        notes=notes,
        listed_tasks=_listed_tasks(path),
        chunk_size=chunk_size,
        video_path=video_path,
    )


def _consistency_notes(
    path: pathlib.Path, info: dict[str, Any], episodes: list[EpisodeRow], notes: list[Note]
) -> None:
    """Notes for `meta/` disagreeing with itself or with the files beside it."""
    if info.get("total_episodes") != len(episodes):
        notes.append(
            Note(
                "problem",
                f"info.json counts {info.get('total_episodes')} episodes, episodes.jsonl lists {len(episodes)}",
            )
        )
    frames = sum(row.frames for row in episodes)
    if info.get("total_frames") != frames:
        notes.append(
            Note("problem", f"info.json counts {info.get('total_frames')} frames, the episodes add up to {frames}")
        )
    if [row.index for row in episodes] != list(range(len(episodes))):
        notes.append(Note("problem", "Episode numbers are not 0, 1, 2, ... in order (Dream requires that)"))
    parquets = sum(1 for _ in (path / "data").rglob("*.parquet"))
    if parquets != len(episodes):
        notes.append(Note("check", f"data/ holds {parquets} parquet files for {len(episodes)} episodes"))
    stats = path / "meta" / "episodes_stats.jsonl"
    if stats.exists() and sum(1 for line in stats.read_text().splitlines() if line.strip()) != len(episodes):
        notes.append(Note("check", "episodes_stats.jsonl does not have one row per episode"))
    used = {row.task for row in episodes}
    listed = _read_jsonl(path / "meta" / "tasks.jsonl", notes)
    listed_tasks = {str(entry.get("task")) for entry in listed}
    for task in sorted(used - listed_tasks - {""}):
        notes.append(Note("problem", f"Episodes use the task “{task}”, which tasks.jsonl does not list"))
    for task in sorted(listed_tasks - used):
        notes.append(Note("note", f"tasks.jsonl lists the task “{task}”, which no episode uses"))
    for extra in sorted(
        child for child in path.iterdir() if child.is_dir() and child.name not in ("meta", "data", "videos")
    ):
        notes.append(
            Note("note", f"Extra folder {extra.name}/ ({_size(_folder_size(extra))}), not part of the dataset")
        )


def _motion_notes(
    path: pathlib.Path, episodes: list[EpisodeRow], arms: dict[str, list[int]], fps: int, notes: list[Note]
) -> None:
    """Notes for episodes whose arms do not move, whose arms act identically, or that Dream would reject."""
    total = len(episodes)
    for arm in arms:
        indices = [row.index for row in episodes if arm in row.still]
        if indices:
            notes.append(
                Note(
                    "check", f"The {arm} arm does not move (action and state are constant) in {_where(indices, total)}"
                )
            )
    mirrored = [row.index for row in episodes if row.mirrored]
    if mirrored:
        a, b = list(arms)[:2]
        notes.append(
            Note("note", f"The {a} and {b} actions are identical in {_where(mirrored, total)} (a mirror recording)")
        )
    shortest = dataset.min_frames(fps) if fps else 0
    short = [row.index for row in episodes if row.frames < shortest]
    if short:
        notes.append(
            Note(
                "check",
                f"{_where(short, total).capitalize()} shorter than the {dataset.MIN_EPISODE_S} s Dream trains on",
            )
        )


def _overlaps(datasets: list[DatasetRow]) -> tuple[list[Overlap], int]:
    """Episodes recorded once but stored in several datasets, and the bytes that the extra copies take."""
    found: list[Overlap] = []
    for a, b in itertools.combinations(datasets, 2):
        by_hash: dict[str, list[EpisodeRow]] = collections.defaultdict(list)
        for row in b.episodes:
            if row.action_hash:
                by_hash[row.action_hash].append(row)
        same = [(x, y) for x in a.episodes if x.action_hash for y in by_hash.get(x.action_hash, [])]
        if same:
            mine, theirs = {x.index: x.size for x, _ in same}, {y.index: y.size for _, y in same}
            found.append(
                Overlap(
                    a.name,
                    sorted(mine),
                    b.name,
                    sorted(theirs),
                    "same recording",
                    min(sum(mine.values()), sum(theirs.values())),
                )
            )
        pairs = {(x.index, y.index) for x, y in same}
        for arm in sorted({arm for row in a.episodes for arm in row.side_hashes}):
            side: dict[str, list[EpisodeRow]] = collections.defaultdict(list)
            for row in b.episodes:
                if arm in row.side_hashes and arm not in row.still:
                    side[row.side_hashes[arm]].append(row)
            match = [
                (x, y)
                for x in a.episodes
                if arm in x.side_hashes and arm not in x.still
                for y in side.get(x.side_hashes[arm], [])
                if (x.index, y.index) not in pairs
            ]
            if match:
                found.append(
                    Overlap(
                        a.name,
                        sorted({x.index for x, _ in match}),
                        b.name,
                        sorted({y.index for _, y in match}),
                        f"same {arm} arm",
                        0,
                    )
                )
    groups: dict[str, list[EpisodeRow]] = collections.defaultdict(list)
    for row in datasets:
        for episode in row.episodes:
            if episode.action_hash:
                groups[episode.action_hash].append(episode)
    extra = sum(sum(e.size for e in group) - max(e.size for e in group) for group in groups.values() if len(group) > 1)
    return found, extra


def _eef_notes(by_name: dict[str, DatasetRow]) -> None:
    """Notes on the `<name>-eef` datasets `tools/add_eef.py` writes beside a dataset: built from it, or out of date."""
    for name, row in by_name.items():
        source = by_name.get(name.removesuffix("-eef")) if name.endswith("-eef") else None
        if source is None:
            continue
        if [e.frames for e in row.episodes] == [e.frames for e in source.episodes]:
            row.notes.append(
                Note("note", f"Built from {source.name} by tools/add_eef.py (the same {len(row.episodes)} episodes)")
            )
        else:
            rebuild = f"tools/add_eef.py {source.path} --arm {source.robot_type} --replace"
            counts = f"{source.name} has {len(source.episodes)} episodes, {row.name} {len(row.episodes)}"
            row.notes.append(Note("check", f"Out of date ({counts}, or their lengths differ): rebuild with {rebuild}"))


def scan(root: pathlib.Path) -> Overview:
    """Reads every dataset folder directly under `root`; files are only read, never written.

    Per dataset it reports `Note`s: a `"problem"` when the files disagree (counts in `info.json`, missing or
    unreadable files, episode numbers out of order, a task `tasks.jsonl` lacks), a `"check"` for what is probably not
    intended (an arm that never moves, an episode shorter than Dream trains on), and a `"note"` for what is merely
    good to know (identical left and right actions from a mirror recording, extra folders, an unused task, episodes
    that another dataset holds too).

    @param root The folder that holds the dataset folders, e.g. `~/hackathon_data`.
    """
    children = sorted(child for child in root.iterdir() if child.is_dir())
    rows = [_scan_dataset(child) for child in children if (child / "meta" / "info.json").exists()]
    skipped = [child.name for child in children if not (child / "meta" / "info.json").exists()]
    overlaps, extra = _overlaps(rows)
    by_name = {row.name: row for row in rows}
    _eef_notes(by_name)
    for overlap in overlaps:
        for mine, theirs, here, there in (
            (overlap.a, overlap.b, overlap.a_episodes, overlap.b_episodes),
            (overlap.b, overlap.a, overlap.b_episodes, overlap.a_episodes),
        ):
            what = (
                "Same recordings as"
                if overlap.kind == "same recording"
                else f"Same {overlap.kind.split()[1]}-arm actions as"
            )
            by_name[mine].notes.append(
                Note("note", f"{what} {theirs}: episodes {span(here)} here, {span(there)} there")
            )
    return Overview(root, datetime.datetime.now().timestamp(), rows, overlaps, extra, skipped)


## ----- what the page loads: `GET /api/overview` -----


def _size(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if count >= 100 or unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return ""


def _clock(seconds: float) -> str:
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 60:.0f} min" if seconds >= 120 else f"{seconds:.0f} s"


def _when(epoch: float | None) -> str:
    if epoch is None:
        return "\u2014"
    moment = datetime.datetime.fromtimestamp(epoch)
    return f"{moment:%b} {moment.day}, {moment:%H:%M}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _short(path: pathlib.Path) -> str:
    """A path with the home folder as `~`."""
    try:
        return "~/" + path.relative_to(pathlib.Path.home()).as_posix()
    except ValueError:
        return str(path)


def _pills(row: DatasetRow, overview: Overview) -> list[dict[str, str]]:
    """The short tags under a dataset's name."""
    pills = [("warning", f"{arm} arm still") for arm in sorted({a for ep in row.episodes for a in ep.still})]
    if any(episode.mirrored for episode in row.episodes):
        pills.append(("info", "left = right (mirror recording)"))
    for overlap in overview.overlaps:
        if row.name in (overlap.a, overlap.b):
            other = overlap.b if row.name == overlap.a else overlap.a
            pills.append(("info", f"{overlap.kind} as {other}"))
    if row.name.endswith("-eef"):
        pills.append(("info", "built by add_eef"))
    if any(note.text.startswith("Extra folder") for note in row.notes):
        pills.append(("info", "backup folder inside"))
    return [{"kind": kind, "text": text} for kind, text in pills]


def _chip(row: DatasetRow) -> str:
    problems = sum(note.level == "problem" for note in row.notes)
    checks = sum(note.level == "check" for note in row.notes)
    if problems:
        return _plural(problems, "problem")
    return f"{checks} to check" if checks else "Consistent"


def overview_json(
    overview: Overview, recording: dict[pathlib.Path, str] | None = None, station_dir: str | None = None
) -> dict[str, Any]:
    """Everything the page draws, as plain JSON: the totals, each task in order of first use, each dataset with its
    episodes, what needs a look, and which datasets a recorder is writing into now.

    @param overview The scan.
    @param recording What `data_edit.recordings()` found.
    @param station_dir The `dataset_dir` of the station config, as written there; None when there is none.
    """
    recording = recording or {}
    slots: dict[str, int] = {}
    for row in overview.datasets:
        for task in row.tasks:
            slots.setdefault(task, len(slots) + 1 if len(slots) < _TASK_SLOTS else 0)
    twins: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for row in overview.datasets:
        for episode in row.episodes:
            if episode.action_hash:
                twins[episode.action_hash].append((row.name, episode.index))
    datasets = []
    attention: list[dict[str, str]] = []
    for row in overview.datasets:
        longest = max((episode.frames for episode in row.episodes), default=1) or 1
        counts = collections.Counter(episode.task for episode in row.episodes)
        episodes = [
            {
                "index": episode.index,
                "frames": episode.frames,
                "task": episode.task,
                "slot": slots.get(episode.task, 0),
                "clock": _clock(episode.frames / row.fps) if row.fps else "0:00",
                "size": _size(episode.size),
                "bytes": episode.size,
                "when": _when(episode.modified),
                "still": list(episode.still),
                "mirrored": episode.mirrored,
                "missing": list(episode.missing),
                "twins": [
                    {"dataset": name, "index": index}
                    for name, index in twins.get(episode.action_hash or "", [])
                    if name != row.name
                ],
                "barPct": episode.frames / longest * 100,
            }
            for episode in row.episodes
        ]
        datasets.append(
            {
                "name": row.name,
                "path": str(row.path),
                "robot": row.robot_type,
                "fps": row.fps,
                "status": row.status,
                "chip": _chip(row),
                "size": _size(row.size),
                "bytes": row.size,
                "when": _when(row.modified),
                "dur": _duration(row.seconds),
                "railSummary": f"{len(row.episodes)} ep \u00b7 {_duration(row.seconds)} \u00b7 {_size(row.size)}",
                "pills": _pills(row, overview),
                "cameras": [
                    {"name": key.removeprefix("observation.images."), "w": w, "h": h}
                    for key, w, h in sorted(
                        row.cameras, key=lambda camera: (not camera[0].endswith("scene"), camera[0])
                    )
                ],
                "notes": [
                    {"level": note.level, "text": note.text}
                    for note in sorted(row.notes, key=lambda note: _LEVELS.index(note.level))
                ],
                "tasks": [{"task": task, "slot": slots.get(task, 0), "count": counts[task]} for task in row.tasks],
                "unused": [task for task in dict.fromkeys(row.listed_tasks) if task not in counts],
                "recording": recording.get(row.path.resolve()),
                "episodes": episodes,
            }
        )
        attention += [
            {"dataset": row.name, "level": note.level, "text": note.text}
            for note in row.notes
            if note.level in ("problem", "check")
        ]
    station_path = pathlib.Path(station_dir).expanduser() if station_dir else None
    here = {row.name for row in overview.datasets}
    if station_path and station_path.parent.resolve() == overview.root.resolve() and station_path.name not in here:
        attention.append(
            {
                "dataset": "",
                "level": "check",
                "text": f"configs/station.yaml records into {station_dir}, which is not a dataset here: change its "
                "dataset_dir, or the next recording starts a new, empty one",
            }
        )
    return {
        "root": str(overview.root),
        "rootShort": _short(overview.root),
        "scanned": _when(overview.scanned_at),
        "summary": {
            "datasets": len(datasets),
            "episodes": overview.episodes,
            "recorded": _duration(overview.seconds),
            "onDisk": _size(overview.size),
            "toCheck": len(attention),
        },
        "datasets": datasets,
        "attention": attention,
        "overlaps": [
            {
                "a": o.a,
                "aEps": span(o.a_episodes),
                "b": o.b,
                "bEps": span(o.b_episodes),
                "kind": o.kind,
                "size": _size(o.size) if o.size else "",
            }
            for o in overview.overlaps
        ],
        "slots": slots,
        "skipped": overview.skipped,
        "stationDir": station_dir or "",
        "stationDirPath": str(station_path.resolve()) if station_path else "",
    }


def video_file(row: DatasetRow, episode: int, camera: str) -> pathlib.Path | None:
    """One episode's video of one camera (`"scene"`, `"left_wrist"`, ...), or None when it is not a file inside the
    dataset's folder."""
    key = next((key for key, _, _ in row.cameras if key.removeprefix("observation.images.") == camera), None)
    if key is None or episode not in {entry.index for entry in row.episodes}:
        return None
    try:
        relative = row.video_path.format(episode_chunk=episode // row.chunk_size, video_key=key, episode_index=episode)
    except (KeyError, IndexError, ValueError):
        return None
    path = (row.path / relative).resolve()
    return path if path.is_relative_to(row.path.resolve()) and path.is_file() else None


def station_dataset_dir(config: pathlib.Path | None) -> str | None:
    """The `dataset_dir` the station config records into, as written there; None when it cannot be read."""
    if config is None or not config.is_file():
        return None
    try:
        value = yaml.safe_load(config.read_text()).get("dataset_dir")
    except (OSError, yaml.YAMLError, AttributeError):
        return None
    return str(value) if value else None


## ----- the server -----

## How long one scan is reused, so a click, or the three videos of one episode, do not each rescan.
_SCAN_TTL_S = 3.0
_PAGE = pathlib.Path(__file__).with_name("overview_page.html")
_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; media-src 'self'; "
    "connect-src 'self'; img-src data:"
)
_VIDEO_URL = re.compile(r"/video/([^/]+)/(\d+)/([A-Za-z0-9_]+)\.mp4")
_CHUNK = 256 * 1024


class _Scanner:
    """The latest scan, reused for a few seconds, and dropped after an edit."""

    def __init__(self, root: pathlib.Path, config: pathlib.Path | None) -> None:
        self.root = root
        self.config = config
        self._at = 0.0
        self._overview: Overview | None = None

    def invalidate(self) -> None:
        self._overview = None

    def overview(self) -> Overview:
        with data_edit.LOCK:
            if self._overview is None or time.monotonic() - self._at > _SCAN_TTL_S:
                self._overview = scan(self.root)
                self._at = time.monotonic()
            return self._overview

    def json(self) -> dict[str, Any]:
        overview = self.overview()
        return overview_json(overview, data_edit.recordings(), station_dataset_dir(self.config))


class _Handler(http.server.BaseHTTPRequestHandler):
    """The page, its data, the episodes' videos and the actions; on loopback only, for this page only."""

    scanner: _Scanner
    actions: dict[str, Callable[[dict[str, Any]], str]]

    def _allowed_host(self) -> bool:
        return (self.headers.get("Host") or "").rsplit(":", 1)[0] in ("localhost", "127.0.0.1", "[::1]")

    def do_GET(self) -> None:  # noqa: N802
        if not self._allowed_host():
            return self._reply(403, "text/plain", b"forbidden")
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._reply(200, "text/html; charset=utf-8", self._page(), {"Content-Security-Policy": _CSP})
            if path == "/api/overview":
                return self._json(200, self.scanner.json())
            if match := _VIDEO_URL.fullmatch(path):
                row = next((r for r in self.scanner.overview().datasets if r.name == match.group(1)), None)
                video = video_file(row, int(match.group(2)), match.group(3)) if row else None
                if video is None:
                    return self._reply(404, "text/plain", b"no such video")
                return self._video(video)
        except Exception as error:  # a vanished folder or a half-written file must not kill the server
            return self._json(500, {"error": f"failed: {error}"})
        return self._reply(404, "text/plain", b"not found")

    def _page(self) -> bytes:
        """The page with the first scan inside it, so it draws at once; a scan that fails leaves `null`, and the
        page then asks for the data itself and says what went wrong."""
        try:
            boot = json.dumps(self.scanner.json()).replace("<", "\\u003c")
        except Exception:
            boot = "null"
        return _PAGE.read_text().replace("__BOOT__", boot).encode()

    def do_POST(self) -> None:  # noqa: N802
        # Only the page itself may act: a custom header, which another website's page cannot add without a preflight
        # this server never answers, and an Origin that names this address.
        origin = self.headers.get("Origin")
        if (
            not self._allowed_host()
            or self.headers.get("X-Dashboard") != "1"
            or (origin and origin != f"http://{self.headers.get('Host')}")
        ):
            return self._json(403, {"error": "actions are taken from the page itself only"})
        action = self.actions.get(urllib.parse.urlsplit(self.path).path)
        if action is None:
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16384:
                return self._json(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the request must be a JSON object")
        except ValueError as error:
            return self._json(400, {"error": f"bad request: {error}"})
        try:
            message = action(body)
        except data_edit.EditError as error:
            self.scanner.invalidate()
            return self._json(409, {"error": str(error)})
        except Exception as error:  # the edit stopped half way: the page shows what is on disk now
            self.scanner.invalidate()
            return self._json(500, {"error": f"failed: {error}"})
        self.scanner.invalidate()
        return self._json(200, {"ok": True, "message": message})

    def _video(self, path: pathlib.Path) -> None:
        """Sends a video, or the byte range of it the browser asks for so that it can seek."""
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
        if match is None or not (match[1] or match[2]):
            status, start, end = 200, 0, size - 1
        else:
            status = 206
            if match[1]:
                start, end = int(match[1]), min(int(match[2] or size - 1), size - 1)
            else:  # the last N bytes
                start, end = max(0, size - int(match[2])), size - 1
            if start > end:
                return self._reply(416, "text/plain", b"range not satisfiable", {"Content-Range": f"bytes */{size}"})
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        with (
            contextlib.suppress(BrokenPipeError, ConnectionResetError),
            path.open("rb") as file,
        ):  # the browser moved on
            file.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = file.read(min(_CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _json(self, status: int, value: Any) -> None:
        self._reply(status, "application/json", json.dumps(value).encode())

    def _reply(self, status: int, kind: str, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):  # the browser moved on
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silent."""


def make_server(root: pathlib.Path, port: int, config: pathlib.Path | None = None) -> http.server.ThreadingHTTPServer:
    """A server for the page on `127.0.0.1:<port>` (loopback only; `port` 0 picks a free one).

    @param root The data root: the folder that holds the dataset folders.
    @param port The port.
    @param config The station config, whose `dataset_dir` the page checks still exists; None for none.
    """
    scanner = _Scanner(root, config)

    def text(body: dict[str, Any], key: str) -> str:
        value = body.get(key)
        return value if isinstance(value, str) else ""

    actions: dict[str, Callable[[dict[str, Any]], str]] = {
        "/api/delete-episodes": lambda b: data_edit.delete_episodes(root, text(b, "dataset"), b.get("episodes")),
        "/api/rename-episodes": lambda b: data_edit.rename_episodes_task(
            root, text(b, "dataset"), b.get("episodes"), text(b, "task")
        ),
        "/api/rename-task": lambda b: data_edit.rename_task(root, text(b, "task"), text(b, "to")),
        "/api/delete-task": lambda b: data_edit.delete_task(root, text(b, "task"), text(b, "dataset") or None),
        "/api/rename-dataset": lambda b: data_edit.rename_dataset(
            root, text(b, "dataset"), text(b, "to"), bool(b.get("with_eef"))
        ),
    }
    handler = type("Handler", (_Handler,), {"scanner": scanner, "actions": actions})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def serve(root: pathlib.Path, port: int, config: pathlib.Path | None = None) -> None:
    """Serves the page until interrupted."""
    server = make_server(root, port, config)
    print(f"serving {root} on http://localhost:{server.server_port}  (Ctrl-C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
