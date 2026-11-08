"""A read-only overview of every LeRobot v2.1 dataset under one folder, as a single self-contained web page.

```
uv run dhk-dataset overview                 # scans ~/hackathon_data, writes the page, prints where
uv run dhk-dataset overview --serve         # serves it on http://localhost:8081, rescanning on every reload
```

The station has no screen: forward the port with `ssh -L 8081:localhost:8081 <user>@<station>` and open
http://localhost:8081, or copy the written file to your laptop and open it there.

The page shows, per dataset: the robot, frame rate, cameras, episodes (their lengths, coloured by task), time, size
on disk, when it was last written, and what looks wrong or worth knowing (see `scan()`). Across datasets it shows
episodes that hold the same recording more than once. It only reads: it never opens a camera or an arm, never
writes into a dataset folder and offers no action.

@see scan(), render_html(), serve()
"""

import collections
import dataclasses
import datetime
import hashlib
import html
import http.server
import itertools
import json
import os
import pathlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from dream_hackathon import dataset

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
        for key, video in videos.items():
            if not video.exists():
                problems[f"video {key.removeprefix('observation.images.')} is missing"].append(index)
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


## ----- the page -----

_e = html.escape


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
    return (
        f"{seconds / 3600:.1f} h"
        if seconds >= 3600
        else f"{seconds / 60:.0f} min"
        if seconds >= 120
        else f"{seconds:.0f} s"
    )


def _when(epoch: float | None) -> str:
    if epoch is None:
        return "—"
    moment = datetime.datetime.fromtimestamp(epoch)
    return f"{moment:%b} {moment.day}, {moment:%H:%M}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


_ICONS = {"good": "✓", "warning": "▲", "critical": "✕", "info": "i"}


def _icon(kind: str) -> str:
    """A status mark; it always comes with words, so colour is never the only signal."""
    return f'<span class="ic {kind}" aria-hidden="true">{_ICONS[kind]}</span>'


def _chip(status: str, problems: int = 0, checks: int = 0) -> str:
    if status == "problem":
        return f'<span class="chip">{_icon("critical")}{_plural(problems, "problem")}</span>'
    if status == "check":
        return f'<span class="chip">{_icon("warning")}{checks} to check</span>'
    return f'<span class="chip">{_icon("good")}Consistent</span>'


def _chip_of(row: DatasetRow) -> str:
    return _chip(row.status, sum(n.level == "problem" for n in row.notes), sum(n.level == "check" for n in row.notes))


def _ticks(longest: float) -> tuple[list[float], float]:
    """Axis ticks in seconds, and the axis end (the first tick at or past the longest bar)."""
    step = next((s for s in (10, 15, 30, 60, 120, 300, 600, 1800, 3600) if longest / s <= 5), 7200)
    end = max(step, -(-longest // step) * step)
    return [i * step for i in range(int(end // step) + 1)], end


def _episode_bars(row: DatasetRow, slots: dict[str, int]) -> str:
    """Horizontal bars, one per episode: length on the axis, task as colour (names in the legend and tooltip)."""
    if not row.episodes:
        return '<p class="muted">No episodes yet.</p>'
    seconds = [episode.frames / row.fps for episode in row.episodes]
    ticks, end = _ticks(max(seconds))
    longest, shortest = seconds.index(max(seconds)), seconds.index(min(seconds))
    grid = "".join(f'<i style="left:{t / end * 100:.2f}%"></i>' for t in ticks)
    labels = "".join(f'<span style="left:{t / end * 100:.2f}%">{_clock(t)}</span>' for t in ticks)
    rows = []
    for position, (episode, length) in enumerate(zip(row.episodes, seconds, strict=True)):
        tip = (
            f'data-v="{_e(_clock(length))} · {episode.frames:,} frames" data-l="{_e(episode.task or "no task")}" '
            f'data-s="Episode {episode.index} · {_size(episode.size)} · written {_when(episode.modified)}"'
        )
        value = f'<span class="val">{_clock(length)}</span>' if position in (longest, shortest) else ""
        slot = slots.get(episode.task, 0)
        rows.append(
            f'<div class="row" tabindex="0" {tip} '
            f'aria-label="Episode {episode.index}, {_clock(length)}, {_e(episode.task)}">'
            f'<span class="idx">{episode.index}</span><span class="plot" style="--w:{length / end * 100:.2f}%">'
            f'<span class="bar s{slot}"></span>{value}</span></div>'
        )
    return (
        f'<div class="bars"><div class="grid" aria-hidden="true">{grid}</div>{"".join(rows)}'
        f'<div class="axis" aria-hidden="true">{labels}</div></div>'
    )


def _legend(row: DatasetRow, slots: dict[str, int]) -> str:
    items = []
    for task in row.tasks:
        mine = [episode for episode in row.episodes if episode.task == task]
        time = _duration(sum(episode.frames for episode in mine) / row.fps)
        items.append(
            f'<li><span class="sw s{slots.get(task, 0)}"></span><span class="t">{_e(task or "no task")}</span>'
            f'<span class="muted">{_plural(len(mine), "episode")} · {time}</span></li>'
        )
    return f'<ul class="legend">{"".join(items)}</ul>'


def _episode_table(row: DatasetRow) -> str:
    body = []
    for episode in row.episodes:
        flags = [f"{arm} arm still" for arm in episode.still] + (["mirrored"] if episode.mirrored else [])
        body.append(
            f"<tr><td>{episode.index}</td><td>{_e(episode.task)}</td><td>{_clock(episode.frames / row.fps)}</td>"
            f"<td>{episode.frames:,}</td><td>{_size(episode.size)}</td><td>{_when(episode.modified)}</td>"
            f"<td>{_e(', '.join(flags))}</td></tr>"
        )
    return (
        '<details><summary>Episode table</summary><div class="scroll"><table><thead><tr><th>#</th><th>Task</th>'
        "<th>Length</th><th>Frames</th><th>Size</th><th>Written</th><th>Flags</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div></details>"
    )


def _notes(row: DatasetRow) -> str:
    if not row.notes:
        return '<p class="muted">Every file the metadata lists is present and agrees with it.</p>'
    kinds = {"problem": "critical", "check": "warning", "note": "info"}
    ordered = sorted(row.notes, key=lambda note: _LEVELS.index(note.level))
    items = "".join(
        f'<li>{_icon(kinds[n.level])}<span class="sr">{n.level.capitalize()}: </span>{_e(n.text)}</li>'
        for n in ordered
    )
    return f'<ul class="notes">{items}</ul>'


def _dataset_card(row: DatasetRow, slots: dict[str, int]) -> str:
    cameras = ", ".join(f"{key.removeprefix('observation.images.')} {w}×{h}" for key, w, h in row.cameras)
    return (
        f'<section class="card" id="ds-{_e(row.name)}"><header><h3>{_e(row.name)}</h3>{_chip_of(row)}</header>'
        f'<p class="muted meta">{_e(row.robot_type)} · {row.fps} fps · {_e(cameras) or "no cameras"} · '
        f"<code>{_e(str(row.path))}</code></p>"
        f"{_notes(row)}{_episode_bars(row, slots)}{_legend(row, slots)}{_episode_table(row)}</section>"
    )


def _tile(label: str, value: str, sub: str) -> str:
    return (
        f'<div class="tile"><div class="k">{_e(label)}</div><div class="v">{value}</div>'
        f'<div class="sub">{_e(sub)}</div></div>'
    )


def _tiles(overview: Overview, slots: dict[str, int]) -> str:
    rows = overview.datasets
    problems = sum(n.level == "problem" for row in rows for n in row.notes)
    checks = sum(n.level == "check" for row in rows for n in row.notes)
    frames = sum(row.frames for row in rows)
    video = sum(row.video_size for row in rows)
    if problems + checks == 0:
        attention = f"{_icon('good')}Nothing"
    else:
        attention = f"{_icon('critical' if problems else 'warning')}{problems + checks}"
    return (
        '<div class="tiles">'
        + "".join(
            [
                _tile("Datasets", str(len(rows)), ", ".join(sorted({row.robot_type for row in rows})) or "none yet"),
                _tile("Episodes", str(overview.episodes), f"across {_plural(len(slots), 'task')}"),
                _tile("Time in datasets", _duration(overview.seconds), f"{frames:,} frames"),
                _tile("On disk", _size(overview.size), f"{_size(video)} of it video"),
                _tile("Needs a look", attention, f"{_plural(problems, 'problem')}, {_plural(checks, 'check')}"),
            ]
        )
        + "</div>"
    )


def _datasets_table(overview: Overview) -> str:
    biggest = max((row.size for row in overview.datasets), default=1) or 1
    body = "".join(
        f'<tr><th scope="row"><a href="#ds-{_e(row.name)}">{_e(row.name)}</a></th><td>{_e(row.robot_type)}</td>'
        f"<td>{len(row.episodes)}</td><td>{_duration(row.seconds)}</td>"
        f'<td><span class="sizecell">{_size(row.size)}<span class="meter" aria-hidden="true">'
        f'<span style="width:{row.size / biggest * 100:.1f}%"></span></span></span></td>'
        f"<td>{len(row.tasks)}</td><td>{_when(row.modified)}</td><td>{_chip_of(row)}</td></tr>"
        for row in overview.datasets
    )
    return (
        '<div class="scroll"><table class="datasets"><thead><tr><th>Dataset</th><th>Robot</th><th>Episodes</th>'
        "<th>Time</th><th>Size</th><th>Tasks</th><th>Last written</th><th>Status</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _overlap_section(overview: Overview) -> str:
    if not overview.overlaps:
        return ""
    body = "".join(
        f"<tr><td>{_e(o.a)}</td><td>{span(o.a_episodes)}</td><td>{_e(o.b)}</td><td>{span(o.b_episodes)}</td>"
        f"<td>{_e(o.kind)}</td><td>{_size(o.size) if o.size else '—'}</td></tr>"
        for o in overview.overlaps
    )
    return (
        '<section class="card"><h2>The same recording in more than one dataset</h2>'
        '<p class="muted">Matched by the episodes\' action data. A <em>same recording</em> match stores the episode '
        f"again (about {_size(overview.duplicate_bytes)} in all); a single-arm match holds the same arm's actions "
        "with other videos.</p>"
        '<div class="scroll"><table><thead><tr><th>Dataset</th><th>Episodes</th><th>Dataset</th><th>Episodes</th>'
        f"<th>Match</th><th>Stored twice</th></tr></thead><tbody>{body}</tbody></table></div></section>"
    )


def render_html(overview: Overview) -> str:
    """The whole page: inline styles and a little script for tooltips, no external files."""
    slots: dict[str, int] = {}
    for row in overview.datasets:
        for task in row.tasks:
            slots.setdefault(task, len(slots) + 1 if len(slots) < _TASK_SLOTS else 0)
    cards = "".join(_dataset_card(row, slots) for row in overview.datasets)
    skipped = (
        f'<p class="muted">Not scanned (no meta/info.json): {_e(", ".join(overview.skipped))}</p>'
        if overview.skipped
        else ""
    )
    empty = '<p class="card muted">No datasets found here yet.</p>' if not overview.datasets else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">'
        f"<title>Hackathon data</title><style>{_CSS}</style></head><body><main>"
        f'<header class="top"><h1>Hackathon data</h1><p class="muted"><code>{_e(str(overview.root))}</code> · scanned '
        f"{_when(overview.scanned_at)} · reload the page to scan again (when served)</p></header>"
        f"{_tiles(overview, slots)}{empty}"
        f'<section class="card"><h2>Datasets</h2>{_datasets_table(overview)}{skipped}</section>'
        f"{_overlap_section(overview)}{cards}</main>"
        '<div id="tip" role="tooltip" hidden><b></b><span></span><span class="muted"></span></div>'
        f"<script>{_JS}</script></body></html>"
    )


_CSP = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"

_CSS = """
:root{color-scheme:light dark;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--track:#cde2fb;--seq:#2a78d6;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s0:#898781;
--good:#0ca30c;--warning:#fab219;--critical:#d03b3b;--info:#898781}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--track:#0d366b;--seq:#3987e5;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px;display:grid;grid-template-columns:minmax(0,1fr);gap:16px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;margin:0 0 12px}h3{font-size:17px;margin:0}
.muted{color:var(--ink2);font-size:13px;margin:0}code{font:12px ui-monospace,monospace;overflow-wrap:anywhere}
a{color:inherit;text-decoration-color:var(--axis)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px}
.card header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between}
.meta{margin:4px 0 12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tile .k{font-size:13px;color:var(--ink2)}.tile .v{font-size:30px;font-weight:600;line-height:1.2;margin:4px 0}
.tile .sub{font-size:12px;color:var(--ink2)}
.ic{display:inline-flex;width:16px;justify-content:center;margin-right:6px;font-size:12px;font-weight:700;flex:none}
.ic.good{color:var(--good)}.ic.warning{color:var(--warning)}.ic.critical{color:var(--critical)}.ic.info{color:var(--info)}
.chip{display:inline-flex;align-items:center;font-size:13px;white-space:nowrap}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:8px 12px 8px 0;border-bottom:1px solid var(--grid);vertical-align:middle;
font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:500}tbody th{color:var(--ink);font-weight:600}
.datasets th,.datasets td{white-space:nowrap}
.sizecell{display:inline-flex;align-items:center;gap:8px;min-width:130px}
.meter{display:inline-block;width:56px;height:8px;border-radius:4px;background:var(--track);overflow:hidden}
.meter span{display:block;height:100%;border-radius:4px;background:var(--seq)}
.notes{list-style:none;margin:0 0 16px;padding:0;display:grid;gap:4px;font-size:14px}
.notes li{display:flex;align-items:baseline}.sr{position:absolute;left:-9999px}
.bars{position:relative;padding-bottom:22px;margin-top:8px}
.grid{position:absolute;left:36px;right:56px;top:0;bottom:22px}
.grid i{position:absolute;top:0;bottom:0;width:1px;background:var(--grid)}
.row{display:flex;align-items:center;height:22px;position:relative;border-radius:4px}
.row:hover,.row:focus{background:color-mix(in srgb,var(--ink) 6%,transparent);outline:none}
.row:focus-visible{outline:2px solid var(--seq)}
.idx{width:36px;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.plot{position:relative;flex:1;margin-right:56px;height:14px}
.bar{position:absolute;left:0;top:0;height:14px;width:var(--w);border-radius:0 4px 4px 0;min-width:2px}
.val{position:absolute;left:calc(var(--w) + 6px);top:-4px;font-size:12px;color:var(--ink2);
font-variant-numeric:tabular-nums}
.axis{position:absolute;left:36px;right:56px;bottom:0;height:18px;font-size:11px;color:var(--muted)}
.axis span{position:absolute;transform:translateX(-50%)}.axis span:first-child{transform:none}
.s1{background:var(--s1)}.s2{background:var(--s2)}.s3{background:var(--s3)}.s4{background:var(--s4)}
.s5{background:var(--s5)}.s6{background:var(--s6)}.s7{background:var(--s7)}.s0{background:var(--s0)}
.legend{list-style:none;margin:8px 0 0;padding:0;display:flex;flex-wrap:wrap;gap:6px 20px;font-size:13px}
.legend li{display:flex;align-items:center;gap:8px}.sw{width:12px;height:12px;border-radius:3px;flex:none}
details{margin-top:16px}summary{cursor:pointer;font-size:13px;color:var(--ink2)}
#tip{position:fixed;z-index:9;pointer-events:none;background:var(--surface);border:1px solid var(--border);
border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.18);display:grid;gap:2px;
max-width:320px}
#tip[hidden]{display:none}
#tip b{font-size:14px}
@media print{body{background:#fff}.row:hover{background:none}}
"""

_JS = """
(function(){
var tip=document.getElementById('tip'),kids=tip.children;
function place(x,y){var w=tip.offsetWidth,h=tip.offsetHeight;
tip.style.left=Math.max(8,Math.min(x+14,innerWidth-w-8))+'px';
tip.style.top=(y+14+h>innerHeight?Math.max(8,y-h-14):y+14)+'px';}
function show(el,x,y){kids[0].textContent=el.dataset.v;kids[1].textContent=el.dataset.l;
kids[2].textContent=el.dataset.s;tip.hidden=false;place(x,y);}
document.querySelectorAll('[data-v]').forEach(function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY)});
el.addEventListener('pointerleave',function(){tip.hidden=true});
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.left+r.width/2,r.top)});
el.addEventListener('blur',function(){tip.hidden=true});});
})();
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Answers `GET /` with a freshly scanned page; nothing else."""

    root: pathlib.Path

    def do_GET(self) -> None:  # noqa: N802
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in ("localhost", "127.0.0.1", "[::1]"):
            return self._reply(403, "text/plain", b"forbidden")
        if self.path.split("?")[0] not in ("/", "/index.html"):
            return self._reply(404, "text/plain", b"not found")
        try:
            page = render_html(scan(self.root)).encode()
        except Exception as error:  # a vanished folder or a half-written file must not kill the server
            return self._reply(500, "text/plain", f"scan failed: {error}".encode())
        return self._reply(200, "text/html; charset=utf-8", page)

    def _reply(self, status: int, kind: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silent."""


def make_server(root: pathlib.Path, port: int) -> http.server.ThreadingHTTPServer:
    """A server for the page on `127.0.0.1:<port>` (loopback only; `port` 0 picks a free one)."""
    handler = type("Handler", (_Handler,), {"root": root})
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(root: pathlib.Path, port: int) -> None:
    """Serves the page until interrupted; every reload rescans."""
    server = make_server(root, port)
    print(f"serving {root} on http://localhost:{server.server_port}  (Ctrl-C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
