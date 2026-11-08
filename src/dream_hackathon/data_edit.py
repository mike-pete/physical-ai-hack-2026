"""What the data page may do to the dataset folders under one root, and the checks it makes first.

Every action here changes recorded data, so each one first refuses when a recorder is writing into a folder it would
touch (`recordings()`), validates its names, and then does the work with `dataset.drop_episodes()`,
`dataset.retask_episodes()` and `dataset.forget_task()`, which keep each dataset's files in step with each other.
Each action returns one sentence for the page to show, or raises `EditError` with the reason.

`dhk-collect` holds the station lock while it records (`station_lock`), but `scripts/mirror_collect.py` does not, so
a recorder is also looked for among the running programs.

@see recordings(), delete_episodes(), rename_task(), delete_task(), rename_dataset()
"""

import datetime
import os
import pathlib
import re
import threading
from collections.abc import Iterable

from dream_hackathon import dataset, station_lock

## Programs that record into a dataset folder, as the basename of one of their first arguments.
_RECORDERS = frozenset({"mirror_collect.py", "mirror_collect_sync.py", "dhk-collect"})
## Where `mirror_collect.py` puts a folder for each task when it is given no `--dataset`.
_DEFAULT_DATA_ROOT = "~/hackathon_data"
## A dataset folder's name: what Dream accepts, and nothing that could leave the data root.
FOLDER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FOLDER_NAME_MAX = 100
LOG_PATH = pathlib.Path("~/.cache/dream_hackathon/data_edits.log").expanduser()
## One edit at a time, and none while a scan reads (the page's scanner takes it too).
LOCK = threading.RLock()


class EditError(RuntimeError):
    """The action was refused, or failed; the message says why, in words for the person at the page."""


def _flag(argv: list[str], *names: str) -> str | None:
    """The value of `--flag value` or `--flag=value` among a program's arguments."""
    for position, argument in enumerate(argv):
        for name in names:
            if argument == name and position + 1 < len(argv):
                return argv[position + 1]
            if argument.startswith(name + "="):
                return argument.split("=", 1)[1]
    return None


def _newest_dataset(root: pathlib.Path) -> pathlib.Path | None:
    """The dataset folder under `root` that was written to last: where a recorder given only a root writes."""
    best: tuple[float, pathlib.Path] | None = None
    if not root.is_dir():
        return None
    for child in root.iterdir():
        info = child / "meta" / "info.json"
        if info.is_file():
            stamp = max(p.stat().st_mtime for p in (info, child / "meta" / "episodes.jsonl") if p.exists())
            if best is None or stamp > best[0]:
                best = (stamp, child)
    return best[1] if best else None


def recordings() -> dict[pathlib.Path, str]:
    """The dataset folders a recorder is writing into right now, each with who.

    Found from the station lock (`dhk-collect`) and from the running programs: `mirror_collect.py` writes into its
    `--dataset` folder, or, given only a `--data-root`, into the one written to last. Linux only: with no `/proc`
    the programs are not seen.

    @return Resolved folder to a phrase such as `mirror_collect.py (pid 4242)`.
    """
    found: dict[pathlib.Path, str] = {}
    held = station_lock.holder()
    if held and held.get("dataset"):
        found[pathlib.Path(held["dataset"]).expanduser().resolve()] = (
            f"{held.get('holder') or 'a collector'} (pid {held.get('pid')})"
        )
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [part.decode(errors="replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            cwd = pathlib.Path(os.readlink(entry / "cwd"))
        except OSError:
            continue
        names = {os.path.basename(argument) for argument in argv[:8]}
        recorder = next(iter(names & _RECORDERS), None)
        if recorder is None:
            continue
        folder = _flag(argv, "--dataset", "--dataset-dir")
        if folder:
            path: pathlib.Path | None = (cwd / pathlib.Path(folder).expanduser()).resolve()
        else:
            data_root = _flag(argv, "--data-root") or _DEFAULT_DATA_ROOT
            path = _newest_dataset((cwd / pathlib.Path(data_root).expanduser()).resolve())
        if path is not None:
            found.setdefault(path, f"{recorder} (pid {entry.name})")
    return found


def _log(line: str) -> None:
    """Appends one line to the edit log; best effort, so a full disk cannot undo a finished edit."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as file:
            file.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}  {line}\n")
    except OSError:
        pass


def _folder(root: pathlib.Path, name: str) -> pathlib.Path:
    """The dataset folder called `name` directly under `root`.

    @throws EditError There is no such dataset (a name with a slash, `.` or `..` never is one).
    """
    if (
        not name
        or "/" in name
        or "\0" in name
        or name in (".", "..")
        or not (root / name / "meta" / "info.json").is_file()
    ):
        raise EditError(f"there is no dataset called {name or '(none)'}")
    return root / name


def _refuse_while_recording(folders: Iterable[pathlib.Path]) -> None:
    busy = recordings()
    for folder in folders:
        who = busy.get(folder.resolve())
        if who:
            raise EditError(f"{folder.name} is being recorded into by {who}; stop it first, then try again")


def _episodes(values: object) -> set[int]:
    """The episode numbers of a request, checked to be a non-empty list of whole numbers."""
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in values)
    ):
        raise EditError("choose at least one episode")
    return set(values)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _done(done: list[str], failed: str, error: Exception) -> EditError:
    """The refusal for an action that ran through several folders and stopped in one."""
    first = f"It was done in {', '.join(done)}, but " if done else ""
    return EditError(f"{first}{failed} failed: {error}. Nothing after it was changed.")


def delete_episodes(root: pathlib.Path, name: str, episodes: object) -> str:
    """Deletes episodes of one dataset; the ones after them are renumbered (`dataset.drop_episodes()`).

    @param root The data root.
    @param name The dataset folder.
    @param episodes The episode numbers.
    @throws EditError A recorder is writing into the folder, or an episode does not exist.
    """
    chosen = _episodes(episodes)
    with LOCK:
        folder = _folder(root, name)
        _refuse_while_recording([folder])
        try:
            left = dataset.drop_episodes(folder, chosen)
        except (dataset.DatasetError, OSError, ValueError, KeyError) as error:
            raise EditError(str(error)) from error
    message = f"Deleted {_plural(len(chosen), 'episode')} of {name}; {left} left, numbered 0 to {max(left - 1, 0)}."
    _log(f"delete-episodes {folder} {sorted(chosen)}")
    return message


def rename_episodes_task(root: pathlib.Path, name: str, episodes: object, task: str) -> str:
    """Gives episodes of one dataset another task text (`dataset.retask_episodes()`).

    @throws EditError A recorder is writing into the folder, the text is empty, or an episode cannot be renamed.
    """
    chosen = _episodes(episodes)
    with LOCK:
        folder = _folder(root, name)
        _refuse_while_recording([folder])
        try:
            dataset.retask_episodes(folder, chosen, task)
        except (dataset.DatasetError, OSError, ValueError, KeyError) as error:
            raise EditError(str(error)) from error
    _log(f"retask {folder} {sorted(chosen)} -> {task.strip()!r}")
    return f"Renamed {_plural(len(chosen), 'episode')} of {name} to “{task.strip()}”."


def _holders(root: pathlib.Path, task: str) -> list[tuple[pathlib.Path, set[int]]]:
    """Every dataset under `root` with episodes of `task`: its folder and their numbers."""
    found = []
    for child in sorted(root.iterdir()):
        if not (child / "meta" / "info.json").is_file():
            continue
        try:
            mine = {e.index for e in dataset.list_episodes(child) if task in e.tasks}
        except (OSError, ValueError, KeyError):
            continue
        if mine:
            found.append((child, mine))
    return found


def rename_task(root: pathlib.Path, old: str, new: str) -> str:
    """Renames a task in every dataset that has episodes of it.

    @throws EditError A recorder is writing into one of them, the text is empty, or the task is nowhere.
    """
    new = new.strip()
    if not new:
        raise EditError("a task needs a text")
    with LOCK:
        targets = _holders(root, old)
        if not targets:
            raise EditError(f"no dataset has episodes of “{old}”")
        _refuse_while_recording(folder for folder, _ in targets)
        done: list[str] = []
        total = 0
        for folder, mine in targets:
            try:
                dataset.retask_episodes(folder, mine, new)
            except (dataset.DatasetError, OSError, ValueError, KeyError) as error:
                raise _done(done, folder.name, error) from error
            done.append(folder.name)
            total += len(mine)
    _log(f"rename-task {old!r} -> {new!r} in {done}")
    return f"Renamed the task to “{new}” in {_plural(len(done), 'dataset')} ({_plural(total, 'episode')})."


def delete_task(root: pathlib.Path, task: str, name: str | None = None) -> str:
    """Deletes a task and the episodes that use it, from one dataset or from every dataset that has it.

    The task's row is removed from each `tasks.jsonl` and the tasks left are numbered 0, 1, 2, ... again.

    @param name The one dataset; None for every dataset that has episodes of the task or lists it.
    @throws EditError A recorder is writing into one of them, or the task is nowhere.
    """
    with LOCK:
        if name is not None:
            folders = [_folder(root, name)]
        else:
            folders = [c for c in sorted(root.iterdir()) if (c / "meta" / "info.json").is_file()]
        targets: list[tuple[pathlib.Path, set[int]]] = []
        for folder in folders:
            try:
                mine = {e.index for e in dataset.list_episodes(folder) if task in e.tasks}
                listed = any(row["task"] == task for row in dataset.task_rows(folder))
            except (OSError, ValueError, KeyError):
                continue
            if mine or listed:
                targets.append((folder, mine))
        if not targets:
            raise EditError(f"no dataset has the task “{task}”")
        _refuse_while_recording(folder for folder, _ in targets)
        done: list[str] = []
        episodes = 0
        for folder, mine in targets:
            try:
                if mine:
                    dataset.drop_episodes(folder, mine)
                dataset.forget_task(folder, task)
            except (dataset.DatasetError, OSError, ValueError, KeyError) as error:
                raise _done(done, folder.name, error) from error
            done.append(folder.name)
            episodes += len(mine)
    _log(f"delete-task {task!r} from {done}: {episodes} episodes")
    return f"Deleted the task “{task}”" + (
        f" and {_plural(episodes, 'episode')} in {_plural(len(done), 'dataset')}." if episodes else "."
    )


def rename_dataset(root: pathlib.Path, name: str, new: str, with_eef: bool = False) -> str:
    """Renames a dataset folder, and with `with_eef` its `<name>-eef` copy (from `tools/add_eef.py`) to `<new>-eef`.

    @throws EditError A recorder is writing into a folder, the new name is not a good one or is taken.
    """
    new = new.strip()
    if not FOLDER_NAME.fullmatch(new) or len(new) > FOLDER_NAME_MAX:
        raise EditError("use letters, digits, “-”, “_” and “.”, starting with a letter or digit")
    with LOCK:
        folder = _folder(root, name)
        moves = [(folder, root / new)]
        sibling = root / f"{name}-eef"
        if with_eef and (sibling / "meta" / "info.json").is_file():
            moves.append((sibling, root / f"{new}-eef"))
        _refuse_while_recording(source for source, _ in moves)
        for source, target in moves:
            if source == target:
                raise EditError("that is the folder's name already")
            if target.exists() or target.is_symlink():
                raise EditError(f"a folder called {target.name} already exists in {root}")
        moved: list[tuple[pathlib.Path, pathlib.Path]] = []
        try:
            for source, target in moves:
                os.rename(source, target)
                moved.append((source, target))
        except OSError as error:
            for source, target in reversed(moved):
                os.rename(target, source)
            raise EditError(f"could not rename the folder: {error}") from error
    _log(f"rename-dataset {[(s.name, t.name) for s, t in moves]}")
    names = " and ".join(f"{s.name} to {t.name}" for s, t in moves)
    return f"Renamed the folder {names}."
