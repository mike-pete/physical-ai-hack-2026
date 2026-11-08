"""Uploads LeRobot datasets to Dimensional's cloud with `dimos data upload`. It touches no arm, and no Dream account.

Needs the `dimos` command (Dimensional's CLI) on this computer and a login: run `dimos login` once. The script never
reads or prints the key; `dimos` keeps it. Python 3.12 or newer, standard library only:

```
# See what would be sent, without packing or sending anything
python3 tools/upload_dimensional.py ~/hackathon_data/stack-blocks ~/hackathon_data/golf-balls --dry-run
# Send them and keep the output next to the code
python3 tools/upload_dimensional.py ~/hackathon_data/stack-blocks --log docs/dimensional_uploads/stack-blocks.log
```

`dimos data upload` takes one file, and a LeRobot dataset is a folder. So each dataset is packed into `<name>.tar`
first: the dataset folder's files, byte for byte, under a `<name>/` folder. The tar is built the same way every time
(sorted, no timestamps or owners), so the same dataset gives the same bytes and Dimensional's content check answers
`already uploaded` instead of storing it twice. `dimos` compresses the tar for the transfer (lz4), and
`dimos data pull` hands the same tar back; `tar -xf` restores the folder.

Steps, per run: check each dataset folder (version 2.1, at least one episode), find `dimos` and check the login
(`dimos data quota`), pack and upload one dataset at a time (the tar is deleted before the next), then list the cloud's
files (`dimos data ls`) and confirm every tar is there. Open the `console:` link `dimos` prints to see the file in
https://console.dimensional.org.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass

LEROBOT_VERSION = "v2.1"
INSTALL_HINT = "install Dimensional's CLI (https://github.com/dimensionalOS/dimos), then run `dimos login` once"


@dataclass(frozen=True)
class DatasetInfo:
    path: pathlib.Path
    name: str
    episodes: int
    frames: int
    fps: int
    robot_type: str | None
    files: int
    size: int


def check_dataset(path: pathlib.Path) -> DatasetInfo:
    """The dataset's summary, or ValueError saying what is wrong. Reads meta/info.json; nothing else is parsed."""
    path = path.expanduser().resolve()
    info_file = path / "meta" / "info.json"
    if not info_file.is_file():
        raise ValueError(f"{path}: not a LeRobot dataset (no meta/info.json)")
    try:
        info = json.loads(info_file.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{info_file}: not valid JSON ({e})") from e
    if info.get("codebase_version") != LEROBOT_VERSION:
        raise ValueError(
            f"{path}: LeRobot {info.get('codebase_version')!r}, this tool packs {LEROBOT_VERSION} datasets"
        )
    episodes = int(info.get("total_episodes") or 0)
    if episodes < 1:
        raise ValueError(f"{path}: the dataset has no episodes")
    if not (path / "meta" / "episodes.jsonl").is_file() or not (path / "data").is_dir():
        raise ValueError(f"{path}: meta/episodes.jsonl or data/ is missing; the dataset is incomplete")
    files = [p for p in path.rglob("*") if p.is_file()]
    return DatasetInfo(
        path=path,
        name=path.name,
        episodes=episodes,
        frames=int(info.get("total_frames") or 0),
        fps=int(info.get("fps") or 0),
        robot_type=info.get("robot_type"),
        files=len(files),
        size=sum(p.stat().st_size for p in files),
    )


def _plain(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Drops what differs between two packs of the same folder (owner, timestamp), so the bytes repeat."""
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mtime = 0
    member.mode = 0o755 if member.isdir() else 0o644
    return member


def pack(dataset: DatasetInfo, out_dir: pathlib.Path) -> pathlib.Path:
    """Writes `<out_dir>/<name>.tar`, the dataset folder's files in sorted order under `<name>/`."""
    out = out_dir / f"{dataset.name}.tar"
    with tarfile.open(out, "w", format=tarfile.GNU_FORMAT) as tar:
        tar.add(dataset.path, arcname=dataset.name, recursive=False, filter=_plain)
        for path in sorted(dataset.path.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"{path}: a symbolic link; the dataset must hold plain files")
            tar.add(
                path,
                arcname=f"{dataset.name}/{path.relative_to(dataset.path).as_posix()}",
                recursive=False,
                filter=_plain,
            )
    return out


def find_dimos(explicit: str | None) -> str:
    """The `dimos` command to run: --dimos, then $DIMOS, then the PATH."""
    candidate = explicit or os.environ.get("DIMOS") or "dimos"
    found = shutil.which(candidate)
    if found is None:
        sys.exit(f"cannot find {candidate!r}: {INSTALL_HINT}")
    return found


class Transcript:
    """Prints to the terminal and, when asked, appends the same lines to a log file."""

    def __init__(self, log: pathlib.Path | None) -> None:
        self.file = None
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            self.file = log.open("a")

    def say(self, line: str = "") -> None:
        print(line, flush=True)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()

    def note(self, line: str) -> None:
        """Writes to the log only: for what the terminal gets anyway (an error `sys.exit` prints)."""
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()


def run_dimos(dimos: str, args: list[str], out: Transcript) -> subprocess.CompletedProcess[str]:
    """Runs `dimos <args>`, echoes what it prints and returns the result. Its progress bar is not shown."""
    out.say(f"$ dimos {' '.join(args)}")
    # A wide terminal for `dimos data ls`, so its table does not cut the file names; NO_COLOR keeps the log plain.
    env = {**os.environ, "COLUMNS": "250", "NO_COLOR": "1"}
    result = subprocess.run([dimos, *args], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip():
            out.say(f"  {line.rstrip()}")
    return result


def human(n: float) -> str:
    for unit in ("B", "kB", "MB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="+", type=pathlib.Path, help="LeRobot v2.1 dataset folders")
    parser.add_argument(
        "--robot", help="the robot id Dimensional files them under (default: each dataset's robot_type)"
    )
    parser.add_argument("--dimos", help="the dimos command (default: $DIMOS, then `dimos` on the PATH)")
    parser.add_argument("--staging-dir", type=pathlib.Path, help="where the tar is built (default: the system temp)")
    parser.add_argument("--log", type=pathlib.Path, help="append everything printed here, with the date, to this file")
    parser.add_argument(
        "--dry-run", action="store_true", help="check the folders and print the plan; pack and send nothing"
    )
    args = parser.parse_args()

    try:
        datasets = [check_dataset(p) for p in args.datasets]
    except ValueError as e:
        sys.exit(str(e))
    if len({d.name for d in datasets}) != len(datasets):
        sys.exit("two of the folders have the same name; their tars would overwrite each other")

    out = Transcript(args.log)
    try:
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        out.say(f"# upload_dimensional.py {'(dry run) ' if args.dry_run else ''}{stamp}")
        for d in datasets:
            out.say(
                f"{d.name}: {d.episodes} episodes, {d.frames} frames at {d.fps} fps, {d.files} files, {human(d.size)}"
                f", robot {args.robot or d.robot_type or '-'}"
            )
        if args.dry_run:
            out.say("dry run: nothing was packed or sent")
            return

        dimos = find_dimos(args.dimos)
        if run_dimos(dimos, ["data", "quota"], out).returncode != 0:
            sys.exit(f"dimos is not logged in or cannot reach Dimensional's cloud; run `dimos login` ({INSTALL_HINT})")

        stage_root = args.staging_dir.expanduser() if args.staging_dir else None
        failed: list[str] = []
        for d in datasets:
            # The tar, and the compressed copy dimos stages beside it: each about the dataset's size.
            free = shutil.disk_usage(stage_root or tempfile.gettempdir()).free
            if free < 2 * d.size + 2**28:
                need = human(2 * d.size)
                sys.exit(f"{d.name}: needs about {need} of free staging space, {human(free)} left; pass --staging-dir")
            with tempfile.TemporaryDirectory(prefix="dhk-dimensional-", dir=stage_root) as tmp:
                tar = pack(d, pathlib.Path(tmp))
                out.say(f"packed {tar.name} ({human(tar.stat().st_size)})")
                robot = args.robot or d.robot_type
                upload = ["data", "upload", str(tar), "--kind", "blob", *(["--robot", robot] if robot else [])]
                if run_dimos(dimos, upload, out).returncode != 0:
                    failed.append(d.name)

        listing = run_dimos(dimos, ["data", "ls"], out)
        # The file column shows the name as stored (`<name>.tar.lz4`); not `my-<name>.tar`.
        missing = [d.name for d in datasets if not re.search(rf"(?<![\w.-]){re.escape(d.name)}\.tar", listing.stdout)]
        if listing.returncode != 0 or missing or failed:
            out.say(f"NOT CONFIRMED: upload failed for {failed or '-'}, not in `dimos data ls`: {missing or '-'}")
            sys.exit(1)
        out.say(f"confirmed in `dimos data ls`: {', '.join(d.name + '.tar' for d in datasets)}")
    except SystemExit as stop:
        # A log that just ends reads like a run that was cut off: say why it stopped.
        if isinstance(stop.code, str):
            out.note(f"stopped: {stop.code}")
        raise
    finally:
        out.close()


if __name__ == "__main__":
    main()
