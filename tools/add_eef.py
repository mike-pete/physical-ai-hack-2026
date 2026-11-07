"""Adds the tool poses Dream's MakerMods models train on to a Metal or Maker Arm dataset.

Run it in ~/dream-env (the Dream library's environment), after collecting and before fine-tuning:

```
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/myteam-cubes --arm metal     # or: --arm maker
```

It writes a new dataset next to the input, `<name>-eef`, and never changes the input. After collecting more episodes,
run it again with `--replace` to rebuild the `-eef` folder. Per arm, per frame, it adds
the tool pose of the measured joints (state) and of the commanded joints (action), computed with Dream's own
kinematics of the arm (`dream_ik`, the arm model and tool center point `dreamcli robot run` uses), and the gripper
as 0 (closed) to 1 (open). The result is laid out as `src/dream_hackathon/embodiments/makermods.json`; it is checked
with the Dream library's dataset checks before this script ends. Videos are hard-linked, not copied, when the output
is on the same disk.
"""

import argparse
import json
import os
import pathlib
import shutil
import sys
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EMBODIMENT = pathlib.Path(__file__).resolve().parents[1] / "src" / "dream_hackathon" / "embodiments" / "makermods.json"
## Per arm: Dream's name for its kinematics, and what the recorded gripper (stroke, mm) reads fully open.
ARMS = {"metal": ("metal", 100.0), "maker": ("makers", 80.0)}
## The collected vectors' layout: per side (left first) six joints in radians, then the gripper in mm.
SIDES = ("left", "right")
JOINTS = 6


def to_eef(
    vectors: np.ndarray,
    fk: Callable[[np.ndarray, str], np.ndarray],
    grip_scale: float,
) -> dict[str, np.ndarray]:
    """Every part of the EEF layout, per frame, from the collected `[frames, 14]` vectors.

    @param vectors `[frames, 14]`: left joints, left gripper (mm), right joints, right gripper (mm).
    @param fk `fk(joints [frames, 6], side) -> [frames, 4, 4]`, Dream's tool pose in that arm's base frame.
    @param grip_scale The gripper's stroke fully open, mm.
    @return Part name to `[frames, width]`: `<side>_arm`, `<side>_gripper`, `<side>_eef_xyz`, `<side>_eef_rot6`
        (the rotation's first two columns, `[R00, R10, R20, R01, R11, R21]`) and `<side>_grip`.
    """
    parts: dict[str, np.ndarray] = {}
    for i, side in enumerate(SIDES):
        start = i * (JOINTS + 1)
        joints = vectors[:, start : start + JOINTS].astype(np.float64)
        gripper_mm = vectors[:, start + JOINTS : start + JOINTS + 1].astype(np.float64)
        poses = fk(joints, side)
        parts[f"{side}_arm"] = joints
        parts[f"{side}_gripper"] = gripper_mm
        parts[f"{side}_eef_xyz"] = poses[:, :3, 3]
        parts[f"{side}_eef_rot6"] = np.concatenate([poses[:, :3, 0], poses[:, :3, 1]], axis=1)
        parts[f"{side}_grip"] = np.clip(gripper_mm / grip_scale, 0.0, 1.0)
    return parts


def pack(parts: dict[str, np.ndarray], spans: dict[str, list[int]]) -> np.ndarray:
    """Lays the parts out by an embodiment's `[start, end)` spans into one `[frames, width]` float32 array."""
    frames = len(next(iter(parts.values())))
    out = np.zeros((frames, max(end for _, end in spans.values())), dtype=np.float32)
    for part, (start, end) in spans.items():
        out[:, start:end] = parts[part]
    return out


def convert(source: pathlib.Path, target: pathlib.Path, arm: str, *, replace: bool = False) -> None:
    """Writes the EEF dataset `target` from the collected dataset `source`.

    @param replace Rebuild `target` if it already holds a dataset (one this script wrote earlier).
    """
    from dream_ik.solver import ChunkIK
    from dream_ik.specs import SERVE

    info = json.loads((source / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"{source} is not a LeRobot v2.1 dataset")
    widths = {key: info["features"][key]["shape"][0] for key in ("observation.state", "action")}
    if widths != {"observation.state": 14, "action": 14}:
        raise SystemExit(f"{source}: expected the 14-wide collected layout, found {widths}")
    if target.exists():
        if not replace:
            raise SystemExit(f"{target} already exists; pass --replace to rebuild it, or pick another --out")
        if not (target / "meta" / "info.json").is_file():
            raise SystemExit(f"{target} exists but is not a dataset; not replacing it")
        shutil.rmtree(target)
    embodiment = json.loads(EMBODIMENT.read_text())
    kinematics, grip_scale = ARMS[arm]
    solver = ChunkIK(kinematics, SERVE.tool, float(info["fps"]))

    def fk(joints: np.ndarray, side: str) -> np.ndarray:
        return np.asarray(solver.fk(joints, side), dtype=np.float64)

    shutil.copytree(source, target, copy_function=_link_or_copy, ignore=shutil.ignore_patterns("*.parquet"))
    stats_rows = [json.loads(line) for line in (source / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    stats_by_episode = {row["episode_index"]: row for row in stats_rows}
    for parquet in sorted(source.glob("data/chunk-*/episode_*.parquet")):
        table = pq.read_table(parquet)
        out = {}
        for key, part in (("observation.state", "state"), ("action", "action")):
            vectors = np.asarray(table.column(key).to_pylist(), dtype=np.float32)
            out[key] = pack(to_eef(vectors, fk, grip_scale), embodiment[part])
        for key, values in out.items():
            index = table.schema.get_field_index(key)
            column = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])
            table = table.set_column(index, key, column)
        destination = target / parquet.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
        episode = int(table.column("episode_index")[0].as_py())
        for key, values in out.items():
            stats_by_episode[episode]["stats"][key] = _stats(values)
    with (target / "meta" / "episodes_stats.jsonl").open("w") as file:
        for episode in sorted(stats_by_episode):
            file.write(json.dumps(stats_by_episode[episode]) + "\n")
    for key, part in (("observation.state", "state"), ("action", "action")):
        names = [f"{name}_{i}" for name, (start, end) in embodiment[part].items() for i in range(end - start)]
        info["features"][key] = {"dtype": "float32", "shape": [len(names)], "names": names}
    (target / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values = values.astype(np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _link_or_copy(source: str, target: str) -> None:
    """Hard-links a video where the disk allows it, and copies every other file.

    Only the videos are shared: they are never written again. The metadata files are rewritten in the new dataset, and
    a hard link would rewrite the collected dataset's copy with them.
    """
    if source.endswith(".mp4"):
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=pathlib.Path, help="the collected dataset folder")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--out", type=pathlib.Path, help="the new dataset folder (default: <dataset>-eef)")
    parser.add_argument("--replace", action="store_true", help="rebuild the output folder if it already exists")
    args = parser.parse_args()
    source = args.dataset.expanduser().resolve()
    target = (args.out or source.with_name(f"{source.name}-eef")).expanduser().resolve()
    if not (source / "meta" / "info.json").is_file():
        sys.exit(
            f"{source}: no episode has been saved here yet. dhk-collect writes the dataset when its first episode is "
            "saved, and a --dry-run session never records. In a live session press A to open the arms, then A to "
            "start an episode and A again to save it (the terminal prints 'saved episode N'); Ctrl+C discards an "
            "episode that is still recording."
        )
    convert(source, target, args.arm, replace=args.replace)

    import dream

    report = dream.datasets.validate_dataset(target, json.loads(EMBODIMENT.read_text()))
    if not report.ok:
        sys.exit(f"{target} was written, but Dream's checks refuse it: {report.error}")
    print(f"wrote {target}: {report.episodes} episodes, {report.frames} frames; Dream's dataset checks pass")


if __name__ == "__main__":
    main()
