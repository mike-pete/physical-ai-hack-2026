"""Copies a Metal or Maker Arm dataset with its left arm frozen: still in the vectors and in what the cameras show.

For datasets recorded with one arm moved by hand as the leader (as `scripts/mirror_collect.py` records them): the left
arm's measured and commanded joints then show a person's motion, and its wrist camera and the scene's left part show
the person moving it. A model trained on that moves its left arm too. This copy holds the left arm at one reference
frame instead, so the model learns a left arm that stays put:

- the left arm's six joints and gripper, in `observation.state` and `action`, are the reference frame's state;
- the left wrist video is the reference frame's image, in every frame;
- the scene video's columns left of `--split-x` are the reference frame's, the rest is the episode's own.

Pick a reference frame with the left arm where it will rest when the model runs (its zero pose, say) and nobody's
hands on it, from a session with the scene camera where it was for the episodes you freeze: the frozen and the live
parts of the scene meet at `--split-x`, and must line up. Choose `--split-x` so that the task never crosses into the
frozen part. Run it in the kit's environment, on the collected dataset, then `tools/add_eef.py` on its output:

```
uv run python tools/freeze_left_arm.py ~/hackathon_data/cables ~/hackathon_data/cables-leftfrozen \
    --reference ~/hackathon_data/cables --episode 2 --frame 0 --split-x 310
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/cables-leftfrozen --arm metal
```

It never changes the input: every rewritten file is a new file, so a hard link to the input (as `tools/add_eef.py`
makes) is never written through. The rewritten videos are H.264 at the dataset's fps, and each changed feature's
per-episode statistics are computed again.
"""

import argparse
import fractions
import json
import os
import pathlib
import shutil
import sys
from collections.abc import Callable

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from dream_hackathon import layouts

FAMILY = "makermods"
## The left arm's slots in the recorded vectors: MakerMods datasets hold the left arm first, six joints then gripper.
LEFT = slice(0, layouts.JOINTS + 1)
SCENE = layouts.video_keys(FAMILY)["overhead"]
LEFT_WRIST = layouts.video_keys(FAMILY)["wrist_l"]
## Every how many frames a rewritten video's statistics sample one, as the collected statistics do.
STATS_EVERY = 10
## The rewritten videos' H.264 quality (lower is better; 18 is close to visually lossless).
CRF = 18


def episode_path(root: pathlib.Path, info: dict, episode: int, video_key: str | None = None) -> pathlib.Path:
    """A file of one episode, from the dataset's own path templates.

    @param root The dataset folder.
    @param info Its `meta/info.json`.
    @param episode The episode index.
    @param video_key A video feature's key for its video, or None for the episode's parquet file.
    @return The path.
    """
    chunk = episode // info["chunks_size"]
    if video_key is None:
        return root / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    return root / info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=episode)


def read_frame(path: pathlib.Path, index: int) -> npt.NDArray[np.uint8]:
    """One frame of a video, RGB."""
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"{path} has no frame {index}")


def keyframe_interval(path: pathlib.Path) -> int:
    """The source video's keyframe spacing, so a rewrite seeks the way the original did."""
    with av.open(str(path)) as container:
        keys = [i for i, packet in enumerate(container.demux(video=0)) if packet.size and packet.is_keyframe]
    return int(np.median(np.diff(keys))) if len(keys) > 1 else 250


def rewrite_video(
    source: pathlib.Path,
    target: pathlib.Path,
    fps: int,
    frame_fn: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
) -> tuple[int, list[npt.NDArray[np.uint8]]]:
    """Writes `target` as `source` with every frame passed through `frame_fn`, one output frame per input frame.

    @return The frame count and every #STATS_EVERY-th output frame, for its statistics.
    """
    rate = fractions.Fraction(fps, 1)
    samples = []
    count = 0
    temporary = target.with_name(target.name + ".tmp.mp4")
    with av.open(str(source)) as reader, av.open(str(temporary), "w") as writer:
        stream_in = reader.streams.video[0]
        stream = writer.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = stream_in.width, stream_in.height, "yuv420p"
        stream.codec_context.time_base = 1 / rate
        stream.options = {"crf": str(CRF), "g": str(keyframe_interval(source))}
        for count, frame in enumerate(reader.decode(stream_in), start=1):
            image = frame_fn(frame.to_ndarray(format="rgb24"))
            if (count - 1) % STATS_EVERY == 0:
                samples.append(image)
            out = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            out.pts, out.time_base = count - 1, 1 / rate
            for packet in stream.encode(out):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    target.unlink(missing_ok=True)  # a hard link to the input goes, the input stays
    os.replace(temporary, target)
    return count, samples


def image_stats(images: list[npt.NDArray[np.uint8]]) -> dict:
    """Per-channel statistics of frames scaled to 0..1, in the layout `meta/episodes_stats.jsonl` holds them."""
    x = np.stack(images).astype(np.float32) / 255.0
    channel = lambda values: [[[float(v)]] for v in values]  # noqa: E731
    return {
        "min": channel(x.min(axis=(0, 1, 2))),
        "max": channel(x.max(axis=(0, 1, 2))),
        "mean": channel(x.mean(axis=(0, 1, 2))),
        "std": channel(x.std(axis=(0, 1, 2))),
        "count": [len(images)],
    }


def vector_stats(values: npt.NDArray[np.float32]) -> dict:
    """Per-slot statistics of a vector feature over one episode."""
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def freeze(
    source: pathlib.Path, target: pathlib.Path, reference: pathlib.Path, episode: int, frame: int, split_x: int
) -> npt.NDArray[np.float32]:
    """Writes `target`, a copy of `source` with the left arm frozen at `reference`'s episode `episode`, frame `frame`.

    @return The left arm's frozen values: six joints and the gripper, in the dataset's units.
    @throws SystemExit `target` exists, `source` is not a MakerMods layout, or a rewritten video lost frames.
    """
    if target.exists():
        sys.exit(f"{target} exists: remove it or name another folder")
    info = json.loads((source / "meta" / "info.json").read_text())
    # A YAM dataset is 14-wide too, but right arm first and with an `overhead` camera instead of the `scene` one.
    features = info["features"]
    if features["action"]["shape"] != [layouts.WIDTH] or SCENE not in features or LEFT_WRIST not in features:
        sys.exit(f"{source} is not a collected Metal or Maker Arm dataset (14-wide, with {SCENE} and {LEFT_WRIST})")
    reference_info = json.loads((reference / "meta" / "info.json").read_text())
    reference_state = pq.read_table(episode_path(reference, reference_info, episode), columns=["observation.state"])
    left = np.asarray(reference_state.column(0).to_pylist(), np.float32)[frame, LEFT]
    scene_reference = read_frame(episode_path(reference, reference_info, episode, SCENE), frame)
    wrist_reference = read_frame(episode_path(reference, reference_info, episode, LEFT_WRIST), frame)
    if not 0 < split_x <= scene_reference.shape[1]:
        sys.exit(f"--split-x must be within the scene's width, 1 to {scene_reference.shape[1]}")

    def frozen_scene(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        image = image.copy()
        image[:, :split_x] = scene_reference[:, :split_x]
        return image

    shutil.copytree(source, target)
    stats_file = target / "meta" / "episodes_stats.jsonl"
    entries = [json.loads(line) for line in stats_file.read_text().splitlines() if line.strip()]
    for entry in entries:
        index = entry["episode_index"]
        parquet = episode_path(target, info, index)
        table = pq.read_table(parquet)
        for name in ("observation.state", "action"):
            values = np.asarray(table.column(name).to_pylist(), np.float32)
            values[:, LEFT] = left
            column = table.schema.get_field_index(name)
            field = table.schema.field(column)
            table = table.set_column(column, field, pa.array(list(values), type=field.type))
            entry["stats"][name] = vector_stats(values)
        temporary = parquet.with_name(parquet.name + ".tmp")
        pq.write_table(table, temporary)
        parquet.unlink()
        os.replace(temporary, parquet)
        for key, frame_fn in ((SCENE, frozen_scene), (LEFT_WRIST, lambda _image: wrist_reference)):
            count, samples = rewrite_video(
                episode_path(source, info, index, key), episode_path(target, info, index, key), info["fps"], frame_fn
            )
            if count != table.num_rows:
                sys.exit(f"episode {index}: {key} has {count} frames for {table.num_rows} rows")
            entry["stats"][key] = image_stats(samples)
        print(f"episode {index}: {table.num_rows} frames frozen", flush=True)
    stats_file.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return left


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=pathlib.Path, help="the collected dataset (not its -eef copy)")
    parser.add_argument("target", type=pathlib.Path, help="the new dataset folder to write")
    parser.add_argument("--reference", type=pathlib.Path, help="the dataset holding the reference frame (the source)")
    parser.add_argument("--episode", type=int, required=True, help="the reference frame's episode index")
    parser.add_argument("--frame", type=int, required=True, help="the reference frame's index in its episode")
    parser.add_argument("--split-x", type=int, required=True, help="scene columns left of this are frozen")
    args = parser.parse_args()
    source = args.source.expanduser()
    reference = (args.reference or args.source).expanduser()
    left = freeze(source, args.target.expanduser(), reference, args.episode, args.frame, args.split_x)
    print(f"left arm held at {np.round(left, 4).tolist()} (six joints in radians, then the gripper)")
    print(f"next: ~/dream-env/bin/python tools/add_eef.py {args.target} --arm <metal|maker>")


if __name__ == "__main__":
    main()
