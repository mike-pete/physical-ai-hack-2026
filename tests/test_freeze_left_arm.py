"""tools/freeze_left_arm.py: the left arm held at one frame in the vectors and the videos, the rest untouched."""

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import write_episodes

from dream_hackathon import layouts

av = pytest.importorskip("av", reason="needs PyAV (the makermods extra)")

_spec = importlib.util.spec_from_file_location(
    "freeze_left_arm", pathlib.Path(__file__).resolve().parents[1] / "tools" / "freeze_left_arm.py"
)
freeze_left_arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze_left_arm)

SCENE = freeze_left_arm.SCENE
LEFT_WRIST = freeze_left_arm.LEFT_WRIST
RIGHT_WRIST = layouts.video_keys("makermods")["wrist_r"]
FRAMES = 60


def frames(path: pathlib.Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)]).astype(np.int16)


def vectors(root: pathlib.Path, episode: int, name: str) -> np.ndarray:
    table = pq.read_table(root / f"data/chunk-000/episode_{episode:06d}.parquet", columns=[name])
    return np.asarray(table.column(0).to_pylist(), np.float32)


def digest(root: pathlib.Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


def video(root: pathlib.Path, key: str, episode: int) -> pathlib.Path:
    return root / f"videos/chunk-000/{key}/episode_{episode:06d}.mp4"


def test_freezes_the_left_arm(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "task"
    write_episodes(source, "makermods", episodes=2, frames=FRAMES, tasks=["unplug the cable"])
    before = digest(source)
    target = tmp_path / "task-leftfrozen"
    split, ref_episode, ref_frame = 100, 1, 5

    left = freeze_left_arm.freeze(source, target, source, ref_episode, ref_frame, split)

    assert digest(source) == before  # the input is left exactly as it was
    np.testing.assert_allclose(left, vectors(source, ref_episode, "observation.state")[ref_frame, :7])
    scene_ref = frames(video(source, SCENE, ref_episode))[ref_frame]
    wrist_ref = frames(video(source, LEFT_WRIST, ref_episode))[ref_frame]
    for episode in range(2):
        for name in ("observation.state", "action"):
            frozen, original = vectors(target, episode, name), vectors(source, episode, name)
            np.testing.assert_allclose(frozen[:, :7], np.broadcast_to(left, (FRAMES, 7)))
            np.testing.assert_array_equal(frozen[:, 7:], original[:, 7:])
        scene, scene_original = frames(video(target, SCENE, episode)), frames(video(source, SCENE, episode))
        wrist = frames(video(target, LEFT_WRIST, episode))
        assert len(scene) == len(wrist) == FRAMES
        # Lossy re-encoding: within a few levels of the intended pixels.
        assert np.abs(scene[:, :, : split - 8] - scene_ref[:, : split - 8]).max() <= 6
        assert np.abs(scene[:, :, split + 8 :] - scene_original[:, :, split + 8 :]).max() <= 6
        assert np.abs(wrist - wrist_ref).max() <= 6
        # The right wrist is not rewritten.
        assert (target / video(target, RIGHT_WRIST, episode).relative_to(target)).read_bytes() == video(
            source, RIGHT_WRIST, episode
        ).read_bytes()

    stats = [json.loads(line) for line in (target / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert stats[0]["stats"]["action"]["std"][:7] == pytest.approx([0.0] * 7, abs=1e-6)
    assert stats[0]["stats"][LEFT_WRIST]["std"][0][0][0] == pytest.approx(0.0, abs=0.01)
    assert stats[0]["stats"][LEFT_WRIST]["count"] == [FRAMES // freeze_left_arm.STATS_EVERY]


def test_refuses_an_existing_target_or_a_yam_dataset(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "task"
    write_episodes(source, "makermods", episodes=1, frames=FRAMES, tasks=["t"])
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        freeze_left_arm.freeze(source, tmp_path / "taken", source, 0, 0, 10)
    yam = tmp_path / "yam"
    write_episodes(yam, "yam", episodes=1, frames=FRAMES, tasks=["t"])
    with pytest.raises(SystemExit, match="not a collected Metal or Maker Arm dataset"):
        freeze_left_arm.freeze(yam, tmp_path / "out", yam, 0, 0, 10)
