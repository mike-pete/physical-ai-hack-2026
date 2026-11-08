"""tools/critic_eval.py: finding episodes, cutting pictures, asking (never for real here) and the report maths."""

import argparse
import csv
import importlib.util
import json
import pathlib
import sys

import av
import numpy as np
import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
TASK_FILE = pathlib.Path(__file__).resolve().parents[1] / "configs" / "tasks" / "usb-cables.yaml"
BASELINE = "Unplug each orange USB cable from the power strip and put them on the table. Go right to left"


@pytest.fixture
def ce(monkeypatch: pytest.MonkeyPatch):
    """The tool's module, loaded the way `python tools/critic_eval.py` does: its folder is on the import path."""
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("critic_eval_tool", TOOLS / "critic_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tc(ce):
    return sys.modules["task_critic"]


@pytest.fixture
def spec(tc):
    return tc.load_task(TASK_FILE)


def make_video(path: pathlib.Path, shades: list[int], size: tuple[int, int] = (320, 180), fps: int = 30) -> None:
    """A small video whose frame i is a flat grey of `shades[i]`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for shade in shades:
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), shade, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def make_rollout(root: pathlib.Path, name: str, task: str = BASELINE, shades=None, status: str = "complete"):
    folder = root / name
    folder.mkdir(parents=True)
    info = {"status": status, "task": task, "checkpoint": "dreamlabs/dream-1-beta", "cameras": ["scene", "wrist_l"]}
    (folder / "rollout.json").write_text(json.dumps(info))
    make_video(
        folder / "videos/chunk-000/observation.images.scene/episode_000000.mp4", shades or list(range(0, 120, 2))
    )
    return folder


def make_demos(root: pathlib.Path, episodes: int = 2) -> pathlib.Path:
    dataset = root / "unplug-cables"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({"fps": 30}))
    lines = [json.dumps({"episode_index": i, "tasks": ["pick up", BASELINE], "length": 90}) for i in range(episodes)]
    (dataset / "meta/episodes.jsonl").write_text("\n".join(lines))
    for i in range(episodes):
        for camera in ("scene", "left_wrist"):
            make_video(
                dataset / f"videos/chunk-000/observation.images.{camera}/episode_{i:06d}.mp4", list(range(10, 100))
            )
    return dataset


# --- finding the episodes ------------------------------------------------------------------------------------


def test_spread_includes_the_first_and_last_frame(ce) -> None:
    assert ce.spread(31, 5, "end") == [0, 8, 15, 22, 30]
    assert ce.spread(31, 1, "end") == [30]
    assert ce.spread(31, 1, "start") == [0]
    assert ce.spread(2, 5, "end") == [0, 1]  # fewer frames than asked for: each once
    assert ce.spread(0, 5, "end") == []


def test_read_frames_takes_the_last_second_or_the_first(ce, tmp_path) -> None:
    video = tmp_path / "v.mp4"
    make_video(video, list(range(0, 200, 2)))  # 100 frames, 3.3 s, shade rising
    end = ce.read_frames(video, "end", 3, 1.0)
    start = ce.read_frames(video, "start", 3, 1.0)
    assert len(end) == len(start) == 3
    shades_end = [int(f.mean()) for f in end]
    shades_start = [int(f.mean()) for f in start]
    assert shades_end == sorted(shades_end) and shades_end[-1] >= 190  # the very last frame is in it
    assert shades_start[0] <= 4 and shades_start[-1] <= 62  # the first second only
    assert shades_end[0] >= 130  # and the end picks do not reach back before the last second


def test_read_frames_names_a_bad_video(ce, tmp_path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ce.EvalError, match="cannot be read"):
        ce.read_frames(broken, "end", 1, 1.0)


def test_discover_rollouts(ce, spec, tmp_path) -> None:
    make_rollout(tmp_path, "2026-11-08T10-00-00_run1")
    make_rollout(tmp_path, "2026-11-08T10-05-00_run1", task="Unplug the orange USB cables.", status="partial")
    samples = ce.discover(tmp_path, spec)
    assert [s.episode for s in samples] == ["2026-11-08T10-00-00_run1", "2026-11-08T10-05-00_run1"]
    first = samples[0]
    assert (first.source, first.at, first.prompt, first.status) == ("rollout", "end", BASELINE, "complete")
    assert list(first.videos) == ["scene"]  # wrist_l has no video in this fixture
    assert first.auto_label is None
    assert ce.prompt_id_for(spec, first.prompt) == "baseline"
    assert ce.prompt_id_for(spec, "UNPLUG THE ORANGE USB CABLES") == "short"
    assert ce.prompt_id_for(spec, "something else") is None


def test_discover_a_single_episode_folder(ce, spec, tmp_path) -> None:
    folder = make_rollout(tmp_path, "run")
    assert [s.id for s in ce.discover(folder, spec)] == ["run"]


def test_discover_demonstrations_gives_a_start_and_an_end_with_labels(ce, spec, tmp_path) -> None:
    samples = ce.discover(make_demos(tmp_path), spec)
    assert [s.id for s in samples] == [
        "unplug-cables-ep000000-start",
        "unplug-cables-ep000000-end",
        "unplug-cables-ep000001-start",
        "unplug-cables-ep000001-end",
    ]
    start, end = samples[0], samples[1]
    assert start.auto_label == {"success": False, "unplugged_count": 0}
    assert end.auto_label == {"success": True, "unplugged_count": spec.cables}
    assert set(start.videos) == {"scene", "wrist_l"}  # left_wrist is renamed as a station names it
    assert end.prompt == BASELINE  # the last of the episode's tasks


def test_discover_refuses_a_folder_it_does_not_know(ce, spec, tmp_path) -> None:
    with pytest.raises(ce.EvalError, match="neither rollout episodes"):
        ce.discover(tmp_path, spec)
    with pytest.raises(ce.EvalError, match="not a folder"):
        ce.discover(tmp_path / "missing", spec)


# --- labels --------------------------------------------------------------------------------------------------


def write_labels(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["episode", "prompt_id", "unplugged_count", "all_on_table", "success", "failure_mode"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_labels_can_leave_success_to_the_counts(ce, spec, tmp_path) -> None:
    path = tmp_path / "labels.csv"
    write_labels(
        path,
        [
            {"episode": "a", "unplugged_count": "4", "all_on_table": "y"},  # every cable out, all on the table
            {"episode": "b", "unplugged_count": "4", "all_on_table": "n"},  # out, but one fell
            {"episode": "c", "unplugged_count": "1", "all_on_table": "y", "failure_mode": "stalled_or_idle"},
            {"episode": "d", "success": "yes", "unplugged_count": "4"},  # said so directly
            {"episode": "e"},  # not labelled yet
        ],
    )
    labels = ce.load_labels(path, spec)
    assert [labels[k].success for k in "abcde"] == [True, False, False, True, None]
    assert labels["c"].failure_mode == "stalled_or_idle" and labels["c"].unplugged == 1
    assert labels["e"].unplugged is None


def test_a_bad_labels_file_says_where(ce, spec, tmp_path) -> None:
    path = tmp_path / "labels.csv"
    write_labels(path, [{"episode": "a", "unplugged_count": "three"}])
    with pytest.raises(ce.EvalError, match="line 2.*three"):
        ce.load_labels(path, spec)
    path.write_text("name,success\nx,y\n")
    with pytest.raises(ce.EvalError, match="`episode` column"):
        ce.load_labels(path, spec)


def args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_the_labels_template_names_each_episode_and_will_not_overwrite(ce, tmp_path) -> None:
    make_rollout(tmp_path / "eps", "run1")
    make_rollout(
        tmp_path / "eps",
        "run2",
        task="One orange cable is still plugged in on the left. Unplug it and put it on the table.",
    )
    out = tmp_path / "labels.csv"
    common = {"task": TASK_FILE, "source": tmp_path / "eps", "out": out, "force": False}
    assert ce.command_labels_template(args(**common)) == 0
    rows = list(csv.DictReader(out.open()))
    assert [r["episode"] for r in rows] == ["run1", "run2"]
    assert [r["prompt_id"] for r in rows] == ["baseline", "corrective"]
    assert [r["scene_start"] for r in rows] == ["full", "partial"]
    assert rows[0]["success"] == "" and rows[0]["status"] == "complete"
    with pytest.raises(ce.EvalError, match="--force"):
        ce.command_labels_template(args(**common))
    assert ce.command_labels_template(args(**{**common, "force": True})) == 0


def test_the_labels_template_is_not_for_demonstrations(ce, tmp_path) -> None:
    make_demos(tmp_path)
    with pytest.raises(ce.EvalError, match="need no labels"):
        ce.command_labels_template(
            args(task=TASK_FILE, source=tmp_path / "unplug-cables", out=tmp_path / "l.csv", force=False)
        )


# --- cutting pictures ----------------------------------------------------------------------------------------


def frames_args(source, out, **kwargs):
    defaults = {"count": 3, "window_s": 1.0, "limit": None, "skip_face_check": False, "only": None}
    return args(task=TASK_FILE, source=source, out=out, **{**defaults, **kwargs})


def test_frames_can_cut_only_the_end_of_each_demonstration(ce, tmp_path) -> None:
    make_demos(tmp_path)
    out = tmp_path / "eval"
    ce.command_frames(frames_args(tmp_path / "unplug-cables", out, only="end"))
    index = json.loads((out / "frames/index.json").read_text())
    assert [e["sample"] for e in index] == ["unplug-cables-ep000000-end", "unplug-cables-ep000001-end"]
    assert {e["at"] for e in index} == {"end"}
    assert not list((out / "frames").glob("*-start"))  # no picture of a start was cut, so none can be sent


def test_frames_writes_cropped_pictures_and_an_index(ce, tc, spec, tmp_path) -> None:
    make_rollout(tmp_path / "eps", "run1")
    out = tmp_path / "eval"
    assert ce.command_frames(frames_args(tmp_path / "eps", out)) == 0
    index = json.loads((out / "frames/index.json").read_text())
    assert len(index) == 1 and index[0]["blocked"] is None and index[0]["prompt"] == BASELINE
    assert len(index[0]["sets"]) == 3
    picture = out / index[0]["sets"][0]["scene"]
    assert picture.exists()
    height, width = ce.decode_jpeg(picture.read_bytes()).shape[:2]
    assert min(height, width) >= 280  # cropped to the table, then scaled into what TypeSafe accepts
    assert width / height == pytest.approx((0.6 * 320) / (0.58 * 180), rel=0.03)


def test_frames_drops_a_picture_set_with_a_face_and_blocks_the_sample(ce, tc, tmp_path, monkeypatch) -> None:
    make_rollout(tmp_path / "eps", "run1")
    monkeypatch.setattr(tc, "count_faces", lambda image: 1)
    out = tmp_path / "eval"
    ce.command_frames(frames_args(tmp_path / "eps", out))
    entry = json.loads((out / "frames/index.json").read_text())[0]
    assert entry["sets"] == [] and entry["faces_skipped"] == 3
    assert "face" in entry["blocked"]
    assert not list((out / "frames").rglob("*.jpg"))  # nothing with a face was ever written


def test_frames_blocks_a_sample_without_the_scene_camera(ce, tmp_path) -> None:
    folder = make_rollout(tmp_path / "eps", "run1")
    (folder / "videos/chunk-000/observation.images.scene/episode_000000.mp4").unlink()
    out = tmp_path / "eval"
    ce.command_frames(frames_args(tmp_path / "eps", out))
    assert "no video for camera" in json.loads((out / "frames/index.json").read_text())[0]["blocked"]


# --- asking --------------------------------------------------------------------------------------------------


def run_args(out, **kwargs):
    defaults = {
        "task": TASK_FILE,
        "out": out,
        "variant": "typesafe",
        "gate": "typesafe",
        "rewriter": "claude",
        "no_rewrite_unsure": False,
        "dry": True,
        "yes": False,
        "limit": None,
    }
    return args(**{**defaults, **kwargs})


@pytest.fixture
def cut(ce, tmp_path):
    """An evaluation folder with the pictures of two demonstrations cut and indexed."""
    make_demos(tmp_path)
    out = tmp_path / "eval"
    ce.command_frames(frames_args(tmp_path / "unplug-cables", out))
    return out


def test_a_dry_run_sends_nothing_and_logs_the_requests(ce, cut, monkeypatch) -> None:
    def refuse(*a, **k):
        raise AssertionError("a dry run must not touch the network")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert ce.command_run(run_args(cut)) == 0
    verdicts = [json.loads(line) for line in (cut / "typesafe/verdicts.jsonl").read_text().splitlines()]
    assert len(verdicts) == 4
    assert all(v["verdict"]["result"]["status"] == "unsure" for v in verdicts)  # a 50/50 stand-in
    requests = [json.loads(line) for line in (cut / "typesafe/requests.jsonl").read_text().splitlines()]
    assert {r["sample"] for r in requests} == {v["sample"] for v in verdicts}
    text = (cut / "typesafe/requests.jsonl").read_text()
    assert 'data_base64": "<' in text  # the payloads are kept without their pictures
    assert "questions" in text


def test_a_second_run_continues_where_the_first_stopped(ce, cut, capsys) -> None:
    ce.command_run(run_args(cut, limit=1))
    assert len((cut / "typesafe/verdicts.jsonl").read_text().splitlines()) == 1
    ce.command_run(run_args(cut))
    assert len((cut / "typesafe/verdicts.jsonl").read_text().splitlines()) == 4
    capsys.readouterr()
    assert ce.command_run(run_args(cut)) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_sending_for_real_asks_first_and_needs_a_person(ce, cut, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    def refuse(*a, **k):
        raise AssertionError("nothing may be sent before the question is answered")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(ce.EvalError, match="rerun with --yes"):
        ce.command_run(run_args(cut, dry=False))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    with pytest.raises(ce.EvalError, match="not sent"):
        ce.command_run(run_args(cut, dry=False))
    assert not (cut / "typesafe/verdicts.jsonl").exists()


def test_the_confirmation_says_what_leaves_the_computer(ce, cut, capsys, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    ce.confirm_send(run_args(cut, dry=False, yes=True), samples=4, sets=20, images=20)
    said = capsys.readouterr().out
    assert "api.typesafe.ai" in said and "api.anthropic.com" in said
    assert "TYPESAFE_API_KEY" in said and "20 cropped pictures" in said
    assert str(cut / "frames") in said
    ce.confirm_send(run_args(cut, dry=False, yes=True, rewriter="none"), samples=4, sets=20, images=20)
    assert "api.anthropic.com" not in capsys.readouterr().out


def test_a_failed_call_is_recorded_as_an_error_not_as_an_answer(ce, tc, cut, monkeypatch) -> None:
    calls = []

    class FlakyGate:
        def ask(self, spec, images):
            calls.append(1)
            if len(calls) == 2:
                raise tc.CriticError("TypeSafe kept failing (503)")
            return tc.GateAnswer(
                nouls={"none_plugged_in": 0.05, "none_off_the_table": 0.05, "strip_visible": 0.9, "failure_fits": 0.9},
                scores={"plugged_in_count": 0.0},
                choices={"failure_type": {"not_grasped": 1.0}},
            )

    def build(arguments, task_spec, log):
        return tc.TaskCritic(task_spec, FlakyGate(), None)

    monkeypatch.setattr(ce, "build_critic", build)
    monkeypatch.setattr(ce, "confirm_send", lambda *a, **k: None)
    # Each sample has three picture sets, so the second call fails inside the first sample.
    assert ce.command_run(run_args(cut, dry=False, yes=True)) == 1
    errors = [json.loads(line) for line in (cut / "typesafe/errors.jsonl").read_text().splitlines()]
    assert len(errors) == 1 and "503" in errors[0]["error"]
    done = (cut / "typesafe/verdicts.jsonl").read_text().splitlines()
    assert errors[0]["sample"] not in {json.loads(line)["sample"] for line in done}
    assert len(done) == 3  # the other samples carried on


# --- the report ----------------------------------------------------------------------------------------------


def row(ce, **kwargs):
    base = {
        "variant": "v",
        "sample": "s",
        "episode": "e",
        "prompt": BASELINE,
        "prompt_id": "baseline",
        "scene_start": None,
        "success": True,
        "auto": False,
        "unplugged": 3,
        "failure_mode": None,
        "status": "done",
        "status_last": "done",
        "p_done": 0.9,
        "p_done_last": 0.9,
        "visible": 0.9,
        "plugged_in": 0.0,
        "predicted_failure": None,
        "rewritten": None,
        "rewrite_source": None,
        "hidden_frames": 0,
        "total_frames": 5,
        "frame": "frames/x/scene_4.jpg",
    }
    return ce.Row(**{**base, **kwargs})


def test_success_by_prompt_comes_from_the_labels_alone(ce) -> None:
    rows = [
        row(ce, prompt_id="baseline", success=True, unplugged=3),
        row(ce, prompt_id="baseline", success=False, unplugged=1),
        row(ce, prompt_id="short", success=False, unplugged=0),
        row(ce, prompt_id="short", success=None, unplugged=None),  # not labelled: left out
        row(ce, prompt_id=None, success=True, unplugged=3),
    ]
    by_prompt = {p["prompt_id"]: p for p in ce.success_by_prompt(rows)}
    assert by_prompt["baseline"] == {
        "prompt_id": "baseline",
        "episodes": 2,
        "successes": 1,
        "success_rate": 0.5,
        "mean_unplugged": 2.0,
    }
    assert by_prompt["short"]["episodes"] == 1 and by_prompt["short"]["success_rate"] == 0.0
    assert "(other)" in by_prompt


def test_decision_summary_counts_the_two_costly_mistakes(ce) -> None:
    rows = [
        row(ce, success=True, status="done"),
        row(ce, success=True, status="not_done"),  # a retry at a finished scene
        row(ce, success=True, status="unsure"),
        row(ce, success=False, status="done"),  # stopped early
        row(ce, success=False, status="not_done"),
        row(ce, success=False, status="not_done"),
        row(ce, success=None, status="done"),  # not labelled: left out
    ]
    d = ce.decision_summary(rows)
    assert d["labelled"] == 6
    assert (d["false_done"], d["false_done_rate"]) == (1, pytest.approx(1 / 3))
    assert (d["false_not_done"], d["false_not_done_rate"]) == (1, pytest.approx(1 / 3))
    assert (d["unsure"], d["unsure_rate"]) == (1, pytest.approx(1 / 6))
    assert d["decided_accuracy"] == pytest.approx(3 / 5)  # done+success, not_done+failure x2 are right
    assert d["confusion"]["success"] == {"done": 1, "not_done": 1, "unsure": 1}
    last = ce.decision_summary([row(ce, success=False, status="not_done", status_last="done")], "status_last")
    assert last["false_done"] == 1


def test_a_decision_summary_with_nothing_labelled_is_empty_not_a_crash(ce) -> None:
    d = ce.decision_summary([row(ce, success=None)])
    assert d["labelled"] == 0 and d["false_done_rate"] is None and d["decided_accuracy"] is None


def test_count_error_reads_the_plugged_in_score(ce) -> None:
    rows = [
        row(ce, unplugged=3, plugged_in=0.2),  # 0 left plugged in = 3 unplugged: exact
        row(ce, unplugged=1, plugged_in=0.9),  # 1 left = 2 unplugged: off by one
        row(ce, unplugged=0, plugged_in=3.4),  # clamped to 3 = 0 unplugged: exact
        row(ce, unplugged=None, plugged_in=1.0),  # not labelled
        row(ce, unplugged=2, plugged_in=None),  # no answer
    ]
    c = ce.count_error(rows, cables=3)
    assert c["n"] == 3 and c["mean_abs_error"] == pytest.approx(1 / 3) and c["exact"] == pytest.approx(2 / 3)


def test_calibration_buckets_include_one(ce) -> None:
    rows = [
        row(ce, p_done=0.05, success=False),
        row(ce, p_done=1.0, success=True),
        row(ce, p_done=0.85, success=False),
    ]
    buckets = {b["bucket"]: b for b in ce.calibration(rows)}
    assert buckets["0.0-0.2"] == {"bucket": "0.0-0.2", "n": 1, "success_rate": 0.0}
    assert buckets["0.8-1.0"]["n"] == 2 and buckets["0.8-1.0"]["success_rate"] == 0.5
    assert buckets["0.4-0.6"]["success_rate"] is None


def test_failure_type_accuracy_only_looks_at_labelled_failures(ce) -> None:
    rows = [
        row(ce, success=False, failure_mode="dropped", predicted_failure="dropped"),
        row(ce, success=False, failure_mode="dropped", predicted_failure="not_grasped"),
        row(ce, success=False, failure_mode="stalled_or_idle", predicted_failure=None),
        row(ce, success=False, failure_mode=None, predicted_failure="dropped"),  # no label for the type
        row(ce, success=True),
    ]
    f = ce.failure_type_accuracy(rows)
    assert f["n"] == 3 and f["accuracy"] == pytest.approx(1 / 3) and f["none_chosen"] == 1
    assert f["pairs"][("dropped", "not_grasped")] == 1 and f["pairs"][("stalled_or_idle", "(none fits)")] == 1


def test_occlusion_is_the_share_of_pictures_with_a_hidden_strip(ce) -> None:
    rows = [row(ce, hidden_frames=1, total_frames=5), row(ce, hidden_frames=4, total_frames=5)]
    assert ce.occlusion(rows) == {"frames": 10, "hidden_rate": 0.5}
    assert ce.occlusion([])["hidden_rate"] is None


def test_the_review_sheet_holds_what_failed(ce) -> None:
    rows = [
        row(ce, sample="ok", success=True),
        row(ce, sample="bad", success=False, status="not_done", rewritten="Pick it up.", rewrite_source="claude"),
        row(ce, sample="new", success=None, status="not_done"),
        row(ce, sample="new-ok", success=None, status="done"),
    ]
    review = ce.review_rows(rows)
    assert [r["sample"] for r in review] == ["bad", "new"]
    assert review[0]["rewritten_prompt"] == "Pick it up." and review[0]["score_1_to_5"] == ""


def verdict_record(sample: str, status: str = "done", p_done: float = 0.9, visible: float = 0.9, **extra):
    judgement = {
        "status": status,
        "p_done": p_done,
        "satisfied": {},
        "visible": visible,
        "plugged_in": extra.pop("plugged_in", 0.0),
        "failure_probabilities": None,
        "failure_fit": None,
        "failure_type": extra.pop("failure_type", None),
    }
    return {
        "sample": sample,
        "verdict": {
            "result": judgement,
            "frames": [judgement, {**judgement, "visible": 0.1}],
            "rewritten_prompt": extra.pop("rewritten", None),
            "rewrite_source": extra.pop("rewrite_source", None),
        },
    }


def test_build_rows_joins_labels_demonstration_labels_and_verdicts(ce, spec) -> None:
    index = [
        {
            "sample": "run1",
            "episode": "run1",
            "at": "end",
            "prompt": BASELINE,
            "auto_label": None,
            "sets": [{"scene": "frames/run1/scene_0.jpg"}],
        },
        {
            "sample": "demo-start",
            "episode": "demo",
            "at": "start",
            "prompt": None,
            "auto_label": {"success": False, "unplugged_count": 0},
            "sets": [],
        },
        {"sample": "unanswered", "episode": "x", "prompt": None, "auto_label": None, "sets": []},
    ]
    verdicts = {
        "run1": verdict_record(
            "run1", "not_done", 0.1, failure_type="dropped", rewritten="Pick up the cable.", rewrite_source="claude"
        ),
        "demo-start": verdict_record("demo-start", "not_done", 0.05),
    }
    labels = {"run1": ce.Label(False, 2, "dropped", None, "full")}
    rows = {r.sample: r for r in ce.build_rows(spec, "v", index, verdicts, labels)}
    assert set(rows) == {"run1", "demo-start"}  # a sample with no verdict is not a row
    run1 = rows["run1"]
    assert (run1.success, run1.unplugged, run1.failure_mode, run1.prompt_id, run1.scene_start) == (
        False,
        2,
        "dropped",
        "baseline",
        "full",
    )
    assert (run1.status, run1.predicted_failure, run1.rewritten) == ("not_done", "dropped", "Pick up the cable.")
    assert (run1.hidden_frames, run1.total_frames, run1.auto) == (1, 2, False)  # one of two pictures hid the strip
    assert run1.frame == "frames/run1/scene_0.jpg"
    demo = rows["demo-start"]
    assert (demo.success, demo.unplugged, demo.auto) == (False, 0, True)


def test_the_report_is_written_for_one_variant_and_for_two(ce, cut) -> None:
    ce.command_run(run_args(cut))
    ce.command_run(run_args(cut, variant="other"))
    common = {"task": TASK_FILE, "out": cut, "labels": None}
    assert ce.command_report(args(**common, variants="typesafe")) == 0
    report = (cut / "report.md").read_text()
    assert "## Variant `typesafe` (4 samples)" in report and "teleop demonstrations" in report
    assert "## Comparison" not in report
    assert ce.command_report(args(**common, variants="typesafe,other")) == 0
    assert "## Comparison" in (cut / "report.md").read_text()
    rows = list(csv.DictReader((cut / "report.csv").open()))
    assert len(rows) == 8 and {r["variant"] for r in rows} == {"typesafe", "other"}
    assert (cut / "review.csv").exists()


def test_the_report_names_a_missing_step(ce, tmp_path) -> None:
    with pytest.raises(ce.EvalError, match="run `frames` first"):
        ce.command_report(args(task=TASK_FILE, out=tmp_path, labels=None, variants="typesafe"))
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/index.json").write_text("[]")
    with pytest.raises(ce.EvalError, match="run --variant typesafe"):
        ce.command_report(args(task=TASK_FILE, out=tmp_path, labels=None, variants="typesafe"))


def test_the_command_line_exits_with_a_plain_message(ce, tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        ce.main(["--task", str(TASK_FILE), "run", "--out", str(tmp_path), "--dry"])
    assert "critic_eval: " in str(error.value) and "run `frames` first" in str(error.value)


def test_your_labels_beat_a_demonstrations_automatic_ones_at_its_end_only(ce, spec) -> None:
    """Episode 4 did not finish: the person's label wins at its end, and its start stays "not done"."""
    index = [
        {
            "sample": "d4-start",
            "episode": "d4",
            "at": "start",
            "prompt": None,
            "sets": [],
            "auto_label": {"success": False, "unplugged_count": 0},
        },
        {
            "sample": "d4-end",
            "episode": "d4",
            "at": "end",
            "prompt": None,
            "sets": [],
            "auto_label": {"success": True, "unplugged_count": spec.cables},
        },
        {
            "sample": "d5-end",
            "episode": "d5",
            "at": "end",
            "prompt": None,
            "sets": [],
            "auto_label": {"success": True, "unplugged_count": spec.cables},
        },
    ]
    verdicts = {s["sample"]: verdict_record(s["sample"]) for s in index}
    labels = {"d4": ce.Label(False, 2, None, None, None)}  # d5 has no label: its automatic one stands
    rows = {r.sample: r for r in ce.build_rows(spec, "v", index, verdicts, labels)}
    assert (rows["d4-end"].success, rows["d4-end"].unplugged, rows["d4-end"].auto) == (False, 2, False)
    assert (rows["d4-start"].success, rows["d4-start"].unplugged, rows["d4-start"].auto) == (False, 0, True)
    assert (rows["d5-end"].success, rows["d5-end"].unplugged, rows["d5-end"].auto) == (True, spec.cables, True)


# --- the Gemini vote gate through critic_eval ----------------------------------------------------------------


def test_frames_also_cut_the_zoomed_strip(ce, tmp_path) -> None:
    make_rollout(tmp_path / "eps", "run1")
    out = tmp_path / "eval"
    ce.command_frames(frames_args(tmp_path / "eps", out))
    one = json.loads((out / "frames/index.json").read_text())[0]["sets"][0]
    assert set(one) == {"scene", "strip"}
    assert (out / one["scene"]).exists()
    assert max(ce.decode_jpeg((out / one["strip"]).read_bytes()).shape[:2]) == 600  # the task file's max_side


def test_the_face_check_looks_at_the_cameras_not_the_zoom(ce, tc, tmp_path, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(tc, "count_faces", lambda image: seen.append(image.shape) or 0)
    make_rollout(tmp_path / "eps", "run1")
    ce.command_frames(frames_args(tmp_path / "eps", tmp_path / "eval"))
    assert len(seen) == 3  # one camera crop for each of the three picture sets, and none for the enlarged strips


def test_a_dry_run_of_the_gemini_gate_sends_one_picture_a_call(ce, cut, monkeypatch) -> None:
    def refuse(*a, **k):
        raise AssertionError("a dry run must not touch the network")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert ce.command_run(run_args(cut, gate="gemini", variant="gemini")) == 0
    verdicts = [json.loads(line) for line in (cut / "gemini/verdicts.jsonl").read_text().splitlines()]
    assert len(verdicts) == 4
    for record in verdicts:
        result = record["verdict"]["result"]
        assert result["status"] == "unsure" and len(result["votes"]) == 6  # three pictures, two votes each
        assert result["votes"][0]["answer"] == "unsure"
    requests = [json.loads(line)["request"] for line in (cut / "gemini/requests.jsonl").read_text().splitlines()]
    asked = [r for r in requests if "contents" in r]
    assert len(asked) == 4 * 3 * 2
    for request in asked:
        picture, question = request["contents"][0]["parts"]  # one picture and the question: never the scene crop too
        assert picture["inline_data"]["data"].startswith("<") and "Four orange USB cables" in question["text"]


def test_the_confirmation_for_gemini_says_what_is_sent_and_checks_the_key(ce, cut, capsys, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gemini = run_args(cut, gate="gemini", dry=False, yes=True, rewriter="none")
    with pytest.raises(ce.EvalError, match="GEMINI_API_KEY"):
        ce.confirm_send(gemini, samples=4, sets=12, images=12, calls=24)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ce.confirm_send(gemini, samples=4, sets=12, images=12, calls=24)
    said = capsys.readouterr().out
    assert "generativelanguage.googleapis.com" in said and "gemini-4-argon" in said and "GEMINI_API_KEY" in said
    assert "12 cropped pictures" in said and "24 calls" in said
    assert "api.anthropic.com" not in said and "api.typesafe.ai" not in said


def test_the_gemini_gate_needs_a_vote_section_in_the_task_file(ce, cut, tmp_path) -> None:
    text = TASK_FILE.read_text()
    bare = tmp_path / "bare.yaml"
    bare.write_text(text[: text.index("# The vote gate")] + text[text.index("# A yes/no answer reads") :])
    with pytest.raises(ce.EvalError, match="no vote_gate section"):
        ce.command_run(run_args(cut, gate="gemini", variant="g", task=bare))


def test_a_dry_run_with_the_gemini_rewriter_gives_it_both_pictures(ce, cut, monkeypatch) -> None:
    def refuse(*a, **k):
        raise AssertionError("a dry run must not touch the network")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert ce.command_run(run_args(cut, gate="gemini", rewriter="gemini", variant="gr")) == 0
    verdicts = [json.loads(line)["verdict"] for line in (cut / "gr/verdicts.jsonl").read_text().splitlines()]
    assert len(verdicts) == 4
    assert all(v["rewrite_source"] == "gemini" and v["rewritten_prompt"] for v in verdicts)  # unsure scenes get one
    requests = [json.loads(line)["request"] for line in (cut / "gr/requests.jsonl").read_text().splitlines()]
    writes = [r for r in requests if r["generationConfig"]["maxOutputTokens"] == 4096]
    assert len(writes) == 4  # one per sample
    for request in writes:
        parts = request["contents"][0]["parts"]
        assert len([p for p in parts if "inline_data" in p]) == 2  # the scene crop and the zoom
        assert "dry run: nothing was sent" in parts[-1]["text"]  # the vote reasons are the hint


def test_the_confirmation_names_gemini_as_the_rewriter_and_its_key(ce, cut, capsys, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    args = run_args(cut, gate="typesafe", rewriter="gemini", dry=False, yes=True)
    with pytest.raises(ce.EvalError, match="GEMINI_API_KEY"):
        ce.confirm_send(args, samples=4, sets=12, images=12)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ce.confirm_send(args, samples=4, sets=12, images=12)
    said = capsys.readouterr().out
    assert "Gemini (generativelanguage.googleapis.com" in said and "to write the next prompt" in said
    assert "api.anthropic.com" not in said


@pytest.mark.parametrize(
    ("gate", "rewriter", "gate_class", "rewriter_class"),
    [
        ("typesafe", "claude", "TypeSafeGate", "ClaudeRewriter"),
        ("claude", "none", "ClaudeGate", None),
        ("gemini", "gemini", "GeminiVoteGate", "GeminiRewriter"),
        ("gemini", "claude", "GeminiVoteGate", "ClaudeRewriter"),
        ("typesafe", "gemini", "TypeSafeGate", "GeminiRewriter"),
    ],
)
def test_the_critic_is_built_from_the_chosen_backends(
    ce, spec, monkeypatch, gate, rewriter, gate_class, rewriter_class
):
    for name in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "k")
    args = argparse.Namespace(dry=False, gate=gate, rewriter=rewriter, no_rewrite_unsure=False, model="gemini-test")
    critic = ce.build_critic(args, spec, lambda payload: None)
    assert type(critic.gate).__name__ == gate_class
    assert (type(critic.rewriter).__name__ if critic.rewriter else None) == rewriter_class
    if gate == "gemini":
        assert critic.gate.model == "gemini-test"  # --model reaches the Gemini gate
    if rewriter == "gemini":
        assert critic.rewriter.model == "gemini-test"
