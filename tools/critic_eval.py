"""Measures how well the task critic (tools/task_critic.py) judges recorded robot episodes.

The workflow, in four steps. Run it in the repo's environment (`uv run`), not ~/dream-env.

```
# 1. After the recording runs (docs/critic_eval.md): a labels sheet to fill in, one row per episode.
uv run python tools/critic_eval.py labels-template ~/hackathon_data/critic-eval/cables --out labels.csv
# 2. Cut the pictures the critic will see, cropped as the task file says. LOOK at them. Nothing is sent except, when
#    TYPESAFE_API_KEY is set, the crops to TypeSafe for a face check (it asks first; --face-check opencv avoids it).
uv run python tools/critic_eval.py frames ~/hackathon_data/critic-eval/cables --out eval
# 3. Ask the critic. This sends the cropped pictures to TypeSafe (and Claude, for rewritten prompts): it asks first.
uv run python tools/critic_eval.py run --out eval --variant typesafe
# 4. Compare with the labels: report.md, report.csv and review.csv in the out folder.
uv run python tools/critic_eval.py report --out eval --labels labels.csv --variants typesafe
```

`run --dry` builds every request and sends nothing (a stand-in answers 50/50), to check the plumbing and read the
exact payloads in `eval/<variant>/requests.jsonl`. The same commands take the teleop demonstrations of the task
(`~/hackathon_data/unplug-cables`): each demonstration gives two pictures, its start (cables plugged in: not done) and
its end (done), which need no labels sheet.

What goes outside this computer is only what `frames` wrote: the crop of each camera's picture named in the task
file (the scene camera also sees the people around the table). `frames` runs two face checks over every crop and
drops the picture set when either finds one: OpenCV's detector (local, finds few faces) and, when TYPESAFE_API_KEY is
set, a TypeSafe question (finds far more; it sends the crops to TypeSafe, after asking). Both are safety nets, so look
at the crops yourself before `run`.
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import pathlib
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import av
import cv2
import numpy as np
import task_critic as tc

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TASK = REPO / "configs" / "tasks" / "usb-cables.yaml"
## A station names its wrist cameras wrist_l and wrist_r; the teleop demonstrations name them left_wrist and
## right_wrist.
CAMERA_ALIASES = {"left_wrist": "wrist_l", "right_wrist": "wrist_r"}
VIDEO_DIR = "videos/chunk-000"
LABEL_COLUMNS = [
    "episode",
    "prompt_id",
    "prompt",
    "status",
    "scene_start",
    "unplugged_count",
    "all_on_table",
    "success",
    "failure_mode",
    "notes",
]
CALIBRATION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class EvalError(RuntimeError):
    """A folder is not what the tool expects, or a step's input is missing; the message says what to do."""


# --- Finding the episodes --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Sample:
    """One thing to judge: the end of a recorded run, or the start or end of a demonstration."""

    id: str
    episode: str
    source: str  # "rollout" or "demo"
    at: str  # "end" or "start": which part of the video the pictures come from
    videos: dict[str, pathlib.Path]  # camera name -> its video
    prompt: str | None
    status: str | None  # a rollout's own status: complete, partial, empty
    auto_label: dict[str, Any] | None  # a demonstration needs no labels sheet: its start is not done, its end is


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _rollout_video(root: pathlib.Path, camera: str) -> pathlib.Path | None:
    base = root / VIDEO_DIR / f"observation.images.{camera}"
    for name in ("episode_000000.mp4", "episode_000000.partial.mp4"):
        if (base / name).exists():
            return base / name
    return None


def discover(root: pathlib.Path, spec: tc.TaskSpec) -> list[Sample]:
    """The samples in a folder of rollout episodes (each with a `rollout.json`), or in a LeRobot v2.1 dataset.

    @throws EvalError The folder holds neither.
    """
    root = root.expanduser()
    if not root.is_dir():
        raise EvalError(f"{root} is not a folder")
    folders = (
        [root]
        if (root / "rollout.json").exists()
        else sorted(p for p in root.iterdir() if p.is_dir() and (p / "rollout.json").exists())
    )
    if folders:
        samples = []
        for folder in folders:
            info = json.loads((folder / "rollout.json").read_text(encoding="utf-8"))
            videos: dict[str, pathlib.Path] = {}
            for camera in info.get("cameras", spec.cameras):
                video = _rollout_video(folder, camera)
                if video is not None:
                    videos[CAMERA_ALIASES.get(camera, camera)] = video
            samples.append(
                Sample(folder.name, folder.name, "rollout", "end", videos, info.get("task"), info.get("status"), None)
            )
        return samples
    if (root / "meta" / "info.json").exists():
        return _demo_samples(root, spec)
    raise EvalError(
        f"{root} holds neither rollout episodes (folders with a rollout.json) nor a LeRobot dataset (meta/info.json)"
    )


def _demo_samples(root: pathlib.Path, spec: tc.TaskSpec) -> list[Sample]:
    cameras = {
        CAMERA_ALIASES.get(d.name.removeprefix("observation.images."), d.name.removeprefix("observation.images.")): d
        for d in sorted((root / VIDEO_DIR).glob("observation.images.*"))
    }
    tasks: dict[int, str] = {}
    episodes_file = root / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        for line in episodes_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if entry.get("tasks"):
                    tasks[int(entry["episode_index"])] = str(entry["tasks"][-1])
    indexes = sorted(
        {
            int(m.group(1))
            for d in cameras.values()
            for f in d.glob("episode_*.mp4")
            if (m := re.match(r"episode_(\d+)", f.name))
        }
    )
    samples = []
    for index in indexes:
        videos = {
            name: d / f"episode_{index:06d}.mp4"
            for name, d in cameras.items()
            if (d / f"episode_{index:06d}.mp4").exists()
        }
        episode = f"{_safe(root.name)}-ep{index:06d}"
        for at, success, unplugged in (("start", False, 0), ("end", True, spec.cables)):
            samples.append(
                Sample(
                    f"{episode}-{at}",
                    episode,
                    "demo",
                    at,
                    videos,
                    tasks.get(index),
                    None,
                    {"success": success, "unplugged_count": unplugged},
                )
            )
    return samples


# --- Cutting pictures out of the video -------------------------------------------------------------------------


def spread(n: int, count: int, at: str) -> list[int]:
    """`count` indexes spread evenly over `n` frames, including the last (at the end) or the first (at the start)."""
    if n <= 0:
        return []
    if count <= 1:
        return [n - 1 if at == "end" else 0]
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


def read_frames(path: pathlib.Path, at: str, count: int, window_s: float) -> list[np.ndarray]:
    """`count` frames from the last (`at="end"`) or first second of a video, as OpenCV (BGR) images, oldest first.

    @throws EvalError The video cannot be read or has no frames.
    """
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 30)
            keep = max(count, math.ceil(window_s * fps) + 1)
            frames: deque[Any] | list[Any] = deque(maxlen=keep) if at == "end" else []
            for frame in container.decode(stream):
                frames.append(frame)
                if at == "start" and len(frames) >= keep:
                    break
            frames = list(frames)
            if not frames:
                raise EvalError(f"{path} has no frames")
            return [frames[i].to_ndarray(format="bgr24") for i in spread(len(frames), count, at)]
    except av.error.FFmpegError as error:
        raise EvalError(f"{path} cannot be read: {error}") from None


def decode_jpeg(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def face_check_mode(args: argparse.Namespace) -> str:
    """Which face checks `frames` runs: "none", "opencv" (local), "typesafe" or "both" (opencv, then typesafe).

    `auto` is "both" when TYPESAFE_API_KEY is set, and "opencv" when it is not.
    """
    if args.skip_face_check:
        return "none"
    if args.face_check == "auto":
        return "both" if os.environ.get("TYPESAFE_API_KEY") else "opencv"
    return args.face_check


def confirm_face_check(args: argparse.Namespace, pictures: int) -> None:
    """Says what the second face check sends and asks; refuses without a person to ask unless --yes."""
    after_opencv = " and none that OpenCV's detector already flagged" if face_check_mode(args) == "both" else ""
    print(
        f"About to send up to {pictures} cropped pictures outside this computer, to check them for faces:\n"
        f"  TypeSafe (api.typesafe.ai, model {tc.TYPESAFE_MODEL}): one question per picture, "
        "'is a person's face visible?'; key from TYPESAFE_API_KEY\n"
        f"Only the cameras' crops go, never the zoomed pictures{after_opencv}."
    )
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise EvalError("no one to ask: rerun with --yes to send the crops, or --face-check opencv to keep them here")
    if input("Send? [y/N] ").strip().lower() not in ("y", "yes"):
        raise EvalError("not sent")


def command_frames(args: argparse.Namespace) -> int:
    spec = tc.load_task(args.task)
    samples = [s for s in discover(args.source, spec) if args.only in (None, s.at)]
    if args.limit:
        samples = samples[: args.limit]
    mode = face_check_mode(args)
    second: tc.TypeSafeFaceCheck | None = None
    if mode in ("typesafe", "both"):
        second = tc.TypeSafeFaceCheck(threshold=args.face_threshold)  # raises if TYPESAFE_API_KEY is not set
        confirm_face_check(args, len(samples) * args.count * len(spec.cameras))
    print(
        {
            "none": "face check: OFF (--skip-face-check)",
            "opencv": "face check: OpenCV's detector only, nothing leaves this computer "
            "(set TYPESAFE_API_KEY for a second check by TypeSafe, which finds far more faces)",
            "typesafe": "face check: TypeSafe only",
            "both": "face check: OpenCV's detector, then TypeSafe on the crops it lets through",
        }[mode]
    )
    folder = args.out / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    totals = Counter()
    for sample in samples:
        entry: dict[str, Any] = {
            "sample": sample.id,
            "episode": sample.episode,
            "source": sample.source,
            "at": sample.at,
            "prompt": sample.prompt,
            "status": sample.status,
            "auto_label": sample.auto_label,
            "sets": [],
            "faces_skipped": 0,
            "face_check": mode,
            "blocked": None,
        }
        missing = [c for c in spec.cameras if c not in sample.videos]
        if missing:
            entry["blocked"] = f"no video for camera {missing}"
        else:
            raw = {c: read_frames(sample.videos[c], sample.at, args.count, args.window_s) for c in spec.cameras}
            # The face checks run on the cameras' crops. A zoom lies inside its camera's crop (the task file checks
            # that), so it shows nothing the crop does not, and a detector run on a few enlarged pixels only finds
            # faces that are not there.
            kept: list[tuple[int, dict[str, bytes]]] = []
            for k in range(min(len(v) for v in raw.values())):
                prepared = {c: tc.prepare_image(raw[c][k], spec.crop.get(c)) for c in spec.cameras}
                if mode in ("opencv", "both") and any(tc.count_faces(decode_jpeg(j)) for j in prepared.values()):
                    entry["faces_skipped"] += 1
                    totals["faces_opencv"] += 1
                    continue
                kept.append((k, prepared))
            if second is not None and kept:
                # One batch for the sample, before anything is written: if the call fails nothing unchecked is kept.
                try:
                    flagged = second.flags([jpeg for _, prepared in kept for jpeg in prepared.values()])
                except tc.CriticError as error:
                    raise EvalError(f"the TypeSafe face check failed, so {sample.id} was not cut: {error}") from None
                checked: list[tuple[int, dict[str, bytes]]] = []
                position = 0
                for k, prepared in kept:
                    hits = flagged[position : position + len(prepared)]
                    position += len(prepared)
                    if any(hits):
                        entry["faces_skipped"] += 1
                        totals["faces_typesafe"] += 1
                    else:
                        checked.append((k, prepared))
                kept = checked
            for k, prepared in kept:
                for name, view in spec.views.items():
                    prepared[name] = tc.prepare_image(raw[view.camera][k], view.box, target_side=view.max_side)
                paths = {}
                for camera, jpeg in prepared.items():
                    path = folder / _safe(sample.id) / f"{camera}_{k}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(jpeg)
                    paths[camera] = str(path.relative_to(args.out))
                entry["sets"].append(paths)
            if not entry["sets"]:
                entry["blocked"] = "a face was found in every picture set" if entry["faces_skipped"] else "no frames"
        totals["sets"] += len(entry["sets"])
        totals["faces"] += entry["faces_skipped"]
        totals["blocked"] += entry["blocked"] is not None
        index.append(entry)
    (folder / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(
        f"{len(index)} samples, {totals['sets']} picture sets written under {folder}\n"
        f"  {totals['faces']} sets dropped because a face was found "
        f"({totals['faces_opencv']} by OpenCV, {totals['faces_typesafe']} by TypeSafe), "
        f"{totals['blocked']} samples blocked\n"
        "Look at the pictures before `run`: they are exactly what will be sent."
    )
    return 0


# --- Labels ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Label:
    success: bool | None
    unplugged: int | None
    failure_mode: str | None
    prompt_id: str | None
    scene_start: str | None


def _flag(text: str | None) -> bool | None:
    value = (text or "").strip().lower()
    if value in ("y", "yes", "true", "1"):
        return True
    if value in ("n", "no", "false", "0"):
        return False
    return None


def load_labels(path: pathlib.Path, spec: tc.TaskSpec) -> dict[str, Label]:
    """Reads labels.csv. `success` may be left blank when `unplugged_count` and `all_on_table` are filled in: it is
    then every cable unplugged and all on the table.

    @throws EvalError The file lacks the `episode` column or has an unreadable number.
    """
    labels: dict[str, Label] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "episode" not in (reader.fieldnames or []):
            raise EvalError(f"{path} needs an `episode` column (see `labels-template`)")
        for number, row in enumerate(reader, 2):
            count_text = (row.get("unplugged_count") or "").strip()
            try:
                unplugged = int(count_text) if count_text else None
            except ValueError:
                raise EvalError(
                    f"{path} line {number}: unplugged_count {count_text!r} is not a whole number"
                ) from None
            success = _flag(row.get("success"))
            on_table = _flag(row.get("all_on_table"))
            if success is None and unplugged is not None and on_table is not None:
                success = unplugged >= spec.cables and on_table
            labels[row["episode"].strip()] = Label(
                success,
                unplugged,
                (row.get("failure_mode") or "").strip() or None,
                (row.get("prompt_id") or "").strip() or None,
                (row.get("scene_start") or "").strip() or None,
            )
    return labels


def prompt_id_for(spec: tc.TaskSpec, prompt: str | None) -> str | None:
    """The id of the task file's matrix prompt with this text, ignoring case and punctuation."""
    if prompt is None:
        return None
    return next((p.id for p in spec.prompts if tc.already_tried(prompt, [p.text])), None)


def command_labels_template(args: argparse.Namespace) -> int:
    spec = tc.load_task(args.task)
    if args.out.exists() and not args.force:
        raise EvalError(f"{args.out} exists; it holds labels. Pass --force to overwrite it")
    samples = [s for s in discover(args.source, spec) if s.source == "rollout"]
    if not samples:
        raise EvalError(f"{args.source} holds no rollout episodes: demonstrations need no labels sheet")
    scene_start = {p.id: p.scene_start for p in spec.prompts}
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, LABEL_COLUMNS)
        writer.writeheader()
        for sample in samples:
            pid = prompt_id_for(spec, sample.prompt)
            writer.writerow(
                {
                    "episode": sample.episode,
                    "prompt_id": pid or "",
                    "prompt": sample.prompt or "",
                    "status": sample.status or "",
                    "scene_start": scene_start.get(pid, "") if pid else "",
                }
            )
    print(
        f"{len(samples)} episodes in {args.out}. For each, fill in unplugged_count (of {spec.cables}), all_on_table "
        "(y/n) and, if it failed, failure_mode:\n  " + ", ".join(t.name for t in spec.failure_types)
    )
    return 0


# --- Asking the critic -----------------------------------------------------------------------------------------


def _fake_answers(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """TypeSafe-shaped answers for a dry run: every yes/no 50/50, the score 0, every option equally likely."""
    answers: dict[str, Any] = {}
    for key, q in questions.items():
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.5}
        elif q["type"] == "score":
            answers[key] = {"type": "score", "score": 0.0}
        else:
            options = list(q["criteria"])
            answers[key] = {
                "type": "choice",
                "choice": options[0],
                "probabilities": {name: 1 / len(options) for name in options},
            }
    return answers


def _gemini_reply(text: str) -> dict[str, Any]:
    """A Gemini-shaped reply carrying `text`, for a dry run."""
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def build_critic(args: argparse.Namespace, spec: tc.TaskSpec, log: Callable[[dict[str, Any]], None]) -> tc.TaskCritic:
    """The critic for `run`: real backends (keys from the environment), or stand-ins that send nothing for --dry."""
    model = getattr(args, "model", None) or tc.GEMINI_MODEL
    gate: tc.Gate | tc.VoteGate
    rewriter: tc.Rewriter | None
    if args.dry:
        if args.gate == "gemini":
            key = spec.vote_gate.answer_key if spec.vote_gate else "answer"
            stand_in = _gemini_reply(json.dumps({key: "unsure", "reason": "dry run: nothing was sent"}))
            gate = tc.GeminiVoteGate("dry", post=lambda url, headers, body: stand_in, log=log)
        else:
            gate = tc.TypeSafeGate(
                "dry", post=lambda url, headers, body: {"answers": _fake_answers(body["questions"])}, log=log
            )
        if args.rewriter == "none":
            rewriter = None
        elif args.rewriter == "gemini":
            written = _gemini_reply("(dry run: nothing was written)")
            rewriter = tc.GeminiRewriter("dry", post=lambda url, headers, body: written, log=log)
        else:
            written = {"content": [{"type": "text", "text": "(dry run: nothing was written)"}]}
            rewriter = tc.ClaudeRewriter("dry", post=lambda url, headers, body: written, log=log)
    else:
        if args.gate == "gemini":
            gate = tc.GeminiVoteGate(model=model, log=log)
        elif args.gate == "claude":
            gate = tc.ClaudeGate(log=log)
        else:
            gate = tc.TypeSafeGate(log=log)
        if args.rewriter == "none":
            rewriter = None
        elif args.rewriter == "gemini":
            rewriter = tc.GeminiRewriter(model=model, log=log)
        else:
            rewriter = tc.ClaudeRewriter(log=log)
    return tc.TaskCritic(spec, gate, rewriter, rewrite_when_unsure=not args.no_rewrite_unsure)


def confirm_send(args: argparse.Namespace, samples: int, sets: int, images: int, calls: int | None = None) -> None:
    """Says what is about to leave this computer and asks; refuses without a person to ask unless --yes.

    @param images The pictures the gate looks at in all (a picture set may hold more than it uses).
    @param calls For the vote gate, how many calls that is: each picture is asked about several times.
    """
    if args.gate == "gemini":
        model = getattr(args, "model", None) or tc.GEMINI_MODEL
        gate = f"Gemini (generativelanguage.googleapis.com, model {model})"
        what = f"the task's one question about each picture, {calls} calls in all"
        gate_key = "GEMINI_API_KEY"
    elif args.gate == "claude":
        gate, what, gate_key = "Claude", "the yes/no questions about each set", "ANTHROPIC_API_KEY"
    else:
        gate = f"TypeSafe (api.typesafe.ai, model {tc.TYPESAFE_MODEL})"
        what, gate_key = "the yes/no questions about each set", "TYPESAFE_API_KEY"
    rewriter_key = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(args.rewriter)
    needed = list(dict.fromkeys([gate_key, *([rewriter_key] if rewriter_key else [])]))
    missing = [name for name in needed if not os.environ.get(name)]
    if missing:
        raise EvalError(
            f"set {' and '.join(missing)} in the environment first (type it with `read -rs`, not in a file)"
        )
    lines = [
        f"About to send {images} cropped pictures ({sets} picture sets of {samples} samples) outside this computer:",
        f"  {gate}: {what}; key from {gate_key}",
    ]
    if args.rewriter == "claude":
        lines.append(f"  Claude (api.anthropic.com, model {tc.ANTHROPIC_MODEL}): one picture set per failed sample")
    elif args.rewriter == "gemini":
        model = getattr(args, "model", None) or tc.GEMINI_MODEL
        lines.append(
            f"  Gemini (generativelanguage.googleapis.com, model {model}): the scene crop and the zoomed strip of "
            "each not-done or unsure sample, to write the next prompt"
        )
    lines += [
        f"The pictures are the files under {args.out / 'frames'}, nothing else is sent.",
        f"The payloads without the pictures are kept in {args.out / args.variant / 'requests.jsonl'}.",
    ]
    print("\n".join(lines))
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise EvalError("no one to ask: look at the pictures, then rerun with --yes to send them")
    if input("Send? [y/N] ").strip().lower() not in ("y", "yes"):
        raise EvalError("not sent")


def command_run(args: argparse.Namespace) -> int:
    spec = tc.load_task(args.task)
    index_path = args.out / "frames" / "index.json"
    if not index_path.exists():
        raise EvalError(f"{index_path} is missing: run `frames` first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    folder = args.out / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    verdicts_path = folder / "verdicts.jsonl"
    done = (
        {json.loads(line)["sample"] for line in verdicts_path.read_text().splitlines() if line.strip()}
        if verdicts_path.exists()
        else set()
    )
    todo = [e for e in index if not e["blocked"] and e["sets"] and e["sample"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do: every sample already has a verdict (delete verdicts.jsonl to redo)")
        return 0
    sets = sum(len(e["sets"]) for e in todo)
    calls = None
    if args.gate == "gemini":
        if spec.vote_gate is None:
            raise EvalError(f"the task file {args.task} has no vote_gate section: the Gemini gate has nothing to ask")
        images, calls = sets, sets * spec.vote_gate.repeats
    else:
        images = sets * len(spec.cameras)  # a question gate sees the cameras' crops, not the zoomed views
    if not args.dry:
        confirm_send(args, len(todo), sets, images, calls)
    current = {"sample": None}
    requests_file = (folder / "requests.jsonl").open("a", encoding="utf-8")

    def log(payload: dict[str, Any]) -> None:
        requests_file.write(json.dumps({"sample": current["sample"], "request": payload}) + "\n")
        requests_file.flush()

    critic = build_critic(args, spec, log)
    errors = 0
    with (
        requests_file,
        verdicts_path.open("a", encoding="utf-8") as out,
        (folder / "errors.jsonl").open("a", encoding="utf-8") as bad,
    ):
        for number, entry in enumerate(todo, 1):
            current["sample"] = entry["sample"]
            image_sets = [{c: (args.out / p).read_bytes() for c, p in one.items()} for one in entry["sets"]]
            history = [entry["prompt"]] if entry.get("prompt") else []
            try:
                verdict = critic.evaluate(image_sets, history)
            except tc.CriticError as error:
                errors += 1
                bad.write(json.dumps({"sample": entry["sample"], "error": str(error)}) + "\n")
                bad.flush()
                print(f"[{number}/{len(todo)}] {entry['sample']}: FAILED: {error}")
                continue
            record = {"sample": entry["sample"], "variant": args.variant, "verdict": dataclasses.asdict(verdict)}
            out.write(json.dumps(record) + "\n")
            out.flush()
            result = verdict.result
            print(f"[{number}/{len(todo)}] {entry['sample']}: {result.status} (p_done {result.p_done:.2f})")
    if errors:
        print(f"{errors} samples failed (errors.jsonl); a failed call is never an answer. Rerun to retry them.")
    return 1 if errors else 0


# --- The report ------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Row:
    """One sample: what the person labelled and what the critic said."""

    variant: str
    sample: str
    episode: str
    prompt: str | None
    prompt_id: str | None
    scene_start: str | None
    success: bool | None
    auto: bool  # the label came from a demonstration's start or end, not from a person
    unplugged: int | None
    failure_mode: str | None
    status: str  # from all the picture sets together
    status_last: str  # from the final picture alone
    p_done: float
    p_done_last: float
    visible: float
    plugged_in: float | None
    predicted_failure: str | None
    rewritten: str | None
    rewrite_source: str | None
    hidden_frames: int
    total_frames: int
    frame: str | None


def build_rows(
    spec: tc.TaskSpec,
    variant: str,
    index: Sequence[Mapping[str, Any]],
    verdicts: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Label],
) -> list[Row]:
    """Joins the index, a variant's verdicts and the labels. A demonstration's label comes from its index entry; a
    rollout's from labels.csv, matched on the episode's folder name."""
    rows = []
    for entry in index:
        record = verdicts.get(entry["sample"])
        if record is None:
            continue
        verdict = record["verdict"]
        result, last = verdict["result"], verdict["frames"][-1]
        # A person's label describes how the episode ENDED, so it applies to the end sample only. It wins over the
        # automatic label of a demonstration; a start sample keeps its automatic one (not done, nothing unplugged).
        label = labels.get(entry["episode"]) if entry.get("at") == "end" else None
        auto = entry.get("auto_label")
        success = label.success if label and label.success is not None else (auto["success"] if auto else None)
        unplugged = (
            label.unplugged if label and label.unplugged is not None else (auto["unplugged_count"] if auto else None)
        )
        by_hand = bool(label and label.success is not None)
        prompt_id = (label.prompt_id if label else None) or prompt_id_for(spec, entry.get("prompt"))
        rows.append(
            Row(
                variant=variant,
                sample=entry["sample"],
                episode=entry["episode"],
                prompt=entry.get("prompt"),
                prompt_id=prompt_id,
                scene_start=label.scene_start if label else None,
                success=success,
                auto=bool(auto) and not by_hand,
                unplugged=unplugged,
                failure_mode=label.failure_mode if label else None,
                status=result["status"],
                status_last=last["status"],
                p_done=result["p_done"],
                p_done_last=last["p_done"],
                visible=result["visible"],
                plugged_in=result["plugged_in"],
                predicted_failure=result["failure_type"],
                rewritten=verdict["rewritten_prompt"],
                rewrite_source=verdict["rewrite_source"],
                hidden_frames=sum(f["visible"] < spec.visibility_min for f in verdict["frames"]),
                total_frames=len(verdict["frames"]),
                frame=entry["sets"][-1].get(spec.cameras[0]) if entry["sets"] else None,
            )
        )
    return rows


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def success_by_prompt(rows: Iterable[Row]) -> list[dict[str, Any]]:
    """How often the base model really succeeded with each prompt, from the labels alone."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.success is not None:
            groups[row.prompt_id or "(other)"].append(row)
    out = []
    for prompt_id, group in sorted(groups.items()):
        counts = [r.unplugged for r in group if r.unplugged is not None]
        out.append(
            {
                "prompt_id": prompt_id,
                "episodes": len(group),
                "successes": sum(bool(r.success) for r in group),
                "success_rate": sum(bool(r.success) for r in group) / len(group),
                "mean_unplugged": statistics.fmean(counts) if counts else None,
            }
        )
    return out


def decision_summary(rows: Iterable[Row], attribute: str = "status") -> dict[str, Any]:
    """The critic against the labels. A false "done" ends a retry loop early; a false "not done" sends a retry at a
    finished scene. `attribute` is `status` (all the picture sets) or `status_last` (the final picture alone)."""
    labelled = [r for r in rows if r.success is not None]
    failures = [r for r in labelled if not r.success]
    successes = [r for r in labelled if r.success]
    confusion = {
        "success": Counter(getattr(r, attribute) for r in successes),
        "not_success": Counter(getattr(r, attribute) for r in failures),
    }
    decided = [r for r in labelled if getattr(r, attribute) != tc.UNSURE]
    right = [r for r in decided if (getattr(r, attribute) == tc.DONE) == r.success]
    false_done = sum(getattr(r, attribute) == tc.DONE for r in failures)
    false_not_done = sum(getattr(r, attribute) == tc.NOT_DONE for r in successes)
    return {
        "labelled": len(labelled),
        "confusion": confusion,
        "false_done": false_done,
        "false_done_rate": _rate(false_done, len(failures)),
        "false_not_done": false_not_done,
        "false_not_done_rate": _rate(false_not_done, len(successes)),
        "unsure": sum(getattr(r, attribute) == tc.UNSURE for r in labelled),
        "unsure_rate": _rate(sum(getattr(r, attribute) == tc.UNSURE for r in labelled), len(labelled)),
        "decided_accuracy": _rate(len(right), len(decided)),
    }


def count_error(rows: Iterable[Row], cables: int) -> dict[str, Any]:
    """The progress answer (cables still plugged in) against the labelled count of unplugged cables."""
    pairs = [
        (cables - min(max(round(r.plugged_in), 0), cables), r.unplugged)
        for r in rows
        if r.unplugged is not None and r.plugged_in is not None
    ]
    errors = [abs(predicted - actual) for predicted, actual in pairs]
    return {
        "n": len(pairs),
        "mean_abs_error": statistics.fmean(errors) if errors else None,
        "exact": _rate(sum(e == 0 for e in errors), len(errors)),
    }


def calibration(rows: Iterable[Row]) -> list[dict[str, Any]]:
    """Whether the critic's p_done means what it says: per bucket of p_done, the share that really succeeded."""
    labelled = [r for r in rows if r.success is not None]
    out = []
    for low, high in zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:], strict=False):
        last = high == CALIBRATION_EDGES[-1]
        bucket = [r for r in labelled if low <= r.p_done < high or (last and r.p_done == high)]
        out.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "n": len(bucket),
                "success_rate": _rate(sum(bool(r.success) for r in bucket), len(bucket)),
            }
        )
    return out


def failure_type_accuracy(rows: Iterable[Row]) -> dict[str, Any]:
    """The chosen failure type against the labelled one, over failed episodes that have a label for it."""
    pool = [r for r in rows if r.success is False and r.failure_mode]
    right = sum(r.predicted_failure == r.failure_mode for r in pool)
    pairs = Counter((r.failure_mode, r.predicted_failure or "(none fits)") for r in pool)
    return {
        "n": len(pool),
        "accuracy": _rate(right, len(pool)),
        "none_chosen": sum(r.predicted_failure is None for r in pool),
        "pairs": pairs,
    }


def occlusion(rows: Iterable[Row]) -> dict[str, Any]:
    """How often an arm or hand hides the strip: the share of pictures where the strip was judged not visible."""
    rows = list(rows)
    total = sum(r.total_frames for r in rows)
    return {"frames": total, "hidden_rate": _rate(sum(r.hidden_frames for r in rows), total)}


def summarize(spec: tc.TaskSpec, rows: list[Row]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "auto_labelled": sum(r.auto for r in rows),
        "by_prompt": success_by_prompt(rows),
        "aggregate": decision_summary(rows, "status"),
        "last_frame": decision_summary(rows, "status_last"),
        "count": count_error(rows, spec.cables),
        "calibration": calibration(rows),
        "failure_type": failure_type_accuracy(rows),
        "occlusion": occlusion(rows),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_report(spec: tc.TaskSpec, summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """The Markdown report: one section per variant, then a comparison when there are several."""
    lines = [
        f"# Task critic report: {spec.name}",
        "",
        f"Thresholds: yes at {spec.yes}, no at {spec.no}. "
        f'{spec.cables} cables. A false "done" ends a retry loop with the task unfinished; a false "not done" '
        "sends a retry at a finished scene.",
        "",
    ]
    for variant, s in summaries.items():
        lines += [f"## Variant `{variant}` ({s['samples']} samples)", ""]
        if s["auto_labelled"]:
            lines += [
                f"> {s['auto_labelled']} samples are teleop demonstrations labelled automatically (start: not done, "
                "end: done). They test the plumbing and the easy cases; the by-prompt table below is about "
                "base-model runs, so read it only for rollouts.",
                "",
            ]
        lines += [
            "### The base model, by prompt (from the labels)",
            "",
            "| prompt | episodes | successes | rate | mean unplugged |",
            "| --- | --- | --- | --- | --- |",
        ]
        for p in s["by_prompt"]:
            cells = [p["prompt_id"], p["episodes"], p["successes"], _pct(p["success_rate"]), _num(p["mean_unplugged"])]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
        if not s["by_prompt"]:
            lines.append("| (no labelled episodes) | | | | |")
        for title, key in (("All picture sets together", "aggregate"), ("Final picture alone", "last_frame")):
            d = s[key]
            lines += [
                "",
                f"### Critic against the labels: {title} ({d['labelled']} labelled)",
                "",
                "| label | done | not done | unsure |",
                "| --- | --- | --- | --- |",
            ]
            for label, name in (("success", "success"), ("not success", "not_success")):
                c = d["confusion"][name]
                lines.append(f"| {label} | {c[tc.DONE]} | {c[tc.NOT_DONE]} | {c[tc.UNSURE]} |")
            lines += [
                "",
                f"- false done: {d['false_done']} ({_pct(d['false_done_rate'])} of failures)",
                f"- false not done: {d['false_not_done']} ({_pct(d['false_not_done_rate'])} of successes)",
                f"- unsure: {d['unsure']} ({_pct(d['unsure_rate'])})",
                f"- accuracy when it decides: {_pct(d['decided_accuracy'])}",
            ]
        c = s["count"]
        o = s["occlusion"]
        lines += [
            "",
            "### Progress, calibration, occlusion",
            "",
            f"- unplugged count: mean absolute error {_num(c['mean_abs_error'])} cables, "
            f"exact on {_pct(c['exact'])} ({c['n']} episodes)",
            f"- strip hidden in {_pct(o['hidden_rate'])} of {o['frames']} pictures "
            "(this is what a better viewpoint could fix)",
            "",
            "| p_done | episodes | really succeeded |",
            "| --- | --- | --- |",
        ]
        lines += [f"| {b['bucket']} | {b['n']} | {_pct(b['success_rate'])} |" for b in s["calibration"]]
        f = s["failure_type"]
        lines += [
            "",
            f"### Failure type ({f['n']} labelled failures)",
            "",
            f"- right: {_pct(f['accuracy'])}; none fitted: {f['none_chosen']}",
        ]
        lines += [f"- labelled `{a}` -> critic said `{b}`: {n}" for (a, b), n in sorted(f["pairs"].items())]
        lines.append("")
    if len(summaries) > 1:
        lines += [
            "## Comparison",
            "",
            "| variant | false done | false not done | unsure | accuracy when decided | count error | strip hidden |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for variant, s in summaries.items():
            d = s["aggregate"]
            cells = [
                variant,
                _pct(d["false_done_rate"]),
                _pct(d["false_not_done_rate"]),
                _pct(d["unsure_rate"]),
                _pct(d["decided_accuracy"]),
                _num(s["count"]["mean_abs_error"]),
                _pct(s["occlusion"]["hidden_rate"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ['Read the rewritten prompts in `review.csv` and score each 1 to 5 for "would this fix it?".', ""]
    return "\n".join(lines)


def review_rows(rows: Iterable[Row]) -> list[dict[str, Any]]:
    """The failed (or, unlabelled, not-done) samples with the critic's rewritten prompt, for a person to score."""
    out = []
    for r in rows:
        if r.success is True or (r.success is None and r.status == tc.DONE):
            continue
        out.append(
            {
                "variant": r.variant,
                "sample": r.sample,
                "prompt_id": r.prompt_id or "",
                "prompt": r.prompt or "",
                "labelled_unplugged": "" if r.unplugged is None else r.unplugged,
                "labelled_failure_mode": r.failure_mode or "",
                "critic_status": r.status,
                "critic_failure_type": r.predicted_failure or "",
                "rewritten_prompt": r.rewritten or "",
                "rewrite_source": r.rewrite_source or "",
                "frame": r.frame or "",
                "score_1_to_5": "",
                "comment": "",
            }
        )
    return out


def _write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)


def command_report(args: argparse.Namespace) -> int:
    spec = tc.load_task(args.task)
    index_path = args.out / "frames" / "index.json"
    if not index_path.exists():
        raise EvalError(f"{index_path} is missing: run `frames` first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    labels = load_labels(args.labels, spec) if args.labels else {}
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    all_rows: list[Row] = []
    summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        path = args.out / variant / "verdicts.jsonl"
        if not path.exists():
            raise EvalError(f"{path} is missing: run `run --variant {variant}` first")
        verdicts = {
            r["sample"]: r for r in map(json.loads, filter(str.strip, path.read_text(encoding="utf-8").splitlines()))
        }
        rows = build_rows(spec, variant, index, verdicts, labels)
        all_rows += rows
        summaries[variant] = summarize(spec, rows)
    unlabelled = sum(r.success is None for r in all_rows)
    (args.out / "report.md").write_text(render_report(spec, summaries), encoding="utf-8")
    columns = [f.name for f in dataclasses.fields(Row)]
    _write_csv(args.out / "report.csv", [dataclasses.asdict(r) for r in all_rows], columns)
    review = review_rows(all_rows)
    _write_csv(args.out / "review.csv", review, list(review[0]) if review else ["variant", "sample"])
    print(f"wrote {args.out / 'report.md'}, report.csv and review.csv ({len(review)} samples to review)")
    if unlabelled:
        print(f"{unlabelled} samples have no label and are left out of the critic-against-labels numbers")
    return 0


# --- Command line ----------------------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    top.add_argument("--task", type=pathlib.Path, default=DEFAULT_TASK, help="the task file (default: usb-cables)")
    commands = top.add_subparsers(dest="command", required=True)

    template = commands.add_parser("labels-template", help="a labels.csv skeleton from rollout episodes")
    template.add_argument("source", type=pathlib.Path, help="the folder of rollout episodes")
    template.add_argument("--out", type=pathlib.Path, required=True, help="the labels.csv to write")
    template.add_argument("--force", action="store_true", help="overwrite an existing file")
    template.set_defaults(handler=command_labels_template)

    cut = commands.add_parser(
        "frames",
        help="cut the cropped pictures the critic will see; sends nothing, except the crops to TypeSafe for a "
        "face check (when TYPESAFE_API_KEY is set, after asking)",
    )
    cut.add_argument("source", type=pathlib.Path, help="rollout episodes, or a LeRobot dataset of demonstrations")
    cut.add_argument("--out", type=pathlib.Path, required=True, help="the evaluation folder")
    cut.add_argument("--count", type=int, default=5, help="pictures per sample, spread over the window (default 5)")
    cut.add_argument("--window-s", type=float, default=1.0, help="seconds from the end (or start) to draw from")
    cut.add_argument(
        "--only",
        choices=["start", "end"],
        help="only the start or only the end of each demonstration (a rollout's pictures are always its end)",
    )
    cut.add_argument("--limit", type=int, help="only the first N samples")
    cut.add_argument("--skip-face-check", action="store_true", help="run no face check at all (not advised)")
    cut.add_argument(
        "--face-check",
        choices=["auto", "opencv", "both", "typesafe"],
        default="auto",
        help="opencv: the local detector only; both: it, then TypeSafe on the crops it lets through; typesafe: "
        "TypeSafe only; auto (default): both when TYPESAFE_API_KEY is set, else opencv. TypeSafe gets the crops, "
        "after asking",
    )
    cut.add_argument(
        "--face-threshold",
        type=float,
        default=tc.FACE_BLOCK_AT,
        help=f"drop a crop when TypeSafe's chance of a face is at least this (default {tc.FACE_BLOCK_AT})",
    )
    cut.add_argument("--yes", action="store_true", help="do not ask before sending crops to TypeSafe")
    cut.set_defaults(handler=command_frames)

    ask = commands.add_parser("run", help="ask the critic about the cut pictures; sends them, after asking")
    ask.add_argument("--out", type=pathlib.Path, required=True, help="the evaluation folder")
    ask.add_argument("--variant", default="typesafe", help="a name for this run's settings (default: typesafe)")
    ask.add_argument(
        "--gate",
        choices=["typesafe", "claude", "gemini"],
        default="typesafe",
        help="who judges: TypeSafe or Claude answer the task's several questions; Gemini votes on one",
    )
    ask.add_argument("--model", help=f"the Gemini model for --gate gemini (default {tc.GEMINI_MODEL})")
    ask.add_argument(
        "--rewriter", choices=["claude", "gemini", "none"], default="claude", help="who writes free-text prompts"
    )
    ask.add_argument("--no-rewrite-unsure", action="store_true", help="rewrite only for not-done, not for unsure")
    ask.add_argument("--dry", action="store_true", help="build the requests and send nothing")
    ask.add_argument("--yes", action="store_true", help="do not ask before sending")
    ask.add_argument("--limit", type=int, help="only the first N samples not yet done")
    ask.set_defaults(handler=command_run)

    report = commands.add_parser("report", help="the critic against the labels")
    report.add_argument("--out", type=pathlib.Path, required=True, help="the evaluation folder")
    report.add_argument("--labels", type=pathlib.Path, help="labels.csv (not needed for demonstrations)")
    report.add_argument("--variants", default="typesafe", help="comma-separated variant names to report")
    report.set_defaults(handler=command_report)
    return top


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        code = args.handler(args)
    except (EvalError, tc.CriticError, tc.TaskFileError) as error:
        sys.exit(f"critic_eval: {error}")
    sys.exit(code)


if __name__ == "__main__":
    main()
