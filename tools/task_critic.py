"""The task critic: judges from camera pictures whether a task is done, says what failed, and rewrites the prompt.

A critic looks at the cameras after a robot run and gives a `Verdict`:

- **done / not_done / unsure.** The questions about the picture are atomic yes/no facts (configs/tasks/*.yaml), and a
  rule in this file, not the model, combines them: done when every fact is in its done state, not done when any fact
  is clearly not, unsure when the picture does not settle it (or when an arm hides what the facts are about).
- **Why it failed.** A TypeSafe Choice picks one of the task file's failure types.
- **A prompt to retry with.** The failure type's own prompt from the task file; or, when none fits, a free-text
  rewrite by Claude that sees the pictures and the prompts already tried. Every rewrite passes the guards in
  clean_prompt() and never repeats a prompt already tried.

There are two kinds of gate. The **question gate** (TypeSafe, or Claude) asks several atomic yes/no questions and
combines them. The **vote gate** (Gemini) asks one plain question about a zoomed picture, several times, and counts the
answers: done when every answer is yes, not done when every answer is no, unsure otherwise (look again). A scene's
verdict from several pictures is done or not done when most agree and none contradicts, unsure otherwise.

The model calls sit behind small protocols, `Gate`, `VoteGate` and `Rewriter`, so a backend can be swapped. All talk
plain HTTPS from the standard library; the keys come from the environment (`TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`) and are never printed or logged.

Pictures are cropped to the region the task file names before they go anywhere, because the scene camera also sees the
people around the table. `count_faces()` is a safety net on that crop, not a guarantee.

```
uv run python tools/task_critic.py questions configs/tasks/usb-cables.yaml   # the exact questions, nothing is sent
```

Run it in the repo's environment (`uv run`), not ~/dream-env: it needs OpenCV's face detector file, which only the
repo's opencv-python carries.
"""

import argparse
import base64
import concurrent.futures
import dataclasses
import functools
import json
import math
import os
import pathlib
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import cv2
import numpy as np
import yaml

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
## Pinned, not the `jev-latest` alias, so a run can be repeated later.
TYPESAFE_MODEL = "jev-1.14.0"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5-5"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-4-argon"

## What a vote may be. Anything else in a reply is an error, never a guess.
VOTE_ANSWERS = ("yes", "no", "unsure")

## What TypeSafe accepts of an image: each side in 280..1568 pixels and at most 2,458,624 pixels in all.
MIN_SIDE = 280
MAX_SIDE = 1568
MAX_PIXELS = 2_458_624

FAILURE_ID = "failure_type"
FAILURE_FIT_ID = "failure_fits"

DONE = "done"
NOT_DONE = "not_done"
UNSURE = "unsure"

## A free-text rewrite is one or two plain sentences, short enough to be a robot instruction.
MAX_PROMPT_CHARS = 200
MAX_PROMPT_SENTENCES = 2

## What Claude's yes / unsure / no answers count as when Claude, which gives no probabilities, is the gate.
CLAUDE_PROBABILITY = {"yes": 0.9, "unsure": 0.5, "no": 0.1}


class CriticError(RuntimeError):
    """A call to a model failed, or a picture or setting cannot be used. The message never carries a key."""


class TaskFileError(ValueError):
    """The task file is not valid; the message names the key."""


# --- The task file ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Item:
    """One atomic yes/no fact about the picture; `done_when` is the answer ("yes" or "no") that means done."""

    id: str
    question: str
    done_when: str


@dataclasses.dataclass(frozen=True)
class FailureType:
    name: str
    describe: str
    prompt: str


@dataclasses.dataclass(frozen=True)
class PromptEntry:
    id: str
    text: str
    scene_start: str


@dataclasses.dataclass(frozen=True)
class View:
    """An extra picture cut from a camera: a zoom on `box` (fractions of the camera's picture), scaled so its longer
    side is `max_side` pixels."""

    camera: str
    box: tuple[float, float, float, float]
    max_side: int


@dataclasses.dataclass(frozen=True)
class VoteGateSpec:
    """What the vote gate asks: `question` of the picture `view`, `repeats` times, the reply carrying `answer_key`."""

    view: str
    repeats: int
    answer_key: str
    question: str


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """A task file, checked."""

    name: str
    goal: str
    cables: int
    cameras: tuple[str, ...]
    crop: Mapping[str, tuple[float, float, float, float]]
    yes: float
    no: float
    items: tuple[Item, ...]
    visibility_id: str
    visibility_question: str
    visibility_min: float
    progress_id: str
    progress_question: str
    failure_min_probability: float
    failure_min_fit: float
    failure_types: tuple[FailureType, ...]
    prompts: tuple[PromptEntry, ...]
    views: Mapping[str, View] = dataclasses.field(default_factory=dict)
    vote_gate: VoteGateSpec | None = None


def _need(mapping: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...]) -> Any:
    if not isinstance(mapping, Mapping):
        raise TaskFileError(f"task file: {where.rstrip('.')} must be a mapping, got {mapping!r}")
    if key not in mapping:
        raise TaskFileError(f"task file: {where}{key} is missing")
    value = mapping[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise TaskFileError(f"task file: {where}{key} must be {kind}, got {value!r}")
    return value


def _fraction(value: Any, where: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise TaskFileError(f"task file: {where} must be a number from 0 to 1, got {value!r}")
    return float(value)


def load_task(path: str | os.PathLike[str]) -> TaskSpec:
    """Reads and checks a task file.

    @param path The YAML file (configs/tasks/*.yaml).
    @return The spec.
    @throws TaskFileError A key is missing or wrong; the message names it.
    """
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TaskFileError("task file: the top level must be a mapping")
    cables = _need(raw, "cables", "", int)
    if cables < 1:
        raise TaskFileError(f"task file: cables must be at least 1, got {cables}")
    cameras = tuple(_need(raw, "cameras", "", list))
    if not cameras or not all(isinstance(c, str) for c in cameras):
        raise TaskFileError("task file: cameras must be a list of camera names")
    crop: dict[str, tuple[float, float, float, float]] = {}
    for camera, box in _need(raw, "crop", "", dict).items():
        if not isinstance(box, list) or len(box) != 4:
            raise TaskFileError(f"task file: crop.{camera} must be [left, top, right, bottom]")
        left, top, right, bottom = (_fraction(v, f"crop.{camera}") for v in box)
        if left >= right or top >= bottom:
            raise TaskFileError(f"task file: crop.{camera} must have left < right and top < bottom")
        crop[camera] = (left, top, right, bottom)
    views: dict[str, View] = {}
    raw_views = raw.get("views") or {}
    if not isinstance(raw_views, dict):
        raise TaskFileError("task file: views must be a mapping of view names")
    for name, body in raw_views.items():
        where = f"views.{name}."
        camera = _need(body, "camera", where, str)
        if camera not in cameras:
            raise TaskFileError(f"task file: {where}camera {camera!r} is not one of cameras {list(cameras)}")
        if name in cameras:
            raise TaskFileError(f"task file: the view {name!r} has the name of a camera; pick another name")
        box = _need(body, "box", where, list)
        if len(box) != 4:
            raise TaskFileError(f"task file: {where}box must be [left, top, right, bottom]")
        left, top, right, bottom = (_fraction(v, f"{where}box") for v in box)
        if left >= right or top >= bottom:
            raise TaskFileError(f"task file: {where}box must have left < right and top < bottom")
        if camera in crop:
            crop_left, crop_top, crop_right, crop_bottom = crop[camera]
            if left < crop_left or top < crop_top or right > crop_right or bottom > crop_bottom:
                raise TaskFileError(
                    f"task file: {where}box must lie inside crop.{camera}, so a zoom shows no more than the crop"
                )
        max_side = _need(body, "max_side", where, int)
        if max_side < 64:
            raise TaskFileError(f"task file: {where}max_side must be at least 64, got {max_side}")
        views[name] = View(camera, (left, top, right, bottom), max_side)
    vote_gate = None
    if "vote_gate" in raw:
        body = raw["vote_gate"]
        view = _need(body, "view", "vote_gate.", str)
        if view not in views:
            raise TaskFileError(f"task file: vote_gate.view {view!r} is not one of views {list(views)}")
        repeats = _need(body, "repeats", "vote_gate.", int)
        if repeats < 1:
            raise TaskFileError(f"task file: vote_gate.repeats must be at least 1, got {repeats}")
        question = _need(body, "question", "vote_gate.", str).strip()
        if not question:
            raise TaskFileError("task file: vote_gate.question must not be empty")
        vote_gate = VoteGateSpec(view, repeats, _need(body, "answer_key", "vote_gate.", str), question)
    thresholds = _need(raw, "thresholds", "", dict)
    yes = _fraction(_need(thresholds, "yes_at_least", "thresholds.", int | float), "thresholds.yes_at_least")
    no = _fraction(_need(thresholds, "no_at_most", "thresholds.", int | float), "thresholds.no_at_most")
    if not no < yes:
        raise TaskFileError("task file: thresholds.no_at_most must be below thresholds.yes_at_least")
    items = []
    for i, entry in enumerate(_need(raw, "items", "", list)):
        where = f"items[{i}]."
        item = Item(
            _need(entry, "id", where, str),
            " ".join(_need(entry, "question", where, str).split()),
            _need(entry, "done_when", where, str),
        )
        if item.done_when not in ("yes", "no"):
            raise TaskFileError(f"task file: {where}done_when must be 'yes' or 'no', got {item.done_when!r}")
        items.append(item)
    if not items:
        raise TaskFileError("task file: items must not be empty")
    visibility = _need(raw, "visibility", "", dict)
    progress = _need(raw, "progress", "", dict)
    failure = _need(raw, "failure_mode", "", dict)
    types = tuple(
        FailureType(
            name,
            " ".join(_need(body, "describe", f"failure_mode.types.{name}.", str).split()),
            " ".join(_need(body, "prompt", f"failure_mode.types.{name}.", str).split()),
        )
        for name, body in _need(failure, "types", "failure_mode.", dict).items()
    )
    prompts = tuple(
        PromptEntry(
            _need(entry, "id", f"prompts[{i}].", str),
            " ".join(_need(entry, "text", f"prompts[{i}].", str).split()),
            _need(entry, "scene_start", f"prompts[{i}].", str),
        )
        for i, entry in enumerate(_need(raw, "prompts", "", list))
    )
    visibility_id = _need(visibility, "id", "visibility.", str)
    progress_id = _need(progress, "id", "progress.", str)
    ids = [item.id for item in items] + [visibility_id, progress_id, FAILURE_ID, FAILURE_FIT_ID]
    if len(set(ids)) != len(ids):
        raise TaskFileError(f"task file: question ids must be different from each other, got {ids}")
    return TaskSpec(
        name=_need(raw, "name", "", str),
        goal=" ".join(_need(raw, "goal", "", str).split()),
        cables=cables,
        cameras=cameras,
        crop=crop,
        yes=yes,
        no=no,
        items=tuple(items),
        visibility_id=visibility_id,
        visibility_question=" ".join(_need(visibility, "question", "visibility.", str).split()),
        visibility_min=_fraction(_need(visibility, "min", "visibility.", int | float), "visibility.min"),
        progress_id=progress_id,
        progress_question=" ".join(_need(progress, "question", "progress.", str).split()),
        failure_min_probability=_fraction(
            _need(failure, "min_probability", "failure_mode.", int | float), "failure_mode.min_probability"
        ),
        failure_min_fit=_fraction(_need(failure, "min_fit", "failure_mode.", int | float), "failure_mode.min_fit"),
        failure_types=types,
        prompts=prompts,
        views=views,
        vote_gate=vote_gate,
    )


# --- The questions, and reading the answers --------------------------------------------------------------------


def progress_levels(spec: TaskSpec) -> list[str]:
    """The Score levels of the progress question: cables still plugged in, 0 to `cables`, each as a situation."""
    levels = ["No orange cable is plugged into the power strip."]
    for count in range(1, spec.cables + 1):
        levels.append(
            "Exactly one orange cable is plugged into the power strip."
            if count == 1
            else f"Exactly {count} orange cables are plugged into the power strip."
        )
    return levels


def gate_questions(spec: TaskSpec) -> dict[str, dict[str, Any]]:
    """Every question the gate asks of one set of pictures, in TypeSafe's request form."""
    questions: dict[str, dict[str, Any]] = {
        item.id: {"type": "noul", "instructions": item.question} for item in spec.items
    }
    questions[spec.visibility_id] = {"type": "noul", "instructions": spec.visibility_question}
    questions[spec.progress_id] = {
        "type": "score",
        "instructions": spec.progress_question,
        "criteria": progress_levels(spec),
    }
    if spec.failure_types:
        questions[FAILURE_ID] = {
            "type": "choice",
            "instructions": "Which of these problems best describes why the orange cables in `scene` are not all "
            "unplugged and lying on the table?",
            "criteria": {t.name: t.describe for t in spec.failure_types},
        }
        questions[FAILURE_FIT_ID] = {
            "type": "noul",
            "instructions": "Does `scene` show at least one of these problems? "
            + " ".join(f"({i}) {t.describe}" for i, t in enumerate(spec.failure_types, 1)),
        }
    return questions


@dataclasses.dataclass
class GateAnswer:
    """The gate's raw answers to one set of pictures: P(yes) per Noul, the position per Score, the probabilities of
    each option per Choice."""

    nouls: dict[str, float]
    scores: dict[str, float]
    choices: dict[str, dict[str, float]]


def parse_answers(answers: Mapping[str, Any]) -> GateAnswer:
    """Reads TypeSafe's `answers` object (keyed by question id).

    @throws CriticError An answer is not of the documented shape or is out of range.
    """
    nouls: dict[str, float] = {}
    scores: dict[str, float] = {}
    choices: dict[str, dict[str, float]] = {}
    for key, answer in answers.items():
        kind = answer.get("type") if isinstance(answer, Mapping) else None
        try:
            if kind == "noul":
                value = float(answer["noul"])
                if not 0.0 <= value <= 1.0:
                    raise CriticError(f"answer {key!r}: a Noul must be from 0 to 1, got {value}")
                nouls[key] = value
            elif kind == "score":
                scores[key] = float(answer["score"])
            elif kind == "choice":
                probabilities = answer.get("probabilities")
                choices[key] = (
                    {str(name): float(p) for name, p in probabilities.items()}
                    if isinstance(probabilities, Mapping) and probabilities
                    else {str(answer["choice"]): 1.0}
                )
        except (KeyError, TypeError, ValueError) as error:
            raise CriticError(f"answer {key!r} is not in the documented shape: {error!r}") from None
    return GateAnswer(nouls, scores, choices)


def aggregate(answers: Sequence[GateAnswer]) -> GateAnswer:
    """One answer from several sets of pictures of the same scene: the median of each Noul and Score, the mean of
    each Choice's probabilities. The median keeps one frame with an arm in the way from deciding the result."""
    if not answers:
        raise CriticError("nothing to aggregate")
    nouls = {k: statistics.median(a.nouls[k] for a in answers) for k in answers[0].nouls}
    scores = {k: statistics.median(a.scores[k] for a in answers) for k in answers[0].scores}
    choices: dict[str, dict[str, float]] = {}
    for key in answers[0].choices:
        names = {name for a in answers for name in a.choices[key]}
        mean = {name: statistics.fmean(a.choices[key].get(name, 0.0) for a in answers) for name in names}
        total = sum(mean.values()) or 1.0
        choices[key] = {name: p / total for name, p in mean.items()}
    return GateAnswer(nouls, scores, choices)


# --- Deciding --------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Judgement:
    """The decision about one scene (one frame, or the aggregate over several)."""

    status: str
    p_done: float  # the lowest probability, over the facts, that a fact is in its done state (vote gate: yes share)
    satisfied: dict[str, float]  # per fact: the probability that it is in its done state
    visible: float  # (vote gate: the share of votes that were not "unsure")
    plugged_in: float | None  # the progress Score: cables still plugged in
    failure_probabilities: dict[str, float] | None
    failure_fit: float | None
    failure_type: str | None
    votes: list[dict[str, str]] | None = None  # the vote gate's answers and reasons, for review


def decide(spec: TaskSpec, satisfied: Mapping[str, float], visible: float) -> str:
    """The truth table: not done when any fact is clearly not in its done state; done when every fact is clearly in it
    and the strip is in view; unsure otherwise (a fact near the middle, or the strip hidden)."""
    if any(p <= spec.no for p in satisfied.values()):
        return NOT_DONE
    if all(p >= spec.yes for p in satisfied.values()) and visible >= spec.visibility_min:
        return DONE
    return UNSURE


def judge(spec: TaskSpec, answer: GateAnswer) -> Judgement:
    """Turns one gate answer into a decision, with the failure type when one clearly fits.

    @throws CriticError The answer lacks a question the spec asks.
    """
    try:
        satisfied = {
            item.id: answer.nouls[item.id] if item.done_when == "yes" else 1.0 - answer.nouls[item.id]
            for item in spec.items
        }
        visible = answer.nouls[spec.visibility_id]
    except KeyError as error:
        raise CriticError(f"the gate's answer has no {error.args[0]!r}") from None
    probabilities = answer.choices.get(FAILURE_ID)
    fit = answer.nouls.get(FAILURE_FIT_ID)
    failure_type = None
    if probabilities and fit is not None:
        top = max(probabilities, key=lambda name: probabilities[name])
        if probabilities[top] >= spec.failure_min_probability and fit >= spec.failure_min_fit:
            failure_type = top
    return Judgement(
        status=decide(spec, satisfied, visible),
        p_done=min(satisfied.values()),
        satisfied=satisfied,
        visible=visible,
        plugged_in=answer.scores.get(spec.progress_id),
        failure_probabilities=probabilities,
        failure_fit=fit,
        failure_type=failure_type,
    )


@dataclasses.dataclass(frozen=True)
class Vote:
    """One answer of the vote gate to one picture: `answer` is yes, no or unsure."""

    answer: str
    reason: str


def judge_votes(votes: Sequence[Vote]) -> Judgement:
    """The decision on one picture from its votes: done when every vote is yes, not done when every vote is no,
    unsure otherwise (the votes disagree, or one says it cannot tell), which means look again.

    The numbers on the result are shares of the votes, not probabilities: `p_done` the yes share, `visible` the share
    that were not unsure.

    @throws CriticError There are no votes.
    """
    if not votes:
        raise CriticError("a picture has no votes")
    answers = [v.answer for v in votes]
    yes, unsure = answers.count("yes"), answers.count("unsure")
    status = DONE if yes == len(votes) else NOT_DONE if answers.count("no") == len(votes) else UNSURE
    return Judgement(
        status=status,
        p_done=yes / len(votes),
        satisfied={"all_unplugged": yes / len(votes)},
        visible=1.0 - unsure / len(votes),
        plugged_in=None,
        failure_probabilities=None,
        failure_fit=None,
        failure_type=None,
        votes=[dataclasses.asdict(v) for v in votes],
    )


def combine_frames(frames: Sequence[Judgement]) -> Judgement:
    """One decision for a scene from the decisions on its pictures: done or not done when more than half of the
    pictures say so and none says the opposite; unsure otherwise. One picture with an arm in the way cannot decide it,
    and two pictures that contradict each other leave it open.

    @throws CriticError There are no pictures.
    """
    if not frames:
        raise CriticError("a scene has no pictures")
    statuses = [f.status for f in frames]
    done, not_done = statuses.count(DONE), statuses.count(NOT_DONE)
    if done and not_done:
        status = UNSURE
    elif done * 2 > len(frames):
        status = DONE
    elif not_done * 2 > len(frames):
        status = NOT_DONE
    else:
        status = UNSURE
    votes = [v for f in frames for v in (f.votes or [])]
    yes = sum(v["answer"] == "yes" for v in votes)
    unsure = sum(v["answer"] == "unsure" for v in votes)
    return Judgement(
        status=status,
        p_done=yes / len(votes) if votes else 0.0,
        satisfied={"all_unplugged": yes / len(votes) if votes else 0.0},
        visible=1.0 - unsure / len(votes) if votes else 0.0,
        plugged_in=None,
        failure_probabilities=None,
        failure_fit=None,
        failure_type=None,
        votes=votes,
    )


# --- The rewritten prompt --------------------------------------------------------------------------------------


def clean_prompt(text: str) -> str | None:
    """A model's rewrite, made safe to hand to the robot model: one or two sentences on one line, short.

    @return The prompt, or None when it is empty or longer than #MAX_PROMPT_CHARS.
    """
    text = " ".join(text.split()).strip("\"'`")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sentences[:MAX_PROMPT_SENTENCES]).strip()
    return text if text and len(text) <= MAX_PROMPT_CHARS else None


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def already_tried(text: str, history: Sequence[str]) -> bool:
    """Whether the prompt is one already tried, ignoring case and punctuation."""
    return _normal(text) in {_normal(h) for h in history}


def next_from_matrix(spec: TaskSpec, history: Sequence[str]) -> str | None:
    """The first prompt of the task file's matrix that was not tried yet, or None when all were."""
    return next((p.text for p in spec.prompts if not already_tried(p.text, history)), None)


def vote_hint(votes: Sequence[Mapping[str, str]] | None) -> str | None:
    """What the vote gate said it saw, for a rewriter: the reasons of the votes that did not say yes, each once, at
    most three, in at most 500 characters. None when there are none (a question gate's result carries no votes)."""
    reasons: list[str] = []
    for vote in votes or []:
        reason = " ".join(str(vote.get("reason", "")).split())
        if vote.get("answer") != "yes" and reason and reason not in reasons:
            reasons.append(reason)
    return " ".join(reasons[:3])[:500] or None


class Gate(Protocol):
    """Answers the yes/no and failure-type questions about one set of pictures (camera name to JPEG bytes)."""

    def ask(self, spec: TaskSpec, images: Mapping[str, bytes]) -> GateAnswer: ...


class VoteGate(Protocol):
    """Answers one plain question about a picture, several times. `wanted_views` names the pictures it looks at."""

    def wanted_views(self, spec: TaskSpec) -> tuple[str, ...]: ...

    def ask_votes(self, spec: TaskSpec, images: Mapping[str, bytes]) -> list[Vote]: ...


class Rewriter(Protocol):
    """Writes a free-text prompt for the robot model from pictures and what was already tried."""

    def rewrite(
        self, spec: TaskSpec, images: Mapping[str, bytes], history: Sequence[str], hint: str | None
    ) -> str: ...


@dataclasses.dataclass
class Verdict:
    """What the critic says about a scene.

    Attributes:
    - result: The decision on the aggregate over all the image sets.
    - frames: The decision on each image set alone, in order; the last is the final picture.
    - rewritten_prompt: A prompt to retry with, when the task is not done (or unsure), else None.
    - rewrite_source: `failure_type`, `claude`, `gemini` or `matrix`: where the prompt came from.
    """

    result: Judgement
    frames: list[Judgement]
    rewritten_prompt: str | None
    rewrite_source: str | None


class TaskCritic:
    """Gate plus rewriter, with the decision rules of this file."""

    def __init__(
        self,
        spec: TaskSpec,
        gate: Gate | VoteGate,
        rewriter: Rewriter | None = None,
        *,
        rewrite_when_unsure: bool = True,
    ):
        self.spec = spec
        self.gate = gate
        self.rewriter = rewriter
        self.rewrite_when_unsure = rewrite_when_unsure

    def evaluate(self, image_sets: Sequence[Mapping[str, bytes]], history: Sequence[str] = ()) -> Verdict:
        """Judges a scene from one or more sets of pictures (typically a few frames from the last second).

        A gate sees only the pictures it wants (the cameras for a question gate, the zoom for the vote gate); the
        rewriter sees them all.

        @param image_sets Each a mapping of picture name to JPEG bytes, as prepare_image() makes them.
        @param history The prompts already tried this session, oldest first.
        @throws CriticError A model call failed, or a set lacks a picture the gate wants. A failed call is an error,
            never an answer.
        """
        wanted = self.gate.wanted_views(self.spec) if hasattr(self.gate, "wanted_views") else self.spec.cameras
        for images in image_sets:
            missing = [name for name in wanted if name not in images]
            if missing:
                raise CriticError(f"a picture set lacks {missing}: cut the pictures again (`critic_eval.py frames`)")
        seen = [{name: images[name] for name in wanted} for images in image_sets]
        if hasattr(self.gate, "ask_votes"):
            frames = [judge_votes(self.gate.ask_votes(self.spec, images)) for images in seen]
            result = combine_frames(frames)
        else:
            answers = [self.gate.ask(self.spec, images) for images in seen]
            frames = [judge(self.spec, a) for a in answers]
            result = judge(self.spec, aggregate(answers))
        prompt = source = None
        if result.status == NOT_DONE or (result.status == UNSURE and self.rewrite_when_unsure):
            prompt, source = self._rewrite(result, image_sets[-1], history)
        return Verdict(result, frames, prompt, source)

    def _rewrite(
        self, result: Judgement, images: Mapping[str, bytes], history: Sequence[str]
    ) -> tuple[str | None, str | None]:
        spec = self.spec
        if result.failure_type is not None:
            chosen = next(t for t in spec.failure_types if t.name == result.failure_type)
            if not already_tried(chosen.prompt, history):
                return chosen.prompt, "failure_type"
        if self.rewriter is not None:
            hint = None
            if result.failure_type is not None:
                hint = next(t.describe for t in spec.failure_types if t.name == result.failure_type)
            else:
                hint = vote_hint(result.votes)
            written = clean_prompt(self.rewriter.rewrite(spec, images, history, hint))
            if written is not None and not already_tried(written, history):
                return written, getattr(self.rewriter, "name", "claude")
        fallback = next_from_matrix(spec, history)
        return (fallback, "matrix") if fallback is not None else (None, None)


# --- Pictures --------------------------------------------------------------------------------------------------


def crop_fractions(bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """The part of a picture inside `box` = (left, top, right, bottom), as fractions of its width and height.

    The edges round inward, so nothing outside the box is ever in the crop, not even a partial pixel column.
    """
    height, width = bgr.shape[:2]
    left, top, right, bottom = box
    x0, x1 = math.ceil(left * width - 1e-9), math.floor(right * width + 1e-9)
    y0, y1 = math.ceil(top * height - 1e-9), math.floor(bottom * height + 1e-9)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise CriticError(f"the crop {box} leaves nothing of a {width}x{height} picture")
    return bgr[y0:y1, x0:x1]


def prepare_image(
    bgr: np.ndarray,
    box: tuple[float, float, float, float] | None = None,
    quality: int = 90,
    target_side: int | None = None,
) -> bytes:
    """Crops a picture to `box`, scales it, and encodes it as a JPEG.

    By default the scale puts it into the sizes TypeSafe takes. With `target_side` the longer side becomes that many
    pixels instead (a zoom for the vote gate), whatever the crop's size.

    @param bgr An OpenCV image, height x width x 3.
    @param box (left, top, right, bottom) as fractions, or None for the whole picture.
    @param target_side The longer side in pixels, or None for TypeSafe's limits.
    @return JPEG bytes.
    @throws CriticError The crop's shape cannot fit the sizes TypeSafe takes.
    """
    image = bgr if box is None else crop_fractions(bgr, box)
    height, width = image.shape[:2]
    if target_side is not None:
        scale = target_side / max(height, width)
        new_width, new_height = int(round(width * scale)), int(round(height * scale))
    else:
        scale = max(1.0, MIN_SIDE / min(height, width))
        scale = min(scale, MAX_SIDE / max(height, width))
        scale = min(scale, (MAX_PIXELS / (height * width)) ** 0.5)
        new_width, new_height = int(round(width * scale)), int(round(height * scale))
        if min(new_width, new_height) < MIN_SIDE or max(new_width, new_height) > MAX_SIDE:
            raise CriticError(f"a {width}x{height} crop cannot be scaled into {MIN_SIDE}..{MAX_SIDE} pixels a side")
    if (new_width, new_height) != (width, height):
        method = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, (new_width, new_height), interpolation=method)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CriticError("JPEG encoding failed")
    return encoded.tobytes()


@functools.cache
def _face_detector() -> cv2.CascadeClassifier:
    path = pathlib.Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise CriticError(
            f"OpenCV's face detector file is missing ({path}); run this with the repo's environment (`uv run`), "
            "not ~/dream-env"
        )
    return detector


def count_faces(bgr: np.ndarray) -> int:
    """Frontal faces OpenCV's detector finds in a picture: a safety net on a crop, not a guarantee of none."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return len(_face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))


# --- Talking to the models -------------------------------------------------------------------------------------


def post_json(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    *,
    retries: int = 4,
    timeout: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """POSTs JSON and returns the JSON reply, asking again after a 429, a 5xx or a lost connection.

    @throws CriticError A refusal that asking again would not change, or the retries ran out. Nothing about the
        request's headers is in the message.
    """
    data = json.dumps(body).encode("utf-8")
    delay = 2.0
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
        try:
            with opener(request, timeout=timeout) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300] if hasattr(error, "read") else ""
            if error.code != 429 and error.code < 500:
                raise CriticError(f"{url} refused the request ({error.code}): {detail}") from None
            if attempt == retries:
                raise CriticError(f"{url} kept failing ({error.code}): {detail}") from None
            retry_after = error.headers.get("retry-after") if error.headers else None
            wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else delay
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries:
                raise CriticError(f"{url} could not be reached: {error}") from None
            wait = delay
        sleep(wait)
        delay *= 2
    raise CriticError(f"{url}: no reply")  # unreachable; keeps the type checker content


def _key(name: str, given: str | None) -> str:
    key = given or os.environ.get(name)
    if not key:
        raise CriticError(f"set {name} in the environment")
    return key


def redact(body: Mapping[str, Any]) -> dict[str, Any]:
    """A request body for a log: every image's bytes replaced by their size, so the payload can be kept and rerun."""
    out = json.loads(json.dumps(body, default=str))
    for asset in out.get("assets", {}).values():
        if isinstance(asset, dict) and "data_base64" in asset:
            asset["data_base64"] = f"<{len(asset['data_base64'])} base64 characters>"
    for message in out.get("messages", []):
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "image":
                block["source"] = {"type": "base64", "data": "<omitted>"}
    for content in out.get("contents", []):  # Gemini
        for part in content.get("parts", []):
            blob = part.get("inline_data") if isinstance(part, dict) else None
            if isinstance(blob, dict) and "data" in blob:
                blob["data"] = f"<{len(blob['data'])} base64 characters>"
    return out


class TypeSafeGate:
    """The gate on TypeSafe's Jev: one request per set of pictures, every question in it, calibrated Noul answers."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = TYPESAFE_MODEL,
        *,
        post: Callable[..., dict[str, Any]] = post_json,
        log: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._key = _key("TYPESAFE_API_KEY", api_key)
        self.model = model
        self._post = post
        self._log = log

    def request(self, spec: TaskSpec, images: Mapping[str, bytes]) -> dict[str, Any]:
        """The request body for one set of pictures. The state holds one entry per camera, named as the camera, so a
        question's `scene` is that camera's picture."""
        assets: dict[str, Any] = {}
        state: dict[str, str] = {}
        for camera, jpeg in images.items():
            reference = uuid.uuid4().hex
            state[camera] = reference
            assets[reference] = {"type": "image", "format": "jpeg", "data_base64": base64.b64encode(jpeg).decode()}
        return {"model": self.model, "state": state, "questions": gate_questions(spec), "assets": assets}

    def ask(self, spec: TaskSpec, images: Mapping[str, bytes]) -> GateAnswer:
        body = self.request(spec, images)
        if self._log is not None:
            self._log(redact(body))
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        }
        reply = self._post(TYPESAFE_URL, headers, body)
        if "answers" not in reply:
            raise CriticError(f"TypeSafe's reply has no answers: {str(reply)[:300]}")
        return parse_answers(reply["answers"])


def parse_vote(reply: Mapping[str, Any], answer_key: str) -> Vote:
    """Reads one vote out of a Gemini reply: the JSON object in its text, with `answer_key` and a `reason`.

    @throws CriticError The reply has no text, no JSON object, or an answer that is not yes, no or unsure. Such a
        reply is an error, never a guess.
    """
    candidates = reply.get("candidates") or []
    if not candidates:
        raise CriticError(f"Gemini's reply has no candidates: {str(reply.get('promptFeedback') or reply)[:300]}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise CriticError(f"Gemini's reply has no JSON object: {text[:300]!r}")
    try:
        found = json.loads(match.group(0))
    except json.JSONDecodeError:
        raise CriticError(f"Gemini's reply is not valid JSON: {text[:300]!r}") from None
    if not isinstance(found, dict):
        raise CriticError(f"Gemini's reply is not a JSON object: {text[:300]!r}")
    answer = str(found.get(answer_key, "")).strip().lower()
    if answer not in VOTE_ANSWERS:
        raise CriticError(f"Gemini's {answer_key!r} is {found.get(answer_key)!r}, not one of {VOTE_ANSWERS}")
    return Vote(answer, str(found.get("reason", "")).strip())


class _Gemini:
    """What the Gemini vote gate and rewriter share: the key from the environment, the model, and one call that logs
    the request (without its pictures) and never puts the key in a message."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        post: Callable[..., dict[str, Any]],
        log: Callable[[dict[str, Any]], None] | None,
    ):
        self._key = _key("GEMINI_API_KEY", api_key)
        self.model = model
        self._post = post
        self._log = log
        self._lock = threading.Lock()

    def _send(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._log is not None:
            with self._lock:
                self._log(redact(body))
        headers = {"x-goog-api-key": self._key, "Content-Type": "application/json"}
        return self._post(GEMINI_URL.format(model=self.model), headers, body)


class GeminiVoteGate(_Gemini):
    """The vote gate on Gemini: the task file's one plain question about the zoom picture, asked `repeats` times of
    the same picture, the repeats in parallel."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        *,
        post: Callable[..., dict[str, Any]] = post_json,
        log: Callable[[dict[str, Any]], None] | None = None,
    ):
        super().__init__(api_key, model, post, log)

    @staticmethod
    def _vote_spec(spec: TaskSpec) -> VoteGateSpec:
        if spec.vote_gate is None:
            raise CriticError(f"the task file for {spec.name!r} has no vote_gate section")
        return spec.vote_gate

    def wanted_views(self, spec: TaskSpec) -> tuple[str, ...]:
        return (self._vote_spec(spec).view,)

    def request(self, spec: TaskSpec, images: Mapping[str, bytes]) -> dict[str, Any]:
        """The request body for one picture: the picture, then the question."""
        vote_spec = self._vote_spec(spec)
        jpeg = images[vote_spec.view]
        picture = {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}
        return {
            "contents": [{"role": "user", "parts": [picture, {"text": vote_spec.question}]}],
            "generationConfig": {"maxOutputTokens": 8192},
        }

    def _one(self, spec: TaskSpec, body: dict[str, Any]) -> Vote:
        return parse_vote(self._send(body), self._vote_spec(spec).answer_key)

    def ask_votes(self, spec: TaskSpec, images: Mapping[str, bytes]) -> list[Vote]:
        vote_spec = self._vote_spec(spec)
        body = self.request(spec, images)
        with concurrent.futures.ThreadPoolExecutor(max_workers=vote_spec.repeats) as pool:
            return list(pool.map(lambda _: self._one(spec, body), range(vote_spec.repeats)))


def _image_blocks(images: Mapping[str, bytes]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for camera, jpeg in images.items():
        blocks.append({"type": "text", "text": f"Picture from the `{camera}` camera:"})
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()},
            }
        )
    return blocks


class _Claude:
    def __init__(self, api_key: str | None, model: str, post: Callable[..., dict[str, Any]], log):
        self._key = _key("ANTHROPIC_API_KEY", api_key)
        self.model = model
        self._post = post
        self._log = log

    def _call(self, system: str, content: list[dict[str, Any]], max_tokens: int) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        if self._log is not None:
            self._log(redact(body))
        headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        reply = self._post(ANTHROPIC_URL, headers, body)
        try:
            return "".join(b["text"] for b in reply["content"] if b.get("type") == "text")
        except (KeyError, TypeError):
            raise CriticError(f"Claude's reply has no text: {str(reply)[:300]}") from None


REWRITE_SYSTEM = (
    "You are the critic in a robot manipulation system. A robot policy was given an instruction and ran, and the task "
    "is not finished. Look at the pictures and write the next instruction for the robot policy. Write one or two "
    "plain imperative sentences. Say what to correct, naming only what you can see in the pictures. Do not repeat an "
    "instruction that was already tried. Reply with the instruction only: no quotes, no explanation."
)


class ClaudeRewriter(_Claude):
    """The free-text rewrite on Claude: sees the pictures and the prompts already tried."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = ANTHROPIC_MODEL,
        *,
        post: Callable[..., dict[str, Any]] = post_json,
        log: Callable[[dict[str, Any]], None] | None = None,
    ):
        super().__init__(api_key, model, post, log)

    def rewrite(self, spec: TaskSpec, images: Mapping[str, bytes], history: Sequence[str], hint: str | None) -> str:
        tried = "\n".join(f"- {p}" for p in history) or "- (none)"
        text = f"The goal: {spec.goal}\nInstructions already tried:\n{tried}\n"
        if hint:
            text += f"A likely problem: {hint}\n"
        text += "Write the next instruction."
        return self._call(REWRITE_SYSTEM, [*_image_blocks(images), {"type": "text", "text": text}], 200)


def gemini_text(reply: Mapping[str, Any]) -> str:
    """The text of a Gemini reply, without its thinking parts.

    @throws CriticError The reply has no candidate or no text (a blocked or empty reply is an error, not an answer).
    """
    candidates = reply.get("candidates") or []
    if not candidates:
        raise CriticError(f"Gemini's reply has no candidates: {str(reply.get('promptFeedback') or reply)[:300]}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
    if not text:
        raise CriticError(f"Gemini's reply has no text: {str(candidates[0])[:300]}")
    return text


class GeminiRewriter(_Gemini):
    """The free-text rewrite on Gemini: sees every picture of the last set (the scene crop and the zoomed strip), the
    prompts already tried, and what the vote gate said it saw, and writes the next instruction for the robot model.

    Like every rewriter it only proposes: TaskCritic keeps it to one or two short sentences, never lets it repeat a
    prompt already tried, and falls back to the task file's prompt list.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        *,
        post: Callable[..., dict[str, Any]] = post_json,
        log: Callable[[dict[str, Any]], None] | None = None,
    ):
        super().__init__(api_key, model, post, log)

    def request(
        self, spec: TaskSpec, images: Mapping[str, bytes], history: Sequence[str], hint: str | None
    ) -> dict[str, Any]:
        """The request body: each picture under its name, then the instructions, goal, what was tried, the hint."""
        tried = "\n".join(f"- {p}" for p in history) or "- (none)"
        text = f"{REWRITE_SYSTEM}\n\nThe goal: {spec.goal}\nInstructions already tried:\n{tried}\n"
        if hint:
            text += f"What the critic saw in the pictures: {hint}\n"
        text += "Write the next instruction."
        parts: list[dict[str, Any]] = []
        for name, jpeg in images.items():
            parts.append({"text": f"Picture `{name}`:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}})
        parts.append({"text": text})
        return {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"maxOutputTokens": 4096}}

    def rewrite(self, spec: TaskSpec, images: Mapping[str, bytes], history: Sequence[str], hint: str | None) -> str:
        return gemini_text(self._send(self.request(spec, images, history, hint)))


GATE_SYSTEM = (
    "You judge pictures of a robot task for a critic. Answer each question about the pictures with yes, no or unsure "
    "(unsure when the pictures do not settle it). Reply with one JSON object and nothing else."
)


class ClaudeGate(_Claude):
    """The gate on Claude, for comparison with TypeSafe. Claude gives no probabilities, so its yes, unsure and no
    count as #CLAUDE_PROBABILITY; read its numbers as coarse."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = ANTHROPIC_MODEL,
        *,
        post: Callable[..., dict[str, Any]] = post_json,
        log: Callable[[dict[str, Any]], None] | None = None,
    ):
        super().__init__(api_key, model, post, log)

    def ask(self, spec: TaskSpec, images: Mapping[str, bytes]) -> GateAnswer:
        questions = gate_questions(spec)
        lines = []
        for key, q in questions.items():
            if q["type"] == "noul":
                lines.append(f'"{key}": {q["instructions"]}  (yes / no / unsure)')
            elif q["type"] == "score":
                lines.append(f'"{key}": {q["instructions"]}  (a whole number from 0 to {spec.cables})')
            else:
                options = "; ".join(f"{name} = {text}" for name, text in q["criteria"].items())
                lines.append(f'"{key}": {q["instructions"]}  (one of: {options})')
        text = "Questions:\n" + "\n".join(lines) + "\nReply with a JSON object keyed by the question ids."
        reply = self._call(GATE_SYSTEM, [*_image_blocks(images), {"type": "text", "text": text}], 400)
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        if match is None:
            raise CriticError(f"Claude's reply has no JSON: {reply[:300]}")
        try:
            raw = json.loads(match.group(0))
            nouls = {
                k: CLAUDE_PROBABILITY[str(raw[k]).strip().lower()] for k, q in questions.items() if q["type"] == "noul"
            }
            scores = {spec.progress_id: float(raw[spec.progress_id])}
            choices: dict[str, dict[str, float]] = {}
            if FAILURE_ID in raw:
                picked = str(raw[FAILURE_ID]).strip()
                choices[FAILURE_ID] = {picked: 1.0}
        except (KeyError, ValueError, TypeError) as error:
            raise CriticError(f"Claude's JSON is not the shape asked for: {error!r}") from None
        return GateAnswer(nouls, scores, choices)


# --- Command line ----------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    shown = commands.add_parser("questions", help="print the exact questions the gate asks; nothing is sent")
    shown.add_argument("task", type=pathlib.Path, help="the task file, e.g. configs/tasks/usb-cables.yaml")
    args = parser.parse_args(argv)
    spec = load_task(args.task)
    print(json.dumps(gate_questions(spec), indent=2))
    print(
        f"\nthresholds: yes >= {spec.yes}, no <= {spec.no}; the strip counts as visible from {spec.visibility_min}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
