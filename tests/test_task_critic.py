"""tools/task_critic.py: the decision rules, the prompt guards, the requests and the pictures. No model is called."""

import base64
import dataclasses
import email.message
import importlib.util
import io
import json
import pathlib
import urllib.error

import cv2
import numpy as np
import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
TASK_FILE = pathlib.Path(__file__).resolve().parents[1] / "configs" / "tasks" / "usb-cables.yaml"


@pytest.fixture(scope="module")
def tc():
    """The tool's module, loaded the way `python tools/task_critic.py` does."""
    spec = importlib.util.spec_from_file_location("task_critic_tool", TOOLS / "task_critic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def spec(tc):
    return tc.load_task(TASK_FILE)


def answer(tc, *, plugged: float = 0.05, off_table: float = 0.05, visible: float = 0.95, **extra):
    """A gate answer: P(yes) for "a cable is still plugged in", for "a cable is off the table", for "strip visible"."""
    nouls = {
        "none_plugged_in": plugged,
        "none_off_the_table": off_table,
        "strip_visible": visible,
        "failure_fits": 0.9,
    }
    nouls.update(extra.pop("nouls", {}))
    return tc.GateAnswer(
        nouls=nouls,
        scores={"plugged_in_count": extra.pop("count", 0.0)},
        choices={"failure_type": extra.pop("choices", {"not_grasped": 0.8, "dropped": 0.2})},
    )


class FakeGate:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def ask(self, spec, images):
        self.asked.append(images)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class FakeRewriter:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def rewrite(self, spec, images, history, hint):
        self.calls.append((list(history), hint))
        return self.text


# --- the task file -------------------------------------------------------------------------------------------


def test_the_shipped_task_file_loads(spec) -> None:
    assert spec.name == "usb-cables"
    assert spec.cables == 4
    assert spec.cameras == ("scene",)
    assert spec.views["strip"].camera == "scene" and spec.views["strip"].max_side == 600
    assert spec.vote_gate.view == "strip" and spec.vote_gate.repeats == 2
    assert spec.vote_gate.answer_key == "all_four_unplugged"
    assert "were plugged into this power strip at the start" in spec.vote_gate.question
    assert spec.yes == 0.7 and spec.no == 0.3
    assert [p.id for p in spec.prompts][0] == "baseline"
    assert {t.name for t in spec.failure_types} >= {"not_grasped", "dropped"}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("cables: 4", "cables: 0", "cables must be at least 1"),
        ("    camera: scene\n", "    camera: overhead\n", "is not one of cameras"),
        ("    box: [0.4531, 0.42, 0.6875, 0.8334]", "    box: [0.4531, 0.2, 0.6875, 0.8334]", "inside crop.scene"),
        ("    max_side: 600", "    max_side: 10", "max_side must be at least 64"),
        ("  view: strip\n", "  view: nowhere\n", "is not one of views"),
        ("  repeats: 2\n", "  repeats: 0\n", "repeats must be at least 1"),
        ("  no_at_most: 0.3", "  no_at_most: 0.9", "no_at_most must be below"),
        ('    done_when: "no"\n  - id: none_off', "    done_when: maybe\n  - id: none_off", "done_when"),
        ("crop:\n  scene: [0.28, 0.42, 0.88, 1.0]", "crop:\n  scene: [0.9, 0.42, 0.88, 1.0]", "left < right"),
        ("  min: 0.5", "  min: 5", "visibility.min"),
    ],
)
def test_a_bad_task_file_names_the_key(tc, tmp_path, old, new, message) -> None:
    text = TASK_FILE.read_text()
    assert old in text, "the test's edit no longer matches the task file"
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace(old, new, 1))
    with pytest.raises(tc.TaskFileError, match=message):
        tc.load_task(bad)


def test_a_missing_key_is_named(tc, tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    with pytest.raises(tc.TaskFileError, match="cables is missing"):
        tc.load_task(bad)


# --- the questions -------------------------------------------------------------------------------------------


def test_the_questions_are_atomic_and_typed(tc, spec) -> None:
    questions = tc.gate_questions(spec)
    assert set(questions) == {
        "none_plugged_in",
        "none_off_the_table",
        "strip_visible",
        "plugged_in_count",
        "failure_type",
        "failure_fits",
    }
    assert questions["plugged_in_count"]["type"] == "score"
    assert len(questions["plugged_in_count"]["criteria"]) == spec.cables + 1
    assert questions["plugged_in_count"]["criteria"][0].startswith("No orange cable")
    assert set(questions["failure_type"]["criteria"]) == {t.name for t in spec.failure_types}
    assert all("`scene`" in q["instructions"] for q in questions.values())
    for question in questions.values():
        assert "\n" not in question["instructions"]


# --- deciding ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plugged", "off_table", "visible", "expected"),
    [
        (0.05, 0.05, 0.95, "done"),  # no cable plugged in, none off the table, strip in view
        (0.9, 0.05, 0.95, "not_done"),  # a cable is clearly still plugged in
        (0.05, 0.9, 0.95, "not_done"),  # a cable is clearly off the table
        (0.5, 0.05, 0.95, "unsure"),  # the plugged-in fact is in the band
        (0.05, 0.05, 0.2, "unsure"),  # looks done, but the strip is hidden by an arm
        (0.9, 0.05, 0.2, "not_done"),  # a plugged-in cable seen through a half-hidden strip is still a cable
        (0.2, 0.2, 0.95, "done"),  # 0.8 satisfied is clearly done
        (0.4, 0.05, 0.95, "unsure"),  # 0.6 satisfied is between the thresholds
    ],
)
def test_the_truth_table(tc, spec, plugged, off_table, visible, expected) -> None:
    judgement = tc.judge(spec, answer(tc, plugged=plugged, off_table=off_table, visible=visible))
    assert judgement.status == expected


def test_p_done_is_the_weakest_fact(tc, spec) -> None:
    judgement = tc.judge(spec, answer(tc, plugged=0.1, off_table=0.4))
    assert judgement.satisfied == pytest.approx({"none_plugged_in": 0.9, "none_off_the_table": 0.6})
    assert judgement.p_done == pytest.approx(0.6)
    assert judgement.status == "unsure"


def test_a_failure_type_needs_a_likely_choice_and_a_fit(tc, spec) -> None:
    assert tc.judge(spec, answer(tc, plugged=0.9)).failure_type == "not_grasped"
    weak = tc.judge(spec, answer(tc, plugged=0.9, choices={"not_grasped": 0.4, "dropped": 0.35, "other": 0.25}))
    assert weak.failure_type is None
    nothing_fits = tc.judge(spec, answer(tc, plugged=0.9, nouls={"failure_fits": 0.2}))
    assert nothing_fits.failure_type is None


def test_a_missing_answer_is_an_error_not_a_no(tc, spec) -> None:
    broken = tc.GateAnswer(nouls={"none_plugged_in": 0.1}, scores={}, choices={})
    with pytest.raises(tc.CriticError, match="none_off_the_table"):
        tc.judge(spec, broken)


def test_one_hidden_frame_does_not_flip_the_aggregate(tc, spec) -> None:
    clear = answer(tc, plugged=0.05)
    hidden = answer(tc, plugged=0.5, visible=0.1)
    merged = tc.aggregate([clear, clear, hidden, clear, clear])
    assert tc.judge(spec, merged).status == "done"
    assert tc.judge(spec, hidden).status == "unsure"


def test_aggregate_means_the_choice_probabilities_and_renormalises(tc) -> None:
    one = tc.GateAnswer({}, {}, {"failure_type": {"a": 1.0}})
    two = tc.GateAnswer({}, {}, {"failure_type": {"b": 1.0}})
    merged = tc.aggregate([one, two, two])
    assert merged.choices["failure_type"] == pytest.approx({"a": 1 / 3, "b": 2 / 3})
    with pytest.raises(tc.CriticError):
        tc.aggregate([])


def test_parse_answers_reads_the_documented_shapes(tc) -> None:
    parsed = tc.parse_answers(
        {
            "a": {"type": "noul", "noul": 0.25},
            "b": {"type": "score", "score": 1.4, "legend": [], "probabilities": {}},
            "c": {"type": "choice", "choice": "x", "confidence": 0.7, "probabilities": {"x": 0.8, "y": 0.2}},
            "d": {"type": "choice", "choice": "z"},
        }
    )
    assert parsed.nouls == {"a": 0.25}
    assert parsed.scores == {"b": 1.4}
    assert parsed.choices == {"c": {"x": 0.8, "y": 0.2}, "d": {"z": 1.0}}
    with pytest.raises(tc.CriticError, match="0 to 1"):
        tc.parse_answers({"a": {"type": "noul", "noul": 1.5}})
    with pytest.raises(tc.CriticError, match="documented shape"):
        tc.parse_answers({"a": {"type": "noul"}})


# --- the rewritten prompt ------------------------------------------------------------------------------------


def test_clean_prompt_guards(tc) -> None:
    assert (
        tc.clean_prompt('  "Pick up the cable.\n Put it on the table."  ') == "Pick up the cable. Put it on the table."
    )
    assert tc.clean_prompt("One. Two. Three. Four.") == "One. Two."
    assert tc.clean_prompt("") is None
    assert tc.clean_prompt("word " * 60) is None


def test_already_tried_ignores_case_and_punctuation(tc) -> None:
    assert tc.already_tried("Unplug the cable!", ["unplug the cable."])
    assert not tc.already_tried("Unplug the cables", ["unplug the cable."])


def test_a_done_scene_gets_no_prompt(tc, spec) -> None:
    critic = tc.TaskCritic(spec, FakeGate(answer(tc)), FakeRewriter("never used"))
    verdict = critic.evaluate([{"scene": b"x"}])
    assert verdict.result.status == "done"
    assert verdict.rewritten_prompt is None and verdict.rewrite_source is None


def test_a_known_failure_gets_its_prompt_from_the_task_file(tc, spec) -> None:
    rewriter = FakeRewriter("should not be asked")
    critic = tc.TaskCritic(spec, FakeGate(answer(tc, plugged=0.95)), rewriter)
    verdict = critic.evaluate([{"scene": b"x"}], history=["Unplug the orange USB cables."])
    chosen = next(t for t in spec.failure_types if t.name == "not_grasped")
    assert (verdict.rewritten_prompt, verdict.rewrite_source) == (chosen.prompt, "failure_type")
    assert rewriter.calls == []


def test_a_failure_whose_prompt_was_tried_goes_to_claude(tc, spec) -> None:
    chosen = next(t for t in spec.failure_types if t.name == "not_grasped")
    rewriter = FakeRewriter("The cable is still in the strip. Grab it and pull.")
    critic = tc.TaskCritic(spec, FakeGate(answer(tc, plugged=0.95)), rewriter)
    verdict = critic.evaluate([{"scene": b"x"}], history=[chosen.prompt])
    assert verdict.rewrite_source == "claude"
    assert verdict.rewritten_prompt == "The cable is still in the strip. Grab it and pull."
    assert rewriter.calls == [([chosen.prompt], chosen.describe)]


@pytest.mark.parametrize("written", ["REPEAT", "word " * 60, ""])
def test_a_repeated_or_unusable_rewrite_falls_back_to_the_matrix(tc, spec, written) -> None:
    chosen = next(t for t in spec.failure_types if t.name == "not_grasped")
    history = [chosen.prompt, spec.prompts[0].text]
    text = spec.prompts[0].text.upper() if written == "REPEAT" else written  # a repeat differs only by case
    critic = tc.TaskCritic(spec, FakeGate(answer(tc, plugged=0.95)), FakeRewriter(text))
    verdict = critic.evaluate([{"scene": b"x"}], history=history)
    assert (verdict.rewrite_source, verdict.rewritten_prompt) == ("matrix", spec.prompts[1].text)


def test_nothing_left_to_try_gives_no_prompt(tc, spec) -> None:
    history = [p.text for p in spec.prompts] + [t.prompt for t in spec.failure_types]
    critic = tc.TaskCritic(spec, FakeGate(answer(tc, plugged=0.95)), None)
    verdict = critic.evaluate([{"scene": b"x"}], history=history)
    assert verdict.result.status == "not_done"
    assert verdict.rewritten_prompt is None


def test_an_unsure_scene_is_rewritten_only_when_asked(tc, spec) -> None:
    unsure = answer(tc, plugged=0.5)
    assert tc.TaskCritic(spec, FakeGate(unsure), None).evaluate([{"scene": b"x"}]).rewritten_prompt is not None
    quiet = tc.TaskCritic(spec, FakeGate(unsure), None, rewrite_when_unsure=False)
    assert quiet.evaluate([{"scene": b"x"}]).rewritten_prompt is None


def test_evaluate_judges_every_frame_and_the_aggregate(tc, spec) -> None:
    gate = FakeGate(answer(tc, plugged=0.05), answer(tc, plugged=0.95), answer(tc, plugged=0.05))
    verdict = tc.TaskCritic(spec, gate).evaluate([{"scene": b"1"}, {"scene": b"2"}, {"scene": b"3"}])
    assert [f.status for f in verdict.frames] == ["done", "not_done", "done"]
    assert verdict.result.status == "done"  # the median of the three
    assert len(gate.asked) == 3


# --- requests ------------------------------------------------------------------------------------------------


def test_the_typesafe_request(tc, spec) -> None:
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return {"answers": {"none_plugged_in": {"type": "noul", "noul": 0.2}}}

    logged = []
    gate = tc.TypeSafeGate("sk-secret", post=post, log=logged.append)
    answers = gate.ask(spec, {"scene": b"\xff\xd8jpeg", "wrist_r": b"other"})
    assert answers.nouls == {"none_plugged_in": 0.2}
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["headers"]["Authorization"] == "Bearer sk-secret"
    assert sent["headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    body = sent["body"]
    assert body["model"] == "jev-1.14.0"
    assert set(body["state"]) == {"scene", "wrist_r"}
    assert set(body["state"].values()) == set(body["assets"])  # every reference has its asset
    asset = body["assets"][body["state"]["scene"]]
    assert asset["type"] == "image" and asset["format"] == "jpeg" and asset["data_base64"]
    assert body["questions"] == tc.gate_questions(spec)
    # What is logged can be kept and rerun, and holds neither the key nor the pictures.
    text = json.dumps(logged)
    assert "sk-secret" not in text and asset["data_base64"] not in text
    assert logged[0]["questions"] == body["questions"]


def test_a_reply_without_answers_is_an_error(tc, spec) -> None:
    gate = tc.TypeSafeGate("k", post=lambda *a: {"error": "nope"})
    with pytest.raises(tc.CriticError, match="no answers"):
        gate.ask(spec, {"scene": b"x"})


def test_a_missing_key_is_named_and_nothing_is_sent(tc, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(tc.CriticError, match="TYPESAFE_API_KEY"):
        tc.TypeSafeGate()
    with pytest.raises(tc.CriticError, match="ANTHROPIC_API_KEY"):
        tc.ClaudeRewriter()
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    assert tc.TypeSafeGate()._key == "from-env"


def test_claude_rewriter_sends_pictures_and_history(tc, spec) -> None:
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return {"content": [{"type": "text", "text": "Pull out the last cable."}]}

    logged = []
    rewriter = tc.ClaudeRewriter("sk-ant-secret", post=post, log=logged.append)
    text = rewriter.rewrite(spec, {"scene": b"jpeg"}, ["first try"], "a cable is still plugged in")
    assert text == "Pull out the last cable."
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["headers"]["x-api-key"] == "sk-ant-secret"
    content = sent["body"]["messages"][0]["content"]
    assert any(block["type"] == "image" for block in content)
    assert "first try" in content[-1]["text"] and "a cable is still plugged in" in content[-1]["text"]
    assert "sk-ant-secret" not in json.dumps(logged)
    assert all(b.get("source", {}).get("data") != "anVwZWc=" for b in logged[0]["messages"][0]["content"])


def test_claude_gate_maps_its_answers_to_coarse_probabilities(tc, spec) -> None:
    reply = {
        "none_plugged_in": "no",
        "none_off_the_table": "no",
        "strip_visible": "yes",
        "plugged_in_count": 0,
        "failure_type": "not_grasped",
        "failure_fits": "unsure",
    }
    post = lambda *a: {"content": [{"type": "text", "text": "Here:\n" + json.dumps(reply)}]}  # noqa: E731
    parsed = tc.ClaudeGate("k", post=post).ask(spec, {"scene": b"x"})
    assert parsed.nouls["none_plugged_in"] == 0.1 and parsed.nouls["strip_visible"] == 0.9
    assert parsed.nouls["failure_fits"] == 0.5
    assert parsed.scores == {"plugged_in_count": 0.0}
    assert parsed.choices == {"failure_type": {"not_grasped": 1.0}}
    bad = lambda *a: {"content": [{"type": "text", "text": "no json here"}]}  # noqa: E731
    with pytest.raises(tc.CriticError, match="no JSON"):
        tc.ClaudeGate("k", post=bad).ask(spec, {"scene": b"x"})


# --- the HTTP helper -----------------------------------------------------------------------------------------


def http_error(code: int, retry_after: str | None = None, body: bytes = b'{"error": "x"}'):
    headers = email.message.Message()
    if retry_after:
        headers["retry-after"] = retry_after
    return urllib.error.HTTPError("https://example.test", code, "reason", headers, io.BytesIO(body))


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_post_json_asks_again_after_429_and_honours_retry_after(tc) -> None:
    outcomes = [http_error(429, "7"), http_error(503), Reply({"ok": True})]
    sleeps = []

    def opener(request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert tc.post_json("https://example.test", {"k": "v"}, {}, opener=opener, sleep=sleeps.append) == {"ok": True}
    assert sleeps == [7.0, 4.0]


def test_post_json_does_not_retry_a_refusal_and_never_echoes_the_key(tc) -> None:
    calls = []

    def opener(request, timeout):
        calls.append(request.get_header("Authorization"))
        raise http_error(401, body=b'{"error": "bad key"}')

    with pytest.raises(tc.CriticError, match="401") as excinfo:
        tc.post_json(
            "https://example.test", {"Authorization": "Bearer sk-secret"}, {}, opener=opener, sleep=lambda s: None
        )
    assert len(calls) == 1
    assert "sk-secret" not in str(excinfo.value)


def test_post_json_gives_up_after_the_retries(tc) -> None:
    def opener(request, timeout):
        raise urllib.error.URLError("down")

    with pytest.raises(tc.CriticError, match="could not be reached"):
        tc.post_json("https://example.test", {}, {}, retries=2, opener=opener, sleep=lambda s: None)


# --- pictures ------------------------------------------------------------------------------------------------


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_a_scene_crop_is_scaled_up_into_the_accepted_sizes(tc, spec) -> None:
    frame = np.full((360, 640, 3), 128, np.uint8)
    out = decode(tc.prepare_image(frame, spec.crop["scene"]))
    height, width = out.shape[:2]
    assert height >= 280 and width >= 280 and max(height, width) <= 1568 and height * width <= 2_458_624
    assert width / height == pytest.approx((0.6 * 640) / (0.58 * 360), rel=0.02)


def test_a_large_picture_is_scaled_down(tc) -> None:
    out = decode(tc.prepare_image(np.zeros((2160, 3840, 3), np.uint8)))
    assert max(out.shape[:2]) <= 1568 and out.shape[0] * out.shape[1] <= 2_458_624


def test_the_crop_is_what_leaves_the_machine(tc) -> None:
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[:, :180] = (0, 0, 255)  # the left of the picture, where a person leans in, is red
    out = decode(tc.prepare_image(frame, (0.28, 0.42, 0.88, 1.0)))
    assert out[..., 2].max() < 30, "the cropped-away red region leaked into the picture"


def test_a_shape_that_cannot_fit_is_refused(tc) -> None:
    with pytest.raises(tc.CriticError, match="cannot be scaled"):
        tc.prepare_image(np.zeros((20, 640, 3), np.uint8))
    with pytest.raises(tc.CriticError, match="leaves nothing"):
        tc.prepare_image(np.zeros((360, 640, 3), np.uint8), (0.5, 0.5, 0.5, 0.5))


def test_the_face_detector_loads_and_finds_nothing_in_noise(tc) -> None:
    noise = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert tc.count_faces(noise) == 0


def test_the_questions_command_prints_json(tc, capsys) -> None:
    tc.main(["questions", str(TASK_FILE)])
    out = capsys.readouterr()
    assert json.loads(out.out)["none_plugged_in"]["type"] == "noul"
    assert "thresholds" in out.err


# --- the vote gate -------------------------------------------------------------------------------------------


def vote(tc, answer: str, reason: str = "because"):
    return tc.Vote(answer, reason)


@pytest.mark.parametrize(
    ("answers", "status", "p_done", "visible"),
    [
        (["yes", "yes"], "done", 1.0, 1.0),
        (["no", "no"], "not_done", 0.0, 1.0),
        (["yes", "no"], "unsure", 0.5, 1.0),  # the votes disagree: look again
        (["yes", "unsure"], "unsure", 0.5, 0.5),
        (["unsure", "unsure"], "unsure", 0.0, 0.0),
        (["no", "unsure"], "unsure", 0.0, 0.5),
        (["yes"], "done", 1.0, 1.0),
    ],
)
def test_a_picture_is_done_only_when_every_vote_says_so(tc, answers, status, p_done, visible) -> None:
    judgement = tc.judge_votes([vote(tc, a) for a in answers])
    assert judgement.status == status
    assert judgement.p_done == pytest.approx(p_done) and judgement.visible == pytest.approx(visible)
    assert [v["answer"] for v in judgement.votes] == answers
    assert judgement.failure_type is None and judgement.plugged_in is None


def test_no_votes_is_an_error(tc) -> None:
    with pytest.raises(tc.CriticError, match="no votes"):
        tc.judge_votes([])


def frame(tc, status: str):
    answers = {"done": ["yes", "yes"], "not_done": ["no", "no"], "unsure": ["yes", "unsure"]}[status]
    return tc.judge_votes([vote(tc, a) for a in answers])


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["done"] * 5, "done"),
        (["done"] * 4 + ["unsure"], "done"),  # one picture with an arm in the way cannot decide it
        (["done"] * 3 + ["unsure"] * 2, "done"),
        (["done"] * 2 + ["unsure"] * 3, "unsure"),  # not more than half
        (["not_done"] * 4 + ["unsure"], "not_done"),
        (["done"] * 4 + ["not_done"], "unsure"),  # two pictures contradict each other
        (["unsure"] * 5, "unsure"),
        (["done"], "done"),
    ],
)
def test_a_scene_follows_most_of_its_pictures_unless_they_contradict(tc, statuses, expected) -> None:
    combined = tc.combine_frames([frame(tc, s) for s in statuses])
    assert combined.status == expected
    assert len(combined.votes) == 2 * len(statuses)


def test_a_scene_without_pictures_is_an_error(tc) -> None:
    with pytest.raises(tc.CriticError, match="no pictures"):
        tc.combine_frames([])


def gemini_reply(text: str, **extra):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}], **extra}


@pytest.mark.parametrize(
    "text",
    [
        '{"all_four_unplugged": "no", "reason": "All four are still plugged in."}',
        '```json\n{"all_four_unplugged": "NO ", "reason": "All four are still plugged in."}\n```',
        'Here you go: {"all_four_unplugged": "no", "reason": "All four are still plugged in."}',
    ],
)
def test_a_vote_is_read_from_the_json_in_the_reply(tc, text) -> None:
    parsed = tc.parse_vote(gemini_reply(text), "all_four_unplugged")
    assert parsed.answer == "no" and parsed.reason == "All four are still plugged in."


def test_thinking_parts_are_not_part_of_the_answer(tc) -> None:
    reply = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{"all_four_unplugged": "yes"}', "thought": True},
                        {"text": '{"all_four_unplugged": "unsure", "reason": "hidden"}'},
                    ]
                }
            }
        ]
    }
    assert tc.parse_vote(reply, "all_four_unplugged").answer == "unsure"


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ({"promptFeedback": {"blockReason": "OTHER"}}, "no candidates"),
        (gemini_reply("I cannot tell."), "no JSON object"),
        (gemini_reply('{"all_four_unplugged": '), "no JSON object|not valid JSON"),
        (gemini_reply('{"all_four_unplugged": "maybe"}'), "not one of"),
        (gemini_reply('{"reason": "no answer key"}'), "not one of"),
        (gemini_reply('["yes"]'), "no JSON object"),
    ],
)
def test_a_reply_that_is_not_a_vote_is_an_error_not_a_guess(tc, reply, message) -> None:
    with pytest.raises(tc.CriticError, match=message):
        tc.parse_vote(reply, "all_four_unplugged")


def test_the_gemini_request_and_its_repeats(tc, spec) -> None:
    sent = []

    def post(url, headers, body):
        sent.append((url, headers, body))
        return gemini_reply('{"all_four_unplugged": "yes", "reason": "ports empty"}')

    logged = []
    gate = tc.GeminiVoteGate("gk-secret", post=post, log=logged.append)
    assert gate.wanted_views(spec) == ("strip",)
    votes = gate.ask_votes(spec, {"scene": b"not used", "strip": b"\xff\xd8zoom"})
    assert [v.answer for v in votes] == ["yes", "yes"]
    assert len(sent) == spec.vote_gate.repeats == 2
    url, headers, body = sent[0]
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent"
    assert headers["x-goog-api-key"] == "gk-secret"
    picture, question = body["contents"][0]["parts"]
    assert picture["inline_data"]["mime_type"] == "image/jpeg" and picture["inline_data"]["data"]
    assert question["text"] == spec.vote_gate.question
    # Only the zoom is sent, never the other picture of the set.
    assert "bm90IHVzZWQ" not in json.dumps(body)
    # What is logged can be kept and rerun, and holds neither the key nor the picture.
    text = json.dumps(logged)
    assert len(logged) == 2 and "gk-secret" not in text and picture["inline_data"]["data"] not in text
    assert logged[0]["contents"][0]["parts"][1]["text"] == spec.vote_gate.question


def test_the_gemini_gate_needs_a_key_and_a_vote_section(tc, spec, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(tc.CriticError, match="GEMINI_API_KEY"):
        tc.GeminiVoteGate()
    text = TASK_FILE.read_text()
    bare = tmp_path / "bare.yaml"
    bare.write_text(text[: text.index("# The vote gate")] + text[text.index("# A yes/no answer reads") :])
    with pytest.raises(tc.CriticError, match="no vote_gate section"):
        tc.GeminiVoteGate("k", post=lambda *a: {}).wanted_views(tc.load_task(bare))


def test_a_failed_gemini_call_is_an_error(tc, spec) -> None:
    def post(url, headers, body):
        raise tc.CriticError("HTTP 429")

    with pytest.raises(tc.CriticError, match="429"):
        tc.GeminiVoteGate("k", post=post).ask_votes(spec, {"strip": b"x"})


class FakeVoteGate:
    """Answers the vote gate's question from a script: one list of answers per picture set."""

    def __init__(self, vote_class, *rounds):
        self.vote_class = vote_class
        self.rounds = list(rounds)
        self.seen = []

    def wanted_views(self, spec):
        return ("strip",)

    def ask_votes(self, spec, images):
        self.seen.append(sorted(images))
        return [self.vote_class(a, "scripted") for a in self.rounds.pop(0)]


def test_the_critic_runs_the_vote_gate_on_the_zoom_only(tc, spec) -> None:
    gate = FakeVoteGate(tc.Vote, *[["yes", "yes"]] * 3)
    sets = [{"scene": b"s", "strip": b"z"}] * 3
    verdict = tc.TaskCritic(spec, gate, FakeRewriter("never used")).evaluate(sets)
    assert gate.seen == [["strip"]] * 3  # the scene picture never reaches the vote gate
    assert verdict.result.status == "done" and [f.status for f in verdict.frames] == ["done"] * 3
    assert verdict.rewritten_prompt is None


def test_the_critic_asks_to_cut_the_zoom_when_it_is_missing(tc, spec) -> None:
    with pytest.raises(tc.CriticError, match="lacks .*strip"):
        tc.TaskCritic(spec, FakeVoteGate(tc.Vote, ["yes"])).evaluate([{"scene": b"s"}])


def test_a_not_done_scene_gets_a_rewrite_without_a_failure_type(tc, spec) -> None:
    gate = FakeVoteGate(tc.Vote, ["no", "no"], ["no", "no"])
    rewriter = FakeRewriter("Pull the cable that is still in the strip out. Put it on the table.")
    sets = [{"scene": b"s", "strip": b"z"}] * 2
    verdict = tc.TaskCritic(spec, gate, rewriter).evaluate(sets, history=[spec.prompts[0].text])
    assert verdict.result.status == "not_done"
    assert (verdict.rewrite_source, verdict.rewritten_prompt) == ("claude", rewriter.text)
    assert rewriter.calls == [([spec.prompts[0].text], "scripted")]  # no failure type: the votes' reason is the hint
    without = tc.TaskCritic(spec, FakeVoteGate(tc.Vote, ["no", "no"])).evaluate(
        sets[:1], history=[spec.prompts[0].text]
    )
    assert (without.rewrite_source, without.rewritten_prompt) == ("matrix", spec.prompts[1].text)


def test_an_unsure_scene_is_looked_at_again_not_decided(tc, spec) -> None:
    gate = FakeVoteGate(tc.Vote, ["yes", "unsure"], ["unsure", "unsure"])
    verdict = tc.TaskCritic(spec, gate, None, rewrite_when_unsure=False).evaluate([{"strip": b"z"}] * 2)
    assert verdict.result.status == "unsure" and verdict.rewritten_prompt is None


def test_prepare_image_can_scale_a_zoom_to_a_side_length(tc, spec) -> None:
    frame_360p = np.full((360, 640, 3), 128, np.uint8)
    view = spec.views["strip"]
    out = decode(tc.prepare_image(frame_360p, view.box, target_side=view.max_side))
    assert max(out.shape[:2]) == 600
    frame_720p = np.full((720, 1280, 3), 128, np.uint8)
    assert max(decode(tc.prepare_image(frame_720p, view.box, target_side=view.max_side)).shape[:2]) == 600
    assert min(decode(tc.prepare_image(frame_360p, view.box, target_side=view.max_side)).shape[:2]) >= 550


def test_the_zoom_is_inside_the_scene_crop(tc, spec) -> None:
    left, top, right, bottom = spec.views["strip"].box
    crop_left, crop_top, crop_right, crop_bottom = spec.crop["scene"]
    assert crop_left <= left and crop_top <= top and right <= crop_right and bottom <= crop_bottom


# --- the Gemini rewriter -------------------------------------------------------------------------------------


def test_the_hint_is_what_the_votes_that_did_not_say_yes_reported(tc) -> None:
    votes = [
        {"answer": "yes", "reason": "ports empty"},
        {"answer": "no", "reason": "One cable is still plugged in."},
        {"answer": "no", "reason": "One cable is still plugged in."},  # said once
        {"answer": "unsure", "reason": "  An arm hides   the ports. "},
        {"answer": "no", "reason": ""},  # nothing to say
    ]
    assert tc.vote_hint(votes) == "One cable is still plugged in. An arm hides the ports."
    assert tc.vote_hint([{"answer": "yes", "reason": "fine"}]) is None
    assert tc.vote_hint(None) is None and tc.vote_hint([]) is None
    four = [{"answer": "no", "reason": f"reason {i}"} for i in range(4)]
    assert tc.vote_hint(four) == "reason 0 reason 1 reason 2"  # at most three
    assert len(tc.vote_hint([{"answer": "no", "reason": "x" * 400}, {"answer": "no", "reason": "y" * 400}])) == 500


def test_gemini_text_skips_thinking_and_refuses_an_empty_reply(tc) -> None:
    reply = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": " Do it. "}]}}]}
    assert tc.gemini_text(reply) == "Do it."
    with pytest.raises(tc.CriticError, match="no candidates"):
        tc.gemini_text({"promptFeedback": {"blockReason": "OTHER"}})
    with pytest.raises(tc.CriticError, match="no text"):
        tc.gemini_text(gemini_reply("   "))


def test_the_gemini_rewriter_request(tc, spec) -> None:
    sent = []

    def post(url, headers, body):
        sent.append((url, headers, body))
        return gemini_reply("Pull the last cable out of the strip.")

    logged = []
    rewriter = tc.GeminiRewriter("gk-secret", post=post, log=logged.append)
    pictures = {"scene": b"\xff\xd8scene", "strip": b"\xff\xd8zoom"}
    text = rewriter.rewrite(spec, pictures, ["first try"], "One cable is still plugged in.")
    assert text == "Pull the last cable out of the strip."
    url, headers, body = sent[0]
    assert url.endswith("/models/gemini-4-argon:generateContent") and headers["x-goog-api-key"] == "gk-secret"
    parts = body["contents"][0]["parts"]
    assert [p["text"] for p in parts if "text" in p][:2] == ["Picture `scene`:", "Picture `strip`:"]
    assert len([p for p in parts if "inline_data" in p]) == 2  # the scene crop and the zoom
    last = parts[-1]["text"]
    assert last.startswith(tc.REWRITE_SYSTEM) and spec.goal in last and "- first try" in last
    assert "One cable is still plugged in." in last and last.endswith("Write the next instruction.")
    # The log can be kept and rerun: no key, no pictures.
    text = json.dumps(logged)
    assert "gk-secret" not in text and "inline_data" in text and "c2NlbmU" not in text


def test_the_gemini_rewriter_without_a_hint_or_history(tc, spec) -> None:
    body = tc.GeminiRewriter("k", post=lambda *a: {}).request(spec, {"scene": b"x"}, [], None)
    last = body["contents"][0]["parts"][-1]["text"]
    assert "- (none)" in last and "What the critic saw" not in last


def test_the_gemini_rewriter_needs_a_key_and_reports_a_failed_call(tc, spec, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(tc.CriticError, match="GEMINI_API_KEY"):
        tc.GeminiRewriter()

    def post(url, headers, body):
        raise tc.CriticError("HTTP 503")

    with pytest.raises(tc.CriticError, match="503"):
        tc.GeminiRewriter("k", post=post).rewrite(spec, {"scene": b"x"}, [], None)


def test_the_critic_names_the_rewriter_and_hints_it_with_the_votes(tc, spec) -> None:
    class Named(FakeRewriter):
        name = "gemini"

    rewriter = Named("Unplug the last cable and put it on the table.")
    gate = FakeVoteGate(tc.Vote, ["no", "no"])
    verdict = tc.TaskCritic(spec, gate, rewriter).evaluate(
        [{"scene": b"s", "strip": b"z"}], history=[spec.prompts[0].text]
    )
    assert (verdict.rewrite_source, verdict.rewritten_prompt) == ("gemini", rewriter.text)
    assert rewriter.calls == [([spec.prompts[0].text], "scripted")]


def test_gemini_judges_the_zoom_and_then_writes_from_both_pictures(tc, spec) -> None:
    posted = []

    def post(url, headers, body):
        posted.append(body)
        if body["generationConfig"]["maxOutputTokens"] == 8192:  # the vote gate's call
            return gemini_reply('{"all_four_unplugged": "no", "reason": "One cable is still in the rightmost port."}')
        return gemini_reply("Pull the cable out of the rightmost port.")

    critic = tc.TaskCritic(spec, tc.GeminiVoteGate("k", post=post), tc.GeminiRewriter("k", post=post))
    verdict = critic.evaluate([{"scene": b"scene-bytes", "strip": b"strip-bytes"}], history=[spec.prompts[0].text])
    assert verdict.result.status == "not_done"
    assert (verdict.rewrite_source, verdict.rewritten_prompt) == (
        "gemini",
        "Pull the cable out of the rightmost port.",
    )
    pictures = lambda b: [p for p in b["contents"][0]["parts"] if "inline_data" in p]  # noqa: E731
    votes = [b for b in posted if b["generationConfig"]["maxOutputTokens"] == 8192]
    writes = [b for b in posted if b["generationConfig"]["maxOutputTokens"] == 4096]
    assert len(votes) == 2 and all(len(pictures(b)) == 1 for b in votes)  # the gate: the zoom only
    assert len(writes) == 1 and len(pictures(writes[0])) == 2  # the rewriter: both
    assert "One cable is still in the rightmost port." in writes[0]["contents"][0]["parts"][-1]["text"]


def test_a_gemini_prompt_that_repeats_one_already_tried_falls_back_to_the_task_list(tc, spec) -> None:
    repeat = spec.prompts[0].text.upper()
    critic = tc.TaskCritic(
        spec, FakeVoteGate(tc.Vote, ["no", "no"]), tc.GeminiRewriter("k", post=lambda *a: gemini_reply(repeat))
    )
    verdict = critic.evaluate([{"scene": b"s", "strip": b"z"}], history=[spec.prompts[0].text])
    assert (verdict.rewrite_source, verdict.rewritten_prompt) == ("matrix", spec.prompts[1].text)


# --- the supervisor section ------------------------------------------------------------------------------------


def test_the_shipped_supervisor_section(tc, spec) -> None:
    config = spec.supervisor
    assert (config.attempts, config.attempt_s, config.warmup_s, config.probe_every_s) == (3, 90.0, 15.0, 10.0)
    assert (config.frames_per_probe, config.done_probes, config.max_stalls) == (3, 2, 3)
    assert (config.stuck_after_s, config.guide_window_s) == (20.0, 45.0)
    motion = config.object_motion
    assert (motion.view, motion.space, motion.low, motion.high, motion.min_change) == (
        "strip", "lab", (0, 140, 128), (255, 255, 255), 0.12
    )  # fmt: skip
    assert config.compare_view == "strip" and config.compare_question is None


def test_a_task_file_without_a_supervisor_section_gets_the_defaults(tc, tmp_path) -> None:
    text = TASK_FILE.read_text()
    start = text.index("supervisor:\n  attempts")
    end = text.index("# A yes/no answer reads as yes")
    bare = tmp_path / "bare.yaml"
    bare.write_text(text[:start] + text[end:])
    config = tc.load_task(bare).supervisor
    assert config == tc.SupervisorSpec()
    assert config.object_motion is None  # no colour, so the supervisor leans on the model compare alone


def test_a_supervisor_section_may_hold_just_a_model_compare(tc, tmp_path) -> None:
    text = TASK_FILE.read_text()
    old = (
        "    object_motion:\n      view: strip\n"
        "      color: {space: lab, low: [0, 140, 128], high: [255, 255, 255]}\n      min_change: 0.12\n"
    )
    assert old in text
    bare = tmp_path / "compare_only.yaml"
    bare.write_text(text.replace(old, "", 1))
    config = tc.load_task(bare).supervisor
    assert config.object_motion is None and config.compare_view == "strip"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  attempts: 3\n", "  attempts: 0\n", "supervisor.attempts must be a whole number of at least 1"),
        ("  attempts: 3\n", "  attempts: 3\n  bogus: 1\n", "unknown keys \\['bogus'\\]"),
        ("  attempt_s: 90\n", "  attempt_s: -5\n", "supervisor.attempt_s must be a number of at least 1"),
        ("  stuck_after_s: 20\n", "  stuck_after_s: 5\n", "stuck_after_s cannot be shorter than probe_every_s"),
        ("  guide_window_s: 45\n", "  guide_window_s: 5\n", "guide_window_s cannot be shorter than probe_every_s"),
        ("  done_probes: 2\n", "  done_probes: true\n", "done_probes must be a whole number"),
        ("  progress:\n", "  progress:\n    telepathy: 1\n", "may hold only object_motion and model_compare"),
        ("      view: strip\n      color", "      view: nowhere\n      color", "object_motion.view 'nowhere' is not"),
        ("space: lab", "space: rgb", "color.space must be lab or hsv"),
        ("space: lab", "space: hsv", "first number 0 to 179"),
        ("low: [0, 140, 128]", "low: [0, 140]", "must be three whole numbers"),
        ("min_change: 0.12", "min_change: 0", "min_change must be above 0"),
        ("min_change: 0.12", "min_change: 1.5", "min_change must be above 0 and at most 1"),
        ("    model_compare:\n      view: strip", "    model_compare:\n      view: nowhere", "not a camera or a view"),
        ("    model_compare:\n      view: strip", "    model_compare:\n      question: ' '", "non-empty string"),
    ],
)  # fmt: skip
def test_a_bad_supervisor_section_names_the_key(tc, tmp_path, old, new, message) -> None:
    text = TASK_FILE.read_text()
    assert text.count(old) >= 1, "the test's edit no longer matches the task file"
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace(old, new, 1))
    with pytest.raises(tc.TaskFileError, match=message):
        tc.load_task(bad)


# --- what the supervisor asks of the critic --------------------------------------------------------------------


def test_judging_without_a_rewrite_leaves_the_prompt_to_rewrite_prompt(tc, spec) -> None:
    rewriter = FakeRewriter("Pull the cable out of the strip.")
    critic = tc.TaskCritic(spec, FakeVoteGate(tc.Vote, ["no", "no"]), rewriter)
    pictures = {"scene": b"s", "strip": b"z"}
    verdict = critic.evaluate([pictures], history=["x"], rewrite=False)
    assert verdict.result.status == "not_done"
    assert verdict.rewritten_prompt is None and rewriter.calls == []  # nothing was asked of the rewriter
    prompt, source = critic.rewrite_prompt(verdict.result, pictures, ["x"])
    assert (prompt, source) == (rewriter.text, "claude")
    assert rewriter.calls == [(["x"], "scripted")]  # the votes' reasons are its hint


def test_rewrite_prompt_runs_the_same_guards_and_says_when_nothing_is_left(tc, spec) -> None:
    tried = [p.text for p in spec.prompts]
    critic = tc.TaskCritic(spec, FakeVoteGate(tc.Vote, ["no", "no"]), FakeRewriter(tried[0]))
    verdict = critic.evaluate([{"strip": b"z"}], rewrite=False)
    assert critic.rewrite_prompt(verdict.result, {"strip": b"z"}, tried) == (None, None)
    prompt, source = critic.rewrite_prompt(verdict.result, {"strip": b"z"}, tried[:1])  # a repeat: use the list
    assert (prompt, source) == (tried[1], "matrix")


def test_cut_pictures_makes_the_scene_crop_and_the_zoom(tc, spec) -> None:
    frame = np.full((360, 640, 3), 128, np.uint8)
    pictures = tc.cut_pictures(spec, {"scene": frame})
    assert sorted(pictures) == ["scene", "strip"]
    strip = cv2.imdecode(np.frombuffer(pictures["strip"], np.uint8), cv2.IMREAD_COLOR)
    assert max(strip.shape[:2]) == 600
    scene = cv2.imdecode(np.frombuffer(pictures["scene"], np.uint8), cv2.IMREAD_COLOR)
    assert min(scene.shape[:2]) >= 280  # scaled into the sizes TypeSafe takes
    with pytest.raises(tc.CriticError, match="no frame for camera"):
        tc.cut_pictures(spec, {"wrist_l": frame})


def test_the_progress_question_is_built_from_the_goal_or_taken_from_the_task_file(tc, spec) -> None:
    built = tc.GeminiProgressCheck.question(spec, 20.0)
    assert spec.goal in built and "about 20 seconds apart" in built and '"progress"' in built
    custom = dataclasses.replace(
        spec, supervisor=dataclasses.replace(spec.supervisor, compare_question="Goal: {goal}. After {seconds} s?")
    )
    assert tc.GeminiProgressCheck.question(custom, 20.0) == f"Goal: {spec.goal}. After 20 s?"


def test_the_progress_compare_sends_both_pictures_in_order_and_reads_the_vote(tc, spec) -> None:
    posted = []

    def post(url, headers, body):
        posted.append((url, headers, body))
        return gemini_reply('{"progress": "no", "reason": "Nothing moved."}')

    vote = tc.GeminiProgressCheck("k", post=post).compare(spec, b"before", b"after", 20.0)
    assert vote == tc.Vote("no", "Nothing moved.")
    ((url, headers, body),) = posted
    assert url.endswith(f"{tc.GEMINI_MODEL}:generateContent") and headers["x-goog-api-key"] == "k"
    parts = body["contents"][0]["parts"]
    assert [next(iter(p)) for p in parts] == ["text", "inline_data", "text", "inline_data", "text"]
    assert [parts[0]["text"], parts[2]["text"]] == ["The earlier picture:", "The later picture:"]
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"before"
    assert base64.b64decode(parts[3]["inline_data"]["data"]) == b"after"
    assert parts[4]["text"] == tc.GeminiProgressCheck.question(spec, 20.0)


def test_the_progress_compare_treats_a_bad_reply_and_a_missing_key_as_errors(tc, spec, monkeypatch) -> None:
    with pytest.raises(tc.CriticError, match="not one of"):
        tc.GeminiProgressCheck("k", post=lambda *a: gemini_reply('{"progress": "maybe"}')).compare(spec, b"a", b"b", 5)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(tc.CriticError, match="GEMINI_API_KEY"):
        tc.GeminiProgressCheck()
