import pytest
from dream_hackathon import lab_db


@pytest.fixture
def store(tmp_path):
    return lab_db.Store(tmp_path / "lab")


def test_prompts_are_added_edited_and_deleted(store):
    first = store.add_prompt("  pick up the cube  ")
    assert first["text"] == "pick up the cube"
    assert first["checkpoint"] == lab_db.DEFAULT_CHECKPOINT
    second = store.add_prompt("stack blocks", "dream-1-ft:team-blocks")
    assert [p["id"] for p in store.prompts()] == [first["id"], second["id"]]
    store.update_prompt(first["id"], "pick up the red cube", "")
    assert store.prompt(first["id"])["text"] == "pick up the red cube"
    store.delete_prompt(first["id"])
    assert [p["id"] for p in store.prompts()] == [second["id"]]
    with pytest.raises(lab_db.LabError):
        store.prompt(first["id"])


@pytest.mark.parametrize("text", ["", "   ", None, 5, "x" * (lab_db.PROMPT_MAX_CHARS + 1)])
def test_a_bad_prompt_is_refused(store, text):
    with pytest.raises(lab_db.LabError):
        store.add_prompt(text)


@pytest.mark.parametrize("checkpoint", ["a b", "x;rm", 7, "m" * 300])
def test_a_bad_checkpoint_is_refused(store, checkpoint):
    with pytest.raises(lab_db.LabError):
        store.add_prompt("do it", checkpoint)


@pytest.mark.parametrize("seconds", [4, 3601, 1.5, "60", True])
def test_a_bad_run_length_is_refused(seconds):
    with pytest.raises(lab_db.LabError):
        lab_db.clean_max_seconds(seconds)


def test_a_run_keeps_its_own_copy_of_the_prompt(store):
    prompt = store.add_prompt("open the drawer")
    run = store.start_run(prompt["text"], prompt["checkpoint"], prompt_id=prompt["id"], dry_run=True, max_seconds=60)
    assert run["status"] == "running" and run["dry_run"] == 1
    assert (store.runs_dir / f"{run['id']:05d}").is_dir() and run["dir"].endswith(f"{run['id']:05d}")
    store.delete_prompt(prompt["id"])
    ended = store.end_run(run["id"], "finished")
    assert ended["prompt_text"] == "open the drawer" and ended["prompt_id"] is None
    assert ended["ended_at"] and ended["status"] == "finished"


def test_runs_come_newest_first_and_unknown_statuses_are_refused(store):
    a = store.start_run("a", "c", prompt_id=None, dry_run=False, max_seconds=10)
    b = store.start_run("b", "c", prompt_id=None, dry_run=False, max_seconds=10)
    assert [r["id"] for r in store.runs()] == [b["id"], a["id"]]
    with pytest.raises(lab_db.LabError):
        store.end_run(a["id"], "exploded")


def test_a_run_left_running_by_a_dead_server_is_closed_as_an_error(tmp_path):
    first = lab_db.Store(tmp_path / "lab")
    run = first.start_run("a", "c", prompt_id=None, dry_run=False, max_seconds=10)
    again = lab_db.Store(tmp_path / "lab")
    assert again.close_unfinished() == 1
    assert again.run(run["id"])["status"] == "error"
    assert again.close_unfinished() == 0


def test_a_sequence_keeps_its_steps_and_loses_those_of_a_deleted_prompt(store):
    a, b = store.add_prompt("pick up the cube"), store.add_prompt("put it down")
    sequence = store.add_sequence("  cube  ", [{"prompt_id": a["id"], "seconds": 30}, {"prompt_id": b["id"]}])
    assert sequence["name"] == "cube"
    assert [(s["prompt_id"], s["seconds"], s["text"]) for s in sequence["steps"]] == [
        (a["id"], 30, "pick up the cube"),
        (b["id"], lab_db.DEFAULT_MAX_SECONDS, "put it down"),
    ]
    store.update_sequence(sequence["id"], "cube 2", [{"prompt_id": b["id"], "seconds": 10}])
    assert store.sequence(sequence["id"])["name"] == "cube 2"
    store.delete_prompt(b["id"])
    assert store.sequence(sequence["id"])["steps"] == []
    store.delete_sequence(sequence["id"])
    assert store.sequences() == []


@pytest.mark.parametrize(
    ("name", "steps"),
    [
        ("", [{"prompt_id": 1, "seconds": 30}]),
        ("x", []),
        ("x", "no"),
        ("x", [{"prompt_id": 99, "seconds": 30}]),
        ("x", [{"prompt_id": 1, "seconds": 2}]),
        ("x", [{"prompt_id": True, "seconds": 30}]),
        ("x", [{"prompt_id": 1, "seconds": 30}] * (lab_db.SEQUENCE_MAX_STEPS + 1)),
    ],
)
def test_a_bad_sequence_is_refused(store, name, steps):
    store.add_prompt("pick up the cube")
    with pytest.raises(lab_db.LabError):
        store.add_sequence(name, steps)
