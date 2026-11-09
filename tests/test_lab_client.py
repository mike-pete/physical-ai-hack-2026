import threading
import time

import pytest
from dream_hackathon import lab_client
from test_lab_server import rig  # noqa: F401  (a pytest fixture)


@pytest.fixture
def client(rig):  # noqa: F811
    the_lab, port = rig
    return the_lab, lab_client.LabClient(f"http://127.0.0.1:{port}")


def test_only_this_computer_is_taken():
    with pytest.raises(ValueError):
        lab_client.LabClient("http://10.0.0.5:9100")


def test_an_unreachable_lab_is_said_in_words():
    assert "cannot reach the lab" in lab_client.LabClient("http://127.0.0.1:9").stop_and_go_home()


def test_stop_and_go_home_stops_waits_then_goes_home(client):
    the_lab, api = client
    the_lab.on_event({"event": "state", "state": "running"})

    def settle():
        time.sleep(0.4)
        the_lab.on_event({"event": "state", "state": "holding"})

    threading.Thread(target=settle).start()
    assert api.stop_and_go_home(wait_s=5) == "Done: stopped the run and sent the arms home slowly."
    assert [c["cmd"] for c in the_lab.worker.sent] == ["stop", "home"]


def test_with_no_run_it_only_goes_home(client):
    the_lab, api = client
    the_lab.on_event({"event": "state", "state": "holding"})
    assert api.stop_and_go_home() == "Done: sent the arms home slowly."
    assert [c["cmd"] for c in the_lab.worker.sent] == ["home"]


def test_it_does_not_go_home_if_the_run_never_settles(client):
    the_lab, api = client
    the_lab.on_event({"event": "state", "state": "running"})
    assert "did not settle" in api.stop_and_go_home(wait_s=0.5)
    assert [c["cmd"] for c in the_lab.worker.sent] == ["stop"]


def test_a_refused_home_is_reported_not_raised(client):
    the_lab, api = client
    the_lab.on_event({"event": "state", "state": "homing"})
    assert api.stop_and_go_home().startswith("Nothing done: home is not possible")
    assert the_lab.worker.sent == []


def test_stop_run_only_stops_a_run(client):
    the_lab, api = client
    assert api.stop_run() == "No run to stop: the arms are idle."
    the_lab.on_event({"event": "state", "state": "running"})
    assert api.stop_run().startswith("Stopping the run")
    assert [c["cmd"] for c in the_lab.worker.sent] == ["stop"]


def test_keep_warm_runs_a_short_dry_run_and_adds_its_prompt_once(client):
    the_lab, api = client
    the_lab.on_event({"event": "state", "state": "holding"})
    assert api.keep_warm() == "Pinged the model with a dry run."
    run = the_lab.worker.sent[-1]
    assert run["cmd"] == "run" and run["dry_run"] is True and run["max_seconds"] == lab_client.KEEP_WARM_SECONDS
    assert run["prompt"] == lab_client.KEEP_WARM_PROMPT
    the_lab.on_event({"event": "run_ended", "status": "finished", "error": ""})
    the_lab.on_event({"event": "state", "state": "holding"})
    assert api.keep_warm() == "Pinged the model with a dry run."
    assert [p["text"] for p in the_lab.store.prompts()].count(lab_client.KEEP_WARM_PROMPT) == 1


def test_keep_warm_skips_unless_the_arms_are_holding(client):
    the_lab, api = client
    assert api.keep_warm().startswith("Skipped: the arms are idle")
    the_lab.on_event({"event": "state", "state": "running"})
    assert api.keep_warm().startswith("Skipped: the arms are running")
    assert the_lab.worker.sent == []
    assert the_lab.store.prompts() == []


def test_keep_warm_says_when_the_lab_is_unreachable():
    assert lab_client.LabClient("http://127.0.0.1:9").keep_warm().startswith("Skipped: cannot reach the lab")
