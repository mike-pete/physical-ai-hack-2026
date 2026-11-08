import http.server
import json
import pathlib
import socket
import threading

import pytest
from dream_hackathon import lab, lab_db


class FakeWorker:
    def __init__(self):
        self.sent = []
        self.up = True

    def send(self, command):
        if not self.up:
            raise lab_db.LabError("the arm worker is not running: restart dhk-lab")
        self.sent.append(command)

    def alive(self):
        return self.up


@pytest.fixture
def rig(tmp_path):
    store = lab_db.Store(tmp_path / "lab")
    frames = tmp_path / "frames"
    frames.mkdir()
    the_lab = lab.Lab(store, frames)
    the_lab.worker = FakeWorker()
    the_lab.on_event({"event": "ready", "cameras": ["overhead", "wrist_l"]})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lab.make_handler(the_lab))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield the_lab, server.server_address[1]
    server.shutdown()
    server.server_close()


def raw(port, method, path, headers=None, body=b"", claimed_length=None):
    """One request with exactly the headers given (http.client would add its own); returns (status, headers, body)."""
    lines = [f"{method} {path} HTTP/1.1"]
    lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
    lines += [
        f"Content-Length: {len(body) if claimed_length is None else claimed_length}",
        "Connection: close",
        "",
        "",
    ]
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall("\r\n".join(lines).encode() + body)
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    head, _, payload = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    parsed = {}
    for line in head.decode().split("\r\n")[1:]:
        key, _, value = line.partition(": ")
        parsed[key.lower()] = value
    return status, parsed, payload


def ok(port, extra=None):
    return {"Host": f"localhost:{port}", lab.ACTION_HEADER: "1", **(extra or {})}


def post(port, path, data=None, headers=None):
    status, _, body = raw(port, "POST", path, headers or ok(port), json.dumps(data or {}).encode())
    return status, json.loads(body)


def get(port, path, headers=None):
    status, head, body = raw(port, "GET", path, headers or {"Host": f"localhost:{port}"})
    return status, head, body


# ---------------------------------------------------------------------- security


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST:9100", "127.0.0.1:9100", "[::1]:9100"])
def test_loopback_hosts_are_allowed(host):
    assert lab.host_allowed(host)


@pytest.mark.parametrize("host", [None, "", "evil.example", "localhost.evil.example", "127.0.0.1.evil.example:9100"])
def test_other_hosts_are_refused(host):
    assert not lab.host_allowed(host)


def test_a_request_with_a_foreign_host_is_refused(rig):
    _, port = rig
    assert get(port, "/api/status", {"Host": "evil.example"})[0] == 403
    assert raw(port, "POST", "/api/prompts", {"Host": "evil.example", lab.ACTION_HEADER: "1"}, b"{}")[0] == 403


def test_a_post_needs_the_action_header_and_a_matching_origin(rig):
    _, port = rig
    body = json.dumps({"text": "x"}).encode()
    assert raw(port, "POST", "/api/prompts", {"Host": f"localhost:{port}"}, body)[0] == 403
    assert raw(port, "POST", "/api/prompts", ok(port, {"Origin": "http://evil.example"}), body)[0] == 403
    assert raw(port, "POST", "/api/prompts", ok(port, {"Origin": f"http://localhost:{port}"}), body)[0] == 200


def test_a_bad_body_is_refused(rig):
    _, port = rig
    headers = ok(port)
    assert raw(port, "POST", "/api/prompts", headers, b"not json")[0] == 400
    assert raw(port, "POST", "/api/prompts", headers, b"[1]")[0] == 400
    assert raw(port, "POST", "/api/prompts", headers, claimed_length=lab.MAX_BODY_BYTES + 1)[0] == 413


# ---------------------------------------------------------------------- prompts


def test_prompts_can_be_added_listed_edited_and_deleted(rig):
    the_lab, port = rig
    assert post(port, "/api/prompts", {"text": "pick up the cube"})[0] == 200
    prompt_id = the_lab.store.prompts()[0]["id"]
    status, _, body = get(port, "/api/prompts")
    assert status == 200 and json.loads(body)[0]["text"] == "pick up the cube"
    assert post(port, f"/api/prompts/{prompt_id}", {"text": "pick up the red cube"})[0] == 200
    assert the_lab.store.prompt(prompt_id)["text"] == "pick up the red cube"
    assert post(port, f"/api/prompts/{prompt_id}/delete")[0] == 200
    assert the_lab.store.prompts() == []
    assert post(port, "/api/prompts", {"text": ""})[0] == 400


# ---------------------------------------------------------------------- arms


def add_prompt(the_lab):
    return the_lab.store.add_prompt("open the drawer")["id"]


def test_a_run_needs_held_arms_and_records_the_prompt(rig):
    the_lab, port = rig
    prompt_id = add_prompt(the_lab)
    status, body = post(port, f"/api/prompts/{prompt_id}/run", {"dry_run": True})
    assert status == 409 and "Hold here" in body["error"]  # idle: no held pose yet

    the_lab.on_event({"event": "state", "state": "holding"})
    status, body = post(port, f"/api/prompts/{prompt_id}/run", {"dry_run": True, "max_seconds": 30})
    assert status == 200
    command = the_lab.worker.sent[-1]
    assert command["cmd"] == "run" and command["prompt"] == "open the drawer" and command["dry_run"] is True
    assert command["max_seconds"] == 30 and command["checkpoint"] == lab_db.DEFAULT_CHECKPOINT
    assert the_lab.status()["state"] == "running" and the_lab.status()["run"]["prompt"] == "open the drawer"
    assert post(port, f"/api/prompts/{prompt_id}/run")[0] == 409  # one run at a time

    the_lab.on_event({"event": "run_started", "run_id": command["run_id"]})
    the_lab.on_event({"event": "run_ended", "run_id": command["run_id"], "status": "stopped"})
    the_lab.on_event({"event": "state", "state": "holding"})
    saved = the_lab.store.run(command["run_id"])
    assert saved["status"] == "stopped" and saved["prompt_text"] == "open the drawer" and saved["dry_run"] == 1
    assert the_lab.status()["run"] is None


def test_a_run_limit_must_be_sane(rig):
    the_lab, port = rig
    the_lab.on_event({"event": "state", "state": "holding"})
    prompt_id = add_prompt(the_lab)
    assert post(port, f"/api/prompts/{prompt_id}/run", {"max_seconds": 1})[0] == 400
    assert post(port, f"/api/prompts/{prompt_id}/run", {"dry_run": "yes"})[0] == 400
    assert the_lab.store.runs() == [] and the_lab.worker.sent == []


@pytest.mark.parametrize(
    ("path", "state", "allowed"),
    [
        ("/api/arms/weightless", "idle", True),
        ("/api/arms/weightless", "running", False),
        ("/api/arms/hold", "weightless", True),
        ("/api/arms/hold", "idle", False),
        ("/api/arms/home", "holding", True),
        ("/api/arms/home", "running", False),
        ("/api/arms/home", "homing", False),
        ("/api/arms/power_off", "holding", True),
        ("/api/stop", "running", True),
        ("/api/stop", "holding", False),
    ],
)
def test_each_arm_command_is_allowed_from_the_right_states_only(rig, path, state, allowed):
    the_lab, port = rig
    the_lab.on_event({"event": "state", "state": state})
    status, _ = post(port, path)
    assert (status == 200) is allowed
    assert bool(the_lab.worker.sent) is allowed


def test_a_command_to_a_dead_worker_is_refused_and_the_state_is_kept(rig):
    the_lab, port = rig
    the_lab.worker.up = False
    status, body = post(port, "/api/arms/weightless")
    assert status == 400 and "not running" in body["error"]
    assert the_lab.status()["state"] == "idle"


def test_when_the_worker_exits_the_run_ends_as_an_error(rig):
    the_lab, port = rig
    the_lab.on_event({"event": "state", "state": "holding"})
    prompt_id = add_prompt(the_lab)
    assert post(port, f"/api/prompts/{prompt_id}/run")[0] == 200
    the_lab.on_exit(1)
    assert the_lab.status()["state"] == "stopped" and the_lab.status()["run"] is None
    assert the_lab.store.runs()[0]["status"] == "error"


# ---------------------------------------------------------------------- cameras and videos


def test_a_camera_frame_is_served_for_known_cameras_only(rig):
    the_lab, port = rig
    assert get(port, "/api/camera/overhead.jpg")[0] == 503  # no frame written yet
    (the_lab.frames_dir / "overhead.jpg").write_bytes(b"\xff\xd8jpeg")
    status, head, body = get(port, "/api/camera/overhead.jpg")
    assert status == 200 and head["content-type"] == "image/jpeg" and body == b"\xff\xd8jpeg"
    assert get(port, "/api/camera/secret.jpg")[0] == 404


def test_a_run_video_is_served_with_ranges(rig):
    the_lab, port = rig
    run = the_lab.store.start_run("p", "c", prompt_id=None, dry_run=False, max_seconds=10)
    (pathlib.Path(run["dir"]) / "overhead.mp4").write_bytes(bytes(range(100)))
    path = f"/api/runs/{run['id']}/video/overhead.mp4"
    status, head, body = get(port, path)
    assert status == 200 and len(body) == 100 and head["accept-ranges"] == "bytes"
    status, head, body = get(port, path, {"Host": f"localhost:{port}", "Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20)) and head["content-range"] == "bytes 10-19/100"
    status, _, body = get(port, path, {"Host": f"localhost:{port}", "Range": "bytes=-5"})
    assert status == 206 and body == bytes(range(95, 100))
    assert get(port, path, {"Host": f"localhost:{port}", "Range": "bytes=500-"})[0] == 416
    assert get(port, f"/api/runs/{run['id']}/video/nothing.mp4")[0] == 404
    assert get(port, "/api/runs/999/video/overhead.mp4")[0] == 404


def test_a_failed_run_raises_an_alert_the_page_can_show(rig):
    the_lab, port = rig
    the_lab.on_event({"event": "state", "state": "holding"})
    assert the_lab.status()["alert"] is None
    prompt_id = add_prompt(the_lab)
    post(port, f"/api/prompts/{prompt_id}/run")
    run_id = the_lab.worker.sent[-1]["run_id"]
    the_lab.on_event(
        {"event": "run_ended", "run_id": run_id, "status": "error", "error": "ValueError: prompt rejected"}
    )
    alert = the_lab.status()["alert"]
    assert "prompt rejected" in alert["text"] and "open the drawer" in alert["text"]
    the_lab.on_event({"event": "error", "message": "shutdown failed"})
    assert the_lab.status()["alert"]["id"] > alert["id"]
    assert get(port, "/api/status")[0] == 200
