"""The dashboard serves its page, the state, the cameras and the episodes' videos, and nothing else."""

import json
import pathlib
import urllib.error
import urllib.request

import numpy as np
import pytest
from conftest import write_episodes

from dream_hackathon import cameras, dashboard, layouts


def _get(board: dashboard.Dashboard, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(f"http://127.0.0.1:{board.port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@pytest.fixture
def board(tmp_path: pathlib.Path):
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=50, tasks=["pick up the cube"])
    frame = np.full((180, 320, 3), 128, np.uint8)
    stalled = {"wrist_r"}

    def frame_of(role: str):
        if role in stalled:
            raise cameras.CameraError(f"camera '{role}' stalled")
        return frame

    board = dashboard.Dashboard(0, frame_of, layouts.video_keys("yam"), root, title="test station")
    board.stalled = stalled
    yield board
    board.close()


def test_serves_the_page_and_the_state(board: dashboard.Dashboard) -> None:
    status, page = _get(board, "/")
    assert status == 200 and b"<title>Station dashboard</title>" in page
    board.publish({"mode": "live", "recording": True})
    board.event("saved episode 0")
    status, body = _get(board, "/api/status")
    state = json.loads(body)
    assert state["snapshot"]["recording"] and state["events"][0].endswith("saved episode 0")
    assert state["cameras"] == list(layouts.CAMERA_ROLES)


def test_cameras_as_jpeg_and_a_stall_as_503(board: dashboard.Dashboard) -> None:
    status, body = _get(board, "/api/camera/wrist_r.jpg")
    assert status == 503 and b"stalled" in body
    assert _get(board, "/api/camera/overhead.jpg")[0] == 200  # one stalled camera leaves the others on the page
    board.stalled.clear()
    status, body = _get(board, "/api/camera/wrist_r.jpg")
    assert status == 200 and body[:2] == b"\xff\xd8"  # a JPEG
    assert _get(board, "/api/camera/kitchen.jpg")[0] == 404


def test_episodes_and_their_videos_with_ranges(board: dashboard.Dashboard) -> None:
    episodes = json.loads(_get(board, "/api/episodes")[1])
    assert [e["index"] for e in episodes] == [0, 1] and episodes[0]["frames"] == 50
    status, video = _get(board, "/api/video/1/wrist_l.mp4")
    assert status == 200 and video[4:8] == b"ftyp"
    status, part = _get(board, "/api/video/1/wrist_l.mp4", {"Range": "bytes=0-99"})
    assert status == 206 and part == video[:100]
    for path in (
        "/api/video/7/wrist_l.mp4",
        "/api/video/0/kitchen.mp4",
        "/api/video/0/../../meta.mp4",
        "/meta/info.json",
    ):
        assert _get(board, path)[0] == 404, path


def test_the_page_gets_each_devices_state_and_the_phase(board: dashboard.Dashboard) -> None:
    from dream_hackathon import devices

    board.devices([devices.Status("Quest"), devices.Status("camera wrist_l", "no frame for 3 s; check its USB cable")])
    board.phase("waiting")
    state = json.loads(_get(board, "/api/status")[1])
    assert state["phase"] == "waiting"
    assert state["devices"] == [
        {"name": "Quest", "ok": True, "problem": "", "fix": None},
        {"name": "camera wrist_l", "ok": False, "problem": "no frame for 3 s; check its USB cable", "fix": None},
    ]
    page = _get(board, "/")[1].decode()
    assert 'id="devices"' in page and "press A on the right controller" in page


def test_the_page_has_no_control_characters(board: dashboard.Dashboard) -> None:
    page = _get(board, "/")[1].decode()
    assert "\\25CF" in page  # the CSS escape of the status dot, as the browser must get it
    assert not any(ord(char) < 32 and char not in "\n\t" for char in page)


def _post(board: dashboard.Dashboard, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{board.port}/api/fix", data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_a_fix_is_taken_from_the_page_only(tmp_path: pathlib.Path) -> None:
    asked = []
    board = dashboard.Dashboard(
        0,
        lambda role: np.zeros((180, 320, 3), np.uint8),
        layouts.video_keys("yam"),
        None,
        title="test station",
        on_fix=lambda fix: asked.append(fix) or (True, f"ran {fix}"),
    )
    try:
        body = json.dumps({"fix": "can-up:can_follower_l"}).encode()
        own = f"http://127.0.0.1:{board.port}"
        assert _post(board, body, {"Content-Type": "application/json"})[0] == 403  # no X-Dashboard header
        assert _post(board, body, {"X-Dashboard": "1", "Origin": "https://example.com"})[0] == 403  # another site
        status, reply = _post(board, body, {"X-Dashboard": "1", "Origin": own})
        assert status == 200 and json.loads(reply) == {"ok": True, "message": "ran can-up:can_follower_l"}
        assert asked == ["can-up:can_follower_l"]
    finally:
        board.close()


def test_a_page_without_fixes_refuses_them(board: dashboard.Dashboard) -> None:
    assert _post(board, b'{"fix": "can-up:x"}', {"X-Dashboard": "1"})[0] == 404


def test_the_episode_buttons_are_taken_from_the_page_only(tmp_path: pathlib.Path) -> None:
    asked = []
    board = dashboard.Dashboard(
        0,
        lambda role: np.zeros((180, 320, 3), np.uint8),
        layouts.video_keys("yam"),
        None,
        title="test station",
        on_session=lambda action: asked.append(action) or (True, f"{action}: requested"),
    )
    try:
        assert json.loads(_get(board, "/api/status")[1])["can_session"]
        own = f"http://127.0.0.1:{board.port}"

        def post(headers: dict[str, str]) -> tuple[int, bytes]:
            request = urllib.request.Request(
                f"{own}/api/session", data=b'{"action": "reset"}', headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()

        assert post({"Content-Type": "application/json"})[0] == 403  # no X-Dashboard header
        assert post({"X-Dashboard": "1", "Origin": "https://example.com"})[0] == 403  # another site
        assert asked == []
        status, reply = post({"X-Dashboard": "1", "Origin": own})
        assert status == 200 and json.loads(reply) == {"ok": True, "message": "reset: requested"}
        assert asked == ["reset"]
        page = _get(board, "/")[1].decode()
        assert all(
            label in page for label in ("Start episode", "Save episode", "Discard episode", "Reset arms to home")
        )
        assert "Reset to home after each episode" in page
    finally:
        board.close()


def test_a_page_without_a_session_has_no_episode_buttons(board: dashboard.Dashboard) -> None:
    assert not json.loads(_get(board, "/api/status")[1])["can_session"]
    request = urllib.request.Request(
        f"http://127.0.0.1:{board.port}/api/session", data=b'{"action": "start"}', headers={"X-Dashboard": "1"}
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)
    assert refused.value.code == 404
