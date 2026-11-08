"""The data page's server: the page, its data, the videos, the actions, and who may send them."""

import http.client
import json
import pathlib
import threading
from collections.abc import Iterator

import pytest
import yaml
from conftest import write_episodes

from dream_hackathon import data_edit, data_overview, dataset_cli, station_lock

HEADERS = {"X-Dashboard": "1", "Content-Type": "application/json"}


@pytest.fixture
def served(tmp_path: pathlib.Path, monkeypatch) -> Iterator[tuple[pathlib.Path, int]]:
    """A data root of two small datasets, served on a free loopback port, with no real lock or edit log involved."""
    monkeypatch.setattr(station_lock, "LOCK_PATH", tmp_path / "arms.lock")
    monkeypatch.setattr(data_edit, "LOG_PATH", tmp_path / "edits.log")
    monkeypatch.setattr(data_edit, "recordings", dict)
    root = tmp_path / "data"
    write_episodes(root / "cubes", "yam", episodes=3, frames=55, tasks=["pick up the cube", "stack the cups"])
    write_episodes(root / "cups", "yam", episodes=2, frames=55, tasks=["stack the cups"])
    server = data_overview.make_server(root, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield root, server.server_port
    server.shutdown()
    server.server_close()


def call(
    port: int, method: str, path: str, body: object = None, headers: dict[str, str] | None = None
) -> tuple[int, dict, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=None if body is None else json.dumps(body), headers=headers or {})
    response = connection.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def overview(port: int) -> dict:
    status, _, body = call(port, "GET", "/api/overview")
    assert status == 200
    return json.loads(body)


def post(port: int, action: str, body: object, headers: dict[str, str] | None = HEADERS) -> tuple[int, dict]:
    status, _, raw = call(port, "POST", f"/api/{action}", body, headers)
    return status, json.loads(raw)


def test_the_page_comes_with_the_first_scan_inside_it(served) -> None:
    _, port = served
    status, headers, body = call(port, "GET", "/")
    page = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert (
        "default-src 'none'" in headers["Content-Security-Policy"]
        and "media-src 'self'" in headers["Content-Security-Policy"]
    )
    assert "__BOOT__" not in page
    boot = page.split('<script id="boot" type="application/json">')[1].split("</script>")[0]
    assert [d["name"] for d in json.loads(boot)["datasets"]] == ["cubes", "cups"]


def test_what_the_data_says_cannot_break_out_of_the_page(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(data_edit, "recordings", dict)
    root = tmp_path / "data"
    evil = "</script><img src=x onerror=alert(1)>"
    write_episodes(root / "cubes", "yam", episodes=2, frames=55, tasks=[evil])
    server = data_overview.make_server(root, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page = call(server.server_port, "GET", "/")[2].decode()
        boot = page.split('<script id="boot" type="application/json">')[1].split("</script>")[0]
        assert "<" not in boot and evil not in page
        assert json.loads(boot)["datasets"][0]["episodes"][0]["task"] == evil  # and it still reads back as it was
        assert json.loads(call(server.server_port, "GET", "/api/overview")[2])["slots"] == {evil: 1}
    finally:
        server.shutdown()
        server.server_close()


def test_the_overview_holds_what_the_page_draws(served) -> None:
    _, port = served
    data = overview(port)
    assert [d["name"] for d in data["datasets"]] == ["cubes", "cups"]
    assert list(data["slots"]) == ["pick up the cube", "stack the cups"] and set(data["slots"].values()) == {1, 2}
    cubes = data["datasets"][0]
    assert [e["task"] for e in cubes["episodes"]] == ["pick up the cube", "stack the cups", "pick up the cube"]
    assert {
        "index",
        "frames",
        "clock",
        "size",
        "bytes",
        "when",
        "still",
        "mirrored",
        "missing",
        "twins",
        "barPct",
        "slot",
    } <= set(cubes["episodes"][0])
    assert [c["name"] for c in cubes["cameras"]] == ["overhead", "wrist_l", "wrist_r"]
    assert cubes["unused"] == [] and cubes["recording"] is None and cubes["status"] == "ok"
    assert data["summary"]["datasets"] == 2 and data["summary"]["episodes"] == 5
    assert data["rootShort"] and data["scanned"]


def test_the_scene_camera_comes_first_then_the_others_by_name(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(data_edit, "recordings", dict)
    write_episodes(tmp_path / "data" / "metal", "makermods", episodes=2, frames=55, tasks=["a"])
    data = data_overview.overview_json(data_overview.scan(tmp_path / "data"))
    assert [c["name"] for c in data["datasets"][0]["cameras"]] == ["scene", "left_wrist", "right_wrist"]


def test_an_empty_root_still_gives_a_page(tmp_path: pathlib.Path) -> None:
    (tmp_path / "data").mkdir()
    server = data_overview.make_server(tmp_path / "data", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert call(server.server_port, "GET", "/")[0] == 200
        assert json.loads(call(server.server_port, "GET", "/api/overview")[2])["datasets"] == []
    finally:
        server.shutdown()
        server.server_close()


def test_only_this_computers_names_may_ask(served) -> None:
    _, port = served
    for method, path in (("GET", "/"), ("GET", "/api/overview"), ("POST", "/api/delete-episodes")):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request(
            method, path, headers={"Host": "evil.example", **HEADERS}, body="{}" if method == "POST" else None
        )
        assert connection.getresponse().status == 403
    assert call(port, "GET", "/nothing")[0] == 404
    assert call(port, "PUT", "/api/overview")[0] == 501


def test_a_video_comes_whole_or_by_range_and_nothing_else_is_served(served) -> None:
    root, port = served
    video = root / "cubes" / "videos" / "chunk-000" / "observation.images.overhead" / "episode_000001.mp4"
    size = video.stat().st_size
    status, headers, body = call(port, "GET", "/video/cubes/1/overhead.mp4")
    assert (status, headers["Content-Type"], headers["Accept-Ranges"], len(body)) == (200, "video/mp4", "bytes", size)
    assert body == video.read_bytes()
    status, headers, body = call(port, "GET", "/video/cubes/1/overhead.mp4", headers={"Range": "bytes=10-19"})
    assert (status, headers["Content-Range"], body) == (206, f"bytes 10-19/{size}", video.read_bytes()[10:20])
    status, headers, body = call(port, "GET", "/video/cubes/1/overhead.mp4", headers={"Range": "bytes=-5"})
    assert (status, body) == (206, video.read_bytes()[-5:])
    status, headers, body = call(port, "GET", "/video/cubes/1/overhead.mp4", headers={"Range": f"bytes={size}-"})
    assert (status, headers["Content-Range"]) == (416, f"bytes */{size}")
    for path in (
        "/video/cubes/9/overhead.mp4",
        "/video/cubes/1/nope.mp4",
        "/video/nope/1/overhead.mp4",
        "/video/..%2f..%2fetc/1/overhead.mp4",
        "/video/cubes/1/../../x.mp4",
    ):
        assert call(port, "GET", path)[0] == 404, path


def test_a_video_path_cannot_leave_the_dataset(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "data"
    write_episodes(root / "cubes", "yam", episodes=2, frames=55, tasks=["a"])
    (tmp_path / "secret.mp4").write_bytes(b"x")
    row = data_overview.scan(root).datasets[0]
    assert data_overview.video_file(row, 0, "overhead") is not None
    row.video_path = "../../secret.mp4"
    assert data_overview.video_file(row, 0, "overhead") is None


def test_an_action_is_taken_from_the_page_only(served) -> None:
    root, port = served
    body = {"dataset": "cubes", "episodes": [0]}
    assert post(port, "delete-episodes", body, headers={})[0] == 403  # no header
    assert post(port, "delete-episodes", body, headers={"X-Dashboard": "1", "Origin": "http://evil.example"})[0] == 403
    status, _ = post(port, "delete-episodes", body, headers={"X-Dashboard": "1", "Origin": f"http://127.0.0.1:{port}"})
    assert status == 200  # the page's own origin is let through
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/api/delete-episodes", body="{not json", headers=HEADERS)
    assert connection.getresponse().status == 400
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/api/delete-episodes", body="[1]", headers=HEADERS)
    assert connection.getresponse().status == 400
    assert post(port, "nothing", {})[0] == 404
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(
        "POST", "/api/delete-episodes", body=" " * 20000, headers={**HEADERS, "Content-Length": "20000"}
    )
    assert connection.getresponse().status == 413


def test_every_action_works_and_the_next_overview_shows_it(served) -> None:
    root, port = served
    status, answer = post(port, "rename-task", {"task": "stack the cups", "to": "stack the bowls"})
    assert (status, answer["ok"]) == (200, True) and "Renamed the task" in answer["message"]
    assert "stack the bowls" in overview(port)["slots"]  # at once, not after the scan is reused

    status, answer = post(port, "rename-episodes", {"dataset": "cubes", "episodes": [0], "task": "lift the cube"})
    assert status == 200 and [e["task"] for e in overview(port)["datasets"][0]["episodes"]][0] == "lift the cube"

    status, answer = post(port, "delete-episodes", {"dataset": "cups", "episodes": [0]})
    assert status == 200 and len(overview(port)["datasets"][1]["episodes"]) == 1

    status, answer = post(port, "delete-task", {"task": "stack the bowls", "dataset": "cubes"})
    assert status == 200 and [e["task"] for e in overview(port)["datasets"][0]["episodes"]] == [
        "lift the cube",
        "pick up the cube",
    ]

    status, answer = post(port, "delete-task", {"task": "stack the bowls"})  # the rest, in every dataset
    assert status == 200 and overview(port)["datasets"][1]["episodes"] == []

    status, answer = post(port, "rename-dataset", {"dataset": "cubes", "to": "blocks", "with_eef": False})
    assert status == 200 and [d["name"] for d in overview(port)["datasets"]] == ["blocks", "cups"]
    assert (root / "blocks").is_dir() and not (root / "cubes").exists()


def test_a_refused_action_says_why_and_changes_nothing(served, monkeypatch) -> None:
    root, port = served
    status, answer = post(port, "delete-episodes", {"dataset": "cubes", "episodes": [9]})
    assert status == 409 and "no episode 9" in answer["error"]
    status, answer = post(port, "rename-dataset", {"dataset": "cubes", "to": "bad name"})
    assert status == 409 and "use letters" in answer["error"]
    status, answer = post(port, "delete-episodes", {"dataset": ["cubes"], "episodes": [0]})  # not a string
    assert status == 409 and "no dataset called" in answer["error"]
    monkeypatch.setattr(data_edit, "recordings", lambda: {(root / "cubes").resolve(): "mirror_collect.py (pid 9)"})
    status, answer = post(port, "delete-episodes", {"dataset": "cubes", "episodes": [0]})
    assert status == 409 and "being recorded into by mirror_collect.py (pid 9)" in answer["error"]
    data = overview(port)
    assert data["datasets"][0]["recording"] == "mirror_collect.py (pid 9)" and data["datasets"][1]["recording"] is None
    assert len(data["datasets"][0]["episodes"]) == 3 and len(data["datasets"][1]["episodes"]) == 2


def test_the_station_config_is_checked_against_the_folders(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(data_edit, "recordings", dict)
    root = tmp_path / "data"
    write_episodes(root / "cubes", "yam", episodes=2, frames=55, tasks=["a"])
    config = tmp_path / "station.yaml"
    for folder, expected in (("cubes", 0), ("elsewhere", 1)):
        config.write_text(yaml.safe_dump({"dataset_dir": str(root / folder)}))
        server = data_overview.make_server(root, 0, config)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            data = overview(server.server_port)
            reminders = [a for a in data["attention"] if "station.yaml" in a["text"]]
            assert len(reminders) == expected and data["stationDir"] == str(root / folder)
        finally:
            server.shutdown()
            server.server_close()
    assert data_overview.station_dataset_dir(tmp_path / "missing.yaml") is None
    (tmp_path / "broken.yaml").write_text(": : :")
    assert data_overview.station_dataset_dir(tmp_path / "broken.yaml") is None


def test_the_command_serves_the_folder_it_is_given(tmp_path: pathlib.Path, monkeypatch) -> None:
    served_with: list[tuple] = []
    monkeypatch.setattr(data_overview, "serve", lambda *args: served_with.append(args))
    (tmp_path / "data").mkdir()
    dataset_cli.main(
        ["overview", str(tmp_path / "data"), "--port", "9000", "--config", str(tmp_path / "station.yaml")]
    )
    assert served_with == [(tmp_path / "data", 9000, tmp_path / "station.yaml")]
    with pytest.raises(SystemExit, match="is not a folder"):
        dataset_cli.main(["overview", str(tmp_path / "nothing")])


def test_dhk_dataset_drop_refuses_for_a_recorder_that_takes_no_lock(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(station_lock, "LOCK_PATH", tmp_path / "arms.lock")
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=55, tasks=["a"])
    monkeypatch.setattr(data_edit, "recordings", lambda: {root.resolve(): "mirror_collect.py (pid 9)"})
    with pytest.raises(SystemExit, match="mirror_collect.py \\(pid 9\\) is recording into"):
        dataset_cli.main(["drop", str(root), "0", "--yes"])
    assert len(list(root.glob("data/chunk-000/*.parquet"))) == 2
