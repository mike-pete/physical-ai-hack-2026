"""tools/station_io.py: the station client against a fake station, and reading a recorder's video while it grows."""

import datetime
import importlib.util
import json
import pathlib

import av
import numpy as np
import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
RECORDER_OPTIONS = {"movflags": "frag_keyframe+empty_moov+default_base_moof", "flush_packets": "1"}


@pytest.fixture(scope="module")
def sio():
    spec = importlib.util.spec_from_file_location("station_io_tool", TOOLS / "station_io.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def station(sio):
    clock = Clock()
    fake = sio.FakeStation(clock=clock, prompt="unplug the cables")
    fake.clock = clock
    base_url = fake.serve()
    yield fake, sio.StationClient(base_url, timeout=5)
    fake.close()


# --- the client and the fake station ---------------------------------------------------------------------------


def test_status_of_an_idle_station(station) -> None:
    fake, client = station
    status = client.status()
    assert status["state"] == "idle"
    assert status["prompt"] == "unplug the cables"
    assert status["checkpoint"] == "dreamlabs/dream-1-beta"
    assert status["run"] == 0


def test_a_run_goes_through_start_stop_and_home(station) -> None:
    fake, client = station
    started = client.start("new prompt", "my/checkpoint")
    assert (started["state"], started["prompt"], started["checkpoint"], started["run"]) == (
        "running",
        "new prompt",
        "my/checkpoint",
        1,
    )
    fake.clock.t = 12.0
    assert client.status()["run_elapsed_s"] == 12.0
    assert client.stop()["state"] == "idle"
    assert client.home()["state"] == "idle"
    assert [route for route, _ in fake.calls] == ["/start", "/stop", "/home"]
    assert fake.calls[0][1] == {"prompt": "new prompt", "checkpoint": "my/checkpoint"}


def test_start_keeps_the_stored_prompt_and_checkpoint_when_given_none(station) -> None:
    fake, client = station
    assert client.start()["prompt"] == "unplug the cables"
    assert fake.calls[0][1] == {}


def test_a_start_while_running_is_refused_with_409(sio, station) -> None:
    _, client = station
    client.start()
    with pytest.raises(sio.StationRefusedError, match="refused") as caught:
        client.start("again")
    assert caught.value.status == 409


def test_home_is_refused_while_a_run_is_going(sio, station) -> None:
    _, client = station
    client.start()
    with pytest.raises(sio.StationRefusedError):
        client.home()


def test_a_prompt_during_a_run_is_applied_and_recorded(station) -> None:
    fake, client = station
    client.start()
    client.set_prompt("pull the cable out")
    assert fake.applied_prompts == ["pull the cable out"]
    assert client.status()["prompt"] == "pull the cable out"


def test_a_prompt_while_idle_is_stored_but_not_applied(station) -> None:
    fake, client = station
    client.set_prompt("later")
    assert fake.applied_prompts == []
    assert fake.prompt == "later"


def test_an_empty_prompt_is_a_plain_error_not_a_refusal(sio, station) -> None:
    _, client = station
    with pytest.raises(sio.StationError, match="non-empty") as caught:
        client.set_prompt("   ")
    assert not isinstance(caught.value, sio.StationRefusedError)


def test_the_scripted_failure_shows_in_the_status_and_start_clears_it(sio, station) -> None:
    fake, client = station
    client.start()
    fake.fail("the model stopped answering")
    status = client.status()
    assert status["state"] == "error"
    assert status["last_error"]["message"] == "the model stopped answering"
    assert client.start("retry")["last_error"] is None


def test_a_scripted_refusal_happens_once(sio, station) -> None:
    fake, client = station
    fake.refuse("/start", "busy")
    with pytest.raises(sio.StationRefusedError, match="busy"):
        client.start("p")
    assert client.start("p")["state"] == "running"


def test_an_unreachable_station_says_how_to_start_it(sio) -> None:
    with pytest.raises(sio.StationError, match="dreamcli robot up"):
        sio.StationClient("http://127.0.0.1:1", timeout=1).status()


def test_the_local_client_raises_the_same_errors(sio) -> None:
    fake = sio.FakeStation(prompt="p")
    client = fake.client()
    assert client.status()["state"] == "idle"
    assert client.start()["state"] == "running"
    with pytest.raises(sio.StationRefusedError):
        client.start()
    with pytest.raises(sio.StationRefusedError):
        client.home()
    assert client.set_prompt("q")["prompt"] == "q"
    with pytest.raises(sio.StationError) as caught:
        client.set_prompt(" ")
    assert not isinstance(caught.value, sio.StationRefusedError)
    assert client.stop()["state"] == "idle"
    assert [route for route, _ in fake.calls] == ["/start", "/start", "/home", "/prompt", "/prompt", "/stop"]


# --- picking frames --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "count", "expected"),
    [(0, 3, []), (1, 3, [0]), (10, 1, [9]), (10, 3, [0, 4, 9]), (2, 5, [0, 1]), (31, 3, [0, 15, 30])],
)
def test_spread_indexes_include_the_last_frame(sio, n, count, expected) -> None:
    assert sio.spread_indexes(n, count) == expected


# --- a video that is still being written -----------------------------------------------------------------------


def shade_frame(shade: int, size=(320, 180)) -> av.VideoFrame:
    return av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), shade, np.uint8), format="rgb24")


class Recorder:
    """The recorder's writer: a fragmented MP4 with a key frame every 10 frames, flushed packet by packet."""

    def __init__(self, path: pathlib.Path, fps: int = 30):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), "w", format="mp4", options=RECORDER_OPTIONS)
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = 320, 180, "yuv420p"
        self.stream.codec_context.gop_size = 10
        self.stream.codec_context.options = {"preset": "ultrafast", "tune": "zerolatency"}
        self.written = 0

    def add(self, shades) -> None:
        for shade in shades:
            self.container.mux(self.stream.encode(shade_frame(shade)))
            self.written += 1

    def close(self) -> None:
        self.container.mux(self.stream.encode())
        self.container.close()


def test_the_tail_of_a_video_is_readable_while_it_grows(sio, tmp_path) -> None:
    path = tmp_path / "episode_000000.partial.mp4"
    recorder = Recorder(path)
    try:
        recorder.add(range(100, 160, 2))  # 30 frames, shades 100..158
        frames, packets, fps = sio.read_tail(path, window_frames=5)
        assert fps == 30.0
        assert len(frames) == 5
        first_packets = packets
        # A fragment is written when the next key frame arrives, so the file is up to one group of pictures (10
        # frames, a third of a second) behind: frames 0 to 19 of the 30 are in it, the newest shade 138.
        newest = int(frames[-1].to_ndarray(format="gray").mean())
        assert 128 <= newest <= 158

        recorder.add(range(200, 230))  # 30 more frames, shades 200..229
        frames, packets, _ = sio.read_tail(path, window_frames=5)
        assert packets > first_packets
        assert int(frames[-1].to_ndarray(format="gray").mean()) >= 200
    finally:
        recorder.close()


def test_an_empty_or_missing_video_is_a_source_error(sio, tmp_path) -> None:
    with pytest.raises(sio.SourceError):
        sio.read_tail(tmp_path / "missing.mp4", 5)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(sio.SourceError):
        sio.read_tail(broken, 5)


def make_run(root: pathlib.Path, name: str, started: datetime.datetime, camera: str = "scene") -> pathlib.Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "rollout.json").write_text(json.dumps({"status": "recording", "started_at_utc": started.isoformat()}))
    return folder / "videos/chunk-000" / f"observation.images.{camera}" / "episode_000000.partial.mp4"


def test_growing_video_finds_the_newest_run_and_returns_spread_frames(sio, tmp_path) -> None:
    now = datetime.datetime.now(datetime.UTC)
    old = make_run(tmp_path, "20261108-090000_run1", now - datetime.timedelta(hours=2))
    new = make_run(tmp_path, "20261108-110000_run2", now - datetime.timedelta(seconds=5))
    Recorder(old).close()
    recorder = Recorder(new)
    try:
        recorder.add(range(40, 100, 2))
        clock = Clock()
        source = sio.GrowingVideo(tmp_path, "scene", clock=clock)
        source.bind(now - datetime.timedelta(seconds=30))
        frames = source.latest(3)
        assert len(frames) == 3
        assert [f.t_s for f in frames] == sorted(f.t_s for f in frames)
        assert frames[0].t_s < frames[-1].t_s
        assert frames[-1].bgr.shape == (180, 320, 3)
        assert 68 <= int(frames[-1].bgr.mean()) <= 98  # this run's, and at most a group of pictures behind
    finally:
        recorder.close()


def test_growing_video_without_a_recording_says_how_to_turn_it_on(sio, tmp_path) -> None:
    source = sio.GrowingVideo(tmp_path / "nothing", "scene")
    source.bind(None)
    with pytest.raises(sio.SourceError, match="rollouts.enabled"):
        source.latest(3)


def test_growing_video_ignores_runs_that_started_before_the_bound_time(sio, tmp_path) -> None:
    now = datetime.datetime.now(datetime.UTC)
    Recorder(make_run(tmp_path, "run1", now - datetime.timedelta(minutes=5))).close()
    source = sio.GrowingVideo(tmp_path, "scene")
    source.bind(now)
    with pytest.raises(sio.SourceError):
        source.latest(3)


def test_growing_video_is_stale_when_the_file_stops_growing(sio, tmp_path) -> None:
    now = datetime.datetime.now(datetime.UTC)
    path = make_run(tmp_path, "run1", now)
    recorder = Recorder(path)
    try:
        recorder.add(range(60))
        clock = Clock()
        source = sio.GrowingVideo(tmp_path, "scene", clock=clock)
        source.bind(now)
        assert source.stale_s() is None
        source.latest(1)
        assert source.stale_s() == 0.0
        clock.t = 10.0
        source.latest(1)  # nothing was written since
        assert source.stale_s() == 10.0
        recorder.add(range(30))
        clock.t = 12.0
        source.latest(1)
        assert source.stale_s() == 0.0
    finally:
        recorder.close()


def test_growing_video_reads_the_finished_file_too(sio, tmp_path) -> None:
    now = datetime.datetime.now(datetime.UTC)
    partial = make_run(tmp_path, "run1", now)
    recorder = Recorder(partial)
    recorder.add(range(60))
    recorder.close()
    partial.rename(partial.with_name("episode_000000.mp4"))
    source = sio.GrowingVideo(tmp_path, "scene")
    source.bind(now)
    assert len(source.latest(3)) == 3


def test_replay_video_plays_against_the_clock_and_rests_on_the_last_picture(sio, tmp_path) -> None:
    path = tmp_path / "demo.mp4"
    recorder = Recorder(path)
    recorder.add([20] * 30 + [220] * 30)  # two seconds: dark, then bright
    recorder.close()
    clock = Clock()
    replay = sio.ReplayVideo(path, clock=clock)
    assert replay.duration_s == pytest.approx(2.0, abs=0.1)
    assert not replay.ended
    assert int(replay.latest(1)[0].bgr.mean()) < 60  # t = 0
    clock.t = 1.5
    assert int(replay.latest(1)[0].bgr.mean()) > 180
    clock.t = 30.0  # long after the end
    assert replay.ended
    assert int(replay.latest(1)[0].bgr.mean()) > 180
    assert replay.stale_s() == 0.0
    clock.t = 31.0
    replay.bind(None)  # a restart: from the first picture again
    assert int(replay.latest(1)[0].bgr.mean()) < 60


def test_growing_video_follows_a_new_episode_after_a_change_of_prompt(sio, tmp_path) -> None:
    """The recorder ends an episode and begins another, in a folder of its own, when the prompt changes mid-run."""
    now = datetime.datetime.now(datetime.UTC)
    first = Recorder(make_run(tmp_path, "2026-11-08T12-00-00_run3", now))
    first.add(range(30, 90, 2))
    clock = Clock()
    source = sio.GrowingVideo(tmp_path, "scene", clock=clock)
    source.bind(now)
    before = source.latest(1)[0]
    assert source.folder.name == "2026-11-08T12-00-00_run3"
    first.close()  # the old episode is finished

    second_path = make_run(tmp_path, "2026-11-08T12-00-40_run3_2", now + datetime.timedelta(seconds=40))
    clock.t = 40.0
    with pytest.raises(sio.SourceError, match="no video of camera 'scene' yet"):
        source.latest(1)  # the new folder has no video yet: it must not fall back to the old picture
    second = Recorder(second_path)
    try:
        second.add([200] * 60)
        frame = source.latest(1)[0]
        assert source.folder.name == "2026-11-08T12-00-40_run3_2"
        assert int(frame.bgr.mean()) > 150 > int(before.bgr.mean())  # the new episode's picture
        assert source.stale_s() == 0.0  # and its recording is counted as growing from now
    finally:
        second.close()


def test_a_pinned_growing_video_stays_on_its_folder(sio, tmp_path) -> None:
    now = datetime.datetime.now(datetime.UTC)
    old = make_run(tmp_path, "2026-11-08T12-00-00_run3", now)
    recorder = Recorder(old)
    recorder.add([40] * 40)
    recorder.close()
    Recorder(make_run(tmp_path, "2026-11-08T12-01-00_run4", now + datetime.timedelta(seconds=60))).close()
    source = sio.GrowingVideo(tmp_path, "scene")
    source.pin(old.parents[3])
    assert int(source.latest(1)[0].bgr.mean()) < 80
    assert source.folder == old.parents[3]
