"""tools/upload_dimensional.py: packs a LeRobot dataset into one repeatable tar and hands it to `dimos data upload`.

A fake `dimos` command stands in for Dimensional's CLI, so nothing here touches the network or a login.
"""

import importlib.util
import json
import os
import pathlib
import stat
import tarfile

import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("upload_dimensional_tool", TOOLS / "upload_dimensional.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(root: pathlib.Path, name: str = "cubes", *, episodes: int = 2, version: str = "v2.1") -> pathlib.Path:
    folder = root / name
    (folder / "meta").mkdir(parents=True)
    (folder / "data" / "chunk-000").mkdir(parents=True)
    (folder / "videos" / "chunk-000" / "cam").mkdir(parents=True)
    info = {
        "codebase_version": version,
        "total_episodes": episodes,
        "total_frames": 60,
        "fps": 30,
        "robot_type": "metal",
    }
    (folder / "meta" / "info.json").write_text(json.dumps(info))
    (folder / "meta" / "episodes.jsonl").write_text('{"episode_index": 0}\n')
    (folder / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"parquet-bytes")
    (folder / "videos" / "chunk-000" / "cam" / "episode_000000.mp4").write_bytes(b"\x00video" * 100)
    return folder


def fake_dimos(directory: pathlib.Path, *, fail_on: str = "") -> pathlib.Path:
    """A `dimos` that logs its arguments, lists every tar it was given, and fails on the subcommand `fail_on`.

    Shell builtins only: the tests run with a PATH that holds nothing but this folder."""
    script = directory / "dimos"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{directory}/calls"\n'
        f'[ "$1 $2" = "{fail_on}" ] && {{ echo "boom" >&2; exit 1; }}\n'
        'case "$1 $2" in\n'
        f'  "data upload") echo "${{3##*/}}: complete (abc123)"; echo "${{3##*/}}" >> "{directory}/uploaded" ;;\n'
        f'  "data ls") [ -f "{directory}/uploaded" ] &&\n'
        f'    while read -r line; do echo "$line"; done < "{directory}/uploaded" ;;\n'
        '  "data quota") echo "1% used (ok)" ;;\n'
        "esac\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_check_dataset_summarises_the_folder(tool, tmp_path):
    info = tool.check_dataset(make_dataset(tmp_path))
    assert (info.name, info.episodes, info.frames, info.fps, info.robot_type) == ("cubes", 2, 60, 30, "metal")
    assert info.files == 4


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        (lambda d: (d / "meta" / "info.json").unlink(), "no meta/info.json"),
        (lambda d: (d / "meta" / "info.json").write_text("{"), "not valid JSON"),
        (lambda d: (d / "meta" / "episodes.jsonl").unlink(), "incomplete"),
    ],
)
def test_check_dataset_refuses_a_broken_folder(tool, tmp_path, breakage, message):
    dataset = make_dataset(tmp_path)
    breakage(dataset)
    with pytest.raises(ValueError, match=message):
        tool.check_dataset(dataset)


def test_check_dataset_refuses_other_versions_and_empty_datasets(tool, tmp_path):
    with pytest.raises(ValueError, match="v3.0"):
        tool.check_dataset(make_dataset(tmp_path, "a", version="v3.0"))
    with pytest.raises(ValueError, match="no episodes"):
        tool.check_dataset(make_dataset(tmp_path, "b", episodes=0))


def test_pack_gives_the_same_bytes_every_time(tool, tmp_path):
    dataset = tool.check_dataset(make_dataset(tmp_path))
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    one = tool.pack(dataset, first)
    # Another owner and another time must not change the tar: Dimensional files equal bytes once.
    for path in dataset.path.rglob("*"):
        os.utime(path, (1_000_000, 1_000_000))
    two = tool.pack(dataset, second)
    assert one.read_bytes() == two.read_bytes()


def test_pack_restores_every_file_byte_for_byte(tool, tmp_path):
    source = make_dataset(tmp_path)
    tar = tool.pack(tool.check_dataset(source), tmp_path)
    restored = tmp_path / "restored"
    with tarfile.open(tar) as archive:
        archive.extractall(restored, filter="data")
    files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(restored / "cubes") for p in (restored / "cubes").rglob("*") if p.is_file())
    assert all((source / f).read_bytes() == (restored / "cubes" / f).read_bytes() for f in files)


def test_pack_refuses_a_symbolic_link(tool, tmp_path):
    dataset = make_dataset(tmp_path)
    (dataset / "meta" / "link").symlink_to(dataset / "meta" / "info.json")
    with pytest.raises(ValueError, match="symbolic link"):
        tool.pack(tool.check_dataset(dataset), tmp_path)


def run_tool(tool, monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["upload_dimensional.py", *argv])
    tool.main()


def test_dry_run_sends_nothing_and_needs_no_dimos(tool, tmp_path, monkeypatch, capsys):
    dataset = make_dataset(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    run_tool(tool, monkeypatch, str(dataset), "--dry-run")
    out = capsys.readouterr().out
    assert "cubes: 2 episodes, 60 frames at 30 fps" in out and "nothing was packed or sent" in out


def test_upload_checks_login_then_uploads_then_confirms(tool, tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_dimos(bin_dir)
    monkeypatch.setenv("PATH", str(bin_dir))
    dataset = make_dataset(tmp_path)
    log = tmp_path / "logs" / "run.log"
    run_tool(tool, monkeypatch, str(dataset), "--robot", "station-3", "--log", str(log))
    calls = (bin_dir / "calls").read_text().splitlines()
    assert [c.split()[:2] for c in calls] == [["data", "quota"], ["data", "upload"], ["data", "ls"]]
    assert calls[1].endswith("cubes.tar --kind blob --robot station-3")
    assert "confirmed in `dimos data ls`: cubes.tar" in capsys.readouterr().out
    # The log holds what was printed, and the temporary tar is gone.
    assert "$ dimos data upload" in log.read_text()
    assert not list(pathlib.Path(tmp_path).glob("**/cubes.tar"))


def test_robot_defaults_to_the_datasets_robot_type(tool, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_dimos(bin_dir)
    monkeypatch.setenv("PATH", str(bin_dir))
    run_tool(tool, monkeypatch, str(make_dataset(tmp_path)))
    assert (bin_dir / "calls").read_text().splitlines()[1].endswith("--kind blob --robot metal")


def test_a_failed_upload_is_reported_and_exits_nonzero(tool, tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_dimos(bin_dir, fail_on="data upload")
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(SystemExit) as stop:
        run_tool(tool, monkeypatch, str(make_dataset(tmp_path)))
    assert stop.value.code == 1
    assert "NOT CONFIRMED" in capsys.readouterr().out


def test_a_longer_name_in_the_listing_does_not_confirm_a_shorter_one(tool, tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_dimos(bin_dir, fail_on="data upload")
    # `my-cubes.tar` is in the cloud already; `cubes.tar` is not, and its upload fails.
    (bin_dir / "uploaded").write_text("my-cubes.tar\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(SystemExit):
        run_tool(tool, monkeypatch, str(make_dataset(tmp_path)))
    assert "not in `dimos data ls`: ['cubes']" in capsys.readouterr().out


def test_not_logged_in_stops_before_packing(tool, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_dimos(bin_dir, fail_on="data quota")
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(SystemExit, match="dimos login"):
        run_tool(tool, monkeypatch, str(make_dataset(tmp_path)))
    assert [c.split()[:2] for c in (bin_dir / "calls").read_text().splitlines()] == [["data", "quota"]]


def test_a_run_that_stops_says_why_in_the_log(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("DIMOS", raising=False)
    log = tmp_path / "run.log"
    with pytest.raises(SystemExit):
        run_tool(tool, monkeypatch, str(make_dataset(tmp_path)), "--log", str(log))
    lines = log.read_text().splitlines()
    assert lines[1].startswith("cubes: 2 episodes") and lines[-1].startswith("stopped: cannot find 'dimos'")


def test_a_missing_dimos_says_how_to_get_it(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("DIMOS", raising=False)
    with pytest.raises(SystemExit, match="cannot find 'dimos'"):
        run_tool(tool, monkeypatch, str(make_dataset(tmp_path)))


def test_two_folders_with_one_name_are_refused(tool, tmp_path, monkeypatch):
    one = make_dataset(tmp_path / "a")
    two = make_dataset(tmp_path / "b")
    with pytest.raises(SystemExit, match="same name"):
        run_tool(tool, monkeypatch, str(one), str(two), "--dry-run")
