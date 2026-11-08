"""The shipped station configs load, and mistakes name the key to fix."""

import pathlib

import pytest
import yaml
from conftest import filled_template

from dream_hackathon import config

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def test_templates_refuse_to_load_until_filled_in(tmp_path: pathlib.Path) -> None:
    with pytest.raises(config.ConfigError, match=r"'cameras\.overhead\.device' still says REPLACE"):
        config.load(CONFIGS / "yam.yaml")
    with pytest.raises(config.ConfigError, match=r"'arms\.left\.port' still says REPLACE"):
        config.load(CONFIGS / "maker.yaml")


def test_filled_yam_and_maker_configs_load(tmp_path: pathlib.Path) -> None:
    yam = config.load(filled_template("yam", tmp_path))
    assert (yam.family, yam.control_hz // yam.fps) == ("yam", 2)
    maker = config.load(filled_template("maker", tmp_path))
    assert maker.family == "makermods"
    assert maker.arms["left"].tcp_xyz_m == (0.0, 0.0, -0.1696387)  # Dream's own point for the stock gripper


def test_metal_config_needs_its_measured_gripper_angles(tmp_path: pathlib.Path) -> None:
    with pytest.raises(config.ConfigError, match="gripper_open_deg"):
        config.load(filled_template("metal", tmp_path))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(fps=25), "fps"),
        (lambda raw: raw.update(record_width=640, record_height=480), "16:9"),
        (lambda raw: raw["cameras"].pop("wrist_l"), "cameras"),
        (lambda raw: raw["arms"]["left"].update(quest_to_base=[[1, 0, 0], [0, 1, 0], [0, 0, -1]]), "rotation"),
        (lambda raw: raw["arms"]["right"].update(workspace_min_m=[1, 1, 1]), "workspace_min_m"),
    ],
)
def test_mistakes_name_the_key(tmp_path: pathlib.Path, change, message: str) -> None:
    raw = yaml.safe_load(filled_template("yam", tmp_path).read_text())
    change(raw)
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(config.ConfigError, match=message):
        config.load(path)


def test_a_missing_file_says_how_to_make_it(tmp_path):
    with pytest.raises(config.ConfigError, match="does not exist: copy your arm's template"):
        config.load(tmp_path / "station.yaml")


def test_the_task_text_is_optional(tmp_path):
    assert "task" not in yaml.safe_load((CONFIGS / "yam.yaml").read_text())
    path = filled_template("yam", tmp_path)
    base = yaml.safe_load(path.read_text())
    for value, expected in ((None, None), ("  ", None), ("  stack the cups ", "stack the cups")):
        path.write_text(yaml.safe_dump(base if value is None else base | {"task": value}))
        assert config.load(path).task == expected


def test_a_task_that_is_not_text_is_refused(tmp_path):
    path = filled_template("yam", tmp_path)
    path.write_text(yaml.safe_dump(yaml.safe_load(path.read_text()) | {"task": 3}))
    with pytest.raises(config.ConfigError, match="'task' is 3"):
        config.load(path)


LEFT_PORT, RIGHT_PORT = "/dev/serial/by-id/usb-left", "/dev/serial/by-id/usb-right"
## Dream-1's home as `dreamcli setup` writes it (a real station's values), radians.
DREAM_HOME = {
    "left": [-0.0633, -0.0011, 0.0101, -0.0179, -0.0342, 0.0114],
    "right": [0.025, 0.002, -0.0411, 0.1766, 0.0348, -0.0292],
}


def _metal_station(tmp_path: pathlib.Path) -> config.StationConfig:
    """A filled-in Metal station with one port per arm (test gripper angles: this config drives nothing)."""
    raw = yaml.safe_load(filled_template("metal", tmp_path).read_text())
    for side, port in (("left", LEFT_PORT), ("right", RIGHT_PORT)):
        raw["arms"][side].update(port=port, gripper_open_deg=40.0, gripper_closed_deg=0.0)
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


def _dream_station(tmp_path: pathlib.Path, **changes) -> pathlib.Path:
    """A `~/.dreamlabs/station.yaml` like the one `dreamcli setup` writes, in tmp_path."""
    raw = {
        "body": "metal",
        "follower_l": {"can_channel": LEFT_PORT, "home": DREAM_HOME["left"]},
        "follower_r": {"can_channel": RIGHT_PORT, "home": DREAM_HOME["right"]},
    }
    for key, value in changes.items():
        section, _, field = key.partition("__")
        if field:
            raw[section][field] = value
        else:
            raw[section] = value
    path = tmp_path / "dreamlabs" / "station.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def test_dreams_home_is_read_per_arm_by_its_port(tmp_path: pathlib.Path) -> None:
    home = config.dream_home(_metal_station(tmp_path), _dream_station(tmp_path))
    assert set(home) == {"left", "right"}
    for side in ("left", "right"):
        assert home[side].shape == (6,) and home[side].tolist() == DREAM_HOME[side]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"body": "maker"}, "is for a 'maker' body"),
        ({"follower_l__can_channel": RIGHT_PORT}, r"'follower_l\.can_channel' is .* but 'arms\.left\.port'"),
        ({"follower_r__home": [0.0] * 5}, r"'follower_r\.home' must be six joint angles"),
        ({"follower_r__home": [0.0] * 5 + [4.0]}, r"'follower_r\.home' must be six joint angles"),
        ({"follower_l__home": ["a", 0, 0, 0, 0, 0]}, r"'follower_l\.home' must be six joint angles"),
        ({"follower_r": None}, "has no 'follower_r' section"),
    ],
)
def test_a_home_that_does_not_fit_is_refused(tmp_path: pathlib.Path, changes: dict, message: str) -> None:
    with pytest.raises(config.ConfigError, match=message):
        config.dream_home(_metal_station(tmp_path), _dream_station(tmp_path, **changes))


def test_a_missing_or_unreadable_dream_config_is_refused(tmp_path: pathlib.Path) -> None:
    station = _metal_station(tmp_path)
    with pytest.raises(config.ConfigError, match="does not exist: `dreamcli setup` writes it"):
        config.dream_home(station, tmp_path / "nowhere.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("body: metal\nfollower_l: [unclosed\n")
    with pytest.raises(config.ConfigError, match="could not be read"):
        config.dream_home(station, broken)


def test_dreams_home_is_used_on_metal_arms_only(tmp_path: pathlib.Path) -> None:
    for arm in ("yam", "maker"):
        with pytest.raises(config.ConfigError, match=f"metal arms only, not {arm}"):
            config.dream_home(config.load(filled_template(arm, tmp_path)), _dream_station(tmp_path))
