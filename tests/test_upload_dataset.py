"""tools/upload_dataset.py: checks, registers the embodiment and uploads, and never queues a fine-tuning job."""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("dream", reason="needs the Dream library (the dream group: uv sync --group dream)")

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def upload_dataset(monkeypatch: pytest.MonkeyPatch):
    """The tool's module, loaded the way `python tools/upload_dataset.py` does: its folder is on the import path."""
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("upload_dataset_tool", TOOLS / "upload_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClient:
    """The slice of dream.DreamClient the tool uses. Fine-tuning is not allowed."""

    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        self.calls: list[str] = []
        self.existing = existing

    def get_datasets(self):
        self.calls.append("get_datasets")
        return [SimpleNamespace(id=f"ds_{i}", name=f"me/{name}") for i, name in enumerate(self.existing)]

    def validate_dataset(self, path, config):
        self.calls.append("validate_dataset")
        return SimpleNamespace(ok=True, episodes=3, frames=90, fps=30, video_keys=["cam"], error=None)

    def add_embodiment(self, name, config):
        self.calls.append("add_embodiment")
        return SimpleNamespace(name=f"me/{name}")

    def upload_dataset(self, path, name, embodiment):
        self.calls.append("upload_dataset")
        return SimpleNamespace(id="ds_new", name=f"me/{name}")

    def get_dataset_status(self, dataset_id):
        self.calls.append("get_dataset_status")
        return SimpleNamespace(id=dataset_id, name="me/cubes", status="processed", error=None)

    def finetune(self, *args, **kwargs):
        raise AssertionError("the upload tool must never queue a fine-tuning job")


def run(upload_dataset, monkeypatch: pytest.MonkeyPatch, client: FakeClient, *argv: str) -> None:
    monkeypatch.setattr(upload_dataset.dream, "DreamClient", lambda: client)
    monkeypatch.setattr(sys, "argv", ["upload_dataset.py", *argv])
    upload_dataset.main()


def test_uploads_and_waits_without_fine_tuning(upload_dataset, monkeypatch, tmp_path, capsys) -> None:
    client = FakeClient()
    run(upload_dataset, monkeypatch, client, str(tmp_path), "--arm", "yam", "--name", "cubes")
    assert client.calls == [
        "get_datasets",
        "validate_dataset",
        "add_embodiment",
        "upload_dataset",
        "get_dataset_status",
    ]
    assert "ds_new" in capsys.readouterr().out


def test_refuses_a_name_already_on_the_account(upload_dataset, monkeypatch, tmp_path) -> None:
    client = FakeClient(existing=("cubes",))
    with pytest.raises(SystemExit, match="already on this account"):
        run(upload_dataset, monkeypatch, client, str(tmp_path), "--arm", "yam", "--name", "cubes")
    assert client.calls == ["get_datasets"]
