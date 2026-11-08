"""DimOS skills that drive the Prompt Lab: stop a run, send the arms home.

Run on the station beside `dhk-lab`, in an environment that has `dimos`. The skills only call the lab's HTTP API
(`dream_hackathon.lab_client`), so its state checks stay in force and nothing here touches an arm directly.

Add `LabSkills` to a blueprint beside `McpServer` and `McpClient`; DimOS finds the `@skill` methods on start.
"""

import pathlib
import sys

from dimos.agents.annotation import skill
from dimos.core.module import Module

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from dream_hackathon.lab_client import LabClient  # noqa: E402


class LabSkills(Module):
    """Skills for the Prompt Lab's arms."""

    @skill
    def stop_and_go_home(self) -> str:
        """Stop the robot's current run, wait for the arms to hold still, then move both arms home slowly."""
        return LabClient().stop_and_go_home()

    @skill
    def stop_run(self) -> str:
        """Stop the robot's current run. The arms hold where they are."""
        return LabClient().stop_run()


## The blueprint DimOS runs: `dimos run` takes it by name once this package is installed, or run_lab_skills.py runs it.
lab_skills = LabSkills.blueprint()
