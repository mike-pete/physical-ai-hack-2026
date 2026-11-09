"""Pings the model on a timer, so it does not have to start cold when a real run begins.

Run beside `dhk-lab` (the repo's own environment: `uv run python scripts/lab_keepwarm.py`). Every INTERVAL seconds,
while the arms are holding a pose and no run is under way, it asks the lab for a 5 second dry run: the model looks at
the scene and answers, and nothing is sent to the arms. Each ping uses the Dream account's quota and appears in the Runs
tab under the prompt "Keep the model warm: stay still."

Stop with Ctrl+C. It touches nothing but the lab's own HTTP API.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dream_hackathon.lab_client import LabClient  # noqa: E402

DEFAULT_INTERVAL_S = 240


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S, help="seconds between pings (60 or more)")
    parser.add_argument("--once", action="store_true", help="ping once and exit")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("--interval is 60 seconds or more")
    client = LabClient()
    while True:
        print(f"{time.strftime('%H:%M:%S')} {client.keep_warm()}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
