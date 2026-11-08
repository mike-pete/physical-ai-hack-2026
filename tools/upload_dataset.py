"""Uploads a dataset to the Dream API and waits until it is processed. It queues no fine-tuning job: no charge.

Run it in ~/dream-env (the Dream library's environment, with your API key saved by the Setup API steps):

```
# YAM: the collected dataset as it is
~/dream-env/bin/python tools/upload_dataset.py ~/hackathon_data/cubes --arm yam --name cubes
# Metal or Maker arm: the dataset tools/add_eef.py wrote
~/dream-env/bin/python tools/upload_dataset.py ~/hackathon_data/cubes-eef --arm metal --name cubes
```

These are the first steps of `tools/finetune.py`, with the same checks and the same embodiment: check the dataset
on this computer, register your robot's embodiment (once; reused after), upload, and wait for the platform to process
it. Then it stops. To fine-tune on the uploaded dataset later, pass its id to `tools/finetune.py --dataset-id ds_...`.
A dataset name must be new on your account: to upload more episodes, use another `--name`.
"""

import argparse
import json
import pathlib
import sys

import dream
from finetune import ARMS, EMBODIMENTS, embodiment_for, wait_for_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=pathlib.Path, help="the LeRobot v2.1 dataset folder")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--name", required=True, help="the dataset's name on the service: lowercase letters, digits, hyphens"
    )
    parser.add_argument("--embodiment", help="the embodiment name to register or reuse (default: hackathon-<arm>)")
    args = parser.parse_args()
    file_name, default_embodiment, _ = ARMS[args.arm]
    config = json.loads((EMBODIMENTS / file_name).read_text())
    path = args.dataset.expanduser()

    client = dream.DreamClient()
    ## The service lists names as `<username>/<name>`.
    taken = next((d for d in client.get_datasets() if d.name.rsplit("/", 1)[-1] == args.name), None)
    if taken is not None:
        sys.exit(f"a dataset named {args.name!r} is already on this account ({taken.id}); pick another --name")
    report = client.validate_dataset(str(path), config)
    if not report.ok:
        sys.exit(f"the dataset needs fixing before it uploads: {report.error}")
    print(f"{report.episodes} episodes, {report.frames} frames at {report.fps} fps, cameras {report.video_keys}")
    embodiment = embodiment_for(client, args.embodiment or default_embodiment, config)
    dataset = client.upload_dataset(str(path), args.name, embodiment)
    print(f"uploaded as {dataset.id} ({dataset.name})", flush=True)
    dataset = wait_for_dataset(client, dataset.id)
    print(f"dataset {dataset.id} ({dataset.name}): {dataset.status}")
    print("nothing was charged. To fine-tune on it later, which queues a paid job:")
    print(
        f"  ~/dream-env/bin/python tools/finetune.py --arm {args.arm} --dataset-id {dataset.id} --suffix <your-suffix>"
    )


if __name__ == "__main__":
    main()
