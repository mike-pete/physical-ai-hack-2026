# Dream Hackathon Kit

## Collect demonstrations by moving the left arm

No Quest? Take the **left arm** in your hand and the **right arm** copies it, while a dashboard in your browser starts, saves and discards the episodes. It records the same LeRobot v2.1 datasets as the Quest, and works on Metal and Maker arm stations. **[How to run it: docs/mirror_collect.md](docs/mirror_collect.md).**

```bash
# 🤖 on the station: REAL RUN, the right arm moves (there is no dry run)
uv run python scripts/mirror_collect.py --config configs/station.yaml --task "pick up the red cube" --gravity
```

Read the guide's safety notes first: the right arm is energized and follows the left arm from the moment the script opens it. Leave out `--gravity` on a Maker arm station.

Put Dream-1, the Dream Labs robot model, to work on a pair of robot arms: run it as it is, build your own project on the Dream API, or teach it your task with demonstrations you record with a Meta Quest and fine-tune it on them (optional).

Built for the [MakerMods Physical AI Hackathon](https://luma.com/pah) (November 7-8, San Francisco), presented with MakerMods and Dream Labs.

This kit drives two follower arms from the Quest's controllers, records each demonstration as a [LeRobot v2.1](https://github.com/huggingface/lerobot) dataset (the format Dream fine-tuning reads), and hands the dataset to the [Dream API](https://platform.dreamlabs.ai/api-documentation/fine-tuning/). It supports three kinds of arms, each driven through its maker's own open-source SDK:

| Arms | Driven by |
| --- | --- |
| [i2rt YAM](https://i2rt.com/) | [i2rt's Python SDK](https://github.com/i2rt-robotics/i2rt) |
| [MakerMods](https://github.com/makermods-robotics) Metal Arm (Damiao motors) | [MakerMods' LeRobot followers](https://github.com/makermods-robotics/lerobot) and [makermodslab](https://github.com/makermods-robotics/makermodslab) |
| MakerMods Maker Arm v1 (RobStride motors) | same |

Every station has one overhead camera and one camera on each wrist (USB webcams or Intel RealSense).

## For participants

Your station is ready: the arms, cameras, Quest and the Dream library are set up, and its config is `configs/station.yaml`. The station has no screen: you work on it over SSH from your laptop, and watch it in your laptop's browser.

Dream-1 runs on your station's arms without any training, so fine-tuning is optional. Pick what suits your project, or combine them:

- **Run Dream-1 as it is.** Give it a task in plain words and it drives the arms: `dreamcli robot run --task "..."` ([docs/running_your_model.md](docs/running_your_model.md#no-fine-tune-yet)).
- **Build with the Dream API.** Write your own program around Dream-1: see [Build with the Dream API](#build-with-the-dream-api) below.
- **Teach it your task (optional).** Collect demonstrations, then fine-tune Dream-1 on them: the two steps below.

### Laptop or station: where each command runs

You use two computers. Every command block below starts with a comment that says which one it is for.

| Where | What runs there |
| --- | --- |
| **💻 Your laptop** | Only `ssh` (to log in to the station), your browser (the dashboard at http://localhost:8080), and copying data off the station (`scp`) |
| **🤖 The station** (the robot's computer, in your SSH session) | Everything else: `tmux`, `dhk-collect`, `tools/add_eef.py`, `tools/finetune.py`, `dreamcli`, and your own Dream API programs |

Log in from your laptop with the station's IP address (from the organizers). The account is always `pah`:

```bash
# 💻 on your laptop
ssh -L 8080:localhost:8080 pah@<ip_address>
```

Type the password the organizers gave you, and wait for the station's prompt, `pah@<station>:~$`. Everything you type at that prompt runs on the station. `-L 8080:localhost:8080` is what lets your laptop's browser open the station's dashboard. To get back to your laptop, type `exit`.

### Dry run or real run

These arms are strong and fast. Every command that drives them has a dry run and a real run, and the only difference on the command line is `--dry-run`, so look for it every time you press Enter.

| | Dry run (`--dry-run`) | Real run (no `--dry-run`) |
| --- | --- | --- |
| **The arms** | Never move: nothing is sent to the motors | **Move**, as fast as the controller, the model or your code tells them |
| **`dhk-collect`** | Reads the arms and the Quest; the dashboard's badge reads **dry run**, and pressing A to record says "a dry run does not record" | Drives the arms and records; the badge reads **live** |
| **`dreamcli robot run`** | Calls the model for real and writes nothing to the arms | Moves both arms to their home pose first, then plays the model's moves |
| **Use it for** | Practice, a new task, a new model, any code you have just written or changed | Recording demonstrations and running a model you have checked in a dry run |

The rules:

1. **Dry run first, every time something is new**: a new session, a new task text, a new model, or new or changed code.
2. **Before a real run**: a hand on the emergency stop (or the arms' power switch), everyone out of the arms' reach, and nothing on the table that should not be hit. Then run the same command again without `--dry-run`.
3. **Your own Dream API code has no `--dry-run` unless you write one.** Give your program a mode that computes the moves and prints them without sending them to the arms, and use it before the first real run ([Your own control loop](https://platform.dreamlabs.ai/api-documentation/control-loop/) shows how to check a loop before it drives the arms).
4. **Not sure which mode you are in?** Look at the dashboard's badge (`dhk-collect`), or stop with Ctrl+C and read the command again.

### Collect demonstrations and fine-tune (optional)

1. **Collect demonstrations** ([docs/data_collection.md](docs/data_collection.md)). Hold a controller's grip to drive that arm and pull its trigger to close its gripper. Press A to start and stop recording an episode, and B to throw one away.

   Log in from your laptop as above (`ssh -L 8080:localhost:8080 pah@<ip_address>`). Then, at the station's prompt:

   ```bash
   # 🤖 on the station
   tmux new -A -s collect          # keeps the session running if your connection drops
   ```

   Run that line on its own. It opens a new screen, still on the station; type the rest there.

   **Practice first, with a dry run.** The arms do not move, and nothing is recorded:

   ```bash
   # 🤖 on the station: DRY RUN, the arms do not move
   cd ~/physical-ai-hacks
   uv run dhk-collect --config configs/station.yaml --dry-run
   ```

   **Then record, for real.** Stop the dry run with Ctrl+C, check the rules above (hand on the emergency stop, people clear), and start it without `--dry-run`:

   ```bash
   # 🤖 on the station: REAL RUN, the arms move
   uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/myteam-cubes
   ```

   The episodes need no labels: Dream labels every dataset from its overhead camera. `--task "..."` adds a task text of your own, if you want one.

   While `dhk-collect` runs, open http://localhost:8080 **in your laptop's browser**: the dashboard. Its badge says **dry run** or **live**. It first lists every device, with what to do for one that is not ready; once all are, press A on the right controller to open the arms. It shows the three cameras as they are recorded, whether you are recording, and both arms. You can type the task there ("Set task"), play back the episodes you saved, and tick the ones you do not want and delete them ("Delete selected").

   Your episodes are saved on the station in the `--dataset-dir` folder, here `~/hackathon_data/myteam-cubes`, as a LeRobot v2.1 dataset: not in the repo, and not uploaded until you fine-tune. [Where your data is stored](docs/data_collection.md#where-your-data-is-stored) shows what is inside and how to copy it to your laptop.

2. **Fine-tune Dream-1 on them** ([docs/fine_tuning.md](docs/fine_tuning.md)). This step is optional, takes hours and costs credits (see [Plan your time](#plan-your-time)). It never moves the arms. First save your team's API key on the station (the guide shows how). Then use `--arm yam`, `metal` or `maker`, whichever your station has:

   ```bash
   # 🤖 on the station (the arms do not move)
   cd ~/physical-ai-hacks
   # Metal or Maker arms only: add the tool poses their models train on (writes ~/hackathon_data/myteam-cubes-eef)
   ~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/myteam-cubes --arm metal
   # Check, upload and fine-tune (on Metal and Maker arms, pass the -eef folder)
   ~/dream-env/bin/python tools/finetune.py ~/hackathon_data/myteam-cubes --arm yam --name myteam-cubes --suffix myteam-cubes
   ```

When your model is ready, [run it on the arms](docs/running_your_model.md): on the station, a dry run first, then a real run:

```bash
# 🤖 on the station: DRY RUN, the model answers and the arms do not move
source ~/dream-env/bin/activate
dreamcli robot run --checkpoint <your model> --task "put the red cube in the bowl" --dry-run
# 🤖 on the station: REAL RUN, the arms move (hand on the emergency stop)
dreamcli robot run --checkpoint <your model> --task "put the red cube in the bowl"
```

On a YAM fine-tune, add `--action-space joint` to both; to run Dream-1 as it is, leave out `--checkpoint`.
### Run a fine-tuned model

The fine-tunes already made at this station are in [docs/trained_models.md](docs/trained_models.md): for each one, the task it was trained on and its number of episodes. They are Metal arm models, so there is no `--action-space joint`. A model runs only under the Dream account that made it (the key in `~/.dreamlabs/credentials`). Run it on the station, with the arms to themselves (close `dhk-collect` first), and do not move the cameras.

**1. List the models** (the arms do not move):

```bash
# 🤖 on the station
~/dream-env/bin/python -c "import dream; print(*[m.id for m in dream.DreamClient().get_models() if m.owner == 'mine'], sep='\n')"
```

**2. Dry run** with the model's id and the task it was trained on. The model answers and the arms do not move. The first call to a model that is not running waits while it starts, up to an hour:

```bash
# 🤖 on the station: DRY RUN, the arms do not move
source ~/dream-env/bin/activate
dreamcli robot run --checkpoint <model id> --task "<task text>" --dry-run
```

Examples, one per task (`--task` is the instruction the model was recorded with):

```bash
# 🤖 on the station: DRY RUN, the arms do not move
# unplug the orange USB cables from the power strip, right to left (10 episodes)
dreamcli robot run --checkpoint dream-1-ft:c8fae656-cables-lf-1000 --task "Unplug each orange USB cable from the power strip and put them on the table. Go right to left" --dry-run
# stack the blocks into a tower (14 episodes)
dreamcli robot run --checkpoint dream-1-ft:c8fae656-blocks-1000 --task "stack the rectangular blocks in a tower" --dry-run
# colorful golf balls into the basket, not the white ones (19 episodes, 8 of golf balls)
dreamcli robot run --checkpoint dream-1-ft:c8fae656-golf-1000 --task "Place the colorful golf balls into the basket. Do not touch white balls." --dry-run
# pick up and place the orange ball in the box (11 episodes)
dreamcli robot run --checkpoint dream-1-ft:c8fae656-orange-1000 --task "pick up and place orange ball in box" --dry-run
```

**3. Real run**, once the dry run looks right: a hand on the emergency stop, everyone out of the arms' reach, then the same command without `--dry-run`. Both arms first move to the home pose in `~/.dreamlabs/station.yaml`, then play the model's moves:

```bash
# 🤖 on the station: REAL RUN, the arms move (hand on the emergency stop)
dreamcli robot run --checkpoint <model id> --task "<task text>"
```

For example, the cables model is the first example above without `--dry-run`. [docs/trained_models.md](docs/trained_models.md) has every model's command, and [docs/running_your_model.md](docs/running_your_model.md) explains the options.

### Build with the Dream API

The Dream library on your station (`~/dream-env`) is the Python client of the [Dream API](https://platform.dreamlabs.ai/api-documentation/). With it you can build your own project around Dream-1, with or without a fine-tune: your program sends the cameras' frames, the arms' joints and a task, gets back the next moves and plays them on the arms, and you decide everything around that (which task comes next, when to stop, what the robot reacts to).

- [Getting started](https://platform.dreamlabs.ai/api-documentation/): API keys and the library.
- [Your own control loop](https://platform.dreamlabs.ai/api-documentation/control-loop/): the two calls, `set_observation()` and `get_action()`, a whole loop in one file, and how to check it before the first real run.
- [Driver template and examples](https://platform.dreamlabs.ai/api-documentation/drivers/): Dream's drivers for the YAM, Metal and Maker arms, to start from.
- [Live View](https://platform.dreamlabs.ai/api-documentation/live-view/): show your robot and its runs on the platform.
- [Client reference](https://platform.dreamlabs.ai/api-documentation/reference/): every class and call.

Your program runs **on the station**, in `~/dream-env`, with your team's API key saved there ([docs/fine_tuning.md](docs/fine_tuning.md#before-you-start) shows how), and needs the arms to itself: close `dhk-collect` and `dreamcli` first. Write it with a dry mode that never sends to the arms, and use that mode first ([Dry run or real run](#dry-run-or-real-run)). Using a coding agent? Dream's documentation is also an MCP server: the platform's Setup API page shows how to add it.

### Plan your time

- **Dream-1 runs without fine-tuning.** Try it on your task first, on the station: `dreamcli robot run --task "..." --dry-run`, then, with a hand on the emergency stop, the same without `--dry-run`: if it does well enough, or your project is built on the API, you may not need a fine-tune at all.
- **If you fine-tune, a job takes about 7 to 8 hours**, and may wait in a queue first. The first run of a new model then takes up to an hour to start. Queue your job early, ideally the evening of day one, and test on day two.
- **A job's price is not fixed: it grows with its training steps.** At the time of writing it is $100 for 500 steps, $200 for 1,000 (the default, and what `tools/finetune.py` runs) and $400 for 2,000; more steps also train longer. The platform's Fine-tune form shows the current price. It is held from the account's prepaid credits when the job is queued. Check with the organizers before you queue one.
- **Tens of good demonstrations beat hundreds of rushed ones.** Plan for 50 or more episodes of one task, each 10 to 30 seconds. [docs/data_collection.md](docs/data_collection.md) has tips.

## Safety

These are real, strong arms. Keep the emergency stop (or the arms' power switch) within reach whenever they are powered, keep people out of their reach while they move, and practice with `--dry-run`, which never commands a motor. A command without `--dry-run` moves the arms: read it before you press Enter ([Dry run or real run](#dry-run-or-real-run)). MakerMods arms have no brakes: start each session with them resting where they can go limp, because that is where they return and are released at the end.

## For organizers: setting up a station

[docs/hardware_setup.md](docs/hardware_setup.md) takes a station from parts on a table to a working dry run: the computer, the arms, the cameras, the Quest and the Dream library.

Claude Code can walk you through it with the repo's `station-setup` skill: start `claude` in the repo and run `/station-setup`. If you have added Dream's documentation server to Claude Code (the Dream platform's Setup API page offers it), Claude reads the Dream docs through it as well. Claude runs the checks and fills in the config with you. You do everything physical, and everything that powers or moves an arm (`CLAUDE.md` holds its rules).

No arms yet? On a YAM config, `uv run dhk-collect --config configs/yam.yaml --sim` drives i2rt's simulated arms with the Quest and records blank camera frames, so you can try the whole flow.

## Repository layout

| Path | What it is |
| --- | --- |
| `src/dream_hackathon/collect.py` | The teleop-and-record loop (`dhk-collect`) |
| `src/dream_hackathon/arms/` | The YAM and MakerMods arm adapters over the vendors' SDKs |
| `src/dream_hackathon/quest.py`, `teleop.py`, `ik.py` | Quest input, the clutched hand-to-tool mapping, and the teleop IK |
| `src/dream_hackathon/cameras.py`, `dataset.py` | Camera capture and the LeRobot v2.1 writer |
| `src/dream_hackathon/dashboard.py`, `dataset_cli.py` | The browser dashboard (over SSH) and `dhk-dataset` (list and drop episodes) |
| `src/dream_hackathon/devices.py` | Each device's state and its fix (Quest, cameras, the arms' links), for the dashboard and the terminal |
| `src/dream_hackathon/station_lock.py` | One program drives the arms at a time (shared with `dreamcli`) |
| `src/dream_hackathon/embodiments/` | The Dream embodiment configs your fine-tunes are registered with |
| `configs/` | One template station config per arm type; a station's own is `configs/station.yaml` (not committed) |
| `tools/` | Dream API steps, run in the Dream library's environment: `add_eef.py`, `upload_dataset.py` (upload only), `finetune.py` |
| `scripts/` | Device checks, the URDF download, and station helpers: `use_dream_tcp.py` (Dream's tool point for Metal and Maker arms), `allow_can_bringup.sh` (the dashboard's CAN buttons), `remove_api_key.sh`, `wipe_station.sh` |
| `docs/` | Collection, fine-tuning, running and setup guides, and `trained_models.md` (the fine-tunes made at this station) |
| `CLAUDE.md`, `.claude/skills/` | Claude Code's rules for this repo and the `station-setup` skill |

Run the tests with `uv sync --group dream && uv run pytest` (no hardware needed).

## Credits and licenses

This kit is licensed under the Apache License 2.0 (see `LICENSE`). It builds on [i2rt](https://github.com/i2rt-robotics/i2rt) (YAM SDK), [makermodslab](https://github.com/makermods-robotics/makermodslab) and [MakerMods' LeRobot fork](https://github.com/makermods-robotics/lerobot) (Metal and Maker arms, Apache-2.0), [oculus_reader](https://github.com/rail-berkeley/oculus_reader) (Quest app and reader, Apache-2.0), [mink](https://github.com/kevinzakka/mink) and [MuJoCo](https://github.com/google-deepmind/mujoco) (IK), and the [Dream library](https://pypi.org/project/dreamlabs/) (Apache-2.0). Each keeps its own license.
