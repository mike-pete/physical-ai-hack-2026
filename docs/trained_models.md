# Fine-tuned models made at this station

The Dream-1 fine-tunes trained from demonstrations recorded on this Metal arm station: what each one was trained on, how many episodes it saw, and the commands to run it. This page says what the models were trained on, not how well they perform. [docs/fine_tuning.md](fine_tuning.md) shows how they were made, and [docs/running_your_model.md](running_your_model.md) is the general guide to running a model.

## The models

Newest first. Every model starts from `dreamlabs/dream-1-beta`. Times are UTC, on 2026-11-08.

| Model id | Task it was trained on | Episodes | Steps | Created |
| --- | --- | --- | --- | --- |
| `dream-1-ft:c8fae656-blocks-1000` | Stack the blocks into a tower | 14 | 1,000 | 16:18 |
| `dream-1-ft:c8fae656-orange-1000` | "pick up and place orange ball in box" | 11 | 1,000 | 15:36 |
| `dream-1-ft:c8fae656-golf-1000` | Put the colorful golf balls into the basket without touching the white ones, and the orange-ball task above | 19 (8 golf balls, 11 orange ball) | 1,000 | 15:36 |
| `dream-1-ft:c8fae656-cables-lf-1000` | "Unplug each orange USB cable from the power strip and put them on the table. Go right to left" | 10 | 500 + 500 | 12:50 |
| `dream-1-ft:c8fae656-cables-raw-1000` | The same cable task | 10 | 500 + 500 | 12:47 |
| `dream-1-ft:c8fae656-cables-lf-500` | The same cable task | 10 | 500 | 07:13 |
| `dream-1-ft:c8fae656-cables-raw-500` | The same cable task | 10 | 500 | 06:59 |

What they share:

- **Embodiment:** `hackathon-metal`, the Metal arm. The models read and answer tool poses beside the joints (`use_eef`), so they run on a Metal arm station, not on a YAM.
- **Cameras:** `scene`, `wrist_l` and `wrist_r`, the same roles as `~/.dreamlabs/station.yaml`.
- **Training settings:** batch size 32, constant learning rate 1e-5.
- **The two `-1000` cable models** continue the `-500` model of the same dataset (`resume_from`) for 500 more steps, 1,000 in all. The other models trained once from `dreamlabs/dream-1-beta`.
- **`lf` and `raw`:** the cable models differ by their dataset, `cables-leftfrozen-v1` (`lf`) and `cables-raw-v1` (`raw`). Each has 10 episodes of the same task.

## Datasets and tasks

The task text is what the episodes were recorded with, so use the same words as `--task` when you run a model. The Dream API lists a job's dataset and its episode count, but not the task text: the text below comes from the recorded datasets in `~/hackathon_data` (`meta/tasks.jsonl` and `meta/episodes.jsonl`) that have the same name and episode count.

| Dataset (uploaded as `<account>/<name>`) | Local folder | Episodes | Task text, with its episode count |
| --- | --- | --- | --- |
| `stack-blocks` | `stack-blocks` | 14 | "stack the rectangular blocks in a tower" (5), "stack the blocks vertically" (3), "stack the rectangular blocks vertically" (2), "stack the rectangular blocks on top of each other" (2), "place the color cubes on top of each other" (2) |
| `one-orange-ball` | none (see below) | 11 | "pick up and place orange ball in box" |
| `golf-balls` | `golf-balls` | 19 | "Place the colorful golf balls into the basket. Do not touch white balls." (6), "Place the three colorful golf balls into the basket. Do not touch white balls." (2), "pick up and place orange ball in box" (11) |
| `cables-raw-v1` | `unplug-cables` | 10 | "Unplug each orange USB cable from the power strip and put them on the table. Go right to left" |
| `cables-leftfrozen-v1` | `my-task-cables-leftfrozen` | 10 | The same text |

`one-orange-ball` has no folder of its own on the station. Its task is inferred: the local `golf-balls` folder holds exactly 11 episodes of "pick up and place orange ball in box", and `one-orange-ball` has 11 episodes. The same goes for the `golf-1000` model, which was trained on the uploaded `golf-balls` dataset (19 episodes, the count of the local folder): if that upload matches the local folder, it also saw the 11 orange-ball episodes. Likewise, `cables-raw-v1` is matched to `unplug-cables` by its episode count and its task text alone.

## Run one

These models run only under the Dream account that made them (the API key in `~/.dreamlabs/credentials`). On another account the API refuses them as not available to run. Before you start:

- Close `dhk-collect`, `mirror_collect.py` and anything else that holds the arms: one program drives them at a time.
- Do not move the cameras: the models learned which view is which.
- Do not add `--action-space joint`; that is for YAM fine-tunes.
- In every new terminal, turn on the Dream library's environment first with `source ~/dream-env/bin/activate`: `dreamcli` is installed there, and without it the shell answers `dreamcli: command not found`.

**1. A dry run: the model answers and the arms do not move.** The Metal arms are opened with torque off, so support them by hand or let them rest. Pick the model's line:

```bash
# 🤖 on the station: DRY RUN, the arms do not move
source ~/dream-env/bin/activate

# cables (the same text for all four cable models; change the checkpoint to cables-lf-500, cables-raw-1000 or cables-raw-500)
dreamcli robot run --checkpoint dream-1-ft:c8fae656-cables-lf-1000 --task "Unplug each orange USB cable from the power strip and put them on the table. Go right to left" --dry-run

# blocks
dreamcli robot run --checkpoint dream-1-ft:c8fae656-blocks-1000 --task "stack the rectangular blocks in a tower" --dry-run

# golf balls
dreamcli robot run --checkpoint dream-1-ft:c8fae656-golf-1000 --task "Place the colorful golf balls into the basket. Do not touch white balls." --dry-run

# orange ball
dreamcli robot run --checkpoint dream-1-ft:c8fae656-orange-1000 --task "pick up and place orange ball in box" --dry-run
```

The first call to a model that is not running waits while it starts, which can take up to an hour (until then the call is refused with `pool_warming`). Later calls answer in well under a second.

**2. The real run.** With a hand on the emergency stop and everyone out of the arms' reach, run the same command without `--dry-run`. Both arms first move to the home pose in `~/.dreamlabs/station.yaml`, then play the model's moves, with the tool's speed capped by `max_ee_speed_m_s` (0.6 m/s in this station's config):

```bash
# 🤖 on the station: REAL RUN, the arms move (hand on the emergency stop)
source ~/dream-env/bin/activate
dreamcli robot run --checkpoint <model id> --task "<task text>"
```

To start runs from the browser and watch the cameras, use Live View: `dreamcli robot up`, then press Start on the platform's Live View page ([docs/running_your_model.md](running_your_model.md#3-run-it)).

## List the account's models and jobs

Both run on the station and neither moves the arms:

```bash
# 🤖 on the station: the account's own fine-tunes
~/dream-env/bin/python -c "import dream; print(*[m.id for m in dream.DreamClient().get_models() if m.owner == 'mine'], sep='\n')"
# 🤖 on the station: each job's model, dataset, steps and the model it continues
~/dream-env/bin/python -c "from dream import fine_tuning as f; [print(j['fine_tuned_model'], j['dataset'].split('/')[-1], j['hyperparameters']['steps'], j['hyperparameters'].get('resume_from')) for j in f.list()]"
```
