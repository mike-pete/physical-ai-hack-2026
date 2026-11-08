# Testing a task critic on recorded runs

A **task critic** looks at the cameras after a robot run and says whether the task is done. When it is not, it says what went wrong and writes a corrected prompt for a retry. This page is how to find out how good the critic is **before** it is allowed to steer a robot: run Dream-1 with several prompts, record the runs, label what happened, and let the critic judge the same runs. The first task is unplugging the orange USB cables from the power strip.

The code is `tools/task_critic.py` (the critic) and `tools/critic_eval.py` (the evaluation). What the critic asks about the picture, how it zooms, and the prompts to try, are in `configs/tasks/usb-cables.yaml`. There are two ways to judge: **question gates** (TypeSafe or Claude answer several yes/no questions about the table, and the code combines them) and the **vote gate** (Gemini answers one plain question about a zoomed picture of the strip, several times, and the code counts the answers). Nothing here moves an arm except the recording runs in step 1, which are ordinary `dreamcli robot run` runs.

To use the critic on a live run, to stop it when the task is done and guide it when it is failing, see [supervising a run](supervisor.md).

## Read this first

- **Step 1 moves the arms.** It is the same real run as in [running your model](running_your_model.md#3-run-it): a hand on the emergency stop, everyone out of the arms' reach, and a dry run first when anything is new. A dry run saves nothing, so it does not record an episode.
- **Pictures leave this computer in step 4, and only then.** The critic sends cropped camera pictures to TypeSafe or Gemini (whichever you pick) and, for rewritten prompts, to Claude or Gemini (`--rewriter`). The Gemini gate sends only a zoom on the strip, which lies inside the table crop; the rewriter also sees the scene crop, for each scene that is not done. The scene camera also sees the people around the table, so the tool crops each picture to the table (`crop` in the task file), drops any picture where a face detector finds a face, and asks before it sends. The detector is a safety net, not a guarantee: **look at the cropped pictures yourself in step 3.**
- **The keys stay in your shell.** Never put `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` in a file you commit or in a command line. Step 4 shows how to type them without saving them in the shell's history.

## 1. Record the runs

**Turn recording on.** The Dream library saves every real run as an episode (a LeRobot v2.1 folder with the camera videos and `rollout.json`, which holds the prompt) when the station file asks it to. Keep a copy of the file, then add this to `~/.dreamlabs/station.yaml`:

```bash
# 🤖 on the station
cp ~/.dreamlabs/station.yaml ~/.dreamlabs/station.yaml.before-rollouts
```

```yaml
rollouts:
  enabled: true
  dir: ~/hackathon_data/critic-eval/cables
```

**Run each prompt several times** from the same starting scene, at least three runs per prompt. The prompts to try are the `prompts` list in `configs/tasks/usb-cables.yaml`; edit it first. Put every orange cable back into the strip before each run. Some runs of the `corrective` prompt should start from a scene where cables are already unplugged: that is the scene a retry would see. Each run is one command (close `dhk-collect` first, and do not move the cameras):

```bash
# 🤖 on the station: REAL RUN, the arms move (hand on the emergency stop)
source ~/dream-env/bin/activate
dreamcli robot run --checkpoint dreamlabs/dream-1-beta --task "<the prompt's text>"
```

Press Ctrl+C when the run has finished or gone wrong: the arms hold, go home and switch off, and the episode is saved. Wait for the line `rollouts: saved ...` before you start the next run.

## 2. Label what happened

```bash
# 🤖 on the station
cd ~/physical-ai-hack-2026
uv run python tools/critic_eval.py labels-template ~/hackathon_data/critic-eval/cables --out ~/hackathon_data/critic-eval/labels.csv
```

The sheet has one row per episode, with the prompt filled in. For each, watch the run (or the scene video in the episode's `videos` folder) and fill in:

| Column | What to write |
| --- | --- |
| `unplugged_count` | How many of the cables ended up unplugged (the task file says how many there are: `cables`) |
| `all_on_table` | `y` if every unplugged cable lies on the table, `n` if one fell or is held up |
| `success` | Leave blank: it is every cable unplugged and all on the table. Write `y` or `n` only to override |
| `failure_mode` | For a failure, one of the types in the task file's `failure_mode`: `not_grasped`, `grasped_not_pulled`, `dropped`, `wrong_cable_or_order`, `stalled_or_idle`, `other`. Add types if the runs show failures that none describes |
| `scene_start` | `full` or `partial` (filled in from the prompt; change it if the run started differently) |

## 3. Cut the pictures, and look at them

```bash
# 🤖 on the station: reads the episodes' videos, sends nothing
uv run python tools/critic_eval.py frames ~/hackathon_data/critic-eval/cables --out ~/hackathon_data/critic-eval/eval
```

This writes five pictures per run, from the last second, to `eval/frames/<episode>/`, cropped as the task file says, and next to each the zoomed `strip` picture the vote gate looks at. **These files are exactly what step 4 sends.** Open a few of each run (copy them to your laptop with `scp -r` if the station has no screen), and check that no one's face is in them. If one is, narrow the `crop` in the task file and cut again. The command also says how many picture sets it dropped because it found a face.

## 4. Ask the critic

```bash
# 🤖 on the station: type the keys without saving them in the shell's history
read -rs TYPESAFE_API_KEY && export TYPESAFE_API_KEY
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

```bash
# 🤖 on the station: SENDS the cropped pictures to TypeSafe and Claude, after asking you
uv run python tools/critic_eval.py run --out ~/hackathon_data/critic-eval/eval --variant typesafe
```

It says what it is about to send and where, and waits for a `y`. To check the plumbing first, add `--dry`: it builds every request, sends nothing, and writes the payloads (without the pictures) to `eval/typesafe/requests.jsonl`, where you can read the exact questions. `uv run python tools/task_critic.py questions configs/tasks/usb-cables.yaml` prints them too. The run is saved as it goes, so a stopped run picks up where it left off, and a failed call is written to `errors.jsonl` and never counted as an answer.

To compare, run again under another name, for example with Claude answering the questions: `--gate claude --variant claude`.

### The Gemini vote gate

```bash
# 🤖 on the station: type the key without saving it in the shell's history
read -rs GEMINI_API_KEY && export GEMINI_API_KEY
```

```bash
# 🤖 on the station: SENDS the zoomed strip pictures to Gemini (and Claude, for rewritten prompts), after asking you
uv run python tools/critic_eval.py run --out ~/hackathon_data/critic-eval/eval --gate gemini --variant gemini
```

It asks the task file's `vote_gate.question` about each zoomed picture twice (`repeats`). A picture is **done** when both answers are yes, **not done** when both are no, and **unsure** when they differ or either says it cannot tell. A scene is the verdict of its five pictures: done or not done when more than half agree and none says the opposite, unsure otherwise. **Unsure means look again, never a guess.** The model is `gemini-4-argon`; `--model` picks another. The question, the zoom and the number of repeats are in the task file; the wording matters, so read the comments there before you change it.

This gate gives no cable count and no failure type: a not-done scene gets its retry prompt from the rewriter or, failing that, from the task file's prompt list. `--rewriter claude` (the default) or `--rewriter gemini` chooses who writes it, and `--rewriter none` leaves it to the prompt list.

**The Gemini rewriter** sees the scene crop and the zoomed strip, the prompts already tried, and the reasons the votes gave for what was wrong ("One cable is still plugged into the rightmost USB port"), and writes the next instruction for the robot model. Its prompt goes through the same guards as Claude's: one or two short sentences, never a repeat of a prompt already tried, and the task file's prompt list when it fails the guards. `review.csv` names which rewriter wrote each prompt (`rewrite_source`), so you can score the two side by side. Whether Dream-1 follows a rewritten prompt and fixes the scene can only be shown by a run.

```bash
# 🤖 on the station: SENDS pictures to Gemini for both the judging and the rewriting, after asking you
uv run python tools/critic_eval.py run --out ~/hackathon_data/critic-eval/eval --gate gemini --rewriter gemini --variant gemini
``` In the report, `p_done` is the share of yes votes (not a calibrated probability) and the "strip hidden" share is the share of unsure votes.

## 5. Read the report

```bash
# 🤖 on the station
uv run python tools/critic_eval.py report --out ~/hackathon_data/critic-eval/eval --labels ~/hackathon_data/critic-eval/labels.csv --variants typesafe
```

`eval/report.md` holds, per variant:

- **The base model, by prompt:** how often each prompt really worked (from your labels alone) and how many cables it unplugged on average. This tells you which wording Dream-1 follows.
- **The critic against your labels.** Two mistakes cost different things. A **false "done"** stops a retry loop with the task unfinished. A **false "not done"** sends a retry at a scene that was already finished, which can undo it. The **unsure** share is how often the pictures did not settle it: those are the cases a better viewpoint could help. The report shows the critic's decision from all five pictures together, and from the final picture alone.
- **Progress, calibration and hidden strip:** how close the cable count is (question gates only), whether `p_done` means what it says (a bucket of 0.8 to 1.0 should mostly be real successes), and how often an arm hides the strip.
- **Failure type:** how often the chosen type matches your label.

`eval/review.csv` lists the failed runs with the critic's **rewritten prompt**. Read each one and fill in `score_1_to_5` for "would this prompt fix what went wrong?". `eval/report.csv` has one row per run for your own analysis.

## Changing what the critic asks

Everything about the task is in `configs/tasks/usb-cables.yaml`: the number of cables, the crop, the thresholds (a yes/no answer is yes from 0.7, no up to 0.3, and unsure in between), the questions, the failure types with the prompt that corrects each, and the prompts to try. Keep each question to one thing a person could answer at a glance, and change the wording there, not in the code. After a change, run `run` again under a new `--variant` name and compare.

`tools/critic_eval.py` also takes the teleop demonstrations of the task (`frames ~/hackathon_data/unplug-cables`): each gives a start sample (labelled "not done" automatically) and an end sample (labelled "done" automatically). Give `report` a `--labels` file and **your label wins at an episode's end**; a start keeps its automatic one. A demonstration can end unfinished (episode 4 of the first ten did), so label them. These demonstrations are 640 by 360 pixels, while a run recorded with `dreamcli robot run` is as large as the camera setting in the station file (1280 by 720): the demonstrations are the hardest case for the critic, and say little about the runs of Dream-1.
