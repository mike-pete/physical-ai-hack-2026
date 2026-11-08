# Supervising a run

`dreamcli robot run` plays the model's moves until someone presses Ctrl+C, whether or not the task is getting done, and the Dream library has no way to tell success. The **supervisor** (`tools/supervise.py`) watches a run, **stops** it when the [task critic](critic_eval.md) says the task is done, and **guides** Dream-1 with a corrected prompt when it is failing. It works for any task that has a task file (`configs/tasks/*.yaml`); the first is unplugging the orange USB cables.

It never opens the arms or the cameras. It calls the station agent's HTTP API (`dreamcli robot up`, port 8091: `status`, `stop`, `home`, `start` and `prompt`) and reads the video the Dream recorder writes while a run goes. So it needs `dreamcli robot up` and the recorder on, and nothing else of Dream.

```
supervisor (uv run) --HTTP--> station agent (dreamcli robot up, in ~/dream-env)
      |                              |
      +---- reads the growing video <---- the recorder (rollouts.enabled)
      +---- Gemini: judges the pictures and writes prompts (after asking you)
```

## Read this first

- **Acting mode moves the arms.** `supervise.py run` (without `--observe-only`) stops the run, sends the arms home and starts the next attempt, through the station agent. Run it only with a hand on the emergency stop and everyone out of the arms' reach, like any real run ([running your model](running_your_model.md#3-run-it)). **Claude Code does not run it for you**; it gives you the command.
- **It attaches to a run you started** (Start in Live View, or one `curl`). It starts a run itself only with `--start --prompt "..."`.
- **Observe-only mode makes no control call at all.** It watches a real run and logs what it *would* do. The first powered session is always observe-only.
- **Pictures leave this computer**, to Google (and to Anthropic with `--rewriter claude`). It says what it will send and asks first (`--yes` skips the question). Each picture is cropped to the table, and a picture with a face in the crop is never sent. Every picture it sent is kept in the log folder.
- **The keys stay in your shell.** Type `GEMINI_API_KEY` with `read -rs`, never into a file or a command line:

  ```bash
  # 🤖 on the station
  read -rs GEMINI_API_KEY && export GEMINI_API_KEY
  ```

- **If the supervisor dies, the run stops.** A crash, Ctrl+C or `kill` makes it send `POST /stop` first, so the arms hold where they are rather than carry on unwatched. (Observe-only mode sends nothing, even then.)

## What it decides

Every second it asks the station for its status. Every 10 seconds (after the first 15) it takes a **probe**: three pictures from the last second of the recording, each judged by the critic (the vote gate: one plain question about a zoom on the strip, asked twice). A picture is **done** only if every answer about it is yes, **not done** only if every answer is no, and **unsure** otherwise; the probe is done or not done when most of its pictures say so and none says the opposite, and unsure otherwise. Probes run in the background, so the loop stays responsive.

| Situation | What it does |
| --- | --- |
| **Two probes in a row say done** | `stop`, then `home`. One probe never ends a run: an arm in the way or a lucky answer would stop a task unfinished. |
| **First failure** of an attempt | **Guides in place**: Gemini writes a corrected prompt from the scene and the strip, and `POST /prompt` starts a fresh episode from where the arms are. No homing. A 45 s guide window starts. |
| **A second failure inside the window**, or the time limit after a guide | `stop`, `home`, then `start` with a new prompt on the same checkpoint: attempt 2, then 3. |
| **Out of attempts** (3) | `stop`, `home`, report. The arms stay home. |
| The station goes **idle** while it watches | Someone stopped the run, so the supervisor ends and starts nothing. |
| The station is in **error** | A failure: restart (no guide: nothing is running to guide). |
| Not done, or unsure | Normal in the middle of a task (an arm covers the strip). Never a failure by itself. |

A **failure** is any of: the attempt passed its time limit (90 s; the demonstrations last 33 to 83 s), the station reports a new error, the stalls rose by more than 3, or the task is **stuck**.

**Stuck** means the task shows no progress for 20 s while the arms are being commanded (the station's `chunks` rise). Progress is measured the way the task file says, so it works for any task:

- With an `object_motion` detector (the task's objects have a colour: the orange cables), it measures how much the pixels of that colour changed between probes. Only that colour counts, so an arm passing through cannot hide a stall. Once nothing moved for 20 s, a **model compare** is asked: Gemini sees the earlier and the latest picture and the task's goal and says whether the task moved forward. **Stuck needs the compare to say no.** A yes or an unsure restarts the 20 s, so a wrong colour range alone can never stop a good run.
- With no colour range, the compare is asked on every other probe, and two noes in a row (about 40 s) mean stuck.
- If the compare cannot be asked (no key, a failed call), a quiet scene is logged and not acted on.

Every prompt, written by Gemini or taken from the task file's `prompts` list, goes through the same guards as in the critic: one or two short sentences, never one already tried.

**A restart cannot judge the scene at home.** The recorder writes video only while a run is going, so nothing can be seen while the arms are homing. Instead, a new attempt is looked at 1 s after it starts, while the arms are still near home; if that look and the next one say done, it ends the attempt early. The next look is at the end of the 15 s warm-up. If the recorder has not written its first picture 1 s in, the early look is skipped (the log says why) and the first look is the one after the warm-up; the first observe-only session shows whether 1 s is late enough.

## 1. Turn the recorder on (once)

The same setting as for [recording runs for the critic](critic_eval.md#1-record-the-runs): a copy of the station file, then in `~/.dreamlabs/station.yaml`

```yaml
rollouts:
  enabled: true
```

and restart `dreamcli robot up` so it takes effect. `dir:` is optional (default `~/.dreamlabs/rollouts`). A dry run records nothing, so there is nothing to watch.

## 2. Look first (no arms, nothing sent)

```bash
# 🤖 on the station: reads the station's status and the recorder's files; changes and sends nothing
cd ~/physical-ai-hack-2026
uv run python tools/supervise.py check
```

It reports the station agent (state, run, prompt, chunks, stalls), whether the recorder is on, the newest picture of the scene camera, and writes the exact pictures it would send under `~/hackathon_data/supervisor/`. **Look at them**: the table and the strip must be in view, with no one's face. With no run going it shows the last picture of the newest recorded run; it is most useful while a run is on.

## 3. Rehearse without a robot

```bash
# 🤖 on the station: replays a recorded demonstration against a FAKE station; nothing moves, nothing is sent
uv run python tools/supervise.py rehearse --dry --video <a demonstration's scene video>.mp4 --speed 20
```

`--dry` uses stand-ins that always answer "unsure", so the supervisor runs its whole ladder (guides, restarts, giving up) on the fake station. Without `--dry`, it sends the replayed pictures to Gemini, after asking, and shows what the real critic makes of a demonstration. Use `--speed 1` for that: model calls are not sped up.

## 4. The first powered session: observe only

Keep a hand on the emergency stop. Run inside tmux (`tmux new -s supervise`).

```bash
# 🤖 on the station, window 1: the station agent (arms stay still until you press Start)
source ~/dream-env/bin/activate
dreamcli robot up
```

```bash
# 🤖 on the station, window 2: watch only. Press Start in Live View after it says it is waiting for a run.
cd ~/physical-ai-hack-2026
read -rs GEMINI_API_KEY && export GEMINI_API_KEY
uv run python tools/supervise.py run --observe-only
```

The arms move because you pressed Start, as in any run. The supervisor only watches. Check, in this order:

1. **It sees.** `attach` appears, then `probe` lines every 10 s or so. A line `skip ... has not grown` or `no recording` means the recorder is off or lagging; fix that before anything else.
2. **The probes make sense.** `yes 0/6` while cables are plugged in, `yes 6/6` when they are all out. Look at the pictures under `pictures/`.
3. **Its latency.** Each `probe` line says how long it took (the number after `yes x/6`). A probe longer than about 10 s means the next one waits for it.
4. **The run is no worse.** Compare `stalls` and the station's timing with a run without the supervisor. It runs at low priority, because heavy CPU on the station can starve the arm thread (Dream's own note on the Metal gripper's 500 ms timeout).
5. **What it would do.** `would call ...` and `decision` lines are the actions it would have taken. Do they fit what you saw?

## 5. Acting mode

Once the observe-only session looked right, with the same hand on the emergency stop:

```bash
# 🤖 on the station: ACTS. It can stop the run, send both arms home and start the next attempt.
uv run python tools/supervise.py run
```

Press Start in Live View as before. Keys, while it runs:

| Key | Does |
| --- | --- |
| `f` | **Finish now**: `stop`, then `home`, as if the task were done |
| `n` | **Next attempt now**: `stop`, `home`, `start` with a new prompt |
| `p` | **Pause or resume** watching (the run goes on; on resume the time limit starts over) |
| `x` | **Abort**: `stop`, and quit. The arms hold where they are (they are not sent home) |

In observe-only mode `f` and `n` log what they would do instead of doing it (`f` and `x` still end the supervisor). If someone else starts a run while it watches (Live View's Start), it follows the new run from its own start.

To start a run too, add `--start --prompt "Unplug each orange USB cable ..."`; with no `--start` it waits up to `--wait-s` (120) seconds for a run to be going.

## What it leaves behind

A folder per session, `~/hackathon_data/supervisor/supervise-<date>-<time>/` (local time):

| File | Holds |
| --- | --- |
| `events.jsonl` | One line per event: `attach`, `probe` (decision, every vote and its reason, the watchdog's reading), `skip` (why a probe saw nothing), `decision`, `guide`, `restart`, `control` (each call it made), `would` (each call it did not make), `refused` (a 409 from the station), `adopt`, `key`, `end` |
| `pictures/` | Every picture sent to a model: `probe003-1-strip.jpg` is the second picture of probe 3 |
| `requests.jsonl` | The model requests without their pictures, so any of them can be rerun |
| `report.json` | The outcome (`done`, `gave_up`, `ended`, `aborted`, `interrupted`, `error`, `crashed`), attempts, guides, every prompt used, probes, control calls |

The exit code is 0 for `done`, `ended`, `aborted` and `interrupted`, 3 when it gave up, and 1 for an error or a crash.

## Changing what it does

All of it is in the `supervisor:` section of the task file, with the reason for each number next to it: `attempts`, `attempt_s`, `warmup_s`, `probe_every_s`, `frames_per_probe`, `done_probes`, `max_stalls`, `stuck_after_s`, `guide_window_s`, and `progress` (`object_motion`: a view, a colour range and how much change counts; `model_compare`: a view and optionally your own question, with `{goal}` and `{seconds}` filled in). Every key has a default, and a task file with no section still works, with the model compare as its only progress detector. `task_critic.py` rejects an unknown key, naming it.

For a new task: write its task file ([critic_eval.md](critic_eval.md#changing-what-the-critic-asks)), leave `progress` empty at first, and add an `object_motion` colour once you have looked at its pictures. The orange range in `usb-cables.yaml` is in Lab colour, where the red-green channel separates the cables' salmon from the table; it was calibrated on the ten demonstrations (the change between probes 10 s apart was 0.22 to 1.0 while the arms worked and about 0.01 on a still scene). Check it on your own camera with `check` and a few probes before trusting it.

## Running several tasks in a row

`tools/sequence.py` runs the supervisor over a list of tasks on one table, say unplugging the cables, stacking the blocks, then putting the golf balls in the box. The arms go home after every step, and the next step starts only when the one before has finished. It is a layer above the supervisor: each step is an ordinary supervised run with its own task file, critic and limits.

```
sequence file --> step 1 (supervisor: up to 3 attempts) --> home --> step 2 --> home --> step 3 --> home --> audit
```

**The sequence file** (`configs/sequences/tabletop.yaml` is a template; it is refused until you fill it in):

```yaml
checkpoint: dreamlabs/dream-1-beta   # every step runs this; leave out to use the station's own
sequence:
  - name: unplug-cables              # labels logs and folders; unique
    task: configs/tasks/usb-cables.yaml
    prompt: "your first instruction for the base model"
    # attempt_s: 120                 # optional: replaces the task file's value for this sequence only
    # max_attempts: 2
  - name: stack-blocks
    task: configs/tasks/stack-blocks.yaml
    prompt: "..."
between_steps: home                  # the only choice for now
on_give_up: continue                 # or stop: end the sequence after the home
audit: final                         # or off
```

**Rules.**

| | |
| --- | --- |
| One task at a time | Step N+1 starts only after step N finished and the station reports `idle` again. The station agent also refuses a start unless it is idle. |
| A step ends done or gave up | Done is two probes in a row where every vote says yes. Gave up is every attempt used. Either way the arms go home. |
| A give-up does not stop the sequence | The step is marked failed, the arms go home, and the next step runs (`on_give_up: continue`). The report keeps the earlier give-up next to the later steps, because a half-unplugged strip or a knocked-over tower is what the next task starts from. |
| Someone else stops the run | The sequence ends and starts nothing, as a single task does. |
| Anything unexpected stops it where it is | A 409 from the station, a station that is not idle after home (or in `error`), a recording that does not appear within 30 s: the sequence ends, the arms stay where they are, and nothing is retried blindly, forced or reset. |

**Between two steps** the layer waits for `idle` (the supervisor has already sent `stop` and `home` at the end of the step; if the station refused the home, the layer sends it once more), writes the step's result, loads the next task file and builds its critic (crop, views, vote gate, colour range), starts the run with the next prompt on the same checkpoint, waits for the run number to rise, and points the video reader at the new run's folder. The 15 s warm-up then starts. At most one control call goes out every 3 s across all steps.

**The final audit.** After the last step the layer cuts the last frames of the last run (the files stay on disk after the home) and asks every task's vote gate its question about them: *still done* (every vote yes), *undone* (every vote no; a step that was done is then flagged **regression**), or *unsure* (the votes disagree, or an arm covers the region; never counted as a failure). It only reports; it moves nothing. It runs only when every step ran.

**What you get back.** One folder per sequence, `sequence-<date>-<time>/`: `sequence.json`, `summary.txt` (the combined table: step, outcome, attempts, seconds, prompts, last probe, audit), one subfolder per step (`01-unplug-cables/` with its `events.jsonl`, `pictures/`, `report.json`), and one `events.jsonl` in which every line carries a `step` field. A step flagged **done at start** had its first probes already say done: the scene looked finished when it began, or the model finished before the first look (the 15 s warm-up cannot tell those apart). The exit code is 0 when no step gave up, 3 when one did, 1 on an error or a crash.

**Keys** (inside tmux): `f` (or `q`) finish this step as done, then home and go on; `n` start the next attempt now; `p` pause or resume; `s` skip this step (stop, home, mark it skipped, go on); `x` abort the sequence (stop, and **home**; a single task's `x` leaves the arms where they are). Ctrl+C or `kill` stop the run and exit without homing.

**Try it, in this order.**

```bash
# 🤖 on the station: validates the file and every task file it names; no network, no station
uv run python tools/sequence.py check configs/sequences/tabletop.yaml
```

`check` prints each step and the worst-case run time (every attempt used, before guide windows and the transitions), and warns when two steps' zoomed views overlap on a camera: an object of one task could then show in the other's zoom or colour mask (golf balls in the cable colour range, say). Look at the pictures of each task alone first (`uv run python tools/supervise.py --task <its task file> check`).

```bash
# 🤖 on the station: replays one recorded demonstration per step against a FAKE station; nothing moves
uv run python tools/sequence.py rehearse configs/sequences/tabletop.yaml --dry --speed 20 \
  --video unplug-cables=<a cables demo>.mp4 --video stack-blocks=<a blocks demo>.mp4 --video golf-balls=<a golf demo>.mp4
```

Without `--dry` it sends the replayed pictures to Gemini, after asking, and shows what the real critic makes of each demonstration.

```bash
# 🤖 on the station, with a hand on the emergency stop: you start every step by hand in Live View
read -rs GEMINI_API_KEY && export GEMINI_API_KEY
uv run python tools/sequence.py run configs/sequences/tabletop.yaml --observe-only
```

Observe-only makes no control call at all. After each step it logs that it would stop and home and start the next one, and then **waits for you to stop, home and start the next step by hand** (`--wait-s`, 120 s by default; raise it). Then, with the same hand on the emergency stop:

```bash
# 🤖 on the station: ACTS. It starts the steps, and every transition moves both arms.
uv run python tools/sequence.py run configs/sequences/tabletop.yaml --start
```

Without `--start` the first step attaches to a run you started in Live View (its prompt is noted if it differs from the file's); every later step is started by the layer. Watch the first few transitions: what *home* does with something still in a gripper after a give-up is not known yet.

**Before it can run the real tasks.** `configs/tasks/stack-blocks.yaml` and `golf-balls.yaml` must exist, each with a vote gate, views and a `supervisor:` block, tried on their demonstrations the way the cable task was. `task_critic.py`'s loader still requires the question gate's sections (`items`, `visibility`, `progress`, `failure_mode`, `thresholds`, `prompts`) in every task file even when a vote gate is present; until someone makes them optional, copy them from `usb-cables.yaml`. The golf demonstrations mix "basket" and "box" in their texts: the task file's question and goal need one word.

## What is not tested yet

Everything above that can run without hardware is tested: the decisions, the watchdog on synthetic and recorded scenes, whole runs against a fake station with a fake clock and scripted models, and reading a video as it is written. These are **not** tested until someone does a powered session:

- Against a real `dreamcli robot up`: the routes are read from the Dream library, not exercised. `POST /prompt` restarting the episode in place, and the station's timing for `stop`, `home` and `start`, need the staged session above (observe-only first, then one failure on purpose with generous limits).
- The probe's latency and its effect on the arm thread, on the station's CPU.
- The critic on **Dream-1's own runs**. It was validated on the teleop demonstrations (the Gemini vote gate got every decision right, with two "unsure"). Runs of Dream-1 at 1280 by 720 may differ.
- The vote gate asks whether the cables are **unplugged**, not whether they are **on the table**. A cable dropped on the floor would read as done.
- The sequence layer has only run against the fake station. Its transitions (stop, home, idle, start, the new run's recording) and the audit on Dream-1's real runs need the staged sessions: each task alone, observe-only and then acting; a two-step chain; then the whole chain, several times, with the table reset by hand between trials. Per-step success multiplies: three steps that each succeed 80% of the time give all three about 51% of the time, which is an illustration and not a measurement.
- The recording lags the robot by up to a third of a second (it is written in fragments of ten frames), and the 3 pictures of a probe come from the last second.
