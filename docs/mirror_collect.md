# Collecting demonstrations by moving the left arm

`scripts/mirror_collect.py` is a second way to record demonstrations, next to the Quest ([data_collection.md](data_collection.md)). You take the **left arm** in your hand and move it; the **right arm** copies it, and the dashboard in your browser starts, saves and discards episodes. No Quest is needed. The episodes are saved in the same LeRobot v2.1 format as `dhk-collect`, so [fine-tuning](fine_tuning.md) works on them the same way.

It works on **Metal and Maker arm stations only** (not the YAM). The station has no screen: you run it over SSH and watch the dashboard in your laptop's browser.

## Read this before you start

- **There is no dry run.** The right arm is energized as soon as the script opens it, and it follows the left arm from the first moment, with no ramp. Hand on the emergency stop, everyone out of the right arm's reach, and nothing on the table that should not be hit.
- **Start with both arms in the same pose**, resting where they can go limp. If the left arm is somewhere else when the script starts, the right arm moves there.
- **Support the right arm** when you start: MakerMods arms have no brakes.
- **One program at a time.** Stop `dhk-collect`, `dreamcli` and any earlier `mirror_collect.py` first: each holds the cameras and the arms, and a second one fails with `cannot open /dev/v4l/...`.

## Start a session

1. **On your laptop**, log in to the station with the dashboard's port forwarded (the account is `pah`; the organizers give you the IP address):

   ```bash
   # 💻 on your laptop
   ssh -L 8080:localhost:8080 pah@<ip_address>
   ```

2. **On the station**, start tmux, so a dropped connection does not end the session. Run it on its own: it opens a new screen, and you type the rest there.

   ```bash
   # 🤖 on the station
   tmux new -A -s mirror
   ```

3. **Start the script.** The task you give names the folder your episodes go in (see [Where the episodes go](#where-the-episodes-go)). On a Metal arm station the left arm is weightless by itself, with nothing to add:

   ```bash
   # 🤖 on the station: REAL RUN, the right arm moves
   cd ~/physical-ai-hacks
   uv run python scripts/mirror_collect.py --config configs/station.yaml --task "pick up the red cube"
   ```

   On a Maker arm station, or when gravity compensation fails to start, the left arm is opened with its motors off, so you carry its weight and the friction in its joints. It works, but it is heavy to move, and **Go to Home** then moves only the right arm.

4. **Open http://localhost:8080 in your laptop's browser.** The page is there only while the script runs. If the script says the port is taken, add `--port 8091` to the command and use the same number in the `ssh -L` line and in the browser.

## The dashboard

| What | Where |
| --- | --- |
| The three cameras, as they are recorded | Top of the page |
| `idle`, `REC` or `HOMING`, then frames and episodes saved, the left arm's pan angle and its gripper | The status bar |
| How the left arm is held | The status bar, next to the mode: **green** `gravity compensation`, **orange** `torque off, no gravity compensation`, **red** `gravity compensation FAILED` |
| The folder the episodes are saved in | Below the status bar (`folder: ...`) |
| **Start Episode** / **Save Episode**, **Discard**, **Go to Home** | The buttons |
| The task, and **Set Task** | The text box |

**Check the left arm's colour before you record.** Red means gravity compensation was expected but did not start, so the arm is running with its motors off. The console says why; call the staff if it keeps happening. Green means the motors hold the arm up, so you move it with much less effort.

To record an episode:

1. Move the left arm; the right arm copies it. Close the left gripper to close the right one.
2. Press **Start Episode**, do the task with the left arm, then press **Save Episode**. **Discard** throws the episode away.
3. **Go to Home** moves **both arms** to Dream-1's home pose (the one `dreamcli setup` saved in `~/.dreamlabs/station.yaml`), the left arm under its own motor power, at half the station's top joint speed. Keep clear of both arms and a hand near the emergency stop. If Dream's home cannot be read, or the left arm is not in gravity compensation, it moves only the right arm, to the pose it had when the script started (the console says which). It is refused while an episode is recording.

You can also type in the tmux screen instead of clicking: Enter starts or saves an episode, `d` discards it, `h` goes home, and `q` quits.

An episode shorter than 1.6 seconds is dropped rather than saved; the console says so.

## Where the episodes go

Each task gets a dataset folder of its own, named after it, under `~/hackathon_data`: the task above is saved in `~/hackathon_data/pick-up-the-red-cube`. The name is the task in lowercase letters, digits and hyphens, at most 50 characters.

- **A new task:** type it in the box and press **Set Task**. The next episodes go in that task's folder, created on the spot. This waits for the episode in progress to be saved or discarded.
- **No `--task`:** the script starts with no folder, and **Start Episode** is refused until you set a task on the page.
- **Everything in one folder:** pass `--dataset ~/hackathon_data/my-task` and every episode goes there, whatever the task says. `--data-root <folder>` moves the place where the per-task folders are made.

List what you have saved (this only reads the folder):

```bash
# 🤖 on the station
uv run dhk-dataset list ~/hackathon_data/pick-up-the-red-cube
```

## Ending the session

**Support the left arm before you end the session.** In gravity compensation, its motors stop holding it up when the session ends, and it can fall.

In the tmux screen, type `q` and press Enter. It saves the episode in progress, returns the right arm to its start pose and releases it. Ctrl+C does the same but **discards** the episode in progress. Do not use `kill -9` or close the window: that skips the shutdown, so the right arm is neither returned nor released.

Your saved episodes are safe either way: each is written to disk the moment you save it.

## When something goes wrong

| What you see | What it means |
| --- | --- |
| `camera 'overhead': cannot open /dev/v4l/...` | Another program holds the cameras, often an earlier `mirror_collect.py`. Find it with `pgrep -af mirror_collect` and end it as above. |
| `could not start dashboard on port 8080` | The port is taken. Use `--port 8091` and match it in `ssh -L` and the browser. |
| `gravity compensation failed (...); falling back to torque-off` | The left arm is running without gravity compensation (the bar is red). Call the staff. |
| `This script is for Metal/Maker arm stations only` | A YAM station: use [`dhk-collect`](data_collection.md). |
| Start Episode is greyed out or refused | Set a task first, or wait for homing to finish. |

If a device does not work and the table does not help, call the staff.

## Next

Fine-tune on what you recorded: [fine_tuning.md](fine_tuning.md). On Metal and Maker arms, run `tools/add_eef.py` on the folder first, as the guide shows.
