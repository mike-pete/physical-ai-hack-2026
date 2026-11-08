# Collecting demonstrations

Your station is already set up; this is where you start. You teach the model by showing it the task many times. Each time, from the start of the task to its end, is one episode. This guide covers the controls, a recording session, where your data is stored, and what makes demonstrations a model learns well from.

## Quick start

1. **On your laptop**, connect to the station with the dashboard's port forwarded. Run this on your laptop only: run on the station, it takes the dashboard's port there and your laptop gets nothing.

   ```bash
   ssh -L 8080:localhost:8080 <user>@<station>
   ```

   Replace `<user>@<station>` with your station's login, which the organizers give you (for example `team3@10.0.0.13`): typed as is, the shell reads `<` and `>` as redirections and says `parse error near '\n'`. Run this line on its own and wait for the station's prompt (`<user>@<station-name>:~$`) before typing anything else; it may ask for the station's password first.

2. **On the station**, at that prompt (tmux is on the station; your laptop does not need it), start a tmux session (or rejoin yours), so a dropped connection does not end it:

   ```bash
   tmux new -A -s collect
   ```

   Run it on its own: it opens a new screen, and lines pasted along with it do not reach that screen. Type the rest in the screen it opens:

   ```bash
   cd ~/physical-ai-hacks
   ```

3. With the arms in their start pose, put on the Quest and practice once with the arms still (Ctrl+C ends it):

   ```bash
   uv run dhk-collect --config configs/station.yaml --dry-run
   ```

   While it runs, open http://localhost:8080 in your laptop's browser: the dashboard shows the cameras, the arms and your episodes. The page is there only while `dhk-collect` runs; before that, the browser cannot connect.

   The page lists every device: the Quest, the three cameras and both arms' links. A red one says what is wrong and what to do, and the session waits for it. Once every device is ready, press A on the right controller: only then are the arms opened.

4. Start the session, with your team's own dataset folder:

   ```bash
   uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/myteam-cubes
   ```

   No task text or other labels are needed: Dream's service labels every dataset from its overhead camera when it processes it. To name the task yourself anyway, type it in the dashboard's Session panel and press "Set task" (it applies to the next episodes), or start with `--task "put the red cube in the bowl"`.

5. For each episode: set the scene, press A, do the task, press A again to save it (B throws it away). Press Ctrl+C when you are done.

Your episodes are saved in `~/hackathon_data/myteam-cubes` on the station ([where your data is stored](#where-your-data-is-stored)). The sections below explain each step.

## The controls

| Control | Does |
| --- | --- |
| Grip (middle finger), held | Drives that controller's arm: the gripper follows your hand |
| Grip released | The arm stops where it is; move your hand and grip again to keep going |
| Index trigger | That arm's gripper: released is open, pulled is closed |
| A | Starts recording an episode; pressed again, saves it. At the start of a session, once every device is ready, it first opens the arms |
| B | Throws away the episode being recorded |
| Right joystick click | Sends both arms back to where they were when the session started (not while recording) |
| Ctrl+C (in the terminal) | Ends the session; an episode still recording is thrown away and the arms return to their start |

The arm follows how your hand moves after you grip, not where your hand is. If you run out of reach, release, move your hand back, and grip again, like lifting a mouse off the table. The app reports each controller relative to the headset, so keep your head still and facing the arms while you grip.

## Working on a station over SSH

The station has no screen. Connect from your laptop with the dashboard's port forwarded. This line runs on your laptop, never on the station:

```bash
ssh -L 8080:localhost:8080 <user>@<station>
```

Replace `<user>@<station>` with your station's login, which the organizers give you (for example `team3@10.0.0.13`): typed as is, the shell reads `<` and `>` as redirections and says `parse error near '\n'`. Run this line on its own and wait for the station's prompt (`<user>@<station-name>:~$`) before typing anything else; it may ask for the station's password first. Then, on the station, start a tmux session, so that a dropped Wi-Fi connection does not end your session:

```bash
tmux new -A -s collect          # after a drop, reconnect from the laptop and run the same line to get back to it
```

Run it on its own, then type the rest in the screen it opens (lines pasted along with it do not reach that screen):

```bash
cd ~/physical-ai-hacks
```

While `dhk-collect` runs, open http://localhost:8080 in your laptop's browser. The dashboard shows:

- Every device: the Quest, the three cameras and both arms' links. A red one says what is wrong and what to do: a camera unplugged (it comes back by itself once plugged in again), the headset asleep or not connected, a CAN link down after a reboot. Some have a button that fixes them: "Bring ... up" for a CAN link that is down (it never takes a link down), "Restart the Quest app" when the controllers' data stops (the app went behind Quest Home, or the headset slept), and "Restart adb" when the headset is `unauthorized` (its prompt shows again).
- At the start, what the session is waiting for, then "press A on the right controller to open the arms" once every device is ready.
- The three cameras, cropped and sized as they are recorded.
- Whether an episode is recording, and for how long.
- Each arm: driving or holding, its tool position and its gripper.
- Warnings (a stalled camera, an arm that cannot follow) and the session's events.
- The task text the next episodes record, with a box to type a new one ("Set task"). An empty one goes back to the default.
- The episodes saved so far, with each camera's video to play back, and a checkbox on each: tick the ones you do not want and press "Delete selected" (it asks first, and is refused while an episode is recording).

Tick "sound cues" and the laptop beeps when recording starts (one beep), when an episode is saved (two) or discarded (a low one). That lets a teammate at the laptop follow the operator. Nothing on the page moves the arms; its buttons fix a device, set the task, or delete the episodes you tick.

If the page does not load:

- **`channel 3: open failed: connect failed: Connection refused` lines in the SSH terminal**: a browser tab is asking for the page while `dhk-collect` is not running yet. They are harmless and stop once it runs; close the tab until then if they get in the way of your typing.
- **"This site can't be reached"**: `dhk-collect` is not running (the page exists only while it runs), or your laptop's SSH connection was opened without `-L 8080:localhost:8080`. Reconnect from the laptop with it.
- **`dhk-collect` says `dashboard: port 8080 is not free`**: something on the station already holds the port, usually an `ssh -L 8080:...` that was run on the station itself. Type `exit` in that terminal to close it (the prompt stays the same: it is the same computer), or run on with `--dashboard-port 8081` and forward that port instead.

If the connection drops outside tmux, the session ends safely: the episode being recorded is discarded, the arms return to where they started and are released, and the rest of the output goes to `~/.cache/dream_hackathon/collect.log`. Only one program can drive the arms at a time. If someone else is using the station, `dhk-collect` says who, from where and since when.

## A recording session

1. Start with the arms resting in their start pose: on a MakerMods station, supported or folded, since they go limp there at the end.
2. Practice the controls without moving the arms. A dry run shows each arm's tool position once a second; hold a grip and watch it follow your hand:

   ```bash
   uv run dhk-collect --config configs/station.yaml --dry-run
   ```

3. Start the session. Give your dataset a folder of its own, named for your team and task:

   ```bash
   uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/myteam-cubes
   ```

   You do not need to label the episodes: when Dream processes the dataset, its service writes dense text labels for it from the overhead camera. A task text of your own is optional: type it on the dashboard ("Set task", from the next episode on) or start with `--task "put the red cube in the bowl"`, and write it the way you will ask the robot. Without it, each episode records the neutral text `do the demonstrated task`, since the dataset format needs one.

4. For each episode: set the scene, press A, do the task, press A to save. If it went wrong, press B instead and do it again.
5. Press Ctrl+C when you are done. The arms return to where they started.

Every session starts the same way: the dashboard lists the devices, and once every one is ready, A opens the arms. Every session adds its episodes to the same `--dataset-dir`, so you can stop and continue later, or switch operators. Use one dataset folder per task; if you name your tasks with `--task`, a different text in the same folder is fine too, when one model should learn several tasks. If a device stays red on the dashboard (an arm, a camera, the Quest), do what it says; if that does not fix it, call the organizers rather than changing the station's config. A camera that was unplugged comes back by itself once it is plugged in again.

Each saved episode is printed with its number. To check the dataset at any point against Dream's own checks, without uploading anything (see [fine-tuning](fine_tuning.md)):

```bash
# YAM
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/myteam-cubes --arm yam --check
# Metal or Maker arm: add_eef checks the dataset it writes; --replace rebuilds it after you collect more
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/myteam-cubes --arm metal --replace
```

## Removing episodes

B throws away the episode you are recording. To remove ones you notice later, after watching them on the dashboard, tick their checkboxes in the Episodes panel and press "Delete selected". It asks first, and is refused while an episode is recording: save (A) or discard (B) it first. The episodes after them are renumbered (Dream wants 0, 1, 2, ...), and the next one you record follows on.

Without a running session, `dhk-dataset` does the same from the terminal:

```bash
uv run dhk-dataset list ~/hackathon_data/myteam-cubes
uv run dhk-dataset drop ~/hackathon_data/myteam-cubes 3        # asks first; several numbers drop several
```

The episodes after it are renumbered, so check `list` again before dropping another. `dhk-dataset` refuses the folder a running `dhk-collect` records into: use the dashboard then. On a Metal or Maker arm, rebuild the dataset you fine-tune on afterwards with `tools/add_eef.py ... --replace`.

## Where your data is stored

Your data stays on the station, in the folder you pass with `--dataset-dir` (without it, the folder the config's `dataset_dir` names). It is not in the repo, and nothing is uploaded until you fine-tune or [send a copy to Dimensional's cloud](#keeping-a-copy-in-dimensionals-cloud). Each folder is one LeRobot v2.1 dataset, and every session adds to it:

```
~/hackathon_data/myteam-cubes/
  meta/
    info.json                       the dataset's shape: frame rate, cameras, state and action
    episodes.jsonl                  one line per episode: its number, task and length
    tasks.jsonl                     the task texts
    episodes_stats.jsonl            each episode's statistics, which Dream's checks read
  data/chunk-000/
    episode_000000.parquet          one row per frame: observation.state, action, timestamp, ...
    episode_000001.parquet
  videos/chunk-000/
    observation.images.overhead/episode_000000.mp4
    observation.images.wrist_l/episode_000000.mp4
    observation.images.wrist_r/episode_000000.mp4
```

On a Metal or Maker arm the video folders are `observation.images.scene`, `observation.images.left_wrist` and `observation.images.right_wrist`, and `tools/add_eef.py` writes the dataset you fine-tune on beside yours, as `~/hackathon_data/myteam-cubes-eef`.

Two more places on the station:

- `~/.cache/dream_hackathon/recording/` holds the episode being recorded. Saving it moves it into your dataset; B or Ctrl+C deletes it.
- `~/.cache/dream_hackathon/collect.log` holds a session's output after the SSH connection drops.

To keep a copy on your laptop, run this on the laptop:

```bash
scp -r <user>@<station>:hackathon_data/myteam-cubes .
```

### Keeping a copy in Dimensional's cloud

`tools/upload_dimensional.py` sends datasets to Dimensional's hosted storage with `dimos data upload`, and lists them in [console.dimensional.org](https://console.dimensional.org). It is separate from Dream: no fine-tuning, no charge, and the dataset folder stays as it is. You need Dimensional's `dimos` command and one login (`dimos login`: it prints a code you approve in any signed-in browser; `dimos` keeps the key, and the script never reads it).

`dimos data upload` takes one file, so the script packs each dataset folder into a `<name>.tar` first (the same files, byte for byte, built the same way every time, so uploading the same dataset twice reports `already uploaded`). Upload the folders `dhk-collect` wrote, not the `-eef` ones `tools/add_eef.py` makes from them, which you can rebuild. It needs only Python 3.12 or newer:

```bash
# First see what would be sent: this packs and sends nothing
python3 tools/upload_dimensional.py ~/hackathon_data/myteam-cubes --dry-run
# Send it, keeping the output
python3 tools/upload_dimensional.py ~/hackathon_data/myteam-cubes --log docs/dimensional_uploads/myteam-cubes.log
```

It checks the login first, uploads one dataset at a time (each tar is deleted after its upload), then runs `dimos data ls` and prints `confirmed` only when every tar is listed; otherwise it says `NOT CONFIRMED` and exits with an error. `dimos` prints a `console:` link for each upload. To get a dataset back on any computer, run `dimos data pull <id>` (the id is in `dimos data ls`), then `tar -xf` the file it writes.

## What the dataset holds

Each episode records 30 frames per second of:

- `observation.state`: both arms' measured joints (radians) and grippers.
- `action`: the joints and grippers commanded at that moment.
- One video per camera, cropped to 16:9 and recorded at 640x360.

On a YAM, the order is the right arm's six joints, its gripper (0 closed to 1 open), then the left arm's. On a Metal or Maker arm, the left arm comes first and the gripper is its opening in millimetres; `tools/add_eef.py` adds the tool poses before fine-tuning.

## Demonstrations that train well

- **Do the same task the same way.** The model copies what you show it. Pick one strategy (which arm grasps, from which side) and keep to it.
- **Vary the scene, not the method.** Move the objects to different positions and orientations across episodes, within the area the model will see when it runs.
- **Plan for 50 or more episodes** of each task, each 10 to 30 seconds. Every episode must be at least 1.6 seconds; a shorter one is not saved.
- **Start and end calmly.** Start recording with the arms still, and save shortly after the task is done.
- **Throw away mistakes.** A failed grasp you recover from is fine. An episode that did not do the task is better discarded with B.
- **Keep the cameras where they are.** Do not move a camera between episodes, or between collecting and running the model.
- **Move smoothly and not too fast.** The arms follow at a capped speed (`teleop.max_joint_speed_rad_s`); jerky hand motion makes jerky demonstrations.
