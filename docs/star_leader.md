# Collecting demonstrations with the Star Arm 102 leader

A Star Arm 102 is a small leader arm you hold in your hand. With it, [`scripts/mirror_collect.py`](mirror_collect.md) records demonstrations the same way as when you move the left arm: the **right arm** copies the Star arm, and the dashboard in your browser starts, saves and discards episodes. The episodes are saved in the same LeRobot v2.1 format as `dhk-collect`, so [fine-tuning](fine_tuning.md) works on them the same way.

It works on **Metal arm stations with a Star arm only**. The station has no screen: you run it over SSH and watch the dashboard in your laptop's browser. Everything in [mirror_collect.md](mirror_collect.md) about folders, the dashboard's buttons and ending a session holds here too; this page covers what is different.

Setting the Star arm up (its USB adapter and its calibration) is the organizers' job: [hardware_setup.md](hardware_setup.md), "Star Arm 102 leader". If the Star arm is not plugged in or the commands below fail, call the staff.

## Read this before you start

- **There is no dry run.** The right arm is energized as soon as the script opens it. Hand on the emergency stop, everyone out of the right arm's reach, and nothing on the table that should not be hit.
- **Only the right arm moves.** The left Metal arm is opened with its motors off and does nothing: **lay it down or support it**, or it falls under its own weight. It is recorded as it rests (still), so a model trained on the episodes keeps that arm still too.
- **Support the right arm** when you start: MakerMods arms have no brakes.
- **The right arm waits for you.** It holds where it is until you bring the Star arm to the same pose. See [Getting the right arm to follow](#getting-the-right-arm-to-follow).
- **One program at a time.** Stop `dhk-collect`, `dreamcli`, makermodslab and any earlier `mirror_collect.py` first: each holds the cameras, the arms or the Star arm's port, and a second one fails (`cannot open /dev/v4l/...` is the usual sign).

## Find the Star arm's port and calibration name

You need two things for the command. Ask the staff, or look (both commands only read):

```bash
# 🤖 on the station
ls -l /dev/serial/by-id/
```

The Star arm is the `usb-1a86_USB_Serial-if00-port0` line (a CH340 serial adapter). The `Protofusion_Labs_CANable` lines are the two Metal arms: do not use them.

```bash
# 🤖 on the station
ls ~/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/
```

Each `<name>.json` there is a calibration; `<name>` is what you pass as `--star-id`. If the folder is empty or does not exist, the Star arm was not calibrated: call the staff.

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

3. **Start the script.** Hold the Star arm near the right arm's pose. The task names the folder your episodes go in (see [Where the episodes go](mirror_collect.md#where-the-episodes-go)). Quote `--star-id` if the name has spaces:

   ```bash
   # 🤖 on the station: REAL RUN, the right arm is energized and will follow the Star arm
   cd ~/physical-ai-hacks
   uv run python scripts/mirror_collect.py \
     --config configs/station.yaml \
     --task "pick up the red cube" \
     --leader-device star \
     --star-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
     --star-id "<name>" \
     --leader left --follower right
   ```

   `--leader left` names the arm that **rests** (the left one), and `--follower right` the arm the Star arm drives. Leave `--gravity` out: it is for a Metal arm used as the leader, and the script refuses it together with `--leader-device star`.

   The console prints `Star leader: connected`, then opens the left arm (`torque off — resting, recorded still`) and the right arm (`energized — follower, THE ARM WILL MOVE`), and ends with `Ready.` and the list of controls.

4. **Open http://localhost:8080 in your laptop's browser.** If the script says the port is taken, add `--port 8091` to the command and use the same number in the `ssh -L` line and in the browser.

## Getting the right arm to follow

The right arm does not jump to wherever the Star arm happens to be. It holds still until **every joint of the Star arm (the gripper excluded) is within 15° of the right arm's**. Then it starts following, from that pose on:

1. Move the Star arm until its pose matches the right arm's. The console says `Not following yet: move the Star arm to the follower's pose (off by 32°)` every two seconds (the number is the largest difference on any joint), and the dashboard shows the same line (`right arm: not following yet: ...`).
2. When the console prints `Star leader engaged (within 15°): the follower follows it.`, the right arm follows, and the dashboard says `right arm: follows the Star Arm 102 leader`.

Match the poses as closely as you can before that moment: the right arm goes straight to the Star arm's pose, with no ramp, so a joint that was 15° off steps 15° at once.

You must go through this again:

- after **Go to Home** (the right arm is back at its start pose, and the Star arm is wherever you left it), and
- if the Star arm's readings fail three times in a row (the console prints `Star leader disengaged after 3 failed reads in a row`). The right arm holds until you bring the Star arm back to its pose.

**Start Episode is refused until the right arm follows** (the console says `cannot start: not following yet`). A frame is recorded only when the right arm is following the Star arm, so a stretch where it held still is left out of the episode.

## Recording

The dashboard works as in [mirror_collect.md](mirror_collect.md#the-dashboard), with these differences:

| What | In Star mode |
| --- | --- |
| The pan angle and the gripper in the status bar | The **Star arm's** |
| The line next to the mode | `right arm: follows the Star Arm 102 leader`, or `right arm: not following yet: ...` while it waits |
| Close the Star arm's gripper | The right gripper closes |
| What an episode records | The right arm's measured pose as its state, and the Star arm's pose as its action; the left arm's own (resting) pose for both |

To record an episode:

1. Check that the dashboard says the right arm follows.
2. Press **Start Episode**, do the task with the Star arm, then press **Save Episode**. **Discard** throws the episode away.

You can also type in the tmux screen: Enter starts or saves an episode, `d` discards it, `h` goes home, and `q` quits. An episode shorter than 1.6 seconds is dropped rather than saved.

## Ending the session

In the tmux screen, type `q` and press Enter. It saves the episode in progress, returns the right arm to its start pose and releases it, and disconnects the Star arm. Ctrl+C does the same but **discards** the episode in progress. Do not use `kill -9` or close the window: that skips the shutdown, so the right arm is neither returned nor released. **Support the right arm** first: it goes limp when released.

Your saved episodes are safe either way: each is written to disk the moment you save it. On Metal arms, run `tools/add_eef.py` on a folder before fine-tuning, as [fine_tuning.md](fine_tuning.md) shows.

## When something goes wrong

| What you see | What it means |
| --- | --- |
| `--leader-device star needs --star-port and --star-id` | Give both. [Find them above.](#find-the-star-arms-port-and-calibration-name) |
| `--gravity is for a Metal leader arm; leave it out with --leader-device star` | Remove `--gravity` from the command. |
| `No Star calibration named '...' in ... (found: [...])` | `--star-id` is not one of the names in `found`. Use one of them; if `found` is `none`, the Star arm was never calibrated: call the staff. |
| `--leader-device star needs the Star driver: uv sync --extra makermods --extra realsense` | The station is missing the Star arm's driver: call the staff. |
| The script stops while opening the Star arm, or `Star leader read error: ...` repeats | The Star arm is not powered, its USB cable is loose, or `--star-port` is wrong. Check that the `usb-1a86_USB_Serial-if00-port0` line shows in `ls -l /dev/serial/by-id/`. A servo that did not answer is named in the error (`Star servo wrist_flex (id=N) did not answer`). The right arm holds still meanwhile. |
| `Star leader disengaged after 3 failed reads in a row` | The Star arm's readings dropped out. Fix the cause above, then bring the Star arm back to the right arm's pose. |
| `Not following yet: move the Star arm to the follower's pose (off by N°)` | Normal at the start and after homing. Match the poses. |
| `camera 'overhead': cannot open /dev/v4l/...` | Another program holds the cameras, often an earlier `mirror_collect.py`. Find it with `pgrep -af mirror_collect` and end it as [mirror_collect.md](mirror_collect.md#ending-the-session) describes. |
| `could not start dashboard on port 8080` | The port is taken. Use `--port 8091` and match it in `ssh -L` and the browser. |

If a device does not work and the table does not help, call the staff.
