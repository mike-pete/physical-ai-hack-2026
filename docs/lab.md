# Prompt Lab: run prompts from a browser

`dhk-lab` replaces typing `dreamcli robot run --task "..."` once per prompt. It keeps a list of prompts, runs one on the arms, shows the live cameras, and saves every run (the prompt and full-resolution video of each camera) so you can compare them later.

It works on Metal arm stations. It needs the arms to itself: close `dhk-collect`, `mirror_collect.py` and `dreamcli` first (they share a station lock, so a second one refuses to start).

## Start and stop

On the station, in the repo:

```bash
uv run dhk-lab
```

No flags. Ctrl+C stops it (or `kill` its process). Stopping ends any run, moves the arms slowly home, then switches them off.

The page is on port **9100** (Mario's data dashboard uses 9000). Stations have no screen, so from your laptop:

```bash
ssh -L 9100:localhost:9100 <user>@<station>
```

then open <http://localhost:9100>. Run it inside tmux for long sessions.

## Using the page

- **Prompts** tab: **+ New prompt** adds a card. Type the prompt, press **Save**, **Run** to test it, **Delete** to remove it. Each card has its own model (default `dreamlabs/dream-1-beta`), a time limit, and a **Dry run** box (the model runs and the trace is saved, but nothing is sent to the arms). Prompts are stored in SQLite.
- **Runs** tab: every past run, newest first, with its prompt, status and duration. Click one to play the three camera videos.
- **Live panel** (top): the three cameras, the arms' state, and the prompt that is running.
- **Weightless**: the arms go limp enough to move by hand, so you can place both in any starting pose. **Hold here** locks them where they are.
- **Home arms (slow)**: moves both arms to the standard home pose over about 10 seconds.
- **Park & power off**: homes nothing; switches the arms off where they are (hold them first).
- **Stop run**: ends the run. The arms **hold where they are** (still energized) until you press Home or Weightless.
- **Home** (top left): back to the Prompts page.

A run starts from where the arms are: place them with Weightless, press Hold here, then Run.

## Where things are saved

`~/hackathon_data/lab/` (set `DHK_LAB_DIR` to move it):

- `lab.db`: the prompts and the run list.
- `runs/<id>/`: `overhead.mp4`, `wrist_l.mp4`, `wrist_r.mp4` (the camera roles in your station config) and `run.json`.

## Try the page without arms

```bash
DHK_LAB_FAKE=1 uv run dhk-lab
```

A fake worker produces test-pattern cameras and moves nothing; the station lock is not taken. Use it on a laptop to see the page.

## Safety

Real runs move the arms: keep a hand on the emergency stop. The first time on a station, go in this order: start `dhk-lab` and check the three feeds; Weightless, move the arms by hand, Hold here; Home arms and watch the slow move; a prompt with Dry run; then the same prompt for real, and Stop.

Weightless is built on Dream's motor driver internals (it drops the joints' stiffness and keeps gravity compensation on); if the library's version changes it may refuse with a message instead of going limp. Call the staff if a button does nothing.
