# Setting up a station (organizers)

A station is one computer driving two follower arms, three cameras and one Meta Quest. This guide is for the organizers who prepare the stations before the event; participants start at a ready station and never need it. It takes you from parts on a table to `dhk-collect --dry-run` showing the right numbers. Do the steps in order: each one checks a single device before the next one depends on it.

Claude Code can walk you through these steps: see "For organizers" in the README (`/station-setup`).

## 1. The computer

You need Ubuntu 26.04 LTS on an x86-64 computer (the version the stations run and CI tests on), with one USB port per camera (USB 3 for RealSense cameras) and one per arm adapter.

```bash
sudo apt install git git-lfs adb tmux curl build-essential python3-dev linux-headers-$(uname -r)
sudo usermod -aG video,dialout,plugdev $USER     # lets SSH sessions open the cameras and arm adapters (from your next login)
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/90-quest-adb.rules
sudo udevadm control --reload-rules              # the rule above lets SSH sessions use the Quest over adb
sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=2833   # and applies it to a headset already plugged in
git lfs install                                  # the Quest app is stored with git LFS
curl -LsSf https://astral.sh/uv/install.sh | sh  # uv brings its own Python 3.13 and every dependency
source $HOME/.local/bin/env                     # puts uv on this terminal's PATH (new terminals have it)
git clone https://github.com/dreamlabs-co/physical-ai-hack-2026.git ~/physical-ai-hacks   # every step uses this folder
cd ~/physical-ai-hacks
```

Already cloned? `cd ~/physical-ai-hacks && git pull` brings it up to date instead. If `uv` is "not found", run the `source` line above; if `~/.local/bin/uv` does not exist, the installer did not run (check that `curl` is installed). Never use the snap the message suggests. Log out and back in once (over SSH, reconnect) so the groups apply; a terminal opened before then, and anything started from it (`dreamcli setup` included), still lacks them, so check with `id -nG`. Without `video`, nothing started over SSH can open a camera, because Ubuntu gives the cameras only to whoever is logged in at the station's own screen. Without `dialout`, nothing can open a Metal or Maker arm's USB adapter (`/dev/ttyACM*`); `dreamcli setup` then says "This user may not open the adapter". The Quest's udev rule does the same for the headset, through the `plugdev` group. Ubuntu 26.04's own Python is 3.14, newer than the MakerMods stack supports, so the kit runs on the Python 3.13 that uv downloads: leave the system Python alone.

Install the dependencies for your arm type. A station installs one arm type, never both:

```bash
uv sync --extra yam --extra realsense          # YAM arms
uv sync --extra makermods --extra realsense    # Metal or Maker arms
```

`--extra realsense` adds the Intel RealSense library; it does no harm on a station with webcams only, so keep it. Always name both extras in one `uv sync`: a `uv sync` without one of them removes it.

Then copy the config for your arms to `configs/station.yaml`, the name the participants' commands use. You will fill it in as you go:

```bash
[ -e configs/station.yaml ] || cp configs/yam.yaml configs/station.yaml   # or metal.yaml, maker.yaml; keeps a filled-in one
```

Set its `dataset_dir` to where datasets should go by default; participants usually name their own with `--dataset-dir`.

## 2. The arms

### YAM arms

The YAM arms talk over CAN, one USB-CAN adapter per arm, at 1 Mbit/s, and the kit expects the two links to be called `can_follower_l` and `can_follower_r`. The Dream setup wizard (`dreamcli setup`, step 6) names them: it writes a udev rule, `/etc/udev/rules.d/90-dreamlabs-can.rules`, that names each link after its adapter's serial number. Without the wizard, i2rt's guide [gives each adapter a fixed name](https://github.com/i2rt-robotics/i2rt/blob/main/docs/guides/set-persistent-can-ids.md) by hand (plug in one adapter at a time while you do).

The rule applies only when an adapter is added, never to a link that already exists, so right after the wizard the links are still `can0` and `can1`. Close the wizard first (it holds the links while it runs), then apply the rule and bring both links up:

```bash
sudo ip link set can0 down
sudo ip link set can1 down
sudo udevadm trigger --action=add --subsystem-match=net
ip -br link show type can        # should now list can_follower_l and can_follower_r
sudo ip link set can_follower_l up type can bitrate 1000000
sudo ip link set can_follower_r up type can bitrate 1000000
```

If `ip -br link show type can` still lists `can0` and `can1`, reboot instead (the rule applies at boot), then run the last two lines.

Let the dashboard bring the links up, so nobody needs `sudo` after a reboot. This allows the station's account to run exactly `ip link set <link> up type can bitrate 1000000` for its two links without a password, and nothing else (`scripts/wipe_station.sh` removes it):

```bash
bash scripts/allow_can_bringup.sh       # the links can_follower_l and can_follower_r; name others as arguments
```

The dashboard then shows a "Bring ... up" button on a link that is down. The links come up down after every reboot: bring them up again with the last two lines (`dreamcli robot up` does this for its own runs), or see i2rt's README to bring them up on every boot. In your config, `arms.left.port` and `arms.right.port` are these names, and `yam.gripper` is your gripper type (`linear_4310` for the stock gripper).

When the collector opens a YAM arm, i2rt holds it where it is. It also sweeps each linear gripper with a small force to find its travel, so keep fingers clear of the jaws.

### Metal and Maker arms

Each arm has its own CAN adapter. List them with `ls -l /dev/serial/by-id/` and put each arm's path in `arms.<side>.port` (`makermods.can_interface: slcan`). With a SocketCAN adapter, use the interface name (`can0`) and `socketcan` instead.

Before the first session:

1. **Calibrate each arm's zero in makermodslab**, MakerMods' own app ([makermodslab](https://github.com/makermods-robotics/makermodslab)). The Metal arm's zero pose is upright with every joint at 0 and the gripper closed; the Maker arm's is folded against its base. Follow the pose makermodslab's calibration shows you. The zero is stored in the motors. This kit never re-zeroes an arm, and a Maker arm whose zero no longer matches refuses to move until it is calibrated again.
2. **Run `dreamcli setup` once** (step 6 below). The setup wizard measures each gripper's motor angle fully open and fully closed. Copy the two numbers into `arms.<side>.gripper_open_deg` and `gripper_closed_deg`: they are in the config `dreamcli setup` wrote (`~/.dreamlabs/station.yaml`). Your recordings and your model then map the gripper the same way. A Metal arm needs both; a Maker arm ships with dreamcli's defaults filled in.
3. **Download the arm's URDF** for the teleop IK: `uv run python scripts/fetch_urdf.py metal` (or `maker`).
4. **Measure each arm's tool center point** (next section).

Both arms are brakeless. Start every session with them resting where they can go limp, supported or folded: that is where they return, and where their torque is released at the end. Only one program can use an arm's adapter at a time, so close makermodslab and `dreamcli` before collecting.

### Tool center point (Metal and Maker arms)

The tool center point (TCP) is the point between the gripper's closed fingertips. The Quest drives that point: when you turn your wrist, the gripper turns about it. The flange is `link6` on a Metal arm and `link_007` on a Maker arm, and `arms.<side>.tcp_xyz_m` is the point in that link's frame, in metres.

**Stock MakerMods gripper: Dream's own point.** The templates carry the tool point of Dream's own model of the arm, the point `dreamcli robot run` drives it about and the one `tools/add_eef.py` records the training data's tool poses about: `[0.135, 0, 0]` in `link6` on a Metal arm, `[0, 0, -0.1696387]` in `link_007` on a Maker arm. (Dream's URDFs and the kit's agree up to the flange, so the numbers mean the same point in both.) On a `configs/station.yaml` made from an older template, set both arms' point from the installed Dream library:

```bash
~/dream-env/bin/python scripts/use_dream_tcp.py configs/station.yaml
```

**Another gripper: measure it.** Do not guess it or copy it from another arm. With the gripper closed, measure from the flange's origin to the point between the fingertips, along the flange's x, y and z axes, in metres, and write the three numbers in `tcp_xyz_m`; or let `dreamcli tcp` place the point on a 3D view of the arm's URDF with its meshes (makermodslab's `frontend/public/<arm>-urdf` folder), which writes the same three numbers.

**Check it on the arm, either way.** In the dry run (step 5), hold an arm's closed fingertips on a mark on the table and turn its wrist by hand, keeping the tips on the mark: that arm's tool position on the dashboard should stay within about a centimetre. If it swings by several, the point is off. The collector refuses to drive a Metal or Maker arm until both arms' `tcp_xyz_m` are filled in.

This point only shapes how the controls feel. The tool poses your Metal or Maker fine-tune trains on come from Dream's own model of the arm (`tools/add_eef.py`).

### Check the arms

```bash
uv run python scripts/check_arms.py --config configs/station.yaml
```

This only reads the arms. YAM arms float in gravity compensation; MakerMods arms are opened with torque off, so support them. Move each arm by hand and check that the numbers for the right side change.

### Star Arm 102 leader (optional, Metal arms)

A Metal station can also record demonstrations with a Star Arm 102 leading the right arm (`scripts/mirror_collect.py --leader-device star`; participants' guide: [star_leader.md](star_leader.md)). Once per station:

1. **Plug in the Star arm's USB-serial adapter and find its port** with `ls -l /dev/serial/by-id/`. It is a CH340 (`usb-1a86_USB_Serial-if00-port0`); the `Protofusion_Labs_CANable` entries are the arms' adapters. The `dialout` group from step 1 lets SSH sessions open it. The port goes on the command line (`--star-port`); nothing about the Star arm goes in `configs/station.yaml`.
2. **Calibrate the Star arm in makermodslab** (Metal arm, "Star Arm 102 leader"), following the steps makermodslab shows. This powers and moves the Star arm, so you run it. The name you give the calibration is what participants pass as `--star-id`. It is saved as `<name>.json` here:

   ```bash
   ls ~/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/
   ```

   Close makermodslab afterwards: only one program can use a port at a time.
3. **Check that the Star arm's driver is installed.** It comes with the `makermods` extra (`uv sync --extra makermods --extra realsense`). This prints nothing when it is there:

   ```bash
   uv run python -c "from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader"
   ```

4. **Run the first session yourself.** `mirror_collect.py` has no dry run: support the right arm, lay the left arm down, keep a hand on the emergency stop, and follow [star_leader.md](star_leader.md). Check that the right arm stays put until the Star arm is brought to its pose, that every joint and the gripper then move the right way, and that **Go to Home** needs the Star arm brought back afterwards. Then write the port and the calibration name on the station's sheet, where participants find them.

## 3. The cameras

```bash
uv run python scripts/check_cameras.py --list
```

This lists the USB cameras by their stable `/dev/v4l/by-id/...` names (model and serial) and their `/dev/v4l/by-path/...` names (the USB port), and the RealSense cameras by serial number. Identical webcams with no serial share one `by-id` name, so only one of them shows there: use their `by-path` names instead, as the setup wizard does (`~/.dreamlabs/station.yaml` has them), quote them in the config (they contain colons), and keep each camera in its USB port from then on. In your config, give each role its camera (the wizard's `scene` camera is the kit's `overhead`):

- `overhead`: a fixed camera that sees the whole workspace.
- `wrist_l` and `wrist_r`: the left and right wrist cameras.

Use `type: uvc` with a `device` for a webcam, or `type: realsense` with a `serial`. Keep the 1280x720 capture mode if the camera has it. Frames are center-cropped to 16:9 and recorded at `record_width` x `record_height` (640x360). Then look at all three:

```bash
uv run python scripts/check_cameras.py --config configs/station.yaml
```

With no screen on the station, this serves the three views to a browser: on your laptop, connect with `ssh -L 8080:localhost:8080 <user>@<station>` and open http://localhost:8080. Move each camera until its view is what you want; then do not move it again for the rest of the hackathon. The model learns from these exact views.

## 4. The Quest

1. Turn on Developer Mode for the headset in the Meta Horizon phone app (the headset's settings, then Developer Mode).
2. Plug the headset into the computer with a USB-C cable, put it on, and accept "Allow USB debugging". `adb devices` should list it as `device`. `unauthorized` means the prompt is still waiting in the headset; `no permissions` means the Quest's udev rule from step 1 is missing (add it, then `adb kill-server` and replug).
3. Run `uv run python scripts/check_quest.py`. The first run installs the teleop app on the headset. With the headset on, move the controllers: their positions, grips and triggers print. Press A, B and the right joystick to see them listed.

To work without the cable, run `adb tcpip 5555` while plugged in, unplug, and pass the headset's address: `--ip 192.168.x.y` (`--quest-ip` for `dhk-collect`).

The app reports each controller **relative to the headset**. Turning your head while you hold a grip moves the arm too, so keep your head still and facing the arms while you drive. If the headset goes to sleep when you take it off, cover its proximity sensor (between the lenses) with tape.

## 5. Check the directions with a dry run

```bash
uv run dhk-collect --config configs/station.yaml --dry-run
```

The session first waits for every device (the Quest, the cameras, both arms' links), printing what each one that is not ready needs, as its dashboard does; once all are ready, press A on the right controller to open the arms. A dry run reads the arms and the Quest and never commands a motor. Its status line prints, once a second, each arm's tool position in that arm's base frame. Check two things:

1. **The arm model.** With the grips released, the tool positions follow the real arms: move an arm by hand, and its tool position changes the way the arm moved. If a joint turns the wrong way on a Metal or Maker arm, flip its entry in `makermods.joint_signs`.
2. **The controller directions.** Hold the right grip and move your hand forward, left and up: the right tool's position should move the same way. Then do the same for the left. If an axis is wrong, change that arm's `quest_to_base`. Its rows say which Quest axis becomes the base's x, y and z; the Quest's axes are x right, y up and z toward you. The base frames: on a YAM and a Metal arm, x points forward from the arm, y to its left and z up; on a Maker arm (its URDF), -x points forward, +y up and +z to its left. The shipped configs already map the Quest this way for an operator standing behind the arms.

Then set each arm's `workspace_min_m` and `workspace_max_m`, the box the tool is kept inside, from the positions you saw at the edges of the table.

## 6. The Dream library (for fine-tuning and running models)

Fine-tuning and running models use the Dream library in its own environment, `~/dream-env`. Its install uses pip, which Ubuntu does not come with, so install that first: `sudo apt install python3-pip`. Then install the library with the two steps on the [Getting started page](https://platform.dreamlabs.ai/api-documentation/), which also save the API key the participants' jobs run under. Then run `dreamcli setup` for the arm type you have: participants' models run from the config it writes. On Metal and Maker arms it also measures the gripper angles step 2 needs.

## 7. Access for participants

Participants work on the station over SSH, with no screen. Give every team the station's one account rather than an account each: the Dream library, its API key and `dreamcli`'s config live in that account's home, and the station's lock, which lets only one program drive the arms at a time, is per account (`~/.dreamlabs/arms.lock`, the same file `dreamcli` locks). Each team names its own dataset folder with `--dataset-dir`. Port 8080 on the station's loopback address must be free for the dashboard (`--dashboard-port` picks another).

## 8. Before handing the station over

Record two short episodes into a throwaway folder (`--dataset-dir ~/hackathon_data/station-test`) and check them with `~/dream-env/bin/python tools/finetune.py ~/hackathon_data/station-test --arm <arm> --check`. On a Metal or Maker arm, run `tools/add_eef.py` first and check the `-eef` folder. A station whose test passes is ready for participants.

Last, remove the setup team's API key, so no team runs or fine-tunes on it: `bash scripts/remove_api_key.sh`. It deletes the key from `~/.dreamlabs/credentials` and its backup, keeps the platform addresses, and warns about a key still set in a shell, a startup file or a running `dreamcli`. Teams then save their own key with the Setup API page's key step before they fine-tune. `dreamcli setup` registered the station for Live View under the setup key's account; `dreamcli robot run` in a terminal does not depend on that, but check Live View with a team's key before relying on it.
