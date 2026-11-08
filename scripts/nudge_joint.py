"""POWERED: nudges chosen joints a few degrees and back, slowly, to check each arm is on and follows commands.

```
uv run python scripts/nudge_joint.py --config configs/station.yaml --side left --joint 6 --dry-run   # read only
uv run python scripts/nudge_joint.py --config configs/station.yaml --side left --joint 6              # moves the arm
```

The second command POWERS the arms and moves one joint of the chosen arm: run it yourself, in a terminal, with a hand
on the emergency stop and the arms resting where they can go limp (they are released when the script ends). It refuses
to run without a terminal, and it asks for Enter before it powers the arms and before every move.

Each move eases the joint out by `--degrees`, holds a second, eases it back, and reports how far the joint really went.
Every other joint of both arms is held where it was. "Positive" is the arm vendor's joint angle increasing; the script
cannot say which way that is on the arm, so start small and flip the sign with `--degrees -3` to go the other way.
The run stops and eases everything back if any joint strays more than a few degrees from where it was told to be.
"""

import argparse
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dream_hackathon import arms, config, station_lock

SIDES = ("left", "right")
## The largest nudge in degrees, and the fastest it may go (the ease's peak speed is 1.5x its mean speed).
MAX_DEGREES = 10.0
MAX_PEAK_DEG_S = 5.0
## A joint further than this from where it was told to be is blocked, limp or knocked: the run stops.
TRACKING_ABORT_DEG = 4.0
## A nudge counts as followed when the joint covered at least this fraction of it.
FOLLOWED_FRACTION = 0.5
## Seconds to hold at the far end, and to let the joint settle before measuring where it came back to.
HOLD_S = 1.0
SETTLE_S = 0.5
## Seconds to ease everything back to where it started when the run ends early.
RETURN_S = 3.0

ENERGIZE_PROMPT = """
This POWERS both arms: each one holds where it is now, and is released again when the script ends.
Check before you go on:
  - your hand is on the emergency stop;
  - both arms are resting where they can go limp;
  - nothing and no one is within reach of either arm.
Press Enter to power the arms, or Ctrl+C to quit: """


class NudgeAbortedError(RuntimeError):
    """A joint strayed from where it was told to be; the message says which."""


@dataclass
class StepResult:
    """What one nudge did.

    Attributes:
    - side, joint: The arm and the joint (0-based, base to wrist).
    - commanded_deg: The nudge asked for.
    - moved_deg: How far the joint really went at the far end.
    - back_error_deg: How far from where it started the joint was after coming back.
    - followed: Whether the joint went most of the way, in the right direction.
    """

    side: str
    joint: int
    commanded_deg: float
    moved_deg: float
    back_error_deg: float
    followed: bool


def smoothstep(x: float) -> float:
    """0 to 1 with zero speed at both ends."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def ramp(start: float, goal: float, seconds: float, hz: float) -> list[float]:
    """The positions to command, one per tick, easing from start to goal; the last one is exactly goal."""
    ticks = max(1, round(seconds * hz))
    return [goal if k == ticks else start + (goal - start) * smoothstep(k / ticks) for k in range(1, ticks + 1)]


def min_seconds(degrees: float) -> float:
    """The shortest time a nudge of this size may take, at MAX_PEAK_DEG_S."""
    return 1.5 * abs(degrees) / MAX_PEAK_DEG_S


def joint_label(station: config.StationConfig, joint: int) -> str:
    """`joint 6 (wrist_roll)`: the joint's number, base to wrist, and the vendor's name for it."""
    if station.family == "makermods":
        from dream_hackathon.arms import makermods

        return f"joint {joint + 1} ({makermods.MOTORS[joint]})"
    return f"joint {joint + 1}"


def check_tracking(measured: arms.Joints, targets: arms.Joints) -> None:
    """@throws NudgeAbortedError Some joint is further than TRACKING_ABORT_DEG from its target."""
    for side, target in targets.items():
        errors = np.degrees(np.abs(measured[side] - target))
        worst = int(np.argmax(errors))
        if errors[worst] > TRACKING_ABORT_DEG:
            raise NudgeAbortedError(
                f"{side} joint {worst + 1} is {errors[worst]:.1f} degrees from where it was told to be"
            )


def move_joint(
    robot: arms.Arms,
    targets: arms.Joints,
    grips: arms.Grips,
    side: str,
    joint: int,
    goal: float,
    seconds: float,
    hz: float,
    sleep: Callable[[float], None],
    *,
    check: bool = True,
) -> None:
    """Eases one joint's target to goal (radians), every other target unchanged, watching all joints as it goes.

    @param targets What is commanded now, per side; updated in place.
    @param check False to skip the tracking watch (when easing back after a stop).
    @throws NudgeAbortedError A joint strayed (only with check).
    """
    for value in ramp(float(targets[side][joint]), goal, seconds, hz):
        targets[side][joint] = value
        robot.command(targets, grips)
        sleep(1.0 / hz)
        if check:
            check_tracking(robot.read()[0], targets)


def hold(
    robot: arms.Arms,
    targets: arms.Joints,
    grips: arms.Grips,
    seconds: float,
    hz: float,
    sleep: Callable[[float], None],
) -> None:
    """Keeps commanding the current targets for a while, watching all joints."""
    for _ in range(max(1, round(seconds * hz))):
        robot.command(targets, grips)
        sleep(1.0 / hz)
        check_tracking(robot.read()[0], targets)


def nudge(
    robot: arms.Arms,
    targets: arms.Joints,
    grips: arms.Grips,
    side: str,
    joint: int,
    degrees: float,
    seconds: float,
    hz: float,
    sleep: Callable[[float], None],
) -> StepResult:
    """Eases one joint out by `degrees`, holds, eases it back, and measures what it did.

    @throws NudgeAbortedError A joint strayed from its target.
    """
    origin = float(targets[side][joint])
    before = float(robot.read()[0][side][joint])
    move_joint(robot, targets, grips, side, joint, origin + math.radians(degrees), seconds, hz, sleep)
    hold(robot, targets, grips, HOLD_S, hz, sleep)
    moved = math.degrees(float(robot.read()[0][side][joint]) - before)
    move_joint(robot, targets, grips, side, joint, origin, seconds, hz, sleep)
    hold(robot, targets, grips, SETTLE_S, hz, sleep)
    back = math.degrees(float(robot.read()[0][side][joint]) - before)
    followed = moved * math.copysign(1.0, degrees) >= FOLLOWED_FRACTION * abs(degrees)
    return StepResult(side, joint, degrees, moved, back, followed)


def return_to_start(
    robot: arms.Arms, targets: arms.Joints, grips: arms.Grips, hz: float, sleep: Callable[[float], None]
) -> None:
    """Eases every joint's target back to where the arms started, whatever happened; never raises."""
    try:
        for side in targets:
            for joint in range(len(targets[side])):
                if targets[side][joint] != robot.start_joints[side][joint]:
                    move_joint(
                        robot,
                        targets,
                        grips,
                        side,
                        joint,
                        float(robot.start_joints[side][joint]),
                        RETURN_S,
                        hz,
                        sleep,
                        check=False,
                    )
    except Exception as error:  # the arms are being shut down either way
        print(f"Could not ease the arms back to where they started: {error}", file=sys.stderr)


def describe(station: config.StationConfig, result: StepResult) -> str:
    """One line saying what a nudge did."""
    label = joint_label(station, result.joint)
    verdict = "FOLLOWED" if result.followed else "DID NOT FOLLOW"
    line = (
        f"{verdict}: {result.side} {label} asked {result.commanded_deg:+.1f} deg, went {result.moved_deg:+.1f} deg, "
        f"came back to within {abs(result.back_error_deg):.1f} deg of where it started."
    )
    if not result.followed:
        line += " It may be stuck, limp, switched off, or already at a limit (try the other sign with --degrees)."
    return line


def run(
    station: config.StationConfig,
    steps: list[tuple[str, int]],
    degrees: float,
    seconds: float,
    *,
    dry_run: bool,
    sleep: Callable[[float], None] = time.sleep,
    ask: Callable[[str], str] = input,
) -> int:
    """Opens the arms, does the nudges, and shuts the arms down; returns the exit code.

    @param steps The (side, joint) pairs, joints 0-based, in order.
    @param dry_run Open the arms with nothing powered and only read them.
    @param ask Waits for the person; it raises KeyboardInterrupt to stop.
    """
    hz = float(station.control_hz)
    if not dry_run:
        ask(ENERGIZE_PROMPT)
    try:
        robot = arms.open_arms(station, energize=not dry_run)
    except arms.ArmsError as error:
        raise SystemExit(str(error)) from None
    targets: arms.Joints = {}
    grips: arms.Grips = {}
    results: list[StepResult] = []
    code = 0
    try:
        joints, grips = robot.read()
        if dry_run:
            for side in SIDES:
                print(
                    f"{side}: joints {np.array2string(np.degrees(joints[side]), precision=1, suppress_small=True)} deg"
                )
            print("Dry run: nothing was powered or moved.")
            return 0
        targets = {side: robot.start_joints[side].copy() for side in SIDES}
        for side, joint in steps:
            ask(
                f"\nEnter to move the {side} arm's {joint_label(station, joint)} {degrees:+.1f} deg over "
                f"{seconds:g} s and back, or Ctrl+C to stop: "
            )
            result = nudge(robot, targets, grips, side, joint, degrees, seconds, hz, sleep)
            print(describe(station, result))
            results.append(result)
        code = 0 if all(result.followed for result in results) else 1
    except KeyboardInterrupt:
        print("\nStopped: easing the arms back to where they started.")
        code = 130
    except NudgeAbortedError as error:
        print(f"\nSTOPPED: {error}. Easing the arms back to where they started.")
        code = 1
    finally:
        try:
            if targets:
                return_to_start(robot, targets, grips, hz, sleep)
        finally:
            robot.close()
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, checked against the limits."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--side", choices=(*SIDES, "both"), required=True)
    parser.add_argument(
        "--joint",
        type=int,
        nargs="+",
        choices=range(1, 7),
        metavar="N",
        required=True,
        help="Joint number(s) 1 to 6, base to wrist, nudged one after another",
    )
    parser.add_argument(
        "--degrees", type=float, default=3.0, help=f"Nudge size, at most {MAX_DEGREES:g}; negative flips it"
    )
    parser.add_argument("--seconds", type=float, default=4.0, help="How long each way takes")
    parser.add_argument(
        "--dry-run", action="store_true", help="Open the arms with nothing powered, read them, move nothing"
    )
    args = parser.parse_args(argv)
    if args.degrees == 0 or abs(args.degrees) > MAX_DEGREES:
        parser.error(f"--degrees must be non-zero and at most {MAX_DEGREES:g} either way")
    if args.seconds < min_seconds(args.degrees):
        parser.error(f"--seconds must be at least {min_seconds(args.degrees):.1f} for {args.degrees:+g} degrees")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        station = config.load(args.config)
    except config.ConfigError as error:
        raise SystemExit(f"config: {error}") from None
    sides = SIDES if args.side == "both" else (args.side,)
    steps = [(side, joint - 1) for side in sides for joint in args.joint]
    print("Plan (positive = the vendor's joint angle increasing; which way that is on the arm is not known here):")
    for side, joint in steps:
        print(
            f"  {side:5s} {joint_label(station, joint)}: {args.degrees:+.1f} deg over {args.seconds:g} s, "
            f"hold {HOLD_S:.0f} s, back (peak {1.5 * abs(args.degrees) / args.seconds:.1f} deg/s)"
        )
    if not args.dry_run and not sys.stdin.isatty():
        raise SystemExit(
            "This script moves the arms: run it yourself, in a terminal, with a hand on the emergency stop."
        )
    try:
        lock = station_lock.acquire("scripts/nudge_joint.py")
    except station_lock.StationBusyError as error:
        raise SystemExit(str(error)) from None
    try:
        return run(station, steps, args.degrees, args.seconds, dry_run=args.dry_run)
    except KeyboardInterrupt:  # only the very first prompt reaches here: run() handles a stop once the arms are open
        print("\nCancelled before the arms were powered.")
        return 130
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
