"""Finds the grasps, lifts and drops of one arm in recorded episodes, and turns them into the policy's poses.

A demonstration of unplugging, seen through one arm's joints and gripper: the gripper, open part way over a plug,
closes (it stalls on the housing, well short of closed), the tool rises with the plug, then somewhere else the
gripper opens again. Each such closing is a GraspEvent. Their grasp points gather around the strip's ports; grouped
(cluster_ports()), each group's most central grasp gives that port's grasp joints, and the group's grips the
widths. The highest point before each grasp gives a ready pose; where cables were let go, the drop spot.

An episode starts with the gripper closed at rest; that first closed stretch is not a grasp.

@see grasp_events(), cluster_ports(), build_poses()
"""

import dataclasses
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from dream_hackathon.unplug import motion
from dream_hackathon.unplug.poses import CablePose, UnplugPoses

Fk = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclasses.dataclass(frozen=True)
class GraspEvent:
    """One closing of the gripper that lifted something.

    Frames are the episode's; joints `[6]` radians; grips 0 (closed) to 1 (open); xyz the tool in the arm's base.
    """

    episode: int
    close_frame: int  # the gripper crossed below the closing threshold
    grasp_frame: int  # the tool lowest, just after closing
    top_frame: int  # the tool highest while closed
    open_frame: int  # the gripper opened again
    grasp_joints: npt.NDArray[np.float64]
    hover_joints: npt.NDArray[np.float64]  # the tool highest in the seconds before closing
    open_joints: npt.NDArray[np.float64]
    grasp_xyz: npt.NDArray[np.float64]
    hover_xyz: npt.NDArray[np.float64]
    open_xyz: npt.NDArray[np.float64]
    approach_grip: float
    held_grip: float
    lift_m: float
    lift_speed_m_s: float
    wrist_turn_deg: float  # how far the tool turned between the grasp and the top of the lift


def grasp_events(
    joints: npt.NDArray[np.float64],
    grips: npt.NDArray[np.float64],
    fk: Fk,
    fps: float,
    episode: int = 0,
    *,
    close_below: float = 0.12,
    open_above: float = 0.16,
    min_closed_s: float = 0.3,
    min_lift_m: float = 0.01,
    lookback_s: float = 3.0,
) -> list[GraspEvent]:
    """Every grasp that lifted at least `min_lift_m` in one episode of one arm.

    @param joints `[N, 6]` radians.
    @param grips `[N]` 0 to 1.
    @param close_below, open_above The gripper is closed below the first and open again above the second.
    """
    n = len(grips)
    poses = np.array([fk(q) for q in joints])
    z = poses[:, 2, 3]
    runs = []
    closed = grips[0] < open_above  # at rest, closed: not a grasp
    start = None
    for i in range(1, n):
        if closed and grips[i] > open_above:
            closed = False
            if start is not None:
                runs.append((start, i))
            start = None
        elif not closed and grips[i] < close_below:
            closed, start = True, i
    events = []
    for start, end in runs:
        if (end - start) / fps < min_closed_s:
            continue
        early = slice(start, min(end, start + int(fps)))
        grasp = start + int(np.argmin(z[early]))
        top = grasp + int(np.argmax(z[grasp:end]))
        lift = float(z[top] - z[grasp])
        if lift < min_lift_m:
            continue
        before = slice(max(0, start - int(lookback_s * fps)), start + 1)
        hover = before.start + int(np.argmax(z[before]))
        approach = grips[max(0, start - int(0.5 * fps)) : max(1, start - int(0.1 * fps))]
        holding = grips[min(start + int(0.3 * fps), top) : top + 1]
        events.append(
            GraspEvent(
                episode=episode,
                close_frame=start,
                grasp_frame=grasp,
                top_frame=top,
                open_frame=end,
                grasp_joints=joints[grasp].copy(),
                hover_joints=joints[hover].copy(),
                open_joints=joints[end].copy(),
                grasp_xyz=poses[grasp, :3, 3].copy(),
                hover_xyz=poses[hover, :3, 3].copy(),
                open_xyz=poses[end, :3, 3].copy(),
                approach_grip=float(np.median(approach)) if len(approach) else float(grips[max(0, start - 1)]),
                held_grip=float(np.median(holding)) if len(holding) else float(grips[grasp]),
                lift_m=lift,
                lift_speed_m_s=lift / max((top - grasp) / fps, 1.0 / fps),
                wrist_turn_deg=math.degrees(motion.rotation_angle(poses[grasp, :3, :3], poses[top, :3, :3])),
            )
        )
    return events


def cluster_ports(events: Sequence[GraspEvent], k: int, iterations: int = 50) -> list[list[GraspEvent]]:
    """The grasps grouped by port (k-means on the grasp points' x and y), ordered by y, smallest first.

    In an arm's base frame y points to the arm's left, so smallest y first is right to left.
    @throws ValueError Fewer grasps than ports.
    """
    if len(events) < k:
        raise ValueError(f"{len(events)} grasps found, fewer than the {k} ports")
    xy = np.array([e.grasp_xyz[:2] for e in events])
    # Start from quantiles along the points' main direction: the ports are in a row.
    centred = xy - xy.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = centred @ axis
    order = np.argsort(along)
    centres = np.array([xy[order[int((i + 0.5) * len(xy) / k)]] for i in range(k)])
    labels = np.zeros(len(xy), dtype=int)
    for _ in range(iterations):
        labels = np.argmin(np.linalg.norm(xy[:, None, :] - centres[None, :, :], axis=2), axis=1)
        moved = np.array([xy[labels == i].mean(axis=0) if np.any(labels == i) else centres[i] for i in range(k)])
        if np.allclose(moved, centres):
            break
        centres = moved
    groups = [[e for e, label in zip(events, labels, strict=True) if label == i] for i in range(k)]
    groups = [g for g in groups if g]
    return sorted(groups, key=lambda g: float(np.mean([e.grasp_xyz[1] for e in g])))


def medoid(items: Sequence[GraspEvent], point: Callable[[GraspEvent], npt.NDArray[np.float64]]) -> GraspEvent:
    """The item whose point is nearest, in sum, to all the others'."""
    points = np.array([point(item) for item in items])
    cost = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).sum(axis=1)
    return items[int(np.argmin(cost))]


def build_poses(
    events: Sequence[GraspEvent], ports: int, side: str, *, min_drop_move_m: float = 0.03
) -> tuple[UnplugPoses, list[str]]:
    """The policy's poses from the grasps found, and a report to read before using them.

    Each port's grasp is its group's medoid grasp (real joints the arm held, not an average of several); its
    widths the group's median approach and held grips. The ready pose is the medoid of the hovers before the
    grasps; the drop spot the medoid of where cables were let go at least `min_drop_move_m` from their grasp.

    @throws ValueError Too few grasps, or no drop.
    """
    groups = cluster_ports(events, ports)
    if len(groups) < ports:
        raise ValueError(f"the grasps fall in only {len(groups)} groups, not {ports}")
    cables = []
    report = [f"{len(events)} grasps in {len({e.episode for e in events})} episodes, grouped into {ports} ports:"]
    for i, group in enumerate(groups):
        best = medoid(group, lambda e: e.grasp_xyz)
        spread = max(float(np.linalg.norm(e.grasp_xyz[:2] - best.grasp_xyz[:2])) for e in group)
        held = float(np.median([e.held_grip for e in group]))
        approach = float(np.median([e.approach_grip for e in group]))
        approach = min(1.0, max(approach, held + 0.05))
        cables.append(
            CablePose(
                name=f"cable_{i + 1}",
                grasp_joints=best.grasp_joints,
                approach_grip=approach,
                held_grip=held,
                tip="usb_a",
                source=f"episode {best.episode} frame {best.grasp_frame} (medoid of {len(group)})",
            )
        )
        x, y, zz = best.grasp_xyz
        report.append(
            f"  cable_{i + 1}: {len(group):2d} grasps, spread {spread * 1000:4.1f} mm, grasp [{x:+.3f} {y:+.3f} "
            f"{zz:+.3f}] m, held {held:.3f}, approach {approach:.3f}, lift "
            f"{np.median([e.lift_m for e in group]) * 1000:.0f} mm at "
            f"{np.median([e.lift_speed_m_s for e in group]) * 100:.1f} cm/s, wrist turn "
            f"{np.median([e.wrist_turn_deg for e in group]):.0f} deg"
        )
    ready = medoid(list(events), lambda e: e.hover_xyz)
    drops = [e for e in events if np.linalg.norm(e.open_xyz[:2] - e.grasp_xyz[:2]) >= min_drop_move_m]
    if not drops:
        raise ValueError("no cable was let go away from its port: set drop_joints_deg by hand")
    drop = medoid(drops, lambda e: e.open_xyz)
    report.append(
        f"  ready: the hover before episode {ready.episode} frame {ready.close_frame}, at "
        f"{np.round(ready.hover_xyz, 3).tolist()} m"
    )
    report.append(
        f"  drop: where episode {drop.episode} let go at frame {drop.open_frame}, at "
        f"{np.round(drop.open_xyz, 3).tolist()} m ({len(drops)} drops found)"
    )
    poses = UnplugPoses(side=side, ready_joints=ready.hover_joints, drop_joints=drop.open_joints, cables=tuple(cables))
    return poses, report
