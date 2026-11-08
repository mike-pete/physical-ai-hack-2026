"""Where the plugs are: the row of them in the right wrist camera's view, and the strip in the overhead camera's.

From the viewing pose (the right gripper closed, above and behind the strip's USB end, looking down at it), each plug
is a near-black vertical bar (its strain relief) standing on the grey strip, with its dim orange shell in the port at
the bar's foot. detect_bars() finds those bars. The ports are rigid in the strip, so the plugs form one row at a
fixed spacing: fit_row() matches the bars found against a reference row (the same view, seen when the taught grasps
were right) and returns how far the whole row has moved in the image, from as few as two bars, with a bar hidden
behind a cable or merged with its neighbour costing nothing. A row offset in pixels becomes a strip offset in metres
through the image Jacobian measured by probe moves at the start of each run (graph.py).

The overhead camera sees the strip small: find_strip_overhead() finds its orange cluster of plugs, to say whether the
strip is there and how far it moved since the reference, which is coarse but needs no calibration.

Everything here is plain OpenCV on BGR frames at the recording size.

@see detect_bars(), fit_row(), find_strip_overhead()
"""

import dataclasses
import itertools
import pathlib
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import yaml

from dream_hackathon import config


@dataclasses.dataclass(frozen=True)
class PerceptionParams:
    """Thresholds, on OpenCV's HSV (hue 0..179, saturation and value 0..255)."""

    black_v_max: int = 32  # a strain relief: darker than this
    strip_v_max: int = 125  # the strip around it: grey, darker than the table ...
    strip_s_max: int = 110  # ... and unsaturated
    bar_width_px: tuple[int, int] = (3, 16)
    bar_height_px: tuple[int, int] = (10, 60)
    bar_min_aspect: float = 1.8  # height over width
    bar_grey_sides: float = 0.4  # this much of the pixels beside a bar must be strip grey
    row_band_px: float = 12.0  # bars of one row: their feet agree within this
    match_px: float = 5.0  # a bar matches a reference slot within this
    min_bars: int = 2  # bars needed to place the row


DEFAULTS = PerceptionParams()


@dataclasses.dataclass(frozen=True)
class Bar:
    """One plug seen from the viewing pose: its centre column and its foot (the port end), pixels."""

    x: float
    foot: float
    top: float
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class RowFit:
    """The row of plugs placed against the reference row.

    Attributes:
    - offset_px: `[du, dv]`, how far the row moved in the image since the reference.
    - slots_px: `[n, 2]`, every port's foot now (the reference row moved by the offset).
    - seen: Per slot, whether a bar stands there (a plug is in it); False may also be a hidden plug.
    - bars: How many bars matched.
    """

    offset_px: npt.NDArray[np.float64]
    slots_px: npt.NDArray[np.float64]
    seen: tuple[bool, ...]
    bars: int


@dataclasses.dataclass(frozen=True)
class ViewReference:
    """The viewing pose and what it saw when the taught grasps were right (configs/unplug_view.yaml).

    Attributes:
    - joints: `[6]` radians, the acting arm at the viewing pose.
    - slots_px: `[n, 2]`, each port's plug foot in the wrist image, ordered by column.
    - overhead_px: The strip's plug cluster in the overhead image, or None.
    - source: Where it came from (a rollout frame, or a capture on the station).
    """

    joints: npt.NDArray[np.float64]
    slots_px: npt.NDArray[np.float64]
    overhead_px: npt.NDArray[np.float64] | None = None
    source: str = ""


def detect_bars(bgr: npt.NDArray[np.uint8], p: PerceptionParams = DEFAULTS) -> list[Bar]:
    """The plugs' strain reliefs in a wrist frame from the viewing pose, as one row, ordered by column."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    black = (v < p.black_v_max).astype(np.uint8)
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
    grey = (v >= p.black_v_max) & (v < p.strip_v_max) & (hsv[..., 1] < p.strip_s_max)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(black)
    bars = []
    for i in range(1, n):
        x, y, w, h, area = (int(value) for value in stats[i])
        if not (p.bar_width_px[0] <= w <= p.bar_width_px[1] and p.bar_height_px[0] <= h <= p.bar_height_px[1]):
            continue
        if h < p.bar_min_aspect * w or area < 25:
            continue
        mask = (labels[y : y + h, max(0, x - 5) : x + w + 5] == i).astype(np.uint8)
        sides = cv2.dilate(mask, np.ones((1, 9), np.uint8)) - mask
        beside = grey[y : y + h, max(0, x - 5) : x + w + 5][sides > 0]
        if beside.size == 0 or beside.mean() < p.bar_grey_sides:
            continue
        bars.append(Bar(float(centroids[i][0]), float(y + h), float(y), w, h))
    if not bars:
        return []
    feet = np.array([bar.foot for bar in bars])
    anchor = max(bars, key=lambda bar: int(np.sum(np.abs(feet - bar.foot) < p.row_band_px)))
    row = [bar for bar in bars if abs(bar.foot - anchor.foot) < p.row_band_px]
    return sorted(row, key=lambda bar: bar.x)


def pick_row(bars: list[Bar], count: int) -> list[Bar]:
    """The `count` bars that look most like one row of ports: evenly spaced, their feet level.

    For a reference row, where every port must be a real plug: a cable's far end lying near the strip, or a bar
    seen twice where a cable crosses, also passes detect_bars().
    """
    if len(bars) <= count:
        return list(bars)
    best, best_cost = list(bars[:count]), float("inf")
    for chosen in itertools.combinations(sorted(bars, key=lambda bar: bar.x), count):
        xs = np.array([bar.x for bar in chosen])
        gaps = np.diff(xs)
        if np.any(gaps < 4.0):
            continue
        feet = np.array([bar.foot for bar in chosen])
        cost = float(np.std(gaps) / max(np.mean(gaps), 1e-6)) + float(np.std(feet)) / 20.0
        if cost < best_cost:
            best, best_cost = list(chosen), cost
    return best


def fit_row(
    bars: list[Bar],
    reference_px: npt.NDArray[np.float64],
    p: PerceptionParams = DEFAULTS,
    occupied: Sequence[bool] | None = None,
):
    """Places the reference row on the bars found: the offset most bars agree with, refined on them.

    @param reference_px `[n, 2]` the reference slots' feet.
    @param occupied Per slot, whether a plug can still stand in it (None: all). Bars are matched to those slots
        only, so the plugs left after some are out still place the row: one bar, when one slot is left.
    @return RowFit, or None when too few bars agree (`min_bars`, or as many as there are occupied slots).
    """
    reference_px = np.asarray(reference_px, dtype=np.float64)
    allowed = np.ones(len(reference_px), bool) if occupied is None else np.asarray(occupied, bool)
    needed = min(p.min_bars, int(allowed.sum()))
    if needed == 0 or len(bars) < needed:
        return None
    candidates = reference_px[allowed]
    points = np.array([[bar.x, bar.foot] for bar in bars])
    # Between fits that match as many bars, the smaller move wins: a row seen in part (two or three bars, its end
    # one hidden) also fits one slot over, and the strip moving a whole port's width is the less likely.
    best: tuple[int, float, npt.NDArray[np.float64]] | None = None
    for point in points:
        for slot in candidates:
            offset = point - slot
            moved = candidates + offset
            # Each bar to its nearest moved slot, along the row (columns) first.
            dists = np.abs(points[:, None, 0] - moved[None, :, 0])
            nearest = np.argmin(dists, axis=1)
            inliers = (dists[np.arange(len(points)), nearest] < p.match_px) & (
                np.abs(points[:, 1] - moved[nearest, 1]) < p.row_band_px
            )
            # A slot takes one bar: two bars in one slot is a wrong offset.
            count = len(set(nearest[inliers].tolist()))
            size = float(np.hypot(*offset))
            if best is None or count > best[0] or (count == best[0] and size < best[1]):
                best = (count, size, offset)
    assert best is not None
    count, _, offset = best
    if count < needed:
        return None
    moved = candidates + offset
    nearest = np.argmin(np.abs(points[:, None, 0] - moved[None, :, 0]), axis=1)
    residual = points - moved[nearest]
    inliers = (np.abs(residual[:, 0]) < p.match_px) & (np.abs(residual[:, 1]) < p.row_band_px)
    offset = offset + residual[inliers].mean(axis=0)
    slots = reference_px + offset
    seen = tuple(bool(np.any(np.abs(points[:, 0] - slot[0]) < p.match_px)) for slot in slots)
    return RowFit(offset, slots, seen, int(np.sum(inliers)))


def annotate_row(
    bgr: npt.NDArray[np.uint8], bars: list[Bar], fit: RowFit | None, reference_px: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.uint8]:
    """A copy of a wrist frame with the bars (yellow), the reference slots (blue) and the fitted slots (green/red)."""
    out = bgr.copy()
    for bar in bars:
        x0 = int(bar.x - bar.width / 2)
        cv2.rectangle(out, (x0, int(bar.top)), (x0 + bar.width, int(bar.foot)), (0, 255, 255), 1)
    if reference_px is not None:
        for u, v in np.asarray(reference_px):
            cv2.circle(out, (int(u), int(v)), 3, (255, 128, 0), -1)
    if fit is not None:
        for k, ((u, v), seen) in enumerate(zip(fit.slots_px, fit.seen, strict=True)):
            cv2.circle(out, (int(u), int(v)), 7, (0, 200, 0) if seen else (0, 0, 255), 2)
            cv2.putText(out, str(k + 1), (int(u) - 4, int(v) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        du, dv = fit.offset_px
        text = f"row moved {du:+.1f} {dv:+.1f} px, {fit.bars} bars"
    else:
        text = f"{len(bars)} bars" + ("" if reference_px is not None else ", row not placed")
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def find_strip_overhead(bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64] | None:
    """The overhead camera's view of the plugs: the centre of the largest orange cluster lying on a dark blob.

    @return `[u, v]` pixels, or None when no such cluster shows.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (0, 70, 70), (24, 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 70))
    near_dark = cv2.dilate(dark, np.ones((9, 9), np.uint8))
    cluster = cv2.dilate(cv2.bitwise_and(orange, near_dark), np.ones((7, 7), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(cluster)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 30:
        return None
    return np.asarray(centroids[best], dtype=np.float64)


def load_view(path: str | pathlib.Path) -> ViewReference:
    """@throws ConfigError A missing file or value."""
    file = pathlib.Path(path).expanduser()
    if not file.is_file():
        raise config.ConfigError(
            f"{file} does not exist: write it with scripts/mine_unplug_rollouts.py --view-episode, or with "
            "dhk-collect --policy graph --capture-view"
        )
    raw = yaml.safe_load(file.read_text()) or {}
    try:
        joints = np.radians(np.asarray(raw["joints_deg"], dtype=np.float64))
        slots = np.asarray(raw["slots_px"], dtype=np.float64)
        overhead = raw.get("overhead_px")
    except (KeyError, TypeError, ValueError) as error:
        raise config.ConfigError(f"{file}: needs joints_deg (six numbers) and slots_px ([u, v] per port)") from error
    if joints.shape != (6,) or slots.ndim != 2 or slots.shape[1] != 2 or len(slots) < 2:
        raise config.ConfigError(f"{file}: needs joints_deg (six numbers) and slots_px ([u, v] per port)")
    return ViewReference(
        joints,
        slots[np.argsort(slots[:, 0])],
        None if overhead is None else np.asarray(overhead, float),
        str(raw.get("source", "")),
    )


def save_view(path: str | pathlib.Path, view: ViewReference, comment: str = "") -> None:
    body: dict[str, Any] = {
        "joints_deg": [round(float(v), 3) for v in np.degrees(view.joints)],
        "slots_px": [[round(float(u), 1), round(float(v), 1)] for u, v in view.slots_px],
        "overhead_px": None if view.overhead_px is None else [round(float(v), 1) for v in view.overhead_px],
        "source": view.source,
    }
    header = "".join(f"# {line}\n" for line in comment.splitlines())
    pathlib.Path(path).expanduser().write_text(
        header + yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=119)
    )
