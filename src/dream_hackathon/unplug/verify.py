"""Whether a pulled plug is out: its metal tip, hidden in the socket before the pull, shows under the fingers.

The acting arm's wrist camera sees the gripper's fingers and what they hold. After a pull, the plug's tip (bright,
unsaturated metal, unlike the orange housing and the black strip) hangs in a box just below the fingertips. A tip
counts when that box holds a metal blob big enough and tip-shaped, when the orange housing is still between the
fingers (the cable came up with the hand), and, given a frame from before the pull, when the box gained that much
metal (the strip's own sockets, also metal, were there already).

Everything here is plain OpenCV on BGR frames; the thresholds are VisionParams. TipVerifier adds the camera, the
votes over several frames and the debug pictures, written off the control loop's thread.

@see detect_tip(), TipVerifier
"""

import contextlib
import dataclasses
import pathlib
import queue
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np
import numpy.typing as npt

from dream_hackathon import cameras
from dream_hackathon.unplug.params import VisionParams

_KERNEL = np.ones((3, 3), np.uint8)


@dataclasses.dataclass(frozen=True)
class TipResult:
    """One frame's verdict, and the measurements behind it.

    Attributes:
    - seen: Whether the tip shows.
    - area_px: The largest metal blob in the tip box.
    - gain_px: Metal pixels in the tip box less the reference frame's (None without one).
    - aspect: That blob's long side over its short side.
    - orange_frac: How much of the housing box is orange.
    - bbox: The blob's box, x, y, w, h pixels, or None.
    - reason: Why it is not seen, or `tip seen`.
    """

    seen: bool
    area_px: int
    gain_px: int | None
    aspect: float
    orange_frac: float
    bbox: tuple[int, int, int, int] | None
    reason: str


def roi_box(shape: tuple[int, ...], roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """A fractional box as pixel bounds x0, y0, x1, y1 in an image of `shape`."""
    h, w = shape[:2]
    x0, y0, x1, y1 = roi
    return int(x0 * w), int(y0 * h), max(int(x0 * w) + 1, int(x1 * w)), max(int(y0 * h) + 1, int(y1 * h))


def orange_mask(bgr: npt.NDArray[np.uint8], p: VisionParams) -> npt.NDArray[np.uint8]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    low = np.array([p.orange_h[0], p.orange_s_min, p.orange_v_min], np.uint8)
    high = np.array([p.orange_h[1], 255, 255], np.uint8)
    return cv2.inRange(hsv, low, high)


def metal_mask(bgr: npt.NDArray[np.uint8], p: VisionParams) -> npt.NDArray[np.uint8]:
    """Bright, unsaturated pixels that are not orange, cleaned of specks."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    metal = cv2.inRange(hsv, np.array([0, 0, p.metal_v_min], np.uint8), np.array([179, p.metal_s_max, 255], np.uint8))
    metal = cv2.bitwise_and(metal, cv2.bitwise_not(orange_mask(bgr, p)))
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, _KERNEL)
    return cv2.morphologyEx(metal, cv2.MORPH_CLOSE, _KERNEL)


def detect_tip(
    bgr: npt.NDArray[np.uint8], p: VisionParams, tip: str, reference: npt.NDArray[np.uint8] | None = None
) -> TipResult:
    """Whether one wrist frame shows the plug's tip out of its socket.

    @param tip `usb_a` or `usb_c`: picks the smallest blob that counts.
    @param reference A frame from just before the pull, or None to skip the gain test.
    """
    x0, y0, x1, y1 = roi_box(bgr.shape, p.tip_roi)
    metal = metal_mask(bgr, p)[y0:y1, x0:x1]
    hx0, hy0, hx1, hy1 = roi_box(bgr.shape, p.housing_roi)
    orange_frac = float(np.count_nonzero(orange_mask(bgr, p)[hy0:hy1, hx0:hx1])) / max(1, (hx1 - hx0) * (hy1 - hy0))
    gain = None
    if reference is not None and reference.shape == bgr.shape:
        gain = int(np.count_nonzero(metal)) - int(np.count_nonzero(metal_mask(reference, p)[y0:y1, x0:x1]))
    contours, _ = cv2.findContours(metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area, aspect, bbox = 0, 0.0, None
    if contours:
        blob = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(blob))
        (_, _), (w, h), _ = cv2.minAreaRect(blob)
        aspect = float(max(w, h) / max(1e-6, min(w, h)))
        bx, by, bw, bh = cv2.boundingRect(blob)
        bbox = (bx + x0, by + y0, bw, bh)
    min_area = p.min_tip_area_px.get(tip, min(p.min_tip_area_px.values()))
    if area < min_area:
        reason = f"no metal blob big enough below the fingers ({area} < {min_area} px)"
    elif not p.tip_aspect[0] <= aspect <= p.tip_aspect[1]:
        reason = f"the metal blob is not tip-shaped (aspect {aspect:.1f})"
    elif orange_frac < p.min_orange_frac:
        reason = f"the orange housing is not between the fingers ({orange_frac:.0%})"
    elif gain is not None and gain < p.min_tip_gain_px:
        reason = f"no more metal than before the pull (gain {gain} px)"
    else:
        reason = "tip seen"
    return TipResult(reason == "tip seen", area, gain, aspect, orange_frac, bbox, reason)


def annotate(bgr: npt.NDArray[np.uint8], result: TipResult | None, p: VisionParams) -> npt.NDArray[np.uint8]:
    """A copy of the frame with the boxes, the blob, the metal pixels tinted, and the verdict."""
    out = bgr.copy()
    out[metal_mask(bgr, p) > 0] = (255, 255, 0)
    x0, y0, x1, y1 = roi_box(bgr.shape, p.tip_roi)
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
    hx0, hy0, hx1, hy1 = roi_box(bgr.shape, p.housing_roi)
    cv2.rectangle(out, (hx0, hy0), (hx1 - 1, hy1 - 1), (0, 140, 255), 1)
    if result is not None:
        if result.bbox is not None:
            bx, by, bw, bh = result.bbox
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0) if result.seen else (0, 0, 255), 2)
        colour = (0, 200, 0) if result.seen else (0, 0, 255)
        cv2.putText(out, result.reason[:70], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


class TipVerifier:
    """The tip check on a live camera: a reference frame before each pull, then a verdict per frame after it.

    `frame_of` raising cameras.CameraError (a camera gone) gives no verdict, never an exception. Each frame judged,
    and each reference, is saved under `debug_dir` as a JPEG (annotated), by a writer thread.
    """

    def __init__(
        self,
        params: VisionParams,
        frame_of: Callable[[], npt.NDArray[np.uint8]] | None,
        debug_dir: pathlib.Path | None = None,
    ) -> None:
        self._params = params
        self._frame_of = frame_of
        self._debug_dir = debug_dir
        self._reference: npt.NDArray[np.uint8] | None = None
        self._writes: queue.Queue[tuple[pathlib.Path, npt.NDArray[np.uint8]]] = queue.Queue(maxsize=32)
        if debug_dir is not None:
            threading.Thread(target=self._write_loop, name="unplug-debug", daemon=True).start()

    @property
    def has_camera(self) -> bool:
        return self._frame_of is not None

    def take_reference(self, tag: str) -> bool:
        """Keeps the newest frame as the before-the-pull picture; False when the camera gives none."""
        frame = self._grab()
        self._reference = frame
        if frame is not None:
            self._save(f"{tag}_reference", annotate(frame, None, self._params))
        return frame is not None

    def check(self, tip: str, tag: str) -> TipResult | None:
        """The newest frame's verdict, or None when the camera gives no frame."""
        frame = self._grab()
        if frame is None:
            return None
        result = detect_tip(frame, self._params, tip, self._reference)
        self._save(tag, annotate(frame, result, self._params))
        return result

    def _grab(self) -> npt.NDArray[np.uint8] | None:
        if self._frame_of is None:
            return None
        try:
            return self._frame_of()
        except cameras.CameraError:
            return None

    def _save(self, tag: str, image: npt.NDArray[np.uint8]) -> None:
        if self._debug_dir is None:
            return
        with contextlib.suppress(queue.Full):  # a slow disk drops pictures, never ticks
            self._writes.put_nowait((self._debug_dir / f"{time.strftime('%H%M%S')}_{tag}.jpg", image))

    def _write_loop(self) -> None:
        assert self._debug_dir is not None
        while True:
            path, image = self._writes.get()
            with contextlib.suppress(OSError):
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), image)
