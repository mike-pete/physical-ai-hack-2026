"""Fakes for the supervisor's tests: a clock that moves when something sleeps, scripted models, a fake camera."""

import datetime

import numpy as np

REWRITES = [
    "Pull the rightmost orange cable straight out of the power strip.",
    "Grab the nearest orange cable and pull it out.",
    "Pick up the orange cable that is still plugged in.",
    "Unplug the orange cable on the left, then put it on the table.",
    "Pull the orange cable out and lay it flat on the table.",
    "Use the other arm to pull the orange cable out.",
]
SALMON = (114, 128, 250)  # BGR; inside the task file's Lab range for the orange cables, the grey table is not


class FakeClock:
    """Time that moves only when something sleeps; hooks run on each move."""

    def __init__(self) -> None:
        self.t = 0.0
        self.base = datetime.datetime(2026, 11, 8, 12, 0, tzinfo=datetime.UTC)
        self.hooks: list = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)
        for hook in list(self.hooks):
            hook(self.t)

    def wall(self) -> datetime.datetime:
        return self.base + datetime.timedelta(seconds=self.t)


class ScriptedGate:
    """The vote gate with a scripted answer: a string, or a function of nothing that returns one."""

    def __init__(self, tc, answer="no") -> None:
        self.tc = tc
        self.answer = answer
        self.asked = 0

    def wanted_views(self, spec):
        return ("strip",)

    def ask_votes(self, spec, images):
        self.asked += 1
        answer = self.answer() if callable(self.answer) else self.answer
        return [self.tc.Vote(answer, f"scripted {answer}")] * spec.vote_gate.repeats


class ScriptedRewriter:
    name = "gemini"

    def __init__(self, texts) -> None:
        self.texts = list(texts)
        self.calls: list = []

    def rewrite(self, spec, images, history, hint):
        self.calls.append((sorted(images), list(history), hint))
        return self.texts.pop(0) if self.texts else "x" * 300  # too long: the guards reject it


class ScriptedCompare:
    def __init__(self, tc, answer="no") -> None:
        self.tc = tc
        self.answer = answer
        self.calls: list = []

    def compare(self, spec, before, after, seconds):
        self.calls.append((before, after, seconds))
        return self.tc.Vote(self.answer, f"scripted {self.answer}")


class FakeSource:
    """Frames of one flat grey, optionally with a salmon-coloured patch where the cables are."""

    def __init__(self, sio, orange: bool = False) -> None:
        self.sio = sio
        self.stale = 0.0
        self.bound: list = []
        self.make = self.orange_frame if orange else self.grey_frame

    @staticmethod
    def grey_frame() -> np.ndarray:
        return np.full((360, 640, 3), 90, np.uint8)

    @classmethod
    def orange_frame(cls) -> np.ndarray:
        frame = cls.grey_frame()
        frame[200:230, 300:330] = SALMON  # inside the strip view
        return frame

    def bind(self, since) -> None:
        self.bound.append(since)

    def latest(self, count):
        return [self.sio.Frame(float(i), self.make()) for i in range(count)]

    def stale_s(self):
        return self.stale
