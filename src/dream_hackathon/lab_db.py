"""The Prompt Lab's SQLite store: the prompts to try and the runs that were made from them.

A run keeps its own copy of the prompt text and the checkpoint, so deleting a prompt never changes what a past run
says it ran. Every function opens its own short connection, so the HTTP threads and the worker's event thread share
nothing.

@see Store
"""

import contextlib
import datetime
import pathlib
import sqlite3
from collections.abc import Iterator
from typing import Any

DEFAULT_CHECKPOINT = "dreamlabs/dream-1-beta"
DEFAULT_MAX_SECONDS = 180
## The longest a prompt may be: Dream's prompts are a few sentences, so this is a guard, not a limit anyone meets.
PROMPT_MAX_CHARS = 4000
CHECKPOINT_MAX_CHARS = 200
MAX_SECONDS_LIMIT = 3600
RUN_STATUSES = ("running", "finished", "stopped", "error")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id   INTEGER,
    prompt_text TEXT NOT NULL,
    checkpoint  TEXT NOT NULL,
    dry_run     INTEGER NOT NULL,
    max_seconds INTEGER NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    status      TEXT NOT NULL,
    error       TEXT NOT NULL DEFAULT '',
    dir         TEXT NOT NULL
);
"""


class LabError(ValueError):
    """A request the store refuses; the message says why, in words for the person at the page."""


def now() -> str:
    """The current local time as an ISO string, to the second."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def clean_prompt(text: object) -> str:
    """A prompt's text, trimmed.

    @throws LabError It is not text, is empty, or is longer than PROMPT_MAX_CHARS.
    """
    if not isinstance(text, str) or not text.strip():
        raise LabError("write the prompt first")
    text = text.strip()
    if len(text) > PROMPT_MAX_CHARS:
        raise LabError(f"a prompt can be at most {PROMPT_MAX_CHARS} characters")
    return text


def clean_checkpoint(value: object) -> str:
    """A checkpoint id, or the default for an empty one.

    @throws LabError It is not text, is over CHECKPOINT_MAX_CHARS long, or holds anything but a model id's characters.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_CHECKPOINT
    if not isinstance(value, str) or len(value.strip()) > CHECKPOINT_MAX_CHARS:
        raise LabError("the checkpoint is not a model id")
    value = value.strip()
    if not all(c.isalnum() or c in "-_.:/" for c in value):
        raise LabError("a checkpoint uses letters, digits and - _ . : / only")
    return value


def clean_max_seconds(value: object) -> int:
    """How long a run may last, whole seconds.

    @throws LabError It is not a whole number from 5 to MAX_SECONDS_LIMIT.
    """
    if value is None:
        return DEFAULT_MAX_SECONDS
    if isinstance(value, bool) or not isinstance(value, int) or not 5 <= value <= MAX_SECONDS_LIMIT:
        raise LabError(f"a run lasts 5 to {MAX_SECONDS_LIMIT} seconds")
    return value


class Store:
    """The database file under the lab's folder, created on first use.

    @param root The lab's folder; `lab.db` lives in it and `runs/<id>/` holds each run's files.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.runs_dir = root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = root / "lab.db"
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    # ------------------------------------------------------------------ prompts

    def add_prompt(self, text: object, checkpoint: object = None) -> dict[str, Any]:
        """Saves a prompt and returns it.

        @throws LabError The text or the checkpoint is not acceptable.
        """
        text, checkpoint = clean_prompt(text), clean_checkpoint(checkpoint)
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO prompts (text, checkpoint, created_at) VALUES (?, ?, ?)", (text, checkpoint, now())
            )
            return self._prompt(db, cursor.lastrowid)

    def update_prompt(self, prompt_id: int, text: object, checkpoint: object = None) -> dict[str, Any]:
        """Replaces a prompt's text and checkpoint.

        @throws LabError There is no such prompt, or the text or the checkpoint is not acceptable.
        """
        text, checkpoint = clean_prompt(text), clean_checkpoint(checkpoint)
        with self._db() as db:
            self._prompt(db, prompt_id)
            db.execute("UPDATE prompts SET text = ?, checkpoint = ? WHERE id = ?", (text, checkpoint, prompt_id))
            return self._prompt(db, prompt_id)

    def delete_prompt(self, prompt_id: int) -> None:
        """Deletes a prompt; the runs made from it keep their own copy of its text.

        @throws LabError There is no such prompt.
        """
        with self._db() as db:
            self._prompt(db, prompt_id)
            db.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
            db.execute("UPDATE runs SET prompt_id = NULL WHERE prompt_id = ?", (prompt_id,))

    def prompts(self) -> list[dict[str, Any]]:
        """Every prompt, oldest first."""
        with self._db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM prompts ORDER BY id")]

    def prompt(self, prompt_id: int) -> dict[str, Any]:
        """One prompt.

        @throws LabError There is no such prompt.
        """
        with self._db() as db:
            return self._prompt(db, prompt_id)

    @staticmethod
    def _prompt(db: sqlite3.Connection, prompt_id: int | None) -> dict[str, Any]:
        row = db.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
        if row is None:
            raise LabError("there is no such prompt")
        return dict(row)

    # ------------------------------------------------------------------ runs

    def start_run(
        self, text: str, checkpoint: str, *, prompt_id: int | None, dry_run: bool, max_seconds: int
    ) -> dict[str, Any]:
        """Records a run that is starting and makes its folder.

        @return The run's row; `dir` is the folder its videos go in.
        """
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO runs (prompt_id, prompt_text, checkpoint, dry_run, max_seconds, started_at, status, dir)"
                " VALUES (?, ?, ?, ?, ?, ?, 'running', '')",
                (prompt_id, text, checkpoint, int(dry_run), max_seconds, now()),
            )
            run_id = cursor.lastrowid
            folder = self.runs_dir / f"{run_id:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            db.execute("UPDATE runs SET dir = ? WHERE id = ?", (str(folder), run_id))
            return self._run(db, run_id)

    def end_run(self, run_id: int, status: str, error: str = "") -> dict[str, Any]:
        """Closes a run with how it ended.

        @throws LabError The status is not one of RUN_STATUSES, or there is no such run.
        """
        if status not in RUN_STATUSES:
            raise LabError(f"unknown run status {status!r}")
        with self._db() as db:
            self._run(db, run_id)
            db.execute(
                "UPDATE runs SET ended_at = ?, status = ?, error = ? WHERE id = ?", (now(), status, error, run_id)
            )
            return self._run(db, run_id)

    def close_unfinished(self) -> int:
        """Marks runs left `running` by a server that died as errors; returns how many."""
        with self._db() as db:
            cursor = db.execute(
                "UPDATE runs SET status = 'error', ended_at = ?, error = 'the server stopped during this run'"
                " WHERE status = 'running'",
                (now(),),
            )
            return cursor.rowcount

    def runs(self, limit: int = 200) -> list[dict[str, Any]]:
        """Past runs, newest first."""
        with self._db() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in rows]

    def run(self, run_id: int) -> dict[str, Any]:
        """One run.

        @throws LabError There is no such run.
        """
        with self._db() as db:
            return self._run(db, run_id)

    @staticmethod
    def _run(db: sqlite3.Connection, run_id: int | None) -> dict[str, Any]:
        row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise LabError("there is no such run")
        return dict(row)
