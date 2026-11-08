"""A small client for the Prompt Lab's HTTP API, for programs that run beside `dhk-lab` (a voice agent's skills).

Everything goes through the lab's own checks, so a refused command comes back as words, never as a half-done move.

@see dream_hackathon.lab
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:9100"
URL_ENV = "DHK_LAB_URL"
ACTION_HEADER = "X-Lab-Dashboard"
## How long a stopped run may take to settle: the worker stops the loop, closes the model and holds the arms.
SETTLE_WAIT_S = 30.0


class LabClient:
    """Talks to a `dhk-lab` on this computer.

    @param url The lab's address; the environment's DHK_LAB_URL, or the default. Only a loopback address is taken,
        because the lab refuses any other.
    """

    def __init__(self, url: str | None = None) -> None:
        self.url = (url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        host = urllib.parse.urlsplit(self.url).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"the lab is reached on this computer only, not at {self.url}")

    def _call(self, method: str, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url + path,
            data=b"{}" if method == "POST" else None,
            method=method,
            headers={ACTION_HEADER: "1", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:
            try:
                return {"ok": False, "error": json.loads(error.read()).get("error", str(error))}
            except ValueError:
                return {"ok": False, "error": str(error)}
        except (urllib.error.URLError, OSError, ValueError) as error:
            return {"ok": False, "error": f"cannot reach the lab at {self.url}: {error}"}

    def status(self) -> dict[str, Any]:
        """The lab's status; `{"ok": False, "error": ...}` when it cannot be reached."""
        return self._call("GET", "/api/status")

    def stop_and_go_home(self, wait_s: float = SETTLE_WAIT_S) -> str:
        """Stops a run, waits for the arms to hold, then sends them home slowly.

        The lab's own rules decide: a home move is refused while a run is under way or the arms are moving. The home
        move is the lab's slow one, and the arms switch nothing off.

        @return What was done, or why it was not, in one sentence.
        """
        status = self.status()
        if status.get("ok") is False:
            return f"Nothing done: {status['error']}"
        done = []
        if status.get("state") == "running":
            reply = self._call("POST", "/api/stop")
            if not reply.get("ok"):
                return f"Nothing done: the lab refused to stop the run ({reply.get('error')})."
            done.append("stopped the run")
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                status = self.status()
                if status.get("state") != "running":
                    break
                time.sleep(0.25)
            else:
                return "Stopped the run, but the arms did not settle in time: not sending them home."
        reply = self._call("POST", "/api/arms/home")
        if not reply.get("ok"):
            said = f"; the lab refused to go home ({reply.get('error')})"
            return ("Stopped the run" + said if done else f"Nothing done: {reply.get('error')}") + "."
        done.append("sent the arms home slowly")
        return "Done: " + " and ".join(done) + "."

    def stop_run(self) -> str:
        """Stops the run under way; the arms hold where they are.

        @return What was done, or why it was not, in one sentence.
        """
        status = self.status()
        if status.get("ok") is False:
            return f"Nothing done: {status['error']}"
        if status.get("state") != "running":
            return f"No run to stop: the arms are {status.get('state')}."
        reply = self._call("POST", "/api/stop")
        return (
            "Stopping the run: the arms hold where they are."
            if reply.get("ok")
            else f"Not stopped: {reply.get('error')}."
        )
