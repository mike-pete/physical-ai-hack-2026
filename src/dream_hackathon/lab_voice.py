"""The Prompt Lab's voice link: reads the LiveKit settings and signs the browser's access token.

The keys live in `.env.local` at the repo root (git-ignored) or in the environment, never in the page or in git. The
token is a standard HS256 JWT, so no LiveKit package is needed here.

@see integrations/livekit/voice_agent.py
"""

import base64
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import time

ENV_FILE = pathlib.Path(__file__).resolve().parents[2] / ".env.local"
AGENT_NAME = "lab-voice"
TOKEN_TTL_S = 600
KEYS = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


def load_env(path: pathlib.Path = ENV_FILE) -> None:
    """Puts the KEY=value lines of an env file into the environment, without replacing what is already set."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _b64(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def make_token(key: str, secret: str, room: str, identity: str, now: float | None = None) -> str:
    """A JWT that lets one browser join one room and asks LiveKit to send the voice agent to it."""
    issued = int(now if now is not None else time.time())
    claims = {
        "iss": key,
        "sub": identity,
        "nbf": issued - 10,
        "exp": issued + TOKEN_TTL_S,
        "video": {"room": room, "roomJoin": True, "canPublish": True, "canSubscribe": True, "canPublishData": True},
        "roomConfig": {"agents": [{"agentName": AGENT_NAME}]},
    }
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps(claims).encode())
    signature = _b64(hmac.new(secret.encode(), head + b"." + body, hashlib.sha256).digest())
    return b".".join((head, body, signature)).decode()


def voice_session() -> dict[str, str]:
    """What the page needs to join: the LiveKit address and a fresh token for a new room.

    @throws ValueError A setting is missing; the message names which.
    """
    missing = [name for name in KEYS if not os.environ.get(name)]
    if missing:
        raise ValueError(f"voice is not set up: put {', '.join(missing)} in {ENV_FILE.name} and restart dhk-lab")
    room = f"lab-{secrets.token_hex(4)}"
    token = make_token(
        os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"], room, f"web-{secrets.token_hex(4)}"
    )
    return {"url": os.environ["LIVEKIT_URL"], "token": token, "room": room}
