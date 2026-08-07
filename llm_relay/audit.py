"""Append-only JSON-lines audit log for auth and admin events.

Best-effort by design: auditing must never take down the request path, so
every failure (unwritable path, full disk, bad field) is swallowed. One line
per event: ``{"ts": <epoch>, "event": "<name>", ...fields}``.

Path resolution: ``LLM_RELAY_AUDIT_LOG`` env var when set, else
``<LLM_RELAY_CONFIG_DIR>/audit.log`` (the config dir already holds
deployment-local state like the key store, and is off-repo).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def _audit_path() -> Path:
    env = os.environ.get("LLM_RELAY_AUDIT_LOG")
    if env:
        return Path(env)
    cfg = os.environ.get("LLM_RELAY_CONFIG_DIR", "config")
    return Path(cfg) / "audit.log"


def audit(event: str, **fields) -> None:
    try:
        line = json.dumps({"ts": time.time(), "event": event, **fields}, sort_keys=True)
        with _lock:
            p = _audit_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(line + "\n")
    except Exception:
        pass
