"""In-memory log buffer + a logging handler, for the /logs and /logs/stream
endpoints (plan 7). The relay manages no upstream processes, so this captures the
relay's OWN log records (routing decisions, errors) for the cockpit to tail.

Streaming is poll-based over a monotonic sequence (the SSE generator re-reads
``since`` on an interval) rather than a push queue, to avoid cross-thread
asyncio-queue hazards from log records emitted off the event loop.
"""
from __future__ import annotations

import collections
import logging


class LogBuffer:
    def __init__(self, maxlen: int = 2000) -> None:
        self._lines: collections.deque[tuple[int, str]] = collections.deque(maxlen=maxlen)
        self._seq = 0

    def append(self, line: str) -> None:
        self._seq += 1
        self._lines.append((self._seq, line))

    def recent(self, limit: int = 200) -> list[str]:
        return [line for _, line in list(self._lines)[-limit:]]

    def since(self, after_seq: int) -> list[tuple[int, str]]:
        return [(s, line) for s, line in list(self._lines) if s > after_seq]

    @property
    def head_seq(self) -> int:
        return self._seq


class BufferHandler(logging.Handler):
    def __init__(self, buf: LogBuffer) -> None:
        super().__init__()
        self.buf = buf

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buf.append(self.format(record))
        except Exception:
            pass


def install_log_buffer(maxlen: int = 2000) -> LogBuffer:
    """Attach a fresh ``LogBuffer`` to the ``llm_relay`` logger and return it.
    Idempotent: removes any prior ``BufferHandler`` first, so repeated calls
    (e.g. one per ``create_app`` in tests) never stack handlers."""
    logger = logging.getLogger("llm_relay")
    for h in list(logger.handlers):
        if isinstance(h, BufferHandler):
            logger.removeHandler(h)
    buf = LogBuffer(maxlen)
    handler = BufferHandler(buf)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return buf
