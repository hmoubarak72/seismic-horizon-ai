"""Structured (JSON-lines) logging so runs can be audited and parsed later."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a package logger. Handlers are configured by :func:`configure_logging`."""
    return logging.getLogger(f"seismic_horizon_ai.{name}")


def configure_logging(level: int = logging.INFO, json_lines: bool = True, path: str | None = None) -> None:
    root = logging.getLogger("seismic_horizon_ai")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler: logging.Handler = logging.FileHandler(path, encoding="utf-8") if path else logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_lines else logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
