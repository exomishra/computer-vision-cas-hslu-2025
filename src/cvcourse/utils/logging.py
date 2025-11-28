"""Minimal logging helpers to keep notebook output clean."""

from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    """Print a timestamped log line."""

    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}")
