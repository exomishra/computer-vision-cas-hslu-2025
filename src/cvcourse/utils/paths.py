"""Central location for repository-aware paths."""

from __future__ import annotations

from pathlib import Path


project_root = Path(__file__).resolve().parents[3]


def _ensure_dir(path: Path) -> Path:
    """Create ``path`` if missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Return the shared data directory, creating it when needed."""

    return _ensure_dir(project_root / "data")


def get_outputs_dir(subdir: str | None = None) -> Path:
    """Return a directory for experiment outputs (optionally nested)."""

    base = _ensure_dir(project_root / "outputs")
    if subdir:
        return _ensure_dir(base / subdir)
    return base
