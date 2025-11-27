"""Lightweight configuration helpers used across notebooks.

The helpers intentionally avoid external frameworks so that students can read the
entire workflow in a single screen.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ExperimentConfig:
    """Container for experiment metadata loaded from YAML."""

    name: str
    dataset: str
    model: str
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-3
    epochs: int = 1
    allow_gpu: bool = False
    extras: Dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Construct a config, routing unknown keys to the ``extras`` dict."""

        remaining = dict(data)
        init_payload: Dict[str, Any] = {}
        for name, field in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
            if name == "extras":
                continue
            if name in remaining:
                init_payload[name] = remaining.pop(name)
                continue
            if field.default is not MISSING:
                init_payload[name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore[assignment]
                init_payload[name] = field.default_factory()  # type: ignore[call-arg]
            else:
                raise KeyError(f"Missing required config key: {name}")
        return cls(extras=remaining or None, **init_payload)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML config file into :class:`ExperimentConfig`."""

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config files must contain a top-level mapping.")
    return ExperimentConfig.from_mapping(data)
