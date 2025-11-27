"""cvcourse
=================

Teaching helpers for the CAS Computer Vision course. This package exposes
reusable utilities so that notebooks stay focused on narrative + orchestration.
"""

from importlib import metadata

try:
    __version__ = metadata.version("computer-vision-cas-hslu-2025")
except metadata.PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
