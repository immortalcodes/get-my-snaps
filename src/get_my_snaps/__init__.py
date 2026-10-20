"""get_my_snaps — local, offline face-recognition photo filter.

Copies every ``input/`` photo containing a chosen reference person into
``output/``, mirroring the input tree. Fully offline (InsightFace, CPU).

Public entry point: :func:`get_my_snaps.cli.main`.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
