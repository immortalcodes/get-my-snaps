"""Shared pytest fixtures and import-path setup.

The package uses a ``src/`` layout, so it must be importable. When the project
is installed (``pip install -e .``) this is automatic. As a fallback for running
``pytest`` without an install, we add ``src/`` to ``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def tmp_cwd(tmp_path, monkeypatch):
    """Run a test inside a throwaway working directory.

    ``config.Paths`` anchors on the current working directory, so tests that
    touch paths get an isolated sandbox instead of the real repo root.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
