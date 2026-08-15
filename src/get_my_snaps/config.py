"""
config.py — all tunable settings and filesystem paths in one place.

Two things live here so the rest of the codebase never hardcodes magic numbers
or path fragments:

- ``Paths``  : where inputs, outputs, caches and reports live (derived from the
               project root, so the tool works regardless of the current dir).
- ``Config`` : every tunable knob (match threshold, batch size, model, detection
               params, reference-quality thresholds, reporting limits).

``Config`` can be overlaid from a JSON file (``--config path.json``) and then
further overridden by explicit CLI flags. Precedence, lowest to highest:

    dataclass defaults  <  JSON config file  <  command-line flags

Keeping these here means "make it configurable" is a one-line field addition,
not a hunt-and-replace across modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional


def default_central_dir() -> Path:
    """Where shared state lives: InsightFace models + reference prototypes.

    Central (not per-run) because enrollment is person-scoped, not run-scoped.
    Resolution, first match wins:
      $GMS_CACHE_DIR        -> explicit override
      $SNAP_USER_COMMON     -> snap runtime (survives snap refreshs)
      $XDG_CACHE_HOME       -> ~/.cache/get-my-snaps
      ~/.cache/get-my-snaps -> plain fallback
    """
    env = os.environ.get("GMS_CACHE_DIR")
    if env:
        return Path(env)
    snap = os.environ.get("SNAP_USER_COMMON")
    if snap:
        return Path(snap)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "get-my-snaps"
    return Path.home() / ".cache" / "get-my-snaps"


def default_model_root() -> Path:
    """Directory InsightFace reads/writes `<root>/models/<name>/from.

    Explicit override first, then reuse an already-populated ~/.insightface so
    an existing download isn't repeated, then fall back to the central dir.
    """
    env = os.environ.get("GMS_MODEL_DIR")
    if env:
        return Path(env)
    legacy = Path.home() / ".insightface"
    if (legacy / "models").is_dir():
        return legacy
    return default_central_dir()


@dataclass(frozen=True)
class Paths:
    """Canonical locations for every file/directory the tool touches.

    Built explicitly per run via :meth:`for_run` -- the source pool, the output
    dir, and the shared central dir -- rather than anchored on cwd at import
    time. This is what lets ``get-my-snaps ~/Pictures --of ~/me.jpg`` work from
    any directory.
    """

    source: Path        # the photo pool to scan (a directory)
    out: Path           # where matches + run-specific state are written
    central: Path       # shared: models/ + references/ (see default_central_dir)
    model_root: Path    # InsightFace root dir (contains models/<name>/)

    @property
    def state_dir(self) -> Path:
        """All run-specific state lives under out/.gms -- disposable."""
        return self.out / ".gms"

    @property
    def checkpoint(self) -> Path:
        return self.state_dir / "checkpoint.json"

    @property
    def embeddings_cache(self) -> Path:
        return self.state_dir / "embeddings"

    @property
    def references(self) -> Path:
        """Per-person prototype cache, keyed by recipe, shared across runs."""
        return self.central / "references"

    @property
    def manifest(self) -> Path:
        return self.out / "manifest.csv"

    @property
    def candidates(self) -> Path:
        return self.out / "candidates.csv"

    @classmethod
    def for_run(cls, source, out, central=None, model_root=None) -> "Paths":
        central = central or default_central_dir()
        model_root = model_root or default_model_root()
        return cls(source=source, out=out, central=central, model_root=model_root)


@dataclass
class Config:
    """Every tunable value, with sensible recall-first defaults."""

    # --- matching ---------------------------------------------------------
    threshold: float = 0.28          # cosine cutoff; lower = higher recall
    batch_size: int = 150            # images between checkpoint flushes
    model_name: str = "buffalo_s"    # or "buffalo_l" (more accurate, ~2x size)

    # --- detection / preprocessing ---------------------------------------
    det_size: int = 640              # detector input square size
    det_thresh: float = 0.35         # detector confidence floor (recall-first)
    max_long_edge: int = 1600        # downscale cap before detection

    # --- reference quality thresholds (used by reference_check._rate) -----
    min_usable_images: int = 3       # fewer than this => "few images" flag
    min_det_score: float = 0.55      # below => "low confidence" flag
    min_face_frac: float = 0.03      # below => "small face" flag (3% of frame)
    min_consistency: float = 0.45    # below => "inconsistent references" flag

    # --- reporting --------------------------------------------------------
    candidates_top_n: int = 15       # rows printed from the candidates ranking

    @classmethod
    def load(cls, path: Optional[Path]) -> "Config":
        """Build a Config from defaults, overlaid with a JSON file if given.

        Unknown keys in the file are ignored (with the assumption the user may
        keep notes there); known keys override the matching defaults.
        """
        cfg = cls()
        if path is None:
            return cfg
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cls)}
        overrides = {k: v for k, v in data.items() if k in valid}
        return replace(cfg, **overrides)

    def with_overrides(self, **kwargs: Any) -> "Config":
        """Return a copy with the given fields overridden (None values ignored).

        Used to layer explicit CLI flags on top of the file/default config.
        """
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)
