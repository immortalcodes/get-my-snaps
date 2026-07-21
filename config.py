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
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

# Project root = the directory this file sits in. Everything is relative to it.
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """Canonical locations for every file/directory the tool touches."""

    root: Path = ROOT
    input: Path = ROOT / "input"
    output: Path = ROOT / "output"
    reference: Path = ROOT / "reference"

    # State + caches (hidden, git-ignored, regenerated on demand).
    checkpoint: Path = ROOT / ".gms_checkpoint.json"
    reference_cache: Path = ROOT / ".gms_reference_cache.json"
    embeddings_cache: Path = ROOT / ".gms_embeddings"

    # Human-facing reports written into output/.
    manifest: Path = ROOT / "output" / "manifest.csv"
    candidates: Path = ROOT / "output" / "candidates.csv"


@dataclass
class Config:
    """Every tunable value, with sensible recall-first defaults."""

    # --- matching ---------------------------------------------------------
    threshold: float = 0.28          # cosine cutoff; lower = higher recall
    batch_size: int = 150            # images between checkpoint flushes
    model_name: str = "buffalo_l"    # or "buffalo_s" (faster, less accurate)

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
