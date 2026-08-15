"""
cache.py -- how embedding cache files and reference samples are keyed.

The whole point of this module is *provenance*. An embedding is only meaningful
together with (a) the recipe that produced it and (b) the exact image it came
from. The old cache hashed only the input-relative path, which silently mixed
embeddings from different models and different source trees. These helpers
encode both dimensions.

  recipe      : what produced the embedding (model + detection settings).
  image_digest: which image, in which source tree, at which content version.
  samples_key : one fingerprint for a whole reference-person photo set.

Everything here is pure and side-effect free so the rules are easy to test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def recipe(
    model_name: str, det_size: int, det_thresh: float, max_long_edge: int
) -> str:
    """A stable id for the exact detection+recognition configuration.

    Switching models (or detection settings) yields a *different* recipe, so
    each configuration keeps its own cache namespace instead of colliding.
    """
    raw = f"{model_name}|{det_size}|{det_thresh}|{max_long_edge}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def image_digest(abs_source: Path, rel_key: str, size: int, mtime_ns: int) -> str:
    """A unique id for one image instance.

    ``rel_key`` is the source-relative POSIX path. Including the absolute source
    root prevents ``wedding/IMG_1.jpg`` and ``party/IMG_1.jpg`` sharing a key;
    including size + mtime makes an edited/replaced image self-invalidate.
    """
    raw = f"{abs_source}\0{rel_key}\0{size}\0{mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def embedding_path(cache_dir: Path, recipe: str, digest: str) -> Path:
    """The on-disk location for one image's cached embeddings.

    Two-level fanout (``<recipe>/<d[:2]>/<digest>.npz``) keeps tens of thousands
    of entries out of a single flat directory.
    """
    return cache_dir / recipe / digest[:2] / f"{digest}.npz"


def samples_key(image_paths: Iterable[Path]) -> str:
    """A stable fingerprint of a reference-person's photo set.

    Order-independent; changes when any sample is added, removed, or edited.
    Used to invalidate a cached prototype without a manual ``--reset``.
    """
    parts = sorted(
        f"{p}\0{p.stat().st_size}\0{p.stat().st_mtime_ns}" for p in image_paths
    )
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
