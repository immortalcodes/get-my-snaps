"""
matching.py — the core matching pipeline and its file I/O.

Responsibilities:
  - Checkpoint      : crash-safe, resumable record of every processed image.
  - Embedding cache : per-image face embeddings on disk so re-scoring is instant.
  - Scoring         : compare an image's faces to each person prototype.
  - Copying         : idempotent copy of matches into the output tree.
  - Reporting       : manifest.csv (matches) and candidates.csv (full ranking).
  - process_images  : the batched loop that ties the above together.

Everything here works in input-relative POSIX paths as keys so arbitrary
filenames stay safe and consistent across dedup, resume, and reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import console
from config import Config, Paths
from face_engine import FaceEngine, load_image_bgr


# ----------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------
@dataclass
class Checkpoint:
    """Resumable run state.

    ``processed`` maps an input-relative POSIX path to a small record:
        {"status": "matched"|"no_match"|"error", "best": {person: score}, ...}
    The same key is used for dedup, resume, and report rows.
    """

    persons: List[str]
    threshold: float
    processed: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Optional["Checkpoint"]:
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                persons=d["persons"],
                threshold=d["threshold"],
                processed=d.get("processed", {}),
            )
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            console.warn(f"Ignoring unreadable checkpoint ({exc}).")
            return None

    def save(self, path: Path) -> None:
        # Write to a temp file then atomically replace, so a crash mid-write
        # can never leave a half-written (corrupt) checkpoint behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "persons": self.persons,
                    "threshold": self.threshold,
                    "processed": self.processed,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    def matches_config(self, persons: List[str], threshold: float) -> bool:
        """True if this checkpoint was made for the same persons + threshold."""
        return sorted(self.persons) == sorted(persons) and abs(
            self.threshold - threshold
        ) <= 1e-9


# ----------------------------------------------------------------------------
# Embedding cache
# ----------------------------------------------------------------------------
def _embedding_cache_path(cache_dir: Path, rel_key: str) -> Path:
    """Deterministic cache file for one image's face embeddings.

    Keyed by a hash of the input-relative path, so arbitrary filenames (spaces,
    unicode, '(1)' collisions) map to safe, unique cache filenames.
    """
    digest = hashlib.sha1(rel_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npz"


def get_face_embeddings(
    engine: FaceEngine, src: Path, rel_key: str, cfg: Config, paths: Paths
) -> Optional[np.ndarray]:
    """Return an (F, 512) float32 array of face embeddings for one image.

    Loads from the on-disk cache when present; otherwise detects + embeds and
    caches the result. This makes changing --threshold/--persons a fast re-score
    instead of a full re-run. Returns None only if the image can't be decoded.
    """
    cache_path = _embedding_cache_path(paths.embeddings_cache, rel_key)
    if cache_path.exists():
        try:
            return np.load(cache_path)["emb"].astype(np.float32)
        except Exception:
            pass  # corrupt cache entry -> recompute below

    bgr = load_image_bgr(src, max_long_edge=cfg.max_long_edge)
    if bgr is None:
        return None

    faces = engine.embed_faces(bgr)
    emb = (
        np.vstack([f.embedding for f in faces]).astype(np.float16)
        if faces
        else np.zeros((0, 512), dtype=np.float16)
    )
    paths.embeddings_cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, emb=emb)
    return emb.astype(np.float32)


# ----------------------------------------------------------------------------
# Scoring + copying
# ----------------------------------------------------------------------------
def score_image(
    face_mat: Optional[np.ndarray], prototypes: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Best similarity to each prototype across all faces in one image.

    Returns {person: best_score} for every prototype (no threshold applied).
    Embeddings are L2-normalized, so cosine similarity is a plain dot product.
    """
    if face_mat is None or len(face_mat) == 0:
        return {name: -1.0 for name in prototypes}
    return {name: float(np.max(face_mat @ proto)) for name, proto in prototypes.items()}


def copy_match(src: Path, rel: Path, output_dir: Path) -> bool:
    """Copy src to output/<rel>, creating parents. Skip if the destination
    already exists. Returns True if a copy actually happened.

    This idempotency is what makes the whole pipeline crash-safe: if the process
    dies after a copy but before the checkpoint is flushed, the next run simply
    re-scores that image (cheap, from cache), sees the file already present, and
    skips it — so no duplicate is ever produced.
    """
    dst = output_dir / rel
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)  # copy2 preserves timestamps/metadata
    return True


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def append_manifest(rows: List[Tuple[str, str, str]], paths: Paths) -> None:
    """Append (relative_path, persons, scores) rows to output/manifest.csv."""
    if not rows:
        return
    paths.output.mkdir(parents=True, exist_ok=True)
    new_file = not paths.manifest.exists()
    with paths.manifest.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["relative_path", "persons", "scores"])
        writer.writerows(rows)


def write_candidates_report(
    processed: Dict[str, dict], threshold: float, cfg: Config, paths: Paths
) -> None:
    """Rank every image by its best score and write output/candidates.csv,
    printing the top rows. Surfaces likely photos even below threshold."""
    rows = []
    for rel_key, rec in processed.items():
        best = rec.get("best", {})
        if not best:
            continue
        top_person = max(best, key=best.get)
        rows.append((best[top_person], top_person, rel_key))
    rows.sort(reverse=True)

    paths.output.mkdir(parents=True, exist_ok=True)
    with paths.candidates.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["best_score", "best_person", "relative_path", "crosses_threshold"])
        for score, person, rel_key in rows:
            writer.writerow([f"{score:.4f}", person, rel_key, score >= threshold])

    if rows:
        console.step("Top candidate images (ranked by best match score):")
        console.detail(f"{'score':>6}  {'person':<12} path")
        for score, person, rel_key in rows[: cfg.candidates_top_n]:
            mark = "*" if score >= threshold else " "
            console.info(f"  {mark}{score:>6.3f}  {person:<12} {rel_key}")
        console.info(f"\nFull ranking written to {paths.candidates}")


# ----------------------------------------------------------------------------
# Batched processing loop
# ----------------------------------------------------------------------------
@dataclass
class RunStats:
    matched: int = 0
    copied: int = 0


def _progress(todo: List[Path]) -> Iterable[Path]:
    """Wrap the work list in a tqdm progress bar when available."""
    try:
        from tqdm import tqdm

        return tqdm(todo, unit="img")
    except Exception:
        return todo


def process_images(
    engine: FaceEngine,
    todo: List[Path],
    prototypes: Dict[str, np.ndarray],
    ckpt: Checkpoint,
    cfg: Config,
    paths: Paths,
    dry_run: bool,
) -> RunStats:
    """Score each pending image, copy matches, and checkpoint in batches.

    Copies happen before the batch checkpoint flush; combined with the idempotent
    copy_match, that ordering guarantees resume safety (see copy_match docstring).
    """
    stats = RunStats()
    since_flush = 0
    pending_manifest: List[Tuple[str, str, str]] = []

    def flush() -> None:
        ckpt.save(paths.checkpoint)
        append_manifest(pending_manifest, paths)
        pending_manifest.clear()

    for src in _progress(todo):
        rel = src.relative_to(paths.input)
        rel_key = rel.as_posix()

        face_mat = get_face_embeddings(engine, src, rel_key, cfg, paths)
        if face_mat is None:
            ckpt.processed[rel_key] = {"status": "error", "best": {}, "error": "unreadable"}
            since_flush += 1
            continue

        best = score_image(face_mat, prototypes)
        matches = {n: s for n, s in best.items() if s >= cfg.threshold}
        best_round = {n: round(s, 4) for n, s in best.items()}

        if matches:
            stats.matched += 1
            ckpt.processed[rel_key] = {"status": "matched", "best": best_round}
            if not dry_run:
                if copy_match(src, rel, paths.output):
                    stats.copied += 1
                pending_manifest.append(
                    (
                        rel_key,
                        ";".join(matches.keys()),
                        ";".join(f"{s:.3f}" for s in matches.values()),
                    )
                )
        else:
            ckpt.processed[rel_key] = {"status": "no_match", "best": best_round}

        since_flush += 1
        if since_flush >= cfg.batch_size:
            flush()
            since_flush = 0

    flush()  # final partial batch
    return stats
