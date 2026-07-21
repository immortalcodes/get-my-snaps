"""
reference_check.py — enroll reference people and score reference-set quality.

For each `reference/<name>/` folder we:
  1. Decode every image and detect faces.
  2. Build a prototype embedding per person (mean of that person's face embeddings).
  3. Score the reference set so we can warn the user *before* scanning the input
     if a person's references are too weak/ambiguous to match reliably.

Quality signals used:
  - usable_images     : how many reference photos yielded exactly-usable faces
  - avg_det_score     : mean detector confidence (blur/occlusion lowers this)
  - avg_face_frac     : mean face-area as a fraction of image area (tiny faces = bad)
  - consistency       : mean pairwise cosine similarity among the person's own faces
                        (low => photos may be of different people / very inconsistent)
  - multi_face_imgs   : reference photos containing >1 face (ambiguous identity)

These roll up into a Good / Fair / Poor rating with actionable advice.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import Config
from face_engine import FaceEngine, DetectedFace, load_image_bgr, is_supported_image

log = logging.getLogger(__name__)


@dataclass
class PersonReference:
    name: str
    prototype: Optional[np.ndarray]  # (512,) float32 mean embedding, or None if empty
    embeddings: List[np.ndarray] = field(default_factory=list)
    # quality metrics
    total_images: int = 0
    usable_images: int = 0
    multi_face_images: int = 0
    zero_face_images: int = 0
    avg_det_score: float = 0.0
    avg_face_frac: float = 0.0
    consistency: float = 0.0
    rating: str = "Poor"
    advice: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.prototype is not None and self.usable_images > 0


def _rate(person: PersonReference, cfg: Config) -> None:
    """Assign a Good/Fair/Poor rating + advice from the collected metrics.

    All thresholds come from Config so they can be tuned via a JSON config file
    without editing this logic.
    """
    advice: List[str] = []

    if person.usable_images == 0:
        person.rating = "Poor"
        person.advice = [
            "No detectable face found in any reference photo. Add clear, "
            "front-facing, well-lit close-ups of this person."
        ]
        return

    # Individual red flags, all driven by configurable thresholds.
    few_images = person.usable_images < cfg.min_usable_images
    low_conf = person.avg_det_score < cfg.min_det_score
    small_face = person.avg_face_frac < cfg.min_face_frac
    low_consistency = person.usable_images >= 2 and person.consistency < cfg.min_consistency
    ambiguous = person.multi_face_images > 0

    if few_images:
        advice.append(
            f"Only {person.usable_images} usable photo(s). Add 2–3 more from "
            "different angles for reliable matching."
        )
    if low_conf:
        advice.append(
            "Reference faces are low-confidence (blurry/occluded). Replace with "
            "sharper, unobstructed photos."
        )
    if small_face:
        advice.append(
            "Faces are small in frame. Use closer crops where the face fills more "
            "of the photo."
        )
    if low_consistency:
        advice.append(
            "Reference photos are inconsistent with each other — they may include "
            "different people or very varied conditions. Remove outliers so all "
            "photos are clearly the same person."
        )
    if ambiguous:
        advice.append(
            f"{person.multi_face_images} reference photo(s) contain multiple faces, "
            "which makes identity ambiguous. Use photos with only this person."
        )

    # Roll-up rating.
    serious = low_consistency or (person.usable_images == 0)
    minor = few_images or low_conf or small_face or ambiguous

    if serious:
        person.rating = "Poor"
    elif minor:
        person.rating = "Fair"
    else:
        person.rating = "Good"

    if not advice:
        advice.append("Reference set looks good.")
    person.advice = advice


def _consistency(embeddings: List[np.ndarray]) -> float:
    """Mean pairwise cosine similarity among a person's normalized embeddings."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    mat = np.vstack(embeddings)  # (n, 512), already L2-normalized
    sims = mat @ mat.T
    # Average of the upper triangle (exclude the diagonal of self-similarities).
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sims[iu]))


def _pick_best_face(faces: List[DetectedFace]) -> DetectedFace:
    """When a reference photo has several faces, take the largest, most confident."""
    return max(faces, key=lambda f: (f.bbox_area, f.det_score))


def enroll_references(
    reference_dir: Path,
    engine: FaceEngine,
    cfg: Config,
) -> Dict[str, PersonReference]:
    """
    Walk reference/<name>/ folders, embed faces, and compute quality metrics.
    Returns a dict name -> PersonReference (including people with 0 usable faces,
    so the caller can report them).
    """
    people: Dict[str, PersonReference] = {}

    person_dirs = sorted(p for p in reference_dir.iterdir() if p.is_dir())
    if not person_dirs:
        log.warning("No <name> subfolders found under %s", reference_dir)
        return people

    for pdir in person_dirs:
        name = pdir.name
        images = [p for p in sorted(pdir.rglob("*")) if p.is_file() and is_supported_image(p)]
        person = PersonReference(name=name, prototype=None, total_images=len(images))

        det_scores: List[float] = []
        face_fracs: List[float] = []

        for img_path in images:
            bgr = load_image_bgr(img_path, max_long_edge=cfg.max_long_edge)
            if bgr is None:
                continue
            img_area = float(bgr.shape[0] * bgr.shape[1])
            faces = engine.embed_faces(bgr)

            if not faces:
                person.zero_face_images += 1
                continue
            if len(faces) > 1:
                person.multi_face_images += 1

            best = _pick_best_face(faces)
            person.embeddings.append(best.embedding)
            person.usable_images += 1
            det_scores.append(best.det_score)
            face_fracs.append(best.bbox_area / img_area if img_area else 0.0)

        if person.embeddings:
            proto = np.mean(np.vstack(person.embeddings), axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            person.prototype = proto.astype(np.float32)
            person.avg_det_score = float(np.mean(det_scores))
            person.avg_face_frac = float(np.mean(face_fracs))
            person.consistency = _consistency(person.embeddings)

        _rate(person, cfg)
        people[name] = person

    return people


def format_report(people: Dict[str, PersonReference]) -> str:
    """Human-readable per-person quality report."""
    lines = ["", "=" * 64, "REFERENCE QUALITY REPORT", "=" * 64]
    for name, p in people.items():
        lines.append(
            f"\n[{p.rating.upper()}] {name}"
            f"  —  usable {p.usable_images}/{p.total_images} photos"
        )
        if p.usable_images:
            lines.append(
                f"    detection conf: {p.avg_det_score:.2f} | "
                f"face size: {p.avg_face_frac * 100:.1f}% of frame | "
                f"self-consistency: {p.consistency:.2f}"
            )
        if p.multi_face_images:
            lines.append(f"    warning: {p.multi_face_images} photo(s) had multiple faces")
        if p.zero_face_images:
            lines.append(f"    warning: {p.zero_face_images} photo(s) had no detectable face")
        for a in p.advice:
            lines.append(f"    → {a}")
    lines.append("=" * 64 + "\n")
    return "\n".join(lines)


def save_reference_cache(people: Dict[str, PersonReference], path: Path) -> None:
    """Persist prototypes + metrics so re-runs can skip re-embedding."""
    payload = {}
    for name, p in people.items():
        d = asdict(p)
        d.pop("embeddings", None)
        d["prototype"] = None if p.prototype is None else p.prototype.tolist()
        payload[name] = d
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_reference_cache(path: Path) -> Optional[Dict[str, PersonReference]]:
    """Load a previously saved reference cache, or None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    people: Dict[str, PersonReference] = {}
    for name, d in payload.items():
        proto = d.get("prototype")
        person = PersonReference(
            name=name,
            prototype=None if proto is None else np.asarray(proto, dtype=np.float32),
            total_images=d.get("total_images", 0),
            usable_images=d.get("usable_images", 0),
            multi_face_images=d.get("multi_face_images", 0),
            zero_face_images=d.get("zero_face_images", 0),
            avg_det_score=d.get("avg_det_score", 0.0),
            avg_face_frac=d.get("avg_face_frac", 0.0),
            consistency=d.get("consistency", 0.0),
            rating=d.get("rating", "Poor"),
            advice=d.get("advice", []),
        )
        people[name] = person
    return people
