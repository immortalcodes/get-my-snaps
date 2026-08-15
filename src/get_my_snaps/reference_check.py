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
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import console
from .cache import samples_key
from .config import Config
from .face_engine import FaceEngine, DetectedFace, load_image_bgr, is_supported_image

log = logging.getLogger(__name__)


@dataclass
class PersonReference:
    name: str
    prototype: Optional[np.ndarray]  # (512,) float32 mean embedding, or None if empty
    embeddings: List[np.ndarray] = field(default_factory=list)
    # fingerprint of the sample set this prototype was computed from, so a
    # changed/added photo invalidates it without a manual reset
    samples_key: str = ""
    # quality metrics (kept below for b/c with cached payloads)
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


def enroll_person(
    name: str,
    image_paths: Iterable[Path],
    engine: FaceEngine,
    cfg: Config,
) -> PersonReference:
    """Embed one person's sample photos and compute quality metrics.

    image_paths may come from a reference/<name>/ folder or a bare --of list;
    enrollment is identical either way. Returns the PersonReference (including
    people with 0 usable faces, so the caller can report them).
    """
    images = sorted(p for p in image_paths if p.is_file() and is_supported_image(p))
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
    person.samples_key = samples_key(images)
    return person


def resolve_ref_items(items: Iterable[Path]) -> Dict[str, List[Path]]:
    """Expand ``--of`` arguments into one-or-more people.

    Rules (documented in the spec):
      * a bare image  -> joins the "loose" bucket
      * a folder with images directly inside -> one person named after the
        folder; any child subfolders become their own people too
      * a folder with ONLY subfolders      -> each subfolder is one person
      * bare-image loose bucket            -> ONE person, named after the first
        image's stem (caller may rename via --name)
    Returns name -> sorted list of image paths, dropping empties.
    """
    people: Dict[str, List[Path]] = {}
    loose: List[Path] = []

    for item in items:
        if item.is_file() and is_supported_image(item):
            loose.append(item)
        elif item.is_dir():
            direct = sorted(
                p for p in item.iterdir() if p.is_file() and is_supported_image(p)
            )
            subs = sorted(p for p in item.iterdir() if p.is_dir())
            if not direct and subs:
                for sd in subs:
                    people[sd.name] = sorted(
                        p for p in sd.rglob("*") if p.is_file() and is_supported_image(p)
                    )
            else:
                if direct:
                    people.setdefault(item.name, []).extend(direct)
                for sd in subs:
                    people[sd.name] = sorted(
                        p for p in sd.rglob("*") if p.is_file() and is_supported_image(p)
                    )
        else:
            console.warn(f"Ignoring non-image, non-folder reference: {item}")

    if loose:
        name = loose[0].stem
        if name in people:  # collision between a bare file and a folder name
            name = f"{name}_ref"
        people[name] = loose + people.get(name, [])

    return OrderedDict((k, v) for k, v in people.items() if v)


def references_dir(central: Path, recipe: str) -> Path:
    """Where per-person prototype files for one recipe live."""
    return central / "references" / recipe


def person_cache_path(central: Path, recipe: str, name: str) -> Path:
    """The on-disk cache file for one person's prototype + metrics."""
    return references_dir(central, recipe) / f"{name}.json"


def save_person(person: PersonReference, central: Path, recipe: str) -> None:
    """Persist one person's prototype/metrics, model-scoped and content-keyed."""
    d = asdict(person)
    d.pop("embeddings", None)
    d["prototype"] = None if person.prototype is None else person.prototype.tolist()
    path = person_cache_path(central, recipe, person.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d), encoding="utf-8")


def load_cached_person(
    central: Path, recipe: str, name: str, expected_key: str
) -> Optional[PersonReference]:
    """Load a person if a cached prototype matches the current sample set.

    Returns None when the file is absent, unreadable, or its samples_key no
    longer matches -- the caller then re-enrolls.
    """
    path = person_cache_path(central, recipe, name)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if d.get("samples_key") != expected_key:
        return None
    person = PersonReference(
        name=name,
        prototype=None if d.get("prototype") is None
        else np.asarray(d["prototype"], dtype=np.float32),
        samples_key=d.get("samples_key", ""),
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
    return person


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
