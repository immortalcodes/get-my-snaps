"""
face_engine.py — format-agnostic image loading + InsightFace detection/embeddings.

This module is the only place that talks to InsightFace and to image decoders.
Everything downstream works with plain NumPy arrays and 512-d float32 embeddings.

Key facts (teaching notes):
- InsightFace `buffalo_l` returns L2-normalized 512-d ArcFace embeddings, so the
  cosine similarity between two faces is just their dot product.
- We decode with Pillow (+ pillow-heif for HEIC) so JPG/PNG/HEIC/WEBP/TIFF/BMP all
  work through one code path, then convert to a BGR NumPy array for InsightFace
  (which, like OpenCV, expects BGR channel order).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Register HEIC/HEIF support with Pillow. Safe if the codec is present; if the
# import fails we simply won't be able to read HEIC files (logged at import).
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception as exc:  # pragma: no cover - environment dependent
    _HEIF_OK = False
    logging.getLogger(__name__).warning(
        "pillow-heif unavailable (%s); HEIC/HEIF files will be skipped.", exc
    )

log = logging.getLogger(__name__)

# Extensions we attempt to decode. Matched case-insensitively against the suffix.
SUPPORTED_EXTS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def is_supported_image(path: Path) -> bool:
    """True if the file's extension is one we know how to decode."""
    return path.suffix.lower() in SUPPORTED_EXTS


def load_image_bgr(path: Path, max_long_edge: int = 1600) -> Optional[np.ndarray]:
    """
    Load any supported image as an RGB->BGR uint8 NumPy array, EXIF-corrected and
    downscaled so its long edge is <= max_long_edge.

    Returns None (and logs) if the file can't be decoded, so a single bad file
    never crashes a long batch run.
    """
    try:
        with Image.open(path) as im:
            # Honor EXIF orientation (phones/cameras rotate via metadata, not pixels).
            im = ImageOps.exif_transpose(im)
            # Flatten alpha/palette/CMYK to plain RGB for consistent downstream ops.
            if im.mode != "RGB":
                im = im.convert("RGB")

            # Downscale large images: full-res 5MB+ shots waste CPU during detection
            # without meaningfully improving recall on normal-sized faces.
            w, h = im.size
            long_edge = max(w, h)
            if long_edge > max_long_edge:
                scale = max_long_edge / float(long_edge)
                im = im.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.LANCZOS,
                )

            rgb = np.asarray(im, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        log.warning("Could not read image %s (%s); skipping.", path, exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unexpected error reading %s (%s); skipping.", path, exc)
        return None

    # PIL gives RGB; InsightFace/OpenCV expect BGR.
    return rgb[:, :, ::-1].copy()


class DetectedFace:
    """A single detected face: its embedding, detection score, and box area."""

    __slots__ = ("embedding", "det_score", "bbox_area")

    def __init__(self, embedding: np.ndarray, det_score: float, bbox_area: float):
        self.embedding = embedding  # L2-normalized float32, shape (512,)
        self.det_score = float(det_score)
        self.bbox_area = float(bbox_area)


class FaceEngine:
    """
    Thin wrapper around InsightFace's FaceAnalysis app, CPU-only.

    The model auto-downloads on first construction (~300MB for buffalo_l) into
    ~/.insightface, then runs fully offline.
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",
        det_size: int = 640,
        det_thresh: float = 0.35,
    ):
        # Imported lazily so `--help` and arg parsing don't pay the heavy import cost.
        from insightface.app import FaceAnalysis

        log.info("Loading InsightFace model '%s' (CPU)...", model_name)
        # we only use detection + recognition; no alignment or segmentation.
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        # ctx_id=-1 forces CPU. det_thresh is intentionally a bit low to favor recall.
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.model_name = model_name

    def embed_faces(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """Detect all faces in an image and return their normalized embeddings."""
        faces = self.app.get(image_bgr)
        out: List[DetectedFace] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = f.embedding / (np.linalg.norm(f.embedding) + 1e-9)
            x1, y1, x2, y2 = f.bbox
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            out.append(
                DetectedFace(
                    embedding=np.asarray(emb, dtype=np.float32),
                    det_score=getattr(f, "det_score", 1.0),
                    bbox_area=area,
                )
            )
        return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for (possibly un-normalized) 1-D vectors."""
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))
