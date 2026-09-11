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

import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Iterator, List, Optional

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


@contextlib.contextmanager
def _quiet_native_io(enabled: bool = True) -> Iterator[None]:
    """Silence stdout/stderr written by native code (onnxruntime/InsightFace).

    InsightFace and onnxruntime write directly to file descriptors 1/2 (bare
    ``print()`` calls in InsightFace, C++ logging in onnxruntime) rather than
    through Python's ``logging`` module, so redirecting ``sys.stdout``/
    ``sys.stderr`` objects has no effect on them. The only reliable way to
    silence this is to swap the underlying OS file descriptors for the
    duration of the call, then restore them.

    When ``enabled`` is False this is a no-op passthrough (used for
    ``--verbose``, so raw InsightFace/onnxruntime output is visible for
    debugging).
    """
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def model_available(model_root: Optional[Path], model_name: str) -> bool:
    """True if the model's weights already exist under ``model_root``.

    InsightFace looks for ``<root>/models/<name>/``. This is a pure filesystem
    check (no download, no engine) so the CLI can warn before the slow path.
    """
    base = Path(model_root) if model_root is not None else Path("~/.insightface").expanduser()
    return (base / "models" / model_name).is_dir()


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
        model_name: str,
        det_size: int = 640,
        det_thresh: float = 0.35,
        model_root: Optional[Path] = None,
        verbose: bool = False,
    ):
        # Imported lazily so `--help` and arg parsing don't pay the heavy import cost.

        # model_name is required: Config is the single source of truth for which
        # model to use. model_root redirects where weights live -- essential
        # under snap strict confinement, where ~/.insightface is unwritable.
        root = str(model_root) if model_root is not None else "~/.insightface"
        log.info("Loading InsightFace model '%s' (CPU) from %s ...", model_name, root)

        if not verbose:
            # InsightFace's face-alignment code calls a scikit-image API
            # that's deprecated since skimage 0.26 (FutureWarning); harmless,
            # but noisy on every run. skimage raises it with a stacklevel that
            # attributes it to the *caller* (insightface.utils.face_align), so
            # filtering by `module` would need to match that caller, not
            # "skimage" -- simpler and more robust to match on the message.
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r".*estimate.*deprecated.*",
            )
            # Pre-fetch the model weights *unsuppressed* so the one-time download
            # progress bar (InsightFace's own tqdm output) stays visible, before
            # we silence the noisy-but-harmless construction/prepare() below.
            try:
                from insightface.utils import ensure_available

                ensure_available("models", model_name, root=root)
            except Exception:  # pragma: no cover - best-effort only
                pass

        # onnxruntime/InsightFace write noisy diagnostics straight to the
        # process's stdout/stderr file descriptors during model construction:
        # onnxruntime's GPU/NPU device-discovery probes (denied under snap
        # strict confinement, since we never request non-CPU providers anyway),
        # per-model thread-pool creation errors (pthread_setaffinity_np denied
        # under confinement), and InsightFace's own bare print() calls ("find
        # model:", "model ignore:", "Applied providers:", "set det-size:").
        # All cosmetic -- we only ever use providers=["CPUExecutionProvider"]
        # and ctx_id=-1 below. Suppressed unless --verbose.
        with _quiet_native_io(enabled=not verbose):
            from insightface.app import FaceAnalysis

            # we only use detection + recognition; no alignment or segmentation.
            self.app = FaceAnalysis(
                name=model_name,
                root=root,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            # ctx_id=-1 forces CPU. det_thresh is a bit low to favor recall.
            self.app.prepare(
                ctx_id=-1, det_size=(det_size, det_size), det_thresh=det_thresh
            )
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
