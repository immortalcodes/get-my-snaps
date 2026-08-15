#!/usr/bin/env python3
"""
profile_run.py — find out what actually eats the time before optimizing.

Runs on a sample of input images WITHOUT using the embedding cache, so it
measures the real cost of the slow path. Reports:

  1. Per-phase wall-clock totals + per-image averages:
       - decode : load_image_bgr (Pillow/HEIC decode + resize)
       - infer  : engine.embed_faces (ONNX detection + per-face recognition)
       - score  : numpy cosine scoring
  2. Faces-per-image stats (recognition cost scales with this).
  3. A cProfile top-20 by cumulative time.

Usage:
    .venv/bin/python scripts/profile_run.py                # 40 images, buffalo_l
    .venv/bin/python scripts/profile_run.py --sample 80
    .venv/bin/python scripts/profile_run.py --model buffalo_s
    .venv/bin/python scripts/profile_run.py --threads 1    # cap ORT/OMP intra-op threads

Nothing is copied and no checkpoint/manifest/cache is written.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=40, help="How many images to time.")
    ap.add_argument("--model", default=None, help="Override model (buffalo_l/buffalo_s).")
    ap.add_argument("--threads", type=int, default=None,
                    help="Cap ORT/OMP intra-op threads (set BEFORE model load).")
    args = ap.parse_args()

    # Thread caps must be set before onnxruntime/numpy import their backends.
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["ONNXRUNTIME_INTRA_OP_NUM_THREADS"] = str(args.threads)

    import numpy as np
    from get_my_snaps.config import Config, Paths
    from get_my_snaps.face_engine import FaceEngine, load_image_bgr, is_supported_image

    source = Path("input")
    paths = Paths.for_run(source=source, out=Path("output"))
    cfg = Config().with_overrides(model_name=args.model)

    images = sorted(
        p for p in paths.source.rglob("*") if p.is_file() and is_supported_image(p)
    )
    if not images:
        print("No input images found.")
        return 1
    step = max(1, len(images) // args.sample)
    sample = images[::step][: args.sample]
    print(f"Profiling {len(sample)} of {len(images)} images "
          f"(model={cfg.model_name}, threads={args.threads or 'default'})\n")

    print("Loading model...")
    t0 = time.perf_counter()
    engine = FaceEngine(
        model_name=cfg.model_name,
        det_size=cfg.det_size,
        det_thresh=cfg.det_thresh,
        model_root=paths.model_root,
    )
    print(f"Model load: {time.perf_counter() - t0:.1f}s\n")

    proto = np.random.randn(512).astype(np.float32)
    proto /= np.linalg.norm(proto)

    t_decode = t_infer = t_score = 0.0
    face_counts = []

    def work() -> None:
        nonlocal t_decode, t_infer, t_score
        for src in sample:
            a = time.perf_counter()
            bgr = load_image_bgr(src, max_long_edge=cfg.max_long_edge)
            b = time.perf_counter()
            if bgr is None:
                continue
            faces = engine.embed_faces(bgr)
            c = time.perf_counter()
            if faces:
                mat = np.vstack([f.embedding for f in faces]).astype(np.float32)
                float(np.max(mat @ proto))
            d = time.perf_counter()
            t_decode += b - a
            t_infer += c - b
            t_score += d - c
            face_counts.append(len(faces))

    prof = cProfile.Profile()
    wall0 = time.perf_counter()
    prof.enable()
    work()
    prof.disable()
    wall = time.perf_counter() - wall0

    n = len(face_counts) or 1
    print("=" * 60)
    print("PHASE TOTALS (wall clock)")
    print("=" * 60)
    print(f"  decode : {t_decode:7.2f}s   {t_decode/n*1000:7.1f} ms/img   {t_decode/wall*100:5.1f}%")
    print(f"  infer  : {t_infer:7.2f}s   {t_infer/n*1000:7.1f} ms/img   {t_infer/wall*100:5.1f}%")
    print(f"  score  : {t_score:7.2f}s   {t_score/n*1000:7.1f} ms/img   {t_score/wall*100:5.1f}%")
    print(f"  TOTAL  : {wall:7.2f}s   {wall/n*1000:7.1f} ms/img")
    print()
    fc = np.array(face_counts)
    print(f"faces/image: mean={fc.mean():.1f}  median={np.median(fc):.0f}  "
          f"max={fc.max()}  total_faces={fc.sum()}")
    print()

    print("=" * 60)
    print("cProfile TOP 20 (by cumulative time)")
    print("=" * 60)
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
