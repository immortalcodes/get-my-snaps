# AGENTS.md — conventions & gotchas for AI coding agents

Context for anyone (human or LLM) working on this repo.

## What this is
A local, offline face-recognition photo filter. `input/` photos containing a
person from `reference/<name>/` are **copied** into `output/`, mirroring the
input tree. See [README.md](README.md) for user-facing docs.

## Module boundaries (keep these clean)
- **config.py** — every tunable (thresholds, batch size, model, detection params,
  reference-quality cutoffs, report limits) + all filesystem `Paths`. No logic.
  Precedence: dataclass defaults < JSON `--config` file < CLI flags. Add a new
  knob here rather than hardcoding a number anywhere else.
- **console.py** — all user-facing output and prompts. No business logic imports
  it for decisions; it only prints/asks. Put wording here, not inline in modules.
- **face_engine.py** — the *only* place that imports `insightface`, `PIL`,
  `pillow_heif`, `cv2`. Owns image decoding and the `FaceEngine` (detect + embed).
  Everything downstream deals only in NumPy arrays and 512-d float32 embeddings.
- **reference_check.py** — reference enrollment + quality scoring. Takes a
  `Config` for its thresholds. No CLI/argparse, no printing decisions.
- **matching.py** — the matching pipeline + its I/O: `Checkpoint`, embedding
  cache, `score_image`, `copy_match`, manifest/candidates reports, and the
  `process_images` batch loop. No argparse.
- **get_my_snaps.py** — CLI orchestration ONLY: argparse, wiring the modules
  together, and small step functions. No model/decoding/scoring logic here.

## Embedding cache
- Per-image face embeddings are cached in `.gms_embeddings/<sha1(rel_key)>.npz`
  (float16). `matching.get_face_embeddings` computes-or-loads. This is what makes
  changing `--threshold`/`--persons` a seconds-long re-score instead of a full
  re-run.
- The cache is input-derived and independent of reference/threshold, so `--reset`
  keeps it; only `--reset-embeddings` (or a changed input) should clear it.

## Hard rules
- **All filesystem paths use `pathlib.Path`, never string concatenation or shell.**
  This is what makes arbitrary filenames (spaces, unicode, parentheses, `(1)`
  collisions) safe without escaping. Do not introduce `os.system`, `subprocess`
  on paths, or f-string path building.
- **Embeddings are L2-normalized**, so similarity is a plain dot product. If you
  add a new embedding source, normalize it or matching math breaks.
- **Checkpoint keys are input-relative POSIX paths** (`rel.as_posix()`). The same
  key is used for dedup, resume, and manifest rows — keep them consistent.
- **Copies must stay idempotent.** `copy_match` skips if the destination exists.
  Preserve this so re-runs/resumes never duplicate.
- **Recall over precision** is intentional (low default `--threshold`, low
  `det_thresh`). Don't "fix" false positives by raising defaults — expose flags.
- **Never modify `input/`.** The tool copies; it must never move or delete sources.

## Things that must stay in sync
- Supported extensions live in `SUPPORTED_EXTS` (face_engine.py). Update there
  only; discovery and reference enrollment both read from it.
- Tunable thresholds live in `Config` (config.py). `reference_check._rate`,
  `FaceEngine`, and the matching loop all read them from a passed `Config` — don't
  reintroduce literal numbers in those modules.
- CLI flags that mirror `Config` fields default to `None` in argparse so
  `with_overrides` can tell "unset" from "set"; keep that pattern when adding flags.
- If you change the checkpoint schema, bump handling in `Checkpoint.load` so old
  checkpoints degrade gracefully instead of crashing.
- Threshold/persons are recorded in the checkpoint; `Checkpoint.matches_config`
  and the fresh-start prompt in `bootstrap_checkpoint` must stay aligned.

## Environment
- Python 3.13, CPU-only. Deps in [requirements.txt](requirements.txt); install
  into `.venv/` (git-ignored). Run everything via `.venv/bin/python`.
- `./run_sample.sh` bootstraps the venv, installs deps, and does a hands-free dry
  run for every reference person (`--real` to actually copy). Good first check.
- InsightFace `buffalo_l` auto-downloads once to `~/.insightface/` then works
  offline. `insightface` prints noisy provider/model lines and a `FutureWarning`
  from scikit-image on import — harmless; filter in logs if needed.

## Gotchas observed
- Input folder names are inconsistent (`Camera -01` with a space vs `Camera-02`)
  and double-nested inside a Google-Takeout wrapper. Never hardcode these; rely on
  `rglob` + relative paths.
- `pillow-heif` must be registered (`register_heif_opener()`) before Pillow can
  open HEIC — done at import in face_engine.py. If HEIC silently skips, that
  registration failed.
- With `--batch-size` larger than the total image count, the checkpoint only
  flushes at the final step. That's fine, but keep the final flush in place.
