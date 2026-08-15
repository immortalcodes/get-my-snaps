# AGENTS.md — conventions & gotchas for AI coding agents

Context for anyone (human or LLM) working on this repo.

## What this is
A local, offline face-recognition photo filter. Source photos containing a
person are **copied** into an output folder, mirroring the source tree. It works
two ways: `get-my-snaps ~/Pictures --of ~/me.jpg` (ad-hoc reference), or the
legacy `input/` + `reference/<name>/` layout. It also ships as a snap
(`sudo snap install get-my-snaps`). See [README.md](README.md) for user-facing
docs.

## Project layout (src/ package)
The code is an installable package under `src/get_my_snaps/`. Runtime state is
split: **per-run** state (checkpoint, input-embedding cache, reports) lives
inside the chosen output dir under `<out>/.gms/`; **shared** state (InsightFace
models, per-person reference prototypes) lives in a central dir —
`$SNAP_USER_COMMON` in a snap, else `~/.cache/get-my-snaps`. Install editable
with `pip install -e .` (or `-e ".[dev]"` for pytest). Entry points:
`get-my-snaps` console script and `python -m get_my_snaps`, both →
`get_my_snaps.cli:main`.

- `src/get_my_snaps/` — the package. Modules import each other **relatively**
  (`from . import console`, `from .config import Config`). Never reintroduce
  top-level absolute imports (`import config`) — they break once installed.
- `scripts/` — non-package helpers: `run_sample.sh` (bootstrap+demo, run from
  repo root as `./scripts/run_sample.sh`) and `profile_run.py` (dev profiler;
  imports the package via `from get_my_snaps... import`).
- `tests/` — pytest suite; `conftest.py` adds `src/` to `sys.path` as a fallback
  when the package isn't installed.
- `pyproject.toml` — authoritative dependency + packaging metadata.
  `requirements.txt` is now just a shim (`-e .`).

## Module boundaries (keep these clean)
- **config.py** — every tunable (thresholds, batch size, model, detection params,
  reference-quality cutoffs, report limits) + the `Paths` factory. No business
  logic. `Paths.for_run(source, out)` derives every path from explicit roots
  (there is **no** cwd-anchored module constant — that's what makes arbitrary
  source dirs work). Central/model-root resolution lives here too
  (`default_central_dir`, `default_model_root`). Precedence: dataclass defaults
  < JSON `--config` file < CLI flags. Add a new knob here rather than hardcoding
  a number anywhere else.
- **cache.py** — *the* place for keying rules: `recipe()` (model + detection
  settings → namespace), `image_digest()` (absolute source + relative key +
  size + mtime → one image instance), `embedding_path()` (fanout layout),
  `samples_key()` (a whole reference set). Pure and side-effect free; extend
  tests here, not in the callers.
- **console.py** — all user-facing output and prompts. No business logic imports
  it for decisions; it only prints/asks. Put wording here, not inline in modules.
- **face_engine.py** — the *only* place that imports `insightface`, `PIL`,
  `pillow_heif`, `cv2`. Owns image decoding and the `FaceEngine` (detect + embed).
  Everything downstream deals only in NumPy arrays and 512-d float32 embeddings.
- **reference_check.py** — reference enrollment + quality scoring + `--of`
  resolution (`resolve_ref_items`). Content-keyed per-person caching lives here
  (`save_person`/`load_cached_person`). Takes a `Config` for its thresholds. No
  CLI/argparse, no printing decisions.
- **matching.py** — the matching pipeline + its I/O: `Checkpoint`, embedding
  cache, `score_image`, `copy_match`, manifest/candidates reports, and the
  `process_images` batch loop. No argparse.
- **get_my_snaps.py** — CLI orchestration ONLY: argparse, wiring the modules
  together, and small step functions. No model/decoding/scoring logic here.
  Resolves SOURCE / `--of` / output default; all reference resolution delegates
  to reference_check. (Lives at `src/get_my_snaps/cli.py`; other module names
  below are also under `src/get_my_snaps/`.)

## Caches & their keys
Everything is **provenance-scoped**, driven by `cache.py`:
- **Per-image input embeddings** cached at
  `<out>/.gms/embeddings/<recipe>/<d[:2]>/<digest>.npz` (float16).
  - `recipe` = model + det_size + det_thresh + max_long_edge. Switching models
    opens a fresh namespace — cross-model embeddings are **incomparable**
    (measured ~−0.05 cosine for the same face), so never mix recipes.
  - `digest` = absolute source root + relative path + size + mtime. Prevents
    two source trees with the same filename colliding, and an edited image
    self-invalidates.
- **Per-person reference prototypes** cached centrally at
  `<central>/references/<recipe>/<person>.json`, keyed on `samples_key` (the
  sorted set of sample paths + sizes + mtimes). Adding/removing/editing a
  reference photo re-enrolls only that person; different recipes stay separate.
- `matching.get_face_embeddings` computes-or-loads with a provenance guard (it
  only trusts a cache entry whose stored recipe+source match). This is what
  makes changing `--threshold`/`--persons` a seconds-long re-score.
- Flags: `--reset` drops only the checkpoint; `--reset-embeddings` also clears
  `<out>/.gms/embeddings/`; `--reset-refs` clears the central reference cache.
  `--reset` does **not** touch references — they self-invalidate on content
  change.

## Hard rules
- **All filesystem paths use `pathlib.Path`, never string concatenation or shell.**
  This is what makes arbitrary filenames (spaces, unicode, parentheses, `(1)`
  collisions) safe without escaping. Do not introduce `os.system`, `subprocess`
  on paths, or f-string path building. Construct paths from the explicit
  `Paths` object, never a fresh `Path.cwd()`-anchored constant.
- **Embeddings are L2-normalized**, so similarity is a plain dot product. If you
  add a new embedding source, normalize it or matching math breaks.
- **Checkpoint keys are source-relative POSIX paths** (`rel.as_posix()`). The same
  key is used for dedup, resume, and manifest rows — keep them consistent.
- **Copies must stay idempotent.** `copy_match` skips if the destination exists.
  Preserve this so re-runs/resumes never duplicate.
- **Recall over precision** is intentional (low default `--threshold`, low
  `det_thresh`). Don't "fix" false positives by raising defaults — expose flags.
- **Never modify the source photos.** The tool copies; it must never move or
  delete sources.
- **Model root matters.** Under snap strict confinement `~/.insightface` is
  unwritable (the `home` interface excludes hidden dirs), so `FaceEngine` takes
  a `model_root` resolved by `config.default_model_root`. Always pass it from
  `Paths`; never hardcode `~/.insightface`.

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
- Threshold/persons/recipe/sources are recorded in the checkpoint;
  `Checkpoint.matches_config` and the fresh-start prompt in `bootstrap_checkpoint`
  must stay aligned. The recipe string must come from the same `cache.recipe`
  everywhere — the cache, the checkpoint, and the reference prototype dir all
  key on it.
- `Config.model_name` and `FaceEngine`'s parameter must match; `FaceEngine` takes
  `model_name` as a required argument (no driftable default).
- The default model is `buffalo_s`. If you change it, re-verify `--threshold`
  still separates positives from negatives on a labelled set, and update the
  README's download size.

## Environment
- Python 3.13, CPU-only. Deps declared in [pyproject.toml](pyproject.toml)
  ([requirements.txt](requirements.txt) is a `-e .` shim). Install into `.venv/`
  (git-ignored) with `pip install -e ".[dev]"`. Run everything via
  `.venv/bin/python` or the `get-my-snaps` console script.
- `./scripts/run_sample.sh` bootstraps the venv, installs the package, and does a
  hands-free dry run for every reference person (`--real` to actually copy).
  Good first check. Run it from the repo root.
- `.venv/bin/python -m pytest` runs the suite (pure-logic tests for config +
  checkpoint; no ML deps needed).
- InsightFace `buffalo_s` (the default) auto-downloads once to the resolved
  `model_root` (`~/.insightface` when already populated, else the central dir)
  then works offline. `insightface` prints noisy provider/model lines and a
  `FutureWarning` from scikit-image on import — harmless; filter in logs if
  needed.

## Gotchas observed
- Input folder names are inconsistent (`Camera -01` with a space vs `Camera-02`)
  and double-nested inside a Google-Takeout wrapper. Never hardcode these; rely on
  `rglob` + relative paths.
- `pillow-heif` must be registered (`register_heif_opener()`) before Pillow can
  open HEIC — done at import in face_engine.py. If HEIC silently skips, that
  registration failed.
- With `--batch-size` larger than the total image count, the checkpoint only
  flushes at the final step. That's fine, but keep the final flush in place.
