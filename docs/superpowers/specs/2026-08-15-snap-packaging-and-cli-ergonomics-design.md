# Snap packaging + CLI ergonomics

**Date:** 2026-08-15
**Status:** approved design, not yet implemented

## Goal

Two commands, total:

```bash
sudo snap install get-my-snaps
get-my-snaps ~/Pictures/wedding --of ~/me.jpg
```

Today it takes a git clone, a venv, `pip install -e .`, a hand-built
`input/` + `reference/<name>/` tree, and an interactive picker. Packaging alone
does not fix that: `config.ROOT = Path.cwd()` hardcodes the folder convention,
so a globally-installed binary would still demand the user build directories
first. Both halves are in scope.

## Verified constraints

These were measured on this machine, not assumed. They drive the design.

| Fact | Evidence | Consequence |
|---|---|---|
| `home` interface denies toplevel hidden dirs | live AppArmor profile: `owner @{HOME}/[^s.]** rwklix` | `~/.insightface` is **unwritable** under strict confinement |
| `FaceAnalysis(root=...)` exists | `signature: (self, name, root='~/.insightface', allowed_modules, **kwargs)` | one-line override; no patching InsightFace |
| Pretrained models are non-commercial research only | [model_zoo README](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) | do **not** bundle weights in a store snap; fetch on first run |
| `insightface` is sdist-only | `pip download` yields `insightface-0.7.3.tar.gz` | snapcraft part needs `build-essential` + `cython` + numpy at build time |
| Same face, `buffalo_l` vs `buffalo_s` embedding → **cosine −0.05** | measured on `reference/colin_powell/colin_powell_1.jpg` | cross-model embeddings are noise; cache MUST be model-scoped |
| Existing cache is a provenance-unknown mix | 876 entries Jul 17–20 (`buffalo_l`), 1478 Jul 22+ (`buffalo_s`, after `be796b9`); neither key nor `.npz` records the model | not safely salvageable |
| `0.28` separates both models on labelled data | `buffalo_l` gap 0.789, `buffalo_s` gap 0.688 over 4 pos / 4 neg | keep `threshold = 0.28` when defaulting to `buffalo_s` |

## Decisions

1. **Distribution:** snap, `base: core24`, `confinement: strict`.
2. **Invocation:** positional sources + `--of`, folder convention retained as fallback.
3. **Storage split:** central = models + reference prototypes; per-run = input
   embeddings, checkpoint, reports.
4. **No legacy cache migration.** Old `.gms_embeddings/` is left on disk, ignored.
5. **Default model → `buffalo_s`**, `Config` and `FaceEngine` reconciled.

---

## 1. CLI surface

```
get-my-snaps SOURCE... --of REF... [-o OUT] [--name NAME] [tuning]
```

**`SOURCE...`** — directories and/or image files. Omitted → `./input` if it
exists, else error with guidance.

**`--of REF...`** — each REF is a photo **or** a folder:

| Invocation | Meaning |
|---|---|
| `--of ~/me.jpg` | one person, single sample |
| `--of ~/me/` | one person `me`, every supported image inside |
| `--of a.jpg b.jpg c.jpg` | **one** person, three samples; name = first stem, override via `--name` |
| `--of reference/` (only subdirs inside) | one person per subfolder — today's semantics |

The last row keeps `scripts/run_sample.sh` and existing habits working unchanged.

**`-o/--out`** — default `./get-my-snaps-out`; stays `./output` when the legacy
`input/` + `reference/` layout is detected.

Explicit `--of` ⇒ no person picker. The only surviving prompt is the
Poor-reference pause, already bypassable via the existing `--skip-ref-check`.
No new flag for that.

**Disambiguation rule** for a folder passed to `--of`: if it contains any
supported image directly, it is one person; if it contains only subdirectories,
each subdirectory is a person. Mixed content → images belong to a person named
after the folder, subdirectories become their own people, and a warning is
printed naming both interpretations.

### Code impact

- `config.ROOT` module constant is deleted. Its import-time `Path.cwd()` is why
  `test_config.py::test_paths_anchor_on_cwd` needs an `importlib.reload` hack;
  that hack goes away.
- `Paths` becomes a factory over explicit roots: `Paths.for_run(sources, out)`.
- `enroll_references` splits into `enroll_person(name, images, engine, cfg)` plus
  a directory walker, so an ad-hoc image list enrolls without a synthetic folder.

## 2. Storage layout

**Central** — `$SNAP_USER_COMMON` (snap) / `$XDG_CACHE_HOME/get-my-snaps`
(pip), resolution order:
`$GMS_CACHE_DIR` → `$SNAP_USER_COMMON` → `$XDG_CACHE_HOME/get-my-snaps` → `~/.cache/get-my-snaps`.

```
<central>/models/<model_name>/          InsightFace weights, fetched once
<central>/references/<recipe>/<person>.json
```

**Per-run** — inside the output dir, so a run is self-contained and disposable:

```
<out>/manifest.csv                      matches (unchanged)
<out>/candidates.csv                    full ranking (unchanged)
<out>/.gms/checkpoint.json              resume state
<out>/.gms/embeddings/<recipe>/xx/<digest>.npz
```

`rm -rf <out>` erases all run state. No `.gms_*` litter in cwd or `$HOME`.
Trade-off accepted: re-running with the same `-o` gets the fast cached
re-score; pointing `-o` at a fresh dir re-embeds the inputs.

Reference prototypes are central because enrollment cost is person-scoped, not
run-scoped: `--of ~/me/` embeds once ever, and every later run against any
source reuses it.

### The recipe key

An embedding is determined by its recipe and its subject:

```
recipe  = sha1(f"{model_name}|{det_size}|{det_thresh}|{max_long_edge}")[:12]
digest  = sha1(f"{abs_source_root}\0{rel_key}\0{size}\0{mtime_ns}")
```

`recipe` is a **directory** level, so switching `--model` opens a fresh
namespace instead of colliding, and switching back finds the old entries intact.
This is what makes `--model buffalo_s` safe given the measured −0.05
cross-model cosine.

`digest` includes the absolute source root, fixing a live collision bug:
`_embedding_cache_path` (matching.py:97) currently hashes only the
source-relative path, so `wedding/IMG_1.jpg` and `party/IMG_1.jpg` both map to
`sha1("IMG_1.jpg")`. Harmless with a fixed `input/`; silent wrong-face
corruption with arbitrary sources. Including size + mtime_ns also makes an
edited or replaced image self-invalidate.

Two-level fanout (`xx/<digest>.npz`) keeps 10k+ entries out of one flat dir.
Each `.npz` also stores `recipe` and `src` arrays, so a stray file is
self-describing and a digest collision is detected on load rather than trusted.

### Reference cache invalidation

`<central>/references/<recipe>/<person>.json` stores `samples_key` =
sha1 over each sample's `(abs path, size, mtime_ns)`, sorted. Mismatch ⇒
re-enroll that person only.

This fixes a second live bug: `.gms_reference_cache.json` is a single blob
invalidated only by `--reset`, so dropping a new photo into `reference/alice/`
silently keeps the stale prototype. Per-person files also mean enrolling `bob`
never rewrites `alice` — which matters once the cache is shared machine-wide.

### Checkpoint

`Checkpoint` gains `recipe` and `sources`; `matches_config` compares them, so a
checkpoint from a different model or source set is rejected rather than resumed.
`Checkpoint.load` tolerates their absence and treats such a checkpoint as
incompatible, per the AGENTS.md rule that schema changes degrade gracefully.

### Flag remap

| Flag | New meaning |
|---|---|
| `--reset` | wipe `<out>/.gms/checkpoint.json` |
| `--reset-embeddings` | also wipe `<out>/.gms/embeddings/` |
| `--reset-refs` (new) | force re-enrollment of selected people |

`--reset` no longer touches references: they self-invalidate on content change.

## 3. Model root

`FaceEngine` gains `model_root`, passed to `FaceAnalysis(root=...)`, resolved:
`$GMS_MODEL_DIR` → `<central>/models` → existing `~/.insightface` when already
populated (so the current 881 MB is reused outside the snap).

Weights are **not** bundled — non-commercial-research licensing is not ours to
redistribute. First run fetches ~159 MB for `buffalo_s` (vs 326 MB for
`buffalo_l`) behind a clear one-time notice.

## 4. Model default reconciliation

`Config.model_name` says `buffalo_l`; `FaceEngine.__init__` defaults to
`buffalo_s` since `be796b9`. Resolution: **`buffalo_s` everywhere**, and
`FaceEngine`'s parameter loses its default so `Config` is the single source of
truth and the two cannot drift again. `scripts/profile_run.py` already passes
`model_name` explicitly via `Config`.

Justified by measurement: on the labelled 4-positive / 4-negative set,
`buffalo_s` keeps a 0.688 score gap at `threshold = 0.28`. README's small/distant
face warning still applies — these were tight crops.

## 5. The snap

`snap/snapcraft.yaml`:

- `base: core24`, `confinement: strict`, `adopt-info` from pyproject version.
- `apps.get-my-snaps.plugs: [home, removable-media, network]`.
  `home` covers `~/Pictures`; `removable-media` covers `/media` for SD cards;
  `network` is first-run model download only.
- Python parts installed with no PEP 517 isolation and ordered pip calls
  (numpy + cython first) because `insightface` is sdist-only.
- Force `opencv-python-headless` and prune the pulled-in full `opencv-python`,
  plus `onnx` and `skimage` where unused — worth roughly 200 MB unsquashed.
  Every prune is smoke-tested, not assumed.

Expected installed size ~200 MB. Publishing needs the maintainer's Ubuntu One
account (`snapcraft register get-my-snaps`); local verification does not.

## Verification plan

Evidence required before this is called done:

1. `snapcraft --use-lxd` builds clean (LXD confirmed present).
2. `snap install --dangerous ./*.snap`, then **as a non-root user**:
   `get-my-snaps ~/Pictures/<dir> --of ~/<face>.jpg` completes and copies
   matches. This is the real confinement test — model root, `home` access, and
   first-run download all exercised at once.
3. Model download lands in `$SNAP_USER_COMMON`, not `~/.insightface`
   (the AppArmor constraint above); verify by path inspection after first run.
4. Regression on the labelled set: 4 `colin_powell` positives matched, 4 guests
   not, at default threshold.
5. Cache correctness: same-basename images in two different source trees produce
   two distinct cache entries and correct per-image scores (the collision bug).
6. Recipe isolation: run `--model buffalo_s`, then `buffalo_l`, then `buffalo_s`
   again; the third run hits cache and agrees with the first.
7. Reference invalidation: add a photo to a person's folder, confirm re-enroll
   without `--reset`.
8. `./scripts/run_sample.sh` still works via the legacy folder convention.
9. `pytest` green, including rewritten `test_config.py` without the reload hack.

## Out of scope

- PyPI publication and Docker images (snap is the blessed channel).
- GPU / non-CPU execution providers.
- Guided first-run folder scaffolding (considered, dropped: `--of` removes the
  need).
- Rewriting the matching algorithm; recall-over-precision stance is unchanged.
