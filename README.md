# get-my-snaps

[![CI](https://github.com/immortalcodes/get-my-snaps/actions/workflows/ci.yml/badge.svg)](https://github.com/immortalcodes/get-my-snaps/actions/workflows/ci.yml)
[![get-my-snaps](https://snapcraft.io/get-my-snaps/badge.svg)](https://snapcraft.io/get-my-snaps)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Local, fully-offline tool that finds every photo containing a specific person and
copies it into an output folder, preserving the original folder structure. It uses
face recognition ([InsightFace](https://github.com/deepinsight/insightface)
`buffalo_s`) running on CPU via ONNX Runtime. After a one-time model fetch
(~160 MB) it needs no internet and sends no data anywhere.

![demo](docs/demo/demo.svg)

*Demo: find one astronaut in a folder of 15 public-domain NASA photos (source: Wikimedia Commons). Replay it in a terminal with `asciinema play docs/demo/demo.cast`.*

## Install

**Snap (recommended)** — one command, no Python needed:

```bash
sudo snap install get-my-snaps
```

**From source (developers)**:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Quick start

Point it at a folder of photos and tell it whose face you're looking for:

```bash
# 'me.jpg' is a sample of the person to find
get-my-snaps ~/Pictures/wedding --of ~/me.jpg

# a whole folder of samples = one person (named after the folder)
get-my-snaps ~/Pictures --of ~/me/

# several bare photos = one person (renamed with --name)
get-my-snaps ~/Pictures --of a.jpg b.jpg --name me

# a folder of subfolders = one person per subfolder
get-my-snaps ~/Pictures --of reference/

# just see what would match, without copying
get-my-snaps ~/Pictures --of ~/me.jpg --dry-run
```

Matches land in `./get-my-snaps-out/` (change with `-o <dir>`). Fully offline
after the first run, which downloads the model once.

## What it does

1. Learns a person's face from the `--of` photo(s) or folder.
2. **Scores the quality** of the reference set and tells you to add or replace
   photos if it's too weak to match reliably.
3. Scans every image under the source folder (JPG, JPEG, PNG, HEIC/HEIF, WEBP,
   BMP, TIFF).
4. **Copies** any photo containing the selected person into the output folder,
   mirroring the source tree, and writes an audit trail to
   `<out>/manifest.csv`.
5. Ranks **every** image by best match score into `<out>/candidates.csv`, so you
   can spot likely photos even if none cross the threshold.

It favors **recall over precision**: a deliberately low match threshold means it
rarely misses a relevant photo, at the cost of some false positives (easy to
delete afterwards).

## Legacy layout (still supported)

If you prefer folders to one-liners, the original convention still works from a
directory containing `input/` and `reference/<name>/`:

```bash
get-my-snaps                 # interactive: pick person(s), uses input/ + reference/
get-my-snaps --persons madhur priya
```

`./scripts/run_sample.sh` in the repo performs a non-destructive demo run of
this flow.

## Output & cache layout

A run is self-contained and disposable. Everything for a single run lives inside
the output dir; shared state (models + reference prototypes) lives centrally so
it's never recomputed across runs.

```
<out>/                         # e.g. ./get-my-snaps-out
├── manifest.csv               # audit trail of every matched, copied photo
├── candidates.csv             # every source image ranked by best match score
└── .gms/
    ├── checkpoint.json        # resume state (which images were processed)
    └── embeddings/            # per-image face embeddings, keyed by recipe
        └── <recipe>/xx/<digest>.npz

<central>/                     # $SNAP_USER_COMMON, or ~/.cache/get-my-snaps
├── models/<model>/            # InsightFace weights, fetched once
└── references/<recipe>/<person>.json
```

- Delete `<out>` to wipe all run-specific state and start clean.
- The **recipe** (model + detection settings) is part of every cache key, so
  switching `--model` never reuses incompatible embeddings.
- Reference prototypes are cached centrally and keyed on the sample set, so
  adding a photo to a person's references re-enrolls just that person.

## Options

| Flag | Default | Purpose |
|------|---------|---------|
| `SOURCE` | `./input` if present | Directory of photos to scan. |
| `-o/--out` | `./output` (legacy) / `./get-my-snaps-out` | Where matches + state go. |
| `--of PATH...` | *(none)* | Reference photo(s)/folder(s): folder w/ images = one person; folder of subfolders = one person each; bare files = one person. |
| `--name NAME` | first stem | Rename the single `--of` person. |
| `--persons NAMES` | *(prompt)* | Pick specific `reference/<name>/` people. |
| `--threshold` | `0.28` | Cosine-similarity cutoff; lower = more recall. |
| `--batch-size` | `150` | Images between checkpoint saves. |
| `--model` | `buffalo_s` | Face model; `buffalo_l` is more accurate but ~2× the download. |
| `--config` | *(none)* | Path to a JSON config file of tunables. |
| `--dry-run` | off | Report matches without copying. |
| `--skip-ref-check` | off | Don't pause on a Poor reference rating. |
| `--reset` | off | Ignore/overwrite the checkpoint. |
| `--reset-embeddings` | off | Also clear cached input embeddings. |
| `--reset-refs` | off | Also force re-enrollment of reference people. |
| `--verbose` | off | Show detailed logs. |

### Config file

Every tunable has a default in `src/get_my_snaps/config.py`. To change several at
once without long command lines, put them in a JSON file and pass `--config`:

```json
{
  "threshold": 0.16,
  "model_name": "buffalo_l",
  "min_face_frac": 0.02,
  "candidates_top_n": 30
}
```

Precedence is: **defaults < config file < command-line flags**. Available keys
match the fields in `src/get_my_snaps/config.py`.

## Embedding cache = instant re-scoring

The slow part (detecting + embedding faces) runs **once per image** and is cached
in `<out>/.gms/embeddings/`. Changing `--threshold` or `--persons` and re-running
re-scores from the cache in seconds. This is the intended tuning loop: run once,
then adjust the threshold quickly using `candidates.csv` as your guide.

## Tuning the threshold (important for small faces)

Face-recognition similarity drops when a person is **small or distant** in a
photo (e.g. wide group shots). Two things help:

1. **Use tight reference crops.** Close-up photos where the face fills much of
   the frame produce a much stronger prototype and better score separation.
2. **Read `<out>/candidates.csv`** — it ranks all images by score. If your real
   matches cluster around, say, `0.18`, re-run with `--threshold 0.16` (fast,
   from cache) to include them, then delete any false positives by hand.

## Resuming after a crash

Progress is saved to `<out>/.gms/checkpoint.json` every batch. If the tool is
interrupted, just run the same command again — it skips already-processed images
and continues. Copies are idempotent (a file already present in the output is
never copied twice).

If you change the model or the source folder, the checkpoint is recognised as
incompatible and you're offered a fresh start, so you never resume against
the wrong embeddings.

## How matching works (brief)

- Each face becomes a 512-dimension embedding (a numeric fingerprint).
- Reference photos of one person are averaged into a single **prototype** vector.
- For each input image, every detected face is compared to the prototype using
  cosine similarity; if any face scores at or above `--threshold`, the photo is
  a match.

## Performance

Measured on the demo set above (15 images, longest side about 1,000 px), an Intel Core Ultra 7 265U laptop CPU, `buffalo_s`, no GPU:

| Run | Time |
|-----|------|
| Cold run (detect + embed 15 images, including startup) | ~3.4 s |
| Re-score from the embedding cache (`--threshold` change) | ~0.8 s |

Result on that set: **10 of the 11** photos of the target person matched, with **0 false positives** among the 4 other photos
(the miss scored 0.169, below the 0.28 threshold; it shows up in `candidates.csv`).
This is a small sample, not a benchmark. Full-size camera photos (12 MP+) take longer
to decode, so expect more per image than shown here.

## Notes & limitations

- CPU-only by design. See [Performance](#performance) for measured timings.
- Very small, blurry, heavily side-profile, or sunglasses-obscured faces are
  harder to match — reflected in the reference quality score. In a quick test on
  1960s film photos where the subject often wears a helmet, a single reference
  photo matched nothing, so use several clear, front-facing references for
  difficult material.
- The pretrained models are available for **non-commercial research**; the tool
  downloads them on first use rather than bundling them.
- Originals are **copied, not moved**; the source folder is never modified.
