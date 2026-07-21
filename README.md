# get-my-snaps

Local, fully-offline tool that finds every photo containing a specific person and
copies it into `output/`, preserving the original folder structure. It uses
face recognition ([InsightFace](https://github.com/deepinsight/insightface)
`buffalo_l`) running on CPU via ONNX Runtime. After a one-time model download
(~280 MB) it needs no internet and sends no data anywhere.

## What it does

1. Learns each person's face from `reference/<name>/` photos.
2. **Scores the quality** of each person's reference set and tells you to add or
   replace photos if the set is too weak to match reliably.
3. Lets you pick which person(s) to filter for.
4. Scans every image under `input/` (JPG, JPEG, PNG, HEIC/HEIF, WEBP, BMP, TIFF).
5. **Copies** any photo containing a selected person into `output/`, mirroring the
   input folder tree, and writes an audit trail to `output/manifest.csv`.
6. Ranks **every** image by best match score into `output/candidates.csv`, so you
   can spot likely photos even if none cross the threshold.

It favors **recall over precision**: a deliberately low match threshold means it
rarely misses a relevant photo, at the cost of some false positives (easy to
delete afterwards).

### Embedding cache = instant re-scoring

The slow part (detecting + embedding faces) runs **once per image ever** and is
cached in `.gms_embeddings/`. Changing `--threshold` or `--persons` and re-running
re-scores from the cache in **seconds** instead of re-processing everything. This
is the intended tuning loop: run once, then adjust the threshold quickly using the
`candidates.csv` ranking as your guide.

## Folder layout

```
get-my-snaps/
├── input/                 # your source photos, in any nested folder structure
├── reference/             # one subfolder per person: reference/<name>/*.jpg|*.heic|...
├── output/                # matches are copied here, mirroring input/ (auto-created)
├── pyproject.toml         # packaging, deps, `get-my-snaps` console-script entry point
├── requirements.txt       # convenience shim → installs this project (editable)
├── src/
│   └── get_my_snaps/      # the importable package
│       ├── cli.py             # CLI orchestration (argparse + wiring)
│       ├── config.py          # all tunables + filesystem paths
│       ├── console.py         # user-facing output / prompts
│       ├── face_engine.py     # image loading + face detection/embeddings
│       ├── reference_check.py # reference enrollment + quality scoring
│       ├── matching.py        # checkpoint, embedding cache, scoring, copy, reports
│       ├── __main__.py        # enables `python -m get_my_snaps`
│       └── __init__.py
├── scripts/
│   ├── run_sample.sh      # one-command setup + demo run
│   └── profile_run.py     # dev-only performance profiler
└── tests/                 # pytest suite
```

## Quick start

The fastest way to see it work (creates the venv, installs the package, runs a
non-destructive dry run for every person in `reference/`):

```bash
./scripts/run_sample.sh          # dry run: reports matches, copies nothing
./scripts/run_sample.sh --real   # actually copy matches into output/
```

## Setup (manual)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .        # or  -e ".[dev]"  to also get pytest
```

Installing the project (`-e .`, editable) puts a `get-my-snaps` command on your
PATH and lets code edits take effect without reinstalling.

Then create a folder per person under `reference/` and drop 3–5 clear,
front-facing photos of that person into each:

```
reference/
├── madhur/
│   ├── photo1.jpg
│   └── photo2.heic
└── priya/
    └── ...
```

## Usage

Interactive (recommended the first time):

```bash
get-my-snaps                       # if the venv is activated
.venv/bin/get-my-snaps             # or call it directly
.venv/bin/python -m get_my_snaps   # equivalent module form
```

You'll see a **reference quality report**, then a prompt to choose who to filter
for. If a selected person's references rate **Poor**, it pauses and asks whether
to continue.

Non-interactive / scripted:

```bash
# Filter for specific people, skip prompts
get-my-snaps --persons madhur priya --skip-ref-check

# See what would match without copying anything
get-my-snaps --persons madhur --dry-run

# Start completely fresh (ignore checkpoint + reference cache)
get-my-snaps --reset
```

### Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--threshold` | `0.28` | Cosine-similarity match cutoff. Lower = more recall (more false positives). Raise toward `0.35–0.4` if you get too many wrong matches. |
| `--batch-size` | `150` | How many images between checkpoint saves. |
| `--model` | `buffalo_l` | Face model. `buffalo_s` is faster but less accurate. |
| `--config` | *(none)* | Path to a JSON config file of tunables (see below). |
| `--persons` | *(prompt)* | Person folder names to filter for; skips the interactive picker. |
| `--reset` | off | Ignore/overwrite the checkpoint and reference cache (keeps the embedding cache). |
| `--reset-embeddings` | off | Also clear cached input embeddings. Use if the input images changed. |
| `--dry-run` | off | Report matches without copying. |
| `--skip-ref-check` | off | Don't pause even if a selected person rates Poor. |
| `--verbose` | off | Show detailed logs. |

### Config file

Every tunable has a default in `src/get_my_snaps/config.py`. To change several at once without
long command lines, put them in a JSON file and pass `--config`:

```json
{
  "threshold": 0.16,
  "model_name": "buffalo_s",
  "min_face_frac": 0.02,
  "candidates_top_n": 30
}
```

```bash
get-my-snaps --config my.json
```

Precedence is: **defaults < config file < command-line flags**, so a flag like
`--threshold` always wins over the file. Available keys match the fields in
`src/get_my_snaps/config.py` (matching, detection/preprocessing, reference-quality thresholds,
and reporting).

## Tuning the threshold (important for small faces)

Face-recognition similarity drops when a person is **small or distant** in a
photo (e.g. wide group shots). In that case even correct matches may score well
below the default `0.28`. Two things help:

1. **Use tight reference crops.** Close-up photos where the face fills much of the
   frame produce a much stronger prototype and better score separation. The
   reference quality report will warn you when your reference faces are too small.
2. **Read `output/candidates.csv`** — it ranks all images by score. If your real
   matches cluster around, say, 0.18, re-run with `--threshold 0.16` (fast, from
   cache) to include them, then delete any false positives by hand.

## Resuming after a crash

Progress is saved to `.gms_checkpoint.json` every batch. If the tool is
interrupted, just run the same command again — it skips already-processed images
and continues where it left off. Copies are idempotent (a file already present in
`output/` is never copied twice).

To change who you're filtering for or the threshold, the tool detects the
mismatch and offers to start fresh.

## How matching works (brief)

- Each face becomes a 512-dimension embedding (a numeric fingerprint).
- Reference photos of one person are averaged into a single **prototype** vector.
- For each input image, every detected face is compared to each prototype using
  cosine similarity; if any face scores at or above `--threshold`, the photo is a
  match.

## Notes & limitations

- CPU-only by design. Expect roughly 1–3 seconds per image; 135 photos take a few
  minutes, larger sets scale linearly. Use `--model buffalo_s` to go faster.
- Very small, blurry, heavily side-profile, or sunglasses-obscured faces are
  harder to match — reflected in the reference quality score.
- Originals are **copied, not moved**; `input/` is never modified.
