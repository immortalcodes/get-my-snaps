#!/usr/bin/env bash
#
# run_sample.sh — one-command way to try get-my-snaps end to end.
#
# It bootstraps an isolated Python virtualenv, installs dependencies (first run
# only), verifies you have reference + input photos, then does a non-destructive
# DRY RUN for every person found under reference/. Nothing is copied — it just
# shows the reference-quality report and the ranked candidate matches so you can
# see the tool working before committing to a real run.
#
# Usage (run from the repo root):
#   ./scripts/run_sample.sh            # dry run for all reference people
#   ./scripts/run_sample.sh --real     # actually copy matches into output/
#
set -euo pipefail

# Operate from the repository root (one level up from this scripts/ folder),
# regardless of where the script is called from. Data dirs (reference/, input/,
# output/) live at the repo root.
cd "$(dirname "$0")/.."

VENV=".venv"
PY="$VENV/bin/python"

# 1. Create the virtualenv on first use. A venv isolates our ML deps from the
#    system Python (and avoids PEP-668 "externally managed" install errors).
if [[ ! -x "$PY" ]]; then
  echo "==> Creating virtualenv in $VENV ..."
  python3 -m venv "$VENV"
fi

# 2. Install/refresh the package (editable) and its dependencies. pip is quiet
#    unless something goes wrong. Editable (-e) means code edits take effect
#    without reinstalling.
echo "==> Installing dependencies (first run downloads a lot; later runs are fast) ..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e .

# 3. Make sure there's something to work with.
if [[ ! -d reference ]] || [[ -z "$(find reference -mindepth 1 -maxdepth 1 -type d 2>/dev/null)" ]]; then
  echo "!! reference/ has no <name> subfolders. Add e.g. reference/alice/*.jpg and re-run." >&2
  exit 1
fi
if [[ -z "$(find input -type f 2>/dev/null | head -n 1)" ]]; then
  echo "!! input/ has no images. Drop some photos under input/ and re-run." >&2
  exit 1
fi

# 4. Collect every person folder name so the run is fully non-interactive.
mapfile -t PERSONS < <(find reference -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "==> Reference people detected: ${PERSONS[*]}"

# 5. Decide dry-run (default) vs real copy.
MODE_ARGS=(--dry-run)
if [[ "${1:-}" == "--real" ]]; then
  MODE_ARGS=()
  echo "==> REAL run: matches WILL be copied into output/"
else
  echo "==> DRY RUN: nothing will be copied (pass --real to copy). "
fi

# 6. Go. --skip-ref-check keeps it hands-free even if a reference set rates Poor.
echo "==> Running get_my_snaps ..."
"$PY" -m get_my_snaps --persons "${PERSONS[@]}" --skip-ref-check "${MODE_ARGS[@]}"

echo
echo "==> Done. See output/candidates.csv for the full ranking."
if [[ "${1:-}" != "--real" ]]; then
  echo "    Re-run with real copying:  ./run_sample.sh --real"
fi
