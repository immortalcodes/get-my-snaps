#!/usr/bin/env bash
# run_sample.sh — dry-run get-my-snaps for every person under reference/.
#   ./scripts/run_sample.sh          # dry run (nothing copied)
#   ./scripts/run_sample.sh --real   # actually copy matches into output/
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=".venv"
PY="$VENV/bin/python"

# Create the venv and install the package once (skipped on later runs).
if [[ ! -x "$PY" ]]; then
  echo "==> Creating virtualenv and installing dependencies ..."
  python3 -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -e .
fi

if [[ -z "$(find reference -mindepth 1 -maxdepth 1 -type d 2>/dev/null)" ]]; then
  echo "!! reference/ has no <name> subfolders (e.g. reference/alice/*.jpg)." >&2
  exit 1
fi
if [[ -z "$(find input -type f 2>/dev/null)" ]]; then
  echo "!! input/ has no images." >&2
  exit 1
fi

mapfile -t PERSONS < <(find reference -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "==> Reference people: ${PERSONS[*]}"

MODE_ARGS=(--dry-run)
if [[ "${1:-}" == "--real" ]]; then
  MODE_ARGS=()
  echo "==> REAL run: matches WILL be copied into output/"
else
  echo "==> DRY RUN: nothing copied (pass --real to copy)."
fi

"$PY" -m get_my_snaps --persons "${PERSONS[@]}" --skip-ref-check "${MODE_ARGS[@]}"

echo "==> Done. See output/candidates.csv for the full ranking."
