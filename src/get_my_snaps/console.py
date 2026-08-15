"""
console.py — one place for all user-facing output and prompts.

Why this exists: the tool used to scatter bare ``print()`` calls with inline
strings everywhere, which made the flow hard to read and impossible to restyle
or redirect. Routing every message through here means:

- consistent formatting (headings, warnings, steps all look the same),
- the wording lives in one file and is easy to review/translate,
- swapping to logging, JSON output, or a richer UI later touches only this file.

Keep the helpers dumb and side-effect-only (they print); keep decisions in the
caller.
"""

from __future__ import annotations

from typing import Iterable

_RULE = "=" * 64


# --- generic output helpers ----------------------------------------------
def heading(title: str) -> None:
    """A boxed section title."""
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def step(msg: str) -> None:
    """A top-level progress line (what the tool is about to do)."""
    print(f"\n{msg}")


def info(msg: str) -> None:
    print(msg)


def detail(msg: str) -> None:
    """Indented secondary line under a step/heading."""
    print(f"    {msg}")


def success(msg: str) -> None:
    print(f"\n✓ {msg}")


def warn(msg: str) -> None:
    print(f"\n⚠  {msg}")


def error(msg: str) -> None:
    print(f"ERROR: {msg}")


def prompt(msg: str) -> str:
    """Ask the user for input and return the trimmed, lower-cased answer.

    When stdin isn't a terminal (piped/scripted/snap runs that must never hang)
    we return the empty string, which callers treat as the default answer.
    """
    import sys

    if not sys.stdin.isatty():
        return ""
    return input(msg).strip().lower()


# --- flow messages (dynamic ones are builders, static ones are constants) -
ENROLLING = "Enrolling reference people and scoring quality..."
REF_CACHE_LOADED = "Loaded cached reference data."
NOTHING_NEW = "Nothing new to process. Done."
ABORTED_POOR = "Aborted. Improve the reference photos (see advice above) and re-run."
MODEL_FETCH_NOTICE = (
    "First run: downloading the face-recognition model to the shared cache "
    "(one time, then fully offline). This can take a moment depending on your "
    "connection."
)


def empty_reference(reference_dir) -> str:
    return (
        f"'{reference_dir}' is empty. Pass --of <photo-or-folder>, or add "
        "reference/<name>/ folders with photos of each person, then re-run."
    )


def missing_input(input_dir) -> str:
    return (
        f"'{input_dir}' was not found. Pass a source directory of photos to "
        "scan, e.g.  get-my-snaps ~/Pictures --of ~/me.jpg"
    )


def missing_source(source) -> str:
    return f"source path '{source}' is not a directory."


def filtering_for(persons: Iterable[str]) -> str:
    return f"Filtering for: {', '.join(persons)}"


def input_summary(total: int, already: int, todo: int) -> str:
    return f"Input images: {total} total | {already} already processed | {todo} to do"
