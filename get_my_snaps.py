#!/usr/bin/env python3
"""
get_my_snaps.py — CLI orchestration.

Copies every input photo that contains a chosen reference person into ``output/``,
mirroring the input folder tree. Fully offline (InsightFace, CPU).

This module is deliberately thin: it wires together the specialised modules and
owns nothing but the command-line flow.

    config.py          all tunables + paths
    console.py         user-facing output/prompts
    face_engine.py     image decoding + face detection/embeddings
    reference_check.py reference enrollment + quality scoring
    matching.py        checkpoint, embedding cache, scoring, copy, reports, loop

Pipeline: enroll references -> pick people -> discover input -> match in resumable
batches -> copy matches + write reports. Recall is favoured over precision (low
default threshold) so relevant photos are rarely missed.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import console
from config import Config, Paths
from face_engine import FaceEngine, is_supported_image
from matching import Checkpoint, RunStats, process_images, write_candidates_report
from reference_check import (
    PersonReference,
    enroll_references,
    format_report,
    load_reference_cache,
    save_reference_cache,
)

log = logging.getLogger("get_my_snaps")


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show defaults in --help *and* keep the epilog's manual line breaks."""


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    Flags that mirror Config fields default to None here so we can tell "user
    left it alone" (use config/default) from "user set it" (override config).
    """
    p = argparse.ArgumentParser(
        prog="get_my_snaps.py",
        description=(
            "Find every photo in input/ that contains a chosen person (from "
            "reference/<name>/) and copy it into output/, preserving the folder "
            "tree. Fully offline, CPU-only face recognition."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  get_my_snaps.py                         interactive, all defaults\n"
            "  get_my_snaps.py --persons madhur priya  skip the picker\n"
            "  get_my_snaps.py --dry-run               report matches, copy nothing\n"
            "  get_my_snaps.py --threshold 0.16        widen recall (re-scores fast)\n"
            "  get_my_snaps.py --config my.json        load tunables from a file\n"
        ),
    )

    tuning = p.add_argument_group("tuning (override the config file / defaults)")
    tuning.add_argument("--threshold", type=float, default=None,
                        help="Cosine-similarity match cutoff; lower = higher recall.")
    tuning.add_argument("--batch-size", type=int, default=None,
                        help="Images processed per checkpoint flush.")
    tuning.add_argument("--model", choices=["buffalo_l", "buffalo_s"], default=None,
                        help="Face model; buffalo_s is faster but less accurate.")
    tuning.add_argument("--config", type=Path, default=None,
                        help="Path to a JSON config file of tunables.")

    selection = p.add_argument_group("selection")
    selection.add_argument("--persons", nargs="*", default=None,
                           help="Person folder names to filter for (skips the prompt).")

    behaviour = p.add_argument_group("behaviour")
    behaviour.add_argument("--dry-run", action="store_true",
                           help="Report matches without copying anything.")
    behaviour.add_argument("--skip-ref-check", action="store_true",
                           help="Don't pause even if a selected person rates Poor.")
    behaviour.add_argument("--reset", action="store_true",
                           help="Ignore/overwrite checkpoint + reference cache "
                                "(keeps the input embedding cache).")
    behaviour.add_argument("--reset-embeddings", action="store_true",
                           help="Also clear cached input embeddings (use if input changed).")
    behaviour.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p


# ----------------------------------------------------------------------------
# Small orchestration steps
# ----------------------------------------------------------------------------
def validate_workspace(paths: Paths) -> Optional[int]:
    """Return an exit code if the workspace isn't ready, else None."""
    if not paths.reference.is_dir() or not any(paths.reference.iterdir()):
        console.error(console.empty_reference(paths.reference))
        return 2
    if not paths.input.is_dir():
        console.error(console.missing_input(paths.input))
        return 2
    return None


def apply_resets(args: argparse.Namespace, paths: Paths) -> None:
    """Honour --reset / --reset-embeddings before anything reads those files."""
    if args.reset:
        paths.checkpoint.unlink(missing_ok=True)
        paths.reference_cache.unlink(missing_ok=True)
    if args.reset_embeddings and paths.embeddings_cache.exists():
        shutil.rmtree(paths.embeddings_cache)


def load_or_enroll_references(
    engine: FaceEngine, cfg: Config, paths: Paths, reset: bool
) -> Dict[str, PersonReference]:
    """Enroll reference people, using the on-disk cache unless --reset was given."""
    people = None if reset else load_reference_cache(paths.reference_cache)
    if people is None:
        console.step(console.ENROLLING)
        people = enroll_references(paths.reference, engine, cfg)
        save_reference_cache(people, paths.reference_cache)
    else:
        console.step(console.REF_CACHE_LOADED)
    console.info(format_report(people))
    return people


def choose_persons(
    people: Dict[str, PersonReference], preselected: Optional[List[str]]
) -> List[str]:
    """Resolve which people to filter for, via --persons or an interactive prompt."""
    usable = {n: p for n, p in people.items() if p.ok}
    if not usable:
        console.error("No usable reference people (no detectable faces). Aborting.")
        sys.exit(2)

    if preselected:
        chosen = [n for n in preselected if n in usable]
        for missing in (n for n in preselected if n not in usable):
            console.warn(f"Requested person '{missing}' has no usable reference; skipping.")
        if not chosen:
            console.error("None of the requested --persons are usable. Aborting.")
            sys.exit(2)
        return chosen

    names = list(usable.keys())
    console.step("Select the person(s) to filter for:")
    for i, name in enumerate(names, 1):
        console.detail(f"{i}. {name}  [{usable[name].rating}]")
    console.detail("a. ALL of the above")
    raw = console.prompt("Enter numbers (comma-separated) or 'a': ")

    if raw in ("a", "all", ""):
        return names
    chosen = []
    for tok in raw.replace(" ", "").split(","):
        # Menu is 1-indexed for humans; convert to 0-indexed list access.
        if tok.isdigit() and 1 <= int(tok) <= len(names):
            chosen.append(names[int(tok) - 1])
    if not chosen:
        console.error("No valid selection made. Aborting.")
        sys.exit(2)
    return chosen


def confirm_poor_references(
    people: Dict[str, PersonReference], persons: List[str], skip_check: bool
) -> bool:
    """Pause if any selected person rates Poor. Return False to abort the run."""
    poor = [n for n in persons if people[n].rating == "Poor"]
    if not poor or skip_check:
        return True
    console.warn(f"These selected people have POOR reference quality: {', '.join(poor)}")
    console.detail("Matching may miss photos or produce many false positives.")
    if console.prompt("   Proceed anyway? [y/N]: ") not in ("y", "yes"):
        console.info(console.ABORTED_POOR)
        return False
    return True


def bootstrap_checkpoint(
    persons: List[str], cfg: Config, paths: Paths, reset: bool
) -> Checkpoint:
    """Load a compatible checkpoint or start a fresh one for this configuration."""
    ckpt = None if reset else Checkpoint.load(paths.checkpoint)
    if ckpt is not None and not ckpt.matches_config(persons, cfg.threshold):
        console.step("Checkpoint was made with different persons/threshold.")
        if console.prompt("Start fresh for this configuration? [Y/n]: ") in ("", "y", "yes"):
            ckpt = None
    if ckpt is None:
        ckpt = Checkpoint(persons=persons, threshold=cfg.threshold, processed={})
    return ckpt


def discover_pending(ckpt: Checkpoint, paths: Paths) -> tuple[List[Path], int]:
    """Return (images still to process, count already processed)."""
    all_images = sorted(
        p for p in paths.input.rglob("*") if p.is_file() and is_supported_image(p)
    )
    todo = [
        p for p in all_images
        if p.relative_to(paths.input).as_posix() not in ckpt.processed
    ]
    return todo, len(all_images) - len(todo)


def report_summary(
    stats: RunStats, ckpt: Checkpoint, cfg: Config, paths: Paths, dry_run: bool
) -> None:
    """Print the final summary and the ranked candidates report."""
    matched = sum(1 for v in ckpt.processed.values() if v.get("status") == "matched")
    write_candidates_report(ckpt.processed, cfg.threshold, cfg, paths)

    suffix = " [dry-run: nothing copied]" if dry_run else ""
    console.success(
        f"Matched {matched} image(s); copied {stats.copied} new file(s) "
        f"to {paths.output}.{suffix}"
    )
    if not dry_run and matched:
        console.info(f"Audit trail: {paths.manifest}")
    if matched == 0:
        console.warn(
            f"No image crossed the threshold ({cfg.threshold}). See the ranked "
            f"near-misses above and in {paths.candidates} — re-run with a lower "
            "--threshold to include them (embeddings are cached, so it's fast)."
        )


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    paths = Paths()
    cfg = Config.load(args.config).with_overrides(
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_name=args.model,
    )

    bad = validate_workspace(paths)
    if bad is not None:
        return bad
    apply_resets(args, paths)

    engine = FaceEngine(
        model_name=cfg.model_name, det_size=cfg.det_size, det_thresh=cfg.det_thresh
    )

    people = load_or_enroll_references(engine, cfg, paths, args.reset)
    persons = choose_persons(people, args.persons)
    console.step(console.filtering_for(persons))

    if not confirm_poor_references(people, persons, args.skip_ref_check):
        return 0

    prototypes = {n: people[n].prototype for n in persons}
    ckpt = bootstrap_checkpoint(persons, cfg, paths, args.reset)

    todo, already = discover_pending(ckpt, paths)
    console.step(console.input_summary(already + len(todo), already, len(todo)))
    if not todo:
        console.info(console.NOTHING_NEW)
        return 0

    stats = process_images(engine, todo, prototypes, ckpt, cfg, paths, args.dry_run)
    report_summary(stats, ckpt, cfg, paths, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
