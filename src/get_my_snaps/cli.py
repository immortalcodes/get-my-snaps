#!/usr/bin/env python3
"""
cli.py -- CLI orchestration.

Copies every photo containing a chosen reference person into an output folder,
mirroring the source's folder tree. Works two ways:

  get-my-snaps ~/Pictures --of ~/me.jpg        # ad-hoc person reference
  get-my-snaps                                 # legacy input//reference/ layout

Fully offline after a one-time model fetch (InsightFace, CPU). This module is
deliberately thin: it wires together the specialised modules and owns nothing
but the command-line flow.

    config.py          all tunables + paths
    console.py         user-facing output/prompts
    cache.py           embedding/prototype keying (provenance)
    face_engine.py     image decoding + face detection/embeddings
    reference_check.py reference enrollment, quality, --of resolution
    matching.py        checkpoint, embedding cache, scoring, copy, reports, loop

Pipeline: resolve source+references -> enroll/load people -> match in resumable
batches -> copy matches + write reports.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import console
from .cache import recipe, samples_key
from .config import Config, Paths
from .face_engine import FaceEngine, model_available, is_supported_image
from .matching import Checkpoint, RunStats, process_images, write_candidates_report
from .reference_check import (
    PersonReference,
    format_report,
    enroll_person,
    load_cached_person,
    resolve_ref_items,
    save_person,
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
        prog="get-my-snaps",
        description=(
            "Find every photo in a source folder that contains a chosen person "
            "and copy it into an output folder, preserving the folder tree. "
            "Fully offline, CPU-only face recognition."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  get-my-snaps ~/Pictures/wedding --of ~/me.jpg   find few photos of\n"
            "                                                  one person from a folder\n"
            "  get-my-snaps ~/Pictures --of ~/me/              folder = all samples of\n"
            "                                                  one person 'me'\n"
            "  get-my-snaps --of a.jpg b.jpg --name me         bare photos = one person\n"
            "  get-my-snaps ~/Pictures --of reference/         folder of subfolders =\n"
            "                                                  one person each\n"
            "  get-my-snaps --persons madhur priya             legacy reference/<name>/\n"
            "  get-my-snaps --dry-run                          report matches, copy nothing\n"
        ),
    )

    loc = p.add_argument_group("source & output")
    loc.add_argument("source", nargs="?", type=Path, default=None,
                     help="Directory of photos to scan (default: ./input if present).")
    loc.add_argument("-o", "--out", type=Path, default=None,
                     help="Output dir. Default: ./output for the legacy input//reference/ "
                          "layout, else ./get-my-snaps-out.")

    people = p.add_argument_group("people (pick ONE of --of or --persons)")
    people.add_argument("--of", dest="of_refs", nargs="+", type=Path, default=None,
                        help="Reference photo(s)/folder(s) for the person(s) to find. "
                             "A folder with images = one person; a folder with only "
                             "subfolders = one person each; multiple bare files = one "
                             "person (renamed with --name).")
    people.add_argument("--name", default=None,
                        help="Name for the single --of person (bare files or a folder).")
    people.add_argument("--persons", nargs="*", default=None,
                        help="Person folder names under reference/ to filter for "
                             "(legacy selection; skips the picker).")

    tuning = p.add_argument_group("tuning (override the config file / defaults)")
    tuning.add_argument("--threshold", type=float, default=None,
                        help="Cosine-similarity match cutoff; lower = higher recall.")
    tuning.add_argument("--batch-size", type=int, default=None,
                        help="Images processed per checkpoint flush.")
    tuning.add_argument("--model", choices=["buffalo_l", "buffalo_s"], default=None,
                        help="Face model; buffalo_s is the default (faster).")
    tuning.add_argument("--config", type=Path, default=None,
                        help="Path to a JSON config file of tunables.")

    behaviour = p.add_argument_group("behaviour")
    behaviour.add_argument("--dry-run", action="store_true",
                           help="Report matches without copying anything.")
    behaviour.add_argument("--skip-ref-check", action="store_true",
                           help="Don't pause even if a selected person rates Poor.")
    behaviour.add_argument("--reset", action="store_true",
                           help="Ignore/overwrite the checkpoint for this run.")
    behaviour.add_argument("--reset-embeddings", action="store_true",
                           help="Also clear cached input embeddings.")
    behaviour.add_argument("--reset-refs", action="store_true",
                           help="Also force re-enrollment of reference people.")
    behaviour.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p


# ----------------------------------------------------------------------------
# Small orchestration steps
# ----------------------------------------------------------------------------
def default_out_dir(source: Path, cwd: Path) -> Path:
    """Choose the output dir, preserving the legacy layout when detected."""
    if source == cwd / "input" and (cwd / "reference").is_dir():
        return cwd / "output"
    return cwd / "get-my-snaps-out"


def resolve_source(args, cwd: Path) -> Path:
    """The explicit SOURCE, or ./input when present, else a fatal error."""
    if args.source is not None:
        source = args.source
        if not source.is_dir():
            console.error(console.missing_source(source))
            raise SystemExit(2)
        return source
    source = cwd / "input"
    if not source.is_dir():
        console.error(console.missing_input(source))
        raise SystemExit(2)
    return source


def apply_resets(args, paths: Paths) -> None:
    """Honour the reset flags before anything reads those files."""
    if args.reset:
        paths.checkpoint.unlink(missing_ok=True)
    if args.reset_embeddings:
        shutil.rmtree(paths.embeddings_cache, ignore_errors=True)
    if args.reset_refs:
        shutil.rmtree(paths.central / "references", ignore_errors=True)


def _reference_people(
    engine: FaceEngine, cfg: Config, paths: Paths, reference_dir: Path
) -> Dict[str, PersonReference]:
    """Enroll/load every reference/<name>/ person into the shared cache.

    Each person is cached under the current recipe, content-keyed so a changed
    photo set re-enrolls only that person.
    """
    rec = recipe(cfg.model_name, cfg.det_size, cfg.det_thresh, cfg.max_long_edge)
    people: Dict[str, PersonReference] = {}
    if not reference_dir.is_dir():
        return people
    for pdir in sorted(reference_dir.iterdir()):
        if not pdir.is_dir():
            continue
        images = sorted(
            p for p in pdir.rglob("*") if p.is_file() and is_supported_image(p)
        )
        if not images:
            continue
        name = pdir.name
        key = samples_key(images)
        person = load_cached_person(paths.central, rec, name, key)
        if person is None:
            person = enroll_person(name, images, engine, cfg)
            save_person(person, paths.central, rec)
        people[name] = person
    return people


def _of_people(
    engine: FaceEngine,
    cfg: Config,
    paths: Paths,
    of_refs: List[Path],
    name_override: Optional[str],
) -> Dict[str, PersonReference]:
    """Enroll/load the people described by --of (no picker)."""
    rec = recipe(cfg.model_name, cfg.det_size, cfg.det_thresh, cfg.max_long_edge)
    resolved = resolve_ref_items(of_refs)

    if name_override is not None:
        if len(resolved) != 1:
            console.error("--name only applies when --of describes exactly one person.")
            raise SystemExit(2)
        only = next(iter(resolved))
        resolved = {name_override: resolved[only]}

    people: Dict[str, PersonReference] = {}
    for name, images in resolved.items():
        key = samples_key(images)
        person = load_cached_person(paths.central, rec, name, key)
        if person is None:
            person = enroll_person(name, images, engine, cfg)
            save_person(person, paths.central, rec)
        people[name] = person
    return people


def load_and_pick_people(
    engine: FaceEngine,
    cfg: Config,
    paths: Paths,
    args,
    reference_dir: Path,
) -> Tuple[Dict[str, PersonReference], List[str]]:
    """Resolve the reference people and the list of whom to filter for."""
    if args.of_refs:
        people = _of_people(engine, cfg, paths, args.of_refs, args.name)
        if not people:
            console.error("No usable reference people. Aborting.")
            raise SystemExit(2)
        persons = list(people.keys())
    else:
        people = _reference_people(engine, cfg, paths, reference_dir)
        if not people:
            console.error(console.empty_reference(reference_dir))
            raise SystemExit(2)
        console.step(console.ENROLLING if args.persons is not None else "")
        persons = choose_persons(people, args.persons)

    console.info(format_report(people))
    return people, persons


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
    rec = recipe(cfg.model_name, cfg.det_size, cfg.det_thresh, cfg.max_long_edge)
    sources = str(paths.source.resolve())
    ckpt = None if reset else Checkpoint.load(paths.checkpoint)
    if ckpt is not None and not ckpt.matches_config(persons, cfg.threshold, rec, sources):
        console.step("Checkpoint was made with different persons/model/source.")
        if console.prompt("Start fresh for this configuration? [Y/n]: ") in ("", "y", "yes"):
            ckpt = None
    if ckpt is None:
        ckpt = Checkpoint(
            persons=persons, threshold=cfg.threshold,
            recipe=rec, sources=sources, processed={},
        )
    return ckpt


def discover_pending(ckpt: Checkpoint, paths: Paths) -> tuple[List[Path], int]:
    """Return (images still to process, count already processed)."""
    all_images = sorted(
        p for p in paths.source.rglob("*") if p.is_file() and is_supported_image(p)
    )
    todo = [
        p for p in all_images
        if p.relative_to(paths.source).as_posix() not in ckpt.processed
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
        f"to {paths.out}.{suffix}"
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

    cfg = Config.load(args.config).with_overrides(
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_name=args.model,
    )
    cwd = Path.cwd()
    source = resolve_source(args, cwd)
    out = args.out or default_out_dir(source, cwd)
    paths = Paths.for_run(source=source, out=out)
    apply_resets(args, paths)

    if not model_available(paths.model_root, cfg.model_name):
        console.warn(
            console.MODEL_FETCH_NOTICE
            + f"\n    -> {cfg.model_name} model will be stored under "
            f"{paths.model_root / 'models'}"
        )

    engine = FaceEngine(
        model_name=cfg.model_name,
        det_size=cfg.det_size,
        det_thresh=cfg.det_thresh,
        model_root=paths.model_root,
    )

    reference_dir = cwd / "reference"
    people, persons = load_and_pick_people(engine, cfg, paths, args, reference_dir)

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
