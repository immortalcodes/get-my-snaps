"""Pure-logic tests for reference_check helpers (no engine, no ML)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from get_my_snaps.reference_check import (
    PersonReference,
    load_cached_person,
    person_cache_path,
    references_dir,
    resolve_ref_items,
    save_person,
)


def _mkdir_images(base: Path, *names: str) -> None:
    for n in names:
        (base / n).touch()


def test_ref_folder_with_images_is_one_person(tmp_path):
    me = tmp_path / "me"
    me.mkdir()
    _mkdir_images(me, "a.jpg", "b.jpg")
    people = resolve_ref_items([me])
    assert set(people) == {"me"}
    assert len(people["me"]) == 2


def test_ref_bare_files_one_person(tmp_path):
    _mkdir_images(tmp_path, "a.jpg", "b.jpg")
    people = resolve_ref_items([tmp_path / "a.jpg", tmp_path / "b.jpg"])
    assert set(people) == {"a"}
    assert len(people["a"]) == 2


def test_ref_folder_of_subdirs_is_many_people(tmp_path):
    base = tmp_path / "ref"
    (base / "ann").mkdir(parents=True)
    (base / "bob").mkdir()
    _mkdir_images(base / "ann", "1.jpg")
    _mkdir_images(base / "bob", "2.jpg")
    people = resolve_ref_items([base])
    assert set(people) == {"ann", "bob"}


def test_ref_mixed_dir_person_plus_subdirs(tmp_path):
    base = tmp_path / "fam"
    base.mkdir()
    _mkdir_images(base, "direct.jpg")  # direct image -> person "fam"
    (base / "kid").mkdir()
    _mkdir_images(base / "kid", "k.jpg")
    people = resolve_ref_items([base])
    assert set(people) == {"fam", "kid"}


def test_ref_ignores_unsupported(tmp_path):
    base = tmp_path / "me"
    base.mkdir()
    (base / "notes.txt").touch()
    (base / "a.jpg").touch()
    people = resolve_ref_items([base])
    assert people["me"] == [base / "a.jpg"]  # txt dropped, jpg kept


def test_references_dir_and_person_path(tmp_path):
    assert references_dir(tmp_path / "c", "abc") == tmp_path / "c" / "references" / "abc"
    assert person_cache_path(tmp_path / "c", "abc", "me") == (
        tmp_path / "c" / "references" / "abc" / "me.json"
    )


def test_person_cache_roundtrip_and_invalidation(tmp_path):
    central = tmp_path / "c"
    person = PersonReference(
        name="me",
        prototype=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        samples_key="key-A",
        usable_images=2,
        rating="Fair",
    )
    save_person(person, central, "recipe1")
    # matching key loads
    loaded = load_cached_person(central, "recipe1", "me", "key-A")
    assert loaded is not None
    assert loaded.samples_key == "key-A"
    assert np.allclose(loaded.prototype, person.prototype)
    # a different key (samples changed) invalidates
    assert load_cached_person(central, "recipe1", "me", "key-B") is None
    # a different recipe is a different namespace
    assert load_cached_person(central, "recipe2", "me", "key-A") is None
