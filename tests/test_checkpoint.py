"""Tests for the resumable Checkpoint (matching.py)."""

from __future__ import annotations

from get_my_snaps.matching import Checkpoint


def test_save_and_load_roundtrip(tmp_path):
    ckpt = Checkpoint(persons=["alice", "bob"], threshold=0.3)
    ckpt.processed["photos/a.jpg"] = {"status": "matched", "best": {"alice": 0.9}}

    path = tmp_path / ".gms_checkpoint.json"
    ckpt.save(path)
    loaded = Checkpoint.load(path)

    assert loaded is not None
    assert loaded.persons == ["alice", "bob"]
    assert loaded.threshold == 0.3
    assert loaded.processed["photos/a.jpg"]["status"] == "matched"


def test_load_missing_returns_none(tmp_path):
    assert Checkpoint.load(tmp_path / "does-not-exist.json") is None


def test_matches_config_is_order_insensitive():
    ckpt = Checkpoint(persons=["bob", "alice"], threshold=0.28)
    assert ckpt.matches_config(["alice", "bob"], 0.28)
    assert not ckpt.matches_config(["alice"], 0.28)
    assert not ckpt.matches_config(["alice", "bob"], 0.5)


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    path = tmp_path / ".gms_checkpoint.json"
    Checkpoint(persons=["a"], threshold=0.1).save(path)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
