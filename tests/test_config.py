"""Tests for config loading and override precedence (config.py)."""

from __future__ import annotations

import json
from pathlib import Path

from get_my_snaps.config import (
    Config,
    Paths,
    default_central_dir,
    default_model_root,
)


def test_defaults_are_recall_first():
    cfg = Config()
    assert cfg.threshold == 0.28
    assert cfg.model_name == "buffalo_s"


def test_json_file_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"threshold": 0.5, "unknown_key": 1}))

    cfg = Config.load(cfg_file)

    assert cfg.threshold == 0.5          # known key applied
    assert not hasattr(cfg, "unknown_key")  # unknown key ignored
    assert cfg.model_name == "buffalo_s"    # untouched default preserved


def test_with_overrides_ignores_none():
    cfg = Config()
    out = cfg.with_overrides(threshold=None, batch_size=42)
    assert out.threshold == cfg.threshold  # None left it alone
    assert out.batch_size == 42            # explicit value applied


def test_precedence_flags_beat_file(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"threshold": 0.4}))

    cfg = Config.load(cfg_file).with_overrides(threshold=0.6)

    assert cfg.threshold == 0.6  # CLI flag wins over file


def test_paths_for_run_layout(tmp_path):
    src = tmp_path / "input"
    out = tmp_path / "out"
    p = Paths.for_run(source=src, out=out, central=tmp_path / "c", model_root=tmp_path / "m")
    assert p.source == src
    assert p.state_dir == out / ".gms"
    assert p.checkpoint == out / ".gms" / "checkpoint.json"
    assert p.embeddings_cache == out / ".gms" / "embeddings"
    assert p.manifest == out / "manifest.csv"
    assert p.candidates == out / "candidates.csv"


def test_default_model_root_reuses_insightface(tmp_path, monkeypatch):
    legacy = tmp_path / ".insightface"
    (legacy / "models").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("GMS_MODEL_DIR", raising=False)
    monkeypatch.delenv("SNAP_USER_COMMON", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert default_model_root() == legacy


def test_default_central_dir_from_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_CACHE_DIR", raising=False)
    monkeypatch.delenv("SNAP_USER_COMMON", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_central_dir() == tmp_path / "get-my-snaps"


def test_central_dir_snap_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CACHE_DIR", str(tmp_path / "forced"))
    monkeypatch.setenv("SNAP_USER_COMMON", str(tmp_path / "snap"))
    assert default_central_dir() == tmp_path / "forced"
