"""Tests for config loading and override precedence (config.py)."""

from __future__ import annotations

import json

from get_my_snaps.config import Config, Paths


def test_defaults_are_recall_first():
    cfg = Config()
    assert cfg.threshold == 0.28
    assert cfg.model_name == "buffalo_l"


def test_json_file_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"threshold": 0.5, "unknown_key": 1}))

    cfg = Config.load(cfg_file)

    assert cfg.threshold == 0.5          # known key applied
    assert not hasattr(cfg, "unknown_key")  # unknown key ignored
    assert cfg.model_name == "buffalo_l"    # untouched default preserved


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


def test_paths_anchor_on_cwd(tmp_cwd):
    # Paths anchors on the working directory at import time, so reload the
    # module after chdir to observe the new root.
    import importlib

    import get_my_snaps.config as config_module

    config_module = importlib.reload(config_module)
    paths = config_module.Paths()
    assert paths.input == tmp_cwd / "input"
    assert paths.output == tmp_cwd / "output"
    assert paths.manifest == tmp_cwd / "output" / "manifest.csv"
