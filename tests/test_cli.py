"""Pure-logic tests for CLI helpers (no engine, no network)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from get_my_snaps.cli import default_out_dir, resolve_source


def test_default_out_detects_legacy_layout(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "reference").mkdir()
    assert default_out_dir(tmp_path / "input", tmp_path) == tmp_path / "output"


def test_default_out_fresh_source(tmp_path):
    (tmp_path / "input").mkdir()
    # input present but no reference/ -> not the legacy layout
    assert default_out_dir(tmp_path / "input", tmp_path) == tmp_path / "get-my-snaps-out"


def test_default_out_explicit_source(tmp_path):
    pics = tmp_path / "Pictures"
    pics.mkdir()
    (tmp_path / "reference").mkdir()
    # a non-default source dir -> fresh name even though reference/ exists
    assert default_out_dir(pics, tmp_path) == tmp_path / "get-my-snaps-out"


def test_resolve_source_explicit_dir(tmp_path):
    pics = tmp_path / "Pictures"
    pics.mkdir()
    args = Namespace(source=pics)
    assert resolve_source(args, tmp_path) == pics


def test_resolve_source_defaults_to_input(tmp_path):
    (tmp_path / "input").mkdir()
    args = Namespace(source=None)
    assert resolve_source(args, tmp_path) == tmp_path / "input"


def test_resolve_source_missing_aborts(tmp_path, capsys):
    args = Namespace(source=None)
    with pytest.raises(SystemExit) as e:
        resolve_source(args, tmp_path)
    assert e.value.code == 2


def test_resolve_source_not_a_dir_aborts(tmp_path, capsys):
    f = tmp_path / "afile"
    f.touch()
    with pytest.raises(SystemExit) as e:
        resolve_source(Namespace(source=f), tmp_path)
    assert e.value.code == 2
