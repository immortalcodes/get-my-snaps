"""Pure-logic tests for face_engine helpers (no model, no inference)."""

from __future__ import annotations

from pathlib import Path

from get_my_snaps.face_engine import (
    SUPPORTED_EXTS,
    is_supported_image,
    model_available,
)


def test_is_supported_image_by_extension(tmp_path):
    for ext in (".jpg", ".JPG", ".heic", ".HEIC", ".png"):
        f = tmp_path / f"x{ext}"
        f.touch()
        assert is_supported_image(f)
    f = tmp_path / "notes.txt"
    f.touch()
    assert not is_supported_image(f)


def test_model_available_true_when_dir_exists(tmp_path):
    (tmp_path / "models" / "buffalo_s").mkdir(parents=True)
    assert model_available(tmp_path, "buffalo_s")
    assert not model_available(tmp_path, "buffalo_l")  # not downloaded


def test_model_available_none_root_defaults_to_insightface(tmp_path, monkeypatch):
    (tmp_path / "models" / "buffalo_s").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert model_available(None, "buffalo_s")


def test_supported_exts_covers_docs():
    assert {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tiff"} <= SUPPORTED_EXTS
