"""Pure-logic tests for face_engine helpers (no model, no inference)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
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


def _run_quiet_io_snippet(body: str) -> subprocess.CompletedProcess:
    """Run a snippet against _quiet_native_io in a fresh subprocess.

    A subprocess (rather than pytest's own capfd) is used because nested
    os.dup2 fd swaps interact unpredictably with pytest's own fd-level output
    capturing in-process; a subprocess gives a clean, realistic fd table.
    """
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {src_dir!r})
        from get_my_snaps.face_engine import _quiet_native_io
        {textwrap.indent(textwrap.dedent(body), "        ")}
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_quiet_native_io_suppresses_fd_writes():
    result = _run_quiet_io_snippet(
        """
        import os
        with _quiet_native_io(enabled=True):
            os.write(1, b"stdout noise\\n")
            os.write(2, b"stderr noise\\n")
            print("python stdout noise")
        """
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_quiet_native_io_restores_fds_after_exception():
    result = _run_quiet_io_snippet(
        """
        import os
        try:
            with _quiet_native_io(enabled=True):
                os.write(2, b"hidden\\n")
                raise ValueError("boom")
        except ValueError:
            pass
        print("visible after restore")
        """
    )
    assert result.returncode == 0
    assert "hidden" not in result.stderr
    assert "visible after restore" in result.stdout


def test_quiet_native_io_passthrough_when_disabled():
    result = _run_quiet_io_snippet(
        """
        import os
        with _quiet_native_io(enabled=False):
            print("visible stdout")
            os.write(2, b"visible stderr\\n")
        """
    )
    assert result.returncode == 0
    assert "visible stdout" in result.stdout
    assert "visible stderr" in result.stderr
