"""Tests for embedding cache keying (cache.py) - pure logic, no ML deps."""

from __future__ import annotations

from pathlib import Path

from get_my_snaps.cache import (
    embedding_path,
    image_digest,
    recipe,
    samples_key,
)


def test_recipe_changes_with_model():
    assert recipe("buffalo_s", 640, 0.35, 1600) != recipe("buffalo_l", 640, 0.35, 1600)


def test_recipe_stable_and_length():
    r = recipe("buffalo_s", 640, 0.35, 1600)
    assert r == recipe("buffalo_s", 640, 0.35, 1600)
    assert len(r) == 12


def test_recipe_sensitive_to_detection_params():
    assert recipe("buffalo_s", 640, 0.35, 1600) != recipe("buffalo_s", 320, 0.35, 1600)
    assert recipe("buffalo_s", 640, 0.35, 1600) != recipe("buffalo_s", 640, 0.5, 1600)


def test_digest_distinguishes_source_roots():
    a = image_digest(Path("/x/wedding"), "a.jpg", 10, 1)
    b = image_digest(Path("/x/party"), "a.jpg", 10, 1)
    assert a != b  # same rel_key, different source -> never collide


def test_digest_sensitive_to_content_mtime():
    a = image_digest(Path("/x"), "a.jpg", 10, 1)
    b = image_digest(Path("/x"), "a.jpg", 10, 2)
    assert a != b  # edited image self-invalidates


def test_embedding_path_fanout():
    p = embedding_path(Path("/c"), "abcd1234abcd", "0123456789abcdef0123")
    assert p == Path("/c/abcd1234abcd/01/0123456789abcdef0123.npz")


def test_samples_key_order_independent(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.touch()
    b.touch()
    assert samples_key([a, b]) == samples_key([b, a])


def test_samples_key_changes_when_file_changes(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.touch()
    b.touch()
    before = samples_key([a, b])
    # Ensure mtime differs; touch may be coarse, so set explicitly.
    import os
    import time
    old = time.time() - 60
    os.utime(b, (old, old))
    after = samples_key([a, b])
    assert before != after
