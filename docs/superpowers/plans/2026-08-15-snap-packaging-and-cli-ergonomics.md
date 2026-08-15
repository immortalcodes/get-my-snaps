# Snap packaging + CLI ergonomics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `sudo snap install get-my-snaps` then `get-my-snaps ~/Pictures --of ~/me.jpg`, with a provenance-correct cache split between central (models + reference prototypes) and per-run (input embeddings, checkpoint, reports).

**Architecture:** CLI gains a positional SOURCE dir + `--of` photo/folder references; `config.Paths` becomes an explicit-root factory; a new `cache.py` owns embedding recipe/digest keys; reference prototypes become content-keyed per-person files; `FaceEngine` gets a configurable model root; a `snap/snapcraft.yaml` wraps it all under strict confinement.

**Tech Stack:** Python 3.10+, InsightFace (`buffalo_s`), ONNX Runtime, snapcraft core24, pytest (pure-logic only — no ML deps per AGENTS.md).

## Global Constraints

- Embeddings are L2-normalized; similarity is a plain dot product. Never break this.
- All filesystem paths use `pathlib.Path`; no string concat / shell on paths.
- Checkpoint keys stay input(source)-relative POSIX paths; the same key drives dedup, resume, and report rows.
- Default model is **`buffalo_s`** everywhere (`Config.model_name` and `FaceEngine`); `FaceEngine` loses its own default so `Config` is the single source of truth.
- Recall over precision: default `threshold = 0.28` unchanged (measured to still separate under `buffalo_s`).
- `insightface` is sdist-only → snapcraft build needs build-essential + cython + numpy, no PEP 517 isolation.
- Pretrained weights are non-commercial research only → never bundle them in the snap; fetch on first run into a writable root.
- Tests are pure-logic: no `FaceEngine`, no network. ML path verified by interactive smoke tests, not unit tests.
- Never modify `input/`. Copies are idempotent (`copy_match` skips existing dest).

---

### Task 1: Reconciled Config + explicit Paths factory

**Files:**
- Modify: `src/get_my_snaps/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Config.model_name` default → `"buffalo_s"`
  - `default_central_dir() -> Path` — `$GMS_CACHE_DIR` → `$SNAP_USER_COMMON` → `$XDG_CACHE_HOME/get-my-snaps` → `~/.cache/get-my-snaps`
  - `default_model_root() -> Path` — `$GMS_MODEL_DIR` → populated `~/.insightface` → `default_central_dir()`
  - `Paths.for_run(source: Path, out: Path, central: Path, model_root: Path) -> Paths`
  - `Paths` fields: `source`, `out`, `central`, `model_root`, `state_dir` (`out/.gms`), `checkpoint`, `embeddings_cache`, `manifest`, `candidates`
- Consumes: nothing.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_config.py::test_paths_anchor_on_cwd` and the buffalo_l assertions.

```python
def test_defaults_are_recall_first():
    cfg = Config()
    assert cfg.threshold == 0.28
    assert cfg.model_name == "buffalo_s"  # was buffalo_l


def test_json_file_overrides_defaults(tmp_path):
    ...  # unchanged except assertion below
    assert cfg.model_name == "buffalo_s"


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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `Paths.for_run` / `default_model_root` not defined; model_name still buffalo_l.

- [ ] **Step 3: Implement**

In `config.py`:
- Delete `ROOT = Path.cwd()`.
- Add `import os`.
- `Config.model_name: str = "buffalo_s"`.
- Add `default_central_dir`, `default_model_root` (as in Interfaces).
- Rewrite `Paths` as frozen dataclass with the new fields and `for_run` classmethod.

```python
@dataclass(frozen=True)
class Paths:
    source: Path
    out: Path
    central: Path
    model_root: Path

    @property
    def state_dir(self) -> Path:
        return self.out / ".gms"

    @property
    def checkpoint(self) -> Path:
        return self.state_dir / "checkpoint.json"

    @property
    def embeddings_cache(self) -> Path:
        return self.state_dir / "embeddings"

    @property
    def manifest(self) -> Path:
        return self.out / "manifest.csv"

    @property
    def candidates(self) -> Path:
        return self.out / "candidates.csv"

    @classmethod
    def for_run(cls, source, out, central=None, model_root=None):
        central = central or default_central_dir()
        model_root = model_root or default_model_root()
        return cls(source=source, out=out, central=central, model_root=model_root)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/get_my_snaps/config.py tests/test_config.py
git commit -m "refactor(config): explicit Paths factory; default model buffalo_s"
```

---

### Task 2: cache.py — recipe, digest, fanout, samples key

**Files:**
- Create: `src/get_my_snaps/cache.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Produces:
  - `recipe(model_name, det_size, det_thresh, max_long_edge) -> str` (12 hex chars)
  - `image_digest(abs_source: Path, rel_key: str, size: int, mtime_ns: int) -> str`
  - `embedding_path(cache_dir: Path, recipe: str, digest: str) -> Path` (`<cache>/<recipe>/<d[:2]>/<d>.npz`)
  - `samples_key(image_paths: Iterable[Path]) -> str`
- Consumes: nothing.

- [ ] **Step 1: Write failing tests**

```python
def test_recipe_changes_with_model():
    a = recipe("buffalo_s", 640, 0.35, 1600)
    b = recipe("buffalo_l", 640, 0.35, 1600)
    assert a != b and len(a) == 12


def test_recipe_stable():
    assert recipe("buffalo_s", 640, 0.35, 1600) == recipe("buffalo_s", 640, 0.35, 1600)


def test_digest_distinguishes_sources():
    a = image_digest(Path("/x/wedding"), "a.jpg", 10, 1)
    b = image_digest(Path("/x/party"), "a.jpg", 10, 1)
    assert a != b  # same rel_key, different source root


def test_digest_sensitive_to_content_mtime():
    a = image_digest(Path("/x"), "a.jpg", 10, 1)
    b = image_digest(Path("/x"), "a.jpg", 10, 2)
    assert a != b


def test_embedding_path_fanout():
    p = embedding_path(Path("/c"), "abc", "0123456789abcdef")
    assert p == Path("/c/abc/01/0123456789abcdef.npz")


def test_samples_key_order_independent(tmp_path):
    x, y = tmp_path / "x.jpg", tmp_path / "y.jpg"
    for f in (x, y):
        f.touch()
    assert samples_key([x, y]) == samples_key([y, x])
    y.write_bytes(b"z")  # mtime may be coarse; force content-only? key uses mtime
    # key must still differ if a file changed -- mtime_ns
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_cache.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
import hashlib
from pathlib import Path
from typing import Iterable


def recipe(model_name, det_size, det_thresh, max_long_edge) -> str:
    raw = f"{model_name}|{det_size}|{det_thresh}|{max_long_edge}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def image_digest(abs_source: Path, rel_key: str, size: int, mtime_ns: int) -> str:
    raw = f"{abs_source}\0{rel_key}\0{size}\0{mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def embedding_path(cache_dir: Path, recipe: str, digest: str) -> Path:
    return cache_dir / recipe / digest[:2] / f"{digest}.npz"


def samples_key(image_paths: Iterable[Path]) -> str:
    parts = sorted(
        f"{p}\0{p.stat().st_size}\0{p.stat().st_mtime_ns}" for p in image_paths
    )
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/get_my_snaps/cache.py tests/test_cache.py
git commit -m "feat(cache): embedding recipe/digest keys, fanout paths, samples key"
```

---

### Task 3: FaceEngine model_root

**Files:**
- Modify: `src/get_my_snaps/face_engine.py`

**Interfaces:**
- Produces: `FaceEngine(model_name, det_size, det_thresh, model_root: Optional[Path] = None)`; `model_available(model_root, model_name) -> bool`
- Consumes: `default_model_root()` (Task 1).

- [ ] **Step 1: Add `model_root` param** — pass `root=str(model_root)` (or `"~/.insightface"` default) to `FaceAnalysis`.

```python
def __init__(self, model_name, det_size=640, det_thresh=0.35, model_root=None):
    from insightface.app import FaceAnalysis
    root = str(model_root) if model_root is not None else "~/.insightface"
    self.app = FaceAnalysis(name=model_name, root=root, providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    ...
```

Also add near module top:

```python
def model_available(model_root: Optional[Path], model_name: str) -> bool:
    base = Path(model_root) if model_root else Path("~/.insightface").expanduser()
    return (base / "models" / model_name).is_dir()
```

- [ ] **Step 2: `model_available` unit test** (pure, no engine) in `tests/test_cache.py` or new — assert `model_available` true when a dir exists.

```python
def test_model_available(tmp_path):
    (tmp_path / "models" / "buffalo_s").mkdir(parents=True)
    assert model_available(tmp_path, "buffalo_s")
    assert not model_available(tmp_path, "nope")
```

- [ ] **Step 3: Verify + commit**

```bash
.venv/bin/python -m pytest tests/test_cache.py -v
git add src/get_my_snaps/face_engine.py tests/test_cache.py
git commit -m "feat(face_engine): configurable model_root + model_available check"
```

---

### Task 4: Content-keyed per-person reference cache

**Files:**
- Modify: `src/get_my_snaps/reference_check.py`
- Modify: `tests/test_checkpoint.py` (or new `tests/test_reference.py`)

**Interfaces:**
- Produces:
  - `enroll_person(name, image_paths, engine, cfg) -> PersonReference`
  - `enroll_references(reference_dir, engine, cfg)` (refactored to use enroll_person)
  - `references_dir(central: Path, recipe: str) -> Path`
  - `person_cache_path(central, recipe, name) -> Path`
  - `load_cached_person(path, samples_key) -> Optional[PersonReference]` (None if absent / key mismatch)
  - `save_person(person, central, recipe, samples_key)`
  - `resolve_ref_items(items: Iterable[Path]) -> Dict[str, List[Path]]` (pure --of expansion)
- Consumes: `samples_key`, `recipe`, `is_supported_image` (Task 2/3).

`PersonReference` gains `samples_key: str = ""`.

- [ ] **Step 1: Write failing pure-logic tests**

```python
def test_resolve_ref_items_single_dir_with_images(tmp_path):
    d = tmp_path / "me"; d.mkdir()
    (d / "a.jpg").touch(); (d / "b.jpg").touch()
    people = resolve_ref_items([d])
    assert set(people) == {"me"} and len(people["me"]) == 2


def test_resolve_ref_items_bare_files_one_person(tmp_path):
    a = tmp_path / "a.jpg"; b = tmp_path / "b.jpg"
    a.touch(); b.touch()
    people = resolve_ref_items([a, b])
    assert set(people) == {"a"} and len(people["a"]) == 2


def test_resolve_ref_items_dir_of_subdirs(tmp_path):
    base = tmp_path / "ref"; (base / "ann").mkdir(parents=True); (base / "bob").mkdir()
    (base / "ann" / "1.jpg").touch(); (base / "bob" / "2.jpg").touch()
    people = resolve_ref_items([base])
    assert set(people) == {"ann", "bob"}


def test_person_cache_key_mismatch_returns_none(tmp_path):
    path = tmp_path / "p.json"
    from get_my_snaps.reference_check import save_person, load_cached_person
    # save with key A, load asking key B -> None
```

(For the last test, build a minimal PersonReference and a FakeEngine-less enrollment is not needed — save/load operate on the dataclass only.)

- [ ] **Step 2: Implement**

Refactor with a private `_enroll_one(name, image_paths, engine, cfg)` reusing the existing per-person scoring body from `enroll_references`. Add the cache file I/O:

```python
def _person_payload(p: PersonReference) -> dict:
    d = asdict(p); d.pop("embeddings", None)
    d["prototype"] = None if p.prototype is None else p.prototype.tolist()
    return d


def save_person(person, central, recipe, samples_key):
    d = _person_payload(person); d["samples_key"] = samples_key
    path = person_cache_path(central, recipe, person.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d), encoding="utf-8")


def load_cached_person(path, samples_key):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if d.get("samples_key") != samples_key:
        return None
    # rebuild PersonReference (existing reconstruction logic)
```

Add `resolve_ref_items` per the rules in the spec:

```python
def resolve_ref_items(items):
    people: Dict[str, List[Path]] = {}
    loose: List[Path] = []
    for item in items:
        if item.is_file() and is_supported_image(item):
            loose.append(item)
        elif item.is_dir():
            direct = sorted(p for p in item.iterdir()
                            if p.is_file() and is_supported_image(p))
            subs = sorted(p for p in item.iterdir() if p.is_dir())
            if not direct and subs:
                for sd in subs:
                    people[sd.name] = sorted(
                        p for p in sd.rglob("*") if p.is_file() and is_supported_image(p))
            else:
                if direct:
                    people[item.name] = direct
                for sd in subs:
                    people[sd.name] = sorted(
                        p for p in sd.rglob("*") if p.is_file() and is_supported_image(p))
        else:
            console.warn(f"Ignoring non-image, non-folder reference: {item}")
    if loose:
        name = loose[0].stem
        if name in people:  # collision: offset
            name = f"{name}_ref"
        existing = people.get(name, [])
        people[name] = existing + loose
    return {k: v for k, v in people.items() if v}
```

- [ ] **Step 3: Run tests + commit**

```bash
.venv/bin/python -m pytest tests/test_reference.py -v
git add src/get_my_snaps/reference_check.py src/get_my_snaps/cache.py tests/
git commit -m "feat(reference_check): per-person content-keyed cache + resolve_ref_items"
```

---

### Task 5: Embedding cache under out/.gms, recipe-scoped; checkpoint recipe/sources

**Files:**
- Modify: `src/get_my_snaps/matching.py`
- Modify: `tests/test_checkpoint.py`

**Interfaces:**
- Produces:
  - `Checkpoint` fields + `recipe: str`, `sources: str`; `matches_config(persons, threshold, recipe, sources)` extended
  - `get_face_embeddings(engine, src, rel_key, cfg, paths)` writes `<paths.embeddings_cache>/<recipe>/<fanout>/<digest>.npz`, storing `recipe`+`src` arrays
- Consumes: `recipe`, `image_digest`, `embedding_path` (Task 2).

- [ ] **Step 1: Update checkpoint tests**

```python
def test_checkpoint_matches_config_recipe():
    a = Checkpoint(["x"], 0.28, {}, recipe="r1", sources="s")
    b = Checkpoint(["x"], 0.28, {}, recipe="r2", sources="s")
    assert not a.matches_config(["x"], 0.28, "r2", "s")
    assert a.matches_config(["x"], 0.28, "r1", "s")


def test_checkpoint_load_old_schema(tmp_path, monkeypatch):  # graceful degrade
    ...old file without recipe -> loads with empty recipe, treated incompatible
```

- [ ] **Step 2: Implement**

In `Checkpoint`: add `recipe: str = ""`, `sources: str = ""`; `load` uses `d.get("recipe", "")`; `save` writes them; `matches_config(self, persons, threshold, recipe, sources)` compares all four.

In `get_face_embeddings`:

```python
st = src.stat()
rec = recipe(cfg.model_name, cfg.det_size, cfg.det_thresh, cfg.max_long_edge)
digest = image_digest(src.absolute(), rel_key, st.st_size, st.st_mtime_ns)
cache_path = embedding_path(paths.embeddings_cache, rec, digest)
if cache_path.exists():
    try:
        z = np.load(cache_path)
        if z.get("recipe") != rec or z.get("src") != str(src):
            pass  # provenance mismatch -> recompute
        else:
            return z["emb"].astype(np.float32)
    except Exception:
        pass
...
paths.embeddings_cache.mkdir(parents=True, exist_ok=True)
np.savez_compressed(cache_path, emb=emb, recipe=rec, src=str(src))
```

In `process_images`: `rel = src.relative_to(paths.source)`; `rel_key = rel.as_posix()`; pass checkpoint recipe/sources at save time (compute once, store on ckpt when created in cli).

- [ ] **Step 3: run + commit** (pure-logic tests only; no engine)

```bash
.venv/bin/python -m pytest tests/test_checkpoint.py -v
git add src/get_my_snaps/matching.py tests/test_checkpoint.py
git commit -m "feat(matching): recipe-scoped embedding cache + recipe/sources checkpoint"
```

---

### Task 6: CLI — SOURCE, --of, output default, first-run notice, flag remap

**Files:**
- Modify: `src/get_my_snaps/cli.py`
- Modify: `src/get_my_snaps/console.py` (first-run notice copy)
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `main()`: positional `SOURCE` (dir, optional), `-o/--out`, `--of`, `--name`, `--reset-refs`, keeps `--persons`, `--skip-ref-check`, `--dry-run`, `--reset`, `--reset-embeddings`, tuning flags.
  - `resolve_workspace(SOURCE, out) -> (source, out, legacy: bool)`
  - `default_out_dir(source: Path, cwd: Path) -> Path`
  - `choose_people(cfg, engine, paths, args) -> Dict[str, PersonReference]` (--of path vs legacy path)
- Consumes: everything above.

- [ ] **Step 1: Write pure-logic tests**

```python
def test_default_out_detects_legacy(tmp_path):
    (tmp_path / "input").mkdir(); (tmp_path / "reference").mkdir()
    from get_my_snaps.cli import default_out_dir
    assert default_out_dir(tmp_path / "input", tmp_path) == tmp_path / "output"


def test_default_out_fresh(tmp_path):
    (tmp_path / "input").mkdir()
    assert default_out_dir(tmp_path / "input", tmp_path) == tmp_path / "get-my-snaps-out"
```

- [ ] **Step 2: Implement**

- argparse: `p.add_argument("SOURCE", nargs="?", type=Path, default=None)`; `-o/--out`; `--of nargs="+"; --name; --reset-refs`.
- `main()` wiring:
  1. `source = args.SOURCE or (cwd/"input" if exists)`; error if neither.
  2. `out = args.out or default_out_dir(source, cwd)`.
  3. `paths = Paths.for_run(source, out)` (central/model_root resolved in config).
  4. First-run model notice: `if not model_available(paths.model_root, cfg.model_name): console.warn(...)`.
  5. `engine = FaceEngine(model_name=..., det..., det_thresh=..., model_root=paths.model_root)`.
  6. People + persons: if `args.of` → `resolve_ref_items(args.of)`, enroll-or-load (central, recipe), `--name` renames the loose person; persons = those with valid refs; no picker. Else legacy → `enroll_references` + `choose_persons`.
  7. Checkpoint recipe/sources set at creation.
- `apply_resets`: `--reset` → unlink checkpoint; `--reset-embeddings` → rmtree embeddings_cache; `--reset-refs` → (reference cache is content-keyed; forcing is a no-op beyond deleting per-person files — implement as deleting the recipe references dir for selected people or full references dir).
- Non-TTY guard: `console.prompt` returns default when stdin isn't a tty (so `--of`/`--skip-ref-check` runs never hang).

- [ ] **Step 3: Run tests + full legacy smoke**

```bash
.venv/bin/python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py -v
./scripts/run_sample.sh   # legacy path must still work (may download buffalo_s)
```

- [ ] **Step 4: Commit**

```bash
git add src/get_my_snaps/cli.py src/get_my_snaps/console.py tests/test_cli.py
git commit -m "feat(cli): positional SOURCE + --of references + output default + flag remap"
```

---

### Task 7: scripts + run_sample.sh + profile_run.py

**Files:**
- Modify: `scripts/run_sample.sh`
- Modify: `scripts/profile_run.py`

- [ ] **Step 1: Update `profile_run.py`** to construct `Paths.for_run(...)` and pass `model_root` to `FaceEngine`; update its loader.

```python
paths = Paths.for_run(source=Path("input"), out=Path("output"))
engine = FaceEngine(model_name=cfg.model_name, det_size=cfg.det_size,
                    det_thresh=cfg.det_thresh, model_root=paths.model_root)
```

- [ ] **Step 2: Update `run_sample.sh`** to keep working (it already uses `--persons` + legacy `input/` `output/`); note `output/` is retained for legacy. No functional change expected; verify.

- [ ] **Step 3: Verify both**

```bash
.venv/bin/python scripts/profile_run.py --sample 2 --model buffalo_s
./scripts/run_sample.sh
```

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "chore(scripts): adapt profile_run.py and run_sample.sh to new Paths/CLI"
```

---

### Task 8: README + AGENTS.md

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: README** — new Quick start (`sudo snap install get-my-snaps`), new Usage (`get-my-snaps ~/Pictures --of ~/me.jpg`), cache layout section, new options table, first-run model note, keep legacy instructions.
- [ ] **Step 2: AGENTS.md** — update cache section (out/.gms + central), model root, `--of` semantics, flag remap, module boundaries (add `cache.py`), `Paths` factory note.
- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: snap install + --of usage, cache layout, module map"
```

---

### Task 9: snap/snapcraft.yaml

**Files:**
- Create: `snap/snapcraft.yaml`

- [ ] **Step 1: Write snapcraft.yaml**

```yaml
name: get-my-snaps
base: core24
version: git
summary: Local, offline face-recognition photo filter
description: |
  Copies every photo containing a chosen person from a source folder into an
  output folder. Fully offline, CPU-only (InsightFace on ONNX Runtime).
grade: stable
confinement: strict

apps:
  get-my-snaps:
    command: bin/get-my-snaps
    plugs: [home, removable-media, network]

parts:
  get-my-snaps:
    plugin: python
    source: .
    python-packages:
      - numpy>=1.26,<2.2
      - onnxruntime>=1.17
      - insightface>=0.7.3
      - opencv-python-headless>=4.9
      - Pillow>=10.2
      - pillow-heif>=0.16
      - tqdm>=4.66
    build-packages:
      - build-essential
      - cython3
      - python3-dev
    override-build: |
      snapcraftctl build
      # insightface is sdist-only and needs numpy+cython present first.
      python3 -m pip install --no-build-isolation \
        --ignore-installed numpy cython
      python3 -m pip install --no-build-isolation --ignore-installed \
        insightface==0.7.3 opencv-python-headless
```

(Exact build recipe is validated in Task 10; may need `--system-site-packages` or a virtualenv part. This file is the *starting point* — snapcraft actual build outcomes drive final form.)

- [ ] **Step 2: Attempt build** `snapcraft --use-lxd` — expect iteration on the python plugin + deps. Do NOT mark done until a `.snap` is produced.
- [ ] **Step 3: Commit** `snap/snapcraft.yaml` (and any `snap/` companion files).

---

### Task 10: Full verification

- [ ] **Step 1: pytest green** — `.venv/bin/python -m pytest -v`
- [ ] **Step 2: Labelled regression on repo dataset** — 4 `colin_powell` positives matched, 4 guests not, at default `0.28` under `buffalo_s`.
- [ ] **Step 3: Recipe isolation smoke** — run `--model buffalo_s`, then `buffalo_l`, then `buffalo_s`; third hits cache and agrees with the first; `out/.gms/embeddings/` shows two recipe dirs.
- [ ] **Step 4: Reference invalidation** — add a photo to a person folder, confirm re-enroll without `--reset`.
- [ ] **Step 5: Collision-free cache** — two source trees with a same-basename image produce distinct cache entries.
- [ ] **Step 6: Snap confinement smoke** — `snapcraft --use-lxd` built; `snap install --dangerous`; as non-root: `get-my-snaps <dir> --of <face>`; model lands in `$SNAP_USER_COMMON`, not `~/.insightface`.
- [ ] **Step 7: Final commit + amend README/AGENTS if verification changed anything.**

---

## Spec coverage cross-check

| Spec section | Task |
|---|---|
| CLI surface (SOURCE, --of, -o, --name) | 6, 4 (resolve_ref_items) |
| Storage layout (central vs out/.gms) | 1, 5 |
| The recipe key | 2, 5 |
| Reference cache invalidation | 4 |
| Checkpoint recipe/sources | 5 |
| Flag remap | 6 |
| Model root + first-run fetch | 1, 3, 6 |
| Model default buffalo_s | 1, 3 |
| The snap | 9, 10 |
| Verification plan | 10 |
| README/AGENTS | 8 |
