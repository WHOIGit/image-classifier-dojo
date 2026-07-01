"""Deterministic defect injectors for failure-mode tests.

Rather than committing one malformed binary fixture per failure mode (opaque,
LFS-heavy, and contradictory to combine in a single table), we keep ONE clean
committed fixture (`plankton-toyset`) and mutate it in-process to manufacture the
exact defect a test needs. Each injector is a pure ``pa.Table -> pa.Table``
transform; ``build_defect`` loads the clean toy, applies one, and writes a
parquet_images directory under a caller-supplied (usually ``tmp_path``) dir.

Why this shape (vs. a multi-column "which test am I" selector fixture):
  * structural defects (empty split, non-contiguous labels, single-instance
    class) are properties of the whole table and cannot coexist in one table;
  * byte-level defects (truncated/absent image bytes, missing target cells)
    must live in the data, not a flag column;
  * a code mutation is diffable, named, isolated, and regenerates for free.

This is a sketch: the injectors map to the `runtime.preflight` checks and
`missing_policy` modes in DESIGN-DOC/12 & /04, but the data/preflight layer that
consumes them is still unbuilt, so signatures may shift once the real error
surface exists.

Wiring (in tests/conftest.py)::

    import pytest
    from tests.fixtures.defects import build_defect

    @pytest.fixture
    def defect_dataset(tmp_path):
        def _make(defect, **kw):
            return build_defect(defect, tmp_path / defect, **kw)
        return _make

    def test_preflight_rejects_empty_eval(defect_dataset):
        path = defect_dataset("empty_eval_class", classname="Trichodesmium")
        # point a parquet_images data config at `path` and assert preflight fails
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# parquet_images fixture columns (see build_fixtures.py / data.yaml).
IMAGE = "image"
LABEL = "label"
CLASSNAME = "classname"
SAMPLE_ID = "ifcb_roi_pid"
SPLIT = "split"


# --- loading / writing -------------------------------------------------------

def load_fixture_table(name: str = "plankton-toyset", fixtures_dir: Path = FIXTURES_DIR) -> pa.Table:
    """Read a committed clean fixture into a single in-memory table."""
    path = Path(fixtures_dir) / name / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data.parquet under fixture {name!r} ({fixtures_dir})")
    return pq.read_table(path)


def write_parquet_images(table: pa.Table, dest_dir: str | Path) -> Path:
    """Write ``table`` as a parquet_images directory (``dest_dir/data/00000.parquet``)."""
    dest = Path(dest_dir)
    (dest / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest / "data" / "00000.parquet")
    return dest


# --- column helpers ----------------------------------------------------------

def _set_column(table: pa.Table, name: str, values: list, typ: pa.DataType) -> pa.Table:
    arr = pa.array(values, type=typ)
    return table.set_column(table.schema.get_field_index(name), name, arr)


def _image_children(table: pa.Table) -> tuple[pa.Array, pa.Array]:
    img = table.column(IMAGE).combine_chunks()
    return img.field("bytes"), img.field("path")


def _set_image_bytes(table: pa.Table, byte_values: list[bytes | None]) -> pa.Table:
    _, path = _image_children(table)
    new = pa.array(byte_values, type=pa.binary())
    struct = pa.StructArray.from_arrays([new, path], names=["bytes", "path"])
    return table.set_column(table.schema.get_field_index(IMAGE), IMAGE, struct)


# --- structural defects (whole-table shape) ----------------------------------

def drop_class_from_split(table: pa.Table, classname: str = "Trichodesmium",
                          split: str = "val") -> pa.Table:
    """Remove every row of ``classname`` in ``split`` -> that split has an empty class.

    ``split="val"`` -> ``empty_eval_classes``; ``split="train"`` -> ``empty_train_classes``.
    """
    drop = pc.and_(pc.equal(table[CLASSNAME], classname), pc.equal(table[SPLIT], split))
    return table.filter(pc.invert(drop))


def make_labels_noncontiguous(table: pa.Table, gap: int | None = None,
                              target_col: str = LABEL) -> pa.Table:
    """Open a hole in the label set so ids are no longer ``0..N-1`` -> ``non_contiguous_class_indices``."""
    labels = table[target_col].to_pylist()
    uniq = sorted({lbl for lbl in labels if lbl is not None})
    if gap is None:
        gap = uniq[len(uniq) // 2]
    shifted = [lbl + 1 if (lbl is not None and lbl >= gap) else lbl for lbl in labels]
    return _set_column(table, target_col, shifted, pa.int64())


def collapse_to_single_instance(table: pa.Table, classname: str = "Skeletonema",
                                keep_split: str = "train", seed: int = 0) -> pa.Table:
    """Reduce ``classname`` to a single row -> a class that cannot be stratified."""
    cns = table[CLASSNAME].to_pylist()
    idx = [i for i, c in enumerate(cns) if c == classname]
    if not idx:
        raise ValueError(f"class {classname!r} not present")
    keep = Random(seed).choice(idx)
    mask = [i == keep or cns[i] != classname for i in range(table.num_rows)]
    table = table.filter(pa.array(mask, type=pa.bool_()))
    cns = table[CLASSNAME].to_pylist()
    splits = [keep_split if c == classname else s
              for c, s in zip(cns, table[SPLIT].to_pylist())]
    return _set_column(table, SPLIT, splits, pa.string())


# --- target / label defects --------------------------------------------------

def null_targets(table: pa.Table, n: int = 3, target_col: str = LABEL,
                 seed: int = 0) -> pa.Table:
    """Null ``n`` random target cells -> ``missing_required_targets`` / ``missing_policy``."""
    labels = table[target_col].to_pylist()
    for i in Random(seed).sample(range(len(labels)), n):
        labels[i] = None
    return _set_column(table, target_col, labels, pa.int64())


def null_class_targets(table: pa.Table, classname: str = "Trichodesmium",
                       target_col: str = LABEL) -> pa.Table:
    """Null every target for one class -> ``no_remaining_valid_target_labels`` for it."""
    cns = table[CLASSNAME].to_pylist()
    labels = [None if c == classname else lbl
              for c, lbl in zip(cns, table[target_col].to_pylist())]
    return _set_column(table, target_col, labels, pa.int64())


def drop_target_column(table: pa.Table, target_col: str = LABEL) -> pa.Table:
    """Remove the target column entirely -> config references a missing column."""
    return table.drop_columns([target_col])


# --- duplicate / byte-level defects ------------------------------------------

def duplicate_rows(table: pa.Table, n: int = 4, new_sample_ids: bool = False,
                   seed: int = 0) -> pa.Table:
    """Append copies of ``n`` rows. ``new_sample_ids=False`` -> exact dup rows
    (same id + bytes); ``True`` -> same image bytes under fresh ids (near-dups)."""
    idx = Random(seed).sample(range(table.num_rows), n)
    dups = table.take(pa.array(idx, type=pa.int64()))
    if new_sample_ids:
        ids = [f"{sid}__dup{i}" for i, sid in enumerate(dups[SAMPLE_ID].to_pylist())]
        dups = _set_column(dups, SAMPLE_ID, ids, pa.string())
    return pa.concat_tables([table, dups])


def corrupt_image_bytes(table: pa.Table, n: int = 2, mode: str = "truncate",
                        seed: int = 0) -> pa.Table:
    """Damage inlined image bytes for ``n`` rows -> decode / missing-bytes paths.

    ``mode``: ``truncate`` (keep first 8 bytes -> undecodable), ``empty`` (b""),
    ``null`` (set the struct ``bytes`` field to null).
    """
    blist, _ = _image_children(table)
    values = blist.to_pylist()
    chosen = set(Random(seed).sample(range(len(values)), n))
    for i in chosen:
        if mode == "truncate":
            values[i] = values[i][:8]
        elif mode == "empty":
            values[i] = b""
        elif mode == "null":
            values[i] = None
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return _set_image_bytes(table, values)


# --- registry / entry point --------------------------------------------------

DEFECTS = {
    "empty_eval_class": partial(drop_class_from_split, split="val"),
    "empty_train_class": partial(drop_class_from_split, split="train"),
    "noncontiguous_labels": make_labels_noncontiguous,
    "single_instance_class": collapse_to_single_instance,
    "missing_targets": null_targets,
    "all_targets_missing_for_class": null_class_targets,
    "missing_target_column": drop_target_column,
    "duplicate_rows": partial(duplicate_rows, new_sample_ids=False),
    "near_duplicate_images": partial(duplicate_rows, new_sample_ids=True),
    "corrupt_image_truncated": partial(corrupt_image_bytes, mode="truncate"),
    "null_image_bytes": partial(corrupt_image_bytes, mode="null"),
}


def build_defect(defect: str, dest_dir: str | Path, *, base: str = "plankton-toyset",
                 **kwargs) -> Path:
    """Load the clean ``base`` fixture, apply ``defect``, write parquet_images to ``dest_dir``."""
    if defect not in DEFECTS:
        raise KeyError(f"unknown defect {defect!r}; known: {sorted(DEFECTS)}")
    table = DEFECTS[defect](load_fixture_table(base), **kwargs)
    return write_parquet_images(table, dest_dir)


if __name__ == "__main__":
    # Smoke-build every defect to a scratch dir and report the row-count delta.
    import tempfile

    base_rows = load_fixture_table().num_rows
    out = Path(tempfile.mkdtemp(prefix="dojo-defects-"))
    print(f"clean plankton-toyset: {base_rows} rows -> writing variants under {out}")
    for name in DEFECTS:
        path = build_defect(name, out / name)
        rows = load_fixture_table(path.name, fixtures_dir=out).num_rows
        print(f"  {name:30s} {rows:4d} rows  ({rows - base_rows:+d})")
