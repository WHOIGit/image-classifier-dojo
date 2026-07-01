#!/usr/bin/env python3
"""Build deterministic Tier-1 ``parquet_images`` fixtures from a local copy of
the NES plankton dataset.

This is dev/test tooling, **not** shipped library code. It reads the local
Tier-3 dataset (gitignored, under ``./datasets/``) and writes small, committed
fixtures under ``tests/fixtures/``. Output is native parquet with image bytes
inlined and a canonical ``split`` column (``train`` / ``val``), directly
consumable by dojo's ``parquet_images`` backend via ``split_column: split``.

Adapted from a Hugging Face ``datasets`` based sampler: the stratified
per-class sampling, train/val split, two-instance-minimum guard, compact label
remapping and ``one_instance_policy`` are preserved, but the implementation
uses pyarrow directly so it needs only ``pyarrow`` + stdlib (no ``datasets`` /
``numpy``) and emits parquet_images directly instead of an Arrow ``DatasetDict``.

Regenerate both fixtures::

    python tests/fixtures/build_fixtures.py

The default source is ``./datasets/NES-plankton-classifier-2022-dataset/data``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

# Canonical split vocabulary used across dojo (see DESIGN-DOC/04). Fixtures only
# ever produce the two supervised splits.
CANONICAL_SPLITS = ("train", "val")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
DEFAULT_SOURCE = DATASETS_DIR / "NES-plankton-classifier-2022-dataset" / "data"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# --- Fixture specs -----------------------------------------------------------
# Toy: committed Tier-1 fixture (tests/fixtures/). 6 visually-distinct classes,
# deliberately imbalanced (~12:1) with a 2-instance class (1 train / 1 val) so the
# clean baseline already covers the minimum-stratification boundary. This is the
# table that tests/fixtures/defects.py mutates to manufacture failure modes.
TOYSET_CLASS_COUNTS = {
    "bubble": 24,
    "fiber": 16,
    "Copepod_nauplii": 10,
    "Skeletonema": 6,
    "Trichodesmium": 4,
    "Dinophysis_acuminata": 2,
}

# Mini: Tier-2 dev dataset, NOT committed -- written under datasets/ (gitignored).
# 30 distinct classes, long-tail. NOTE: bubble has only 34 rows in the source, so
# at strict=False it is capped (see build_all); raise/redistribute to taste.
MINISET_CLASS_COUNTS = {
    "bubble": 120, "fiber": 95, "bead": 80, "camera_spot": 70, "detritus": 60,
    "detritus_transparent": 52, "fecal_pellet": 45, "pollen": 40,
    "Copepod_nauplii": 36, "zooplankton": 32, "shellfish_larvae": 30, "Amoeba": 28,
    "Favella": 26, "Eutintinnus": 24, "Tintinnopsis": 22, "Stenosemella_pacifica": 20,
    "Laboea_strobila": 18, "Euplotes": 16, "Trichodesmium": 15, "Skeletonema": 14,
    "Thalassionema": 13, "Pseudo-nitzschia": 12, "Rhizosolenia": 11,
    "Ditylum_brightwellii": 10, "Corethron_hystrix": 9, "Coscinodiscus": 8,
    "Acanthoica_quattrospina": 7, "Dictyocha": 6, "Tripos_fusus": 5,
    "Dinophysis_acuminata": 4,
}


def long_tail_counts(
    n_classes: int,
    total: int,
    *,
    min_count: int = 2,
    decay: float = 0.84,
) -> list[int]:
    """Return ``n_classes`` per-class counts forming a long-tail distribution.

    Counts are non-increasing, every class gets at least ``min_count`` examples,
    and the counts sum to exactly ``total``. The head follows a geometric decay;
    ``decay`` closer to 1 flattens the curve, lower values steepen the tail.
    """
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if min_count < 2:
        raise ValueError("min_count must be >= 2 so each class can fill train and val")
    floor_total = n_classes * min_count
    if total < floor_total:
        raise ValueError(
            f"total={total} is below the floor n_classes*min_count={floor_total}"
        )

    weights = [decay**i for i in range(n_classes)]
    wsum = sum(weights)
    extra = total - floor_total  # examples to distribute above the per-class floor
    counts = [min_count + round(w / wsum * extra) for w in weights]

    # Rounding drift lands on the head class, which has the most slack.
    counts[0] += total - sum(counts)
    if counts[0] < min_count:
        raise ValueError("decay too steep for this total; raise decay or total")
    return counts


def _resolve_class_counts(
    class_counts: Mapping[str, int] | None,
    class_names: Sequence[str] | None,
    instances_per_class: int | Sequence[int] | None,
) -> dict[str, int]:
    if class_counts is not None:
        resolved = dict(class_counts)
    else:
        if class_names is None or instances_per_class is None:
            raise ValueError(
                "Provide either class_counts={name: count}, or both "
                "class_names=[...] and instances_per_class=..."
            )
        if isinstance(instances_per_class, int):
            resolved = {name: instances_per_class for name in class_names}
        else:
            if len(class_names) != len(instances_per_class):
                raise ValueError(
                    "class_names and instances_per_class must have the same length"
                )
            resolved = dict(zip(class_names, instances_per_class))

    if not resolved:
        raise ValueError("No classes requested.")
    bad = {k: v for k, v in resolved.items() if v <= 0}
    if bad:
        raise ValueError(f"All requested counts must be positive. Bad counts: {bad}")
    return resolved


def select_top_classes(source: pads.Dataset, n: int, class_name_col: str) -> list[str]:
    """The ``n`` most populous class names, ties broken alphabetically."""
    names = source.to_table(columns=[class_name_col]).column(class_name_col).to_pylist()
    counts = Counter(names)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < n:
        raise ValueError(f"Dataset has only {len(ranked)} classes, need {n}")
    return [name for name, _ in ranked[:n]]


def build_fixture(
    *,
    source_dir: str | Path,
    out_dir: str | Path,
    class_counts: Mapping[str, int] | None = None,
    class_names: Sequence[str] | None = None,
    instances_per_class: int | Sequence[int] | None = None,
    train_fraction: float = 0.8,
    seed: int = 0,
    label_col: str = "label",
    class_name_col: str = "classname",
    sample_id_col: str = "ifcb_roi_pid",
    image_col: str = "image",
    split_col: str = "split",
    remap_labels: bool = True,
    one_instance_policy: Literal["train", "val", "error"] = "error",
    strict: bool = True,
    file_pattern: str = "*.parquet",
) -> dict:
    """Build a stratified parquet_images fixture and write it under ``out_dir``.

    Writes ``out_dir/data.parquet`` (image bytes inlined, plus a canonical
    ``split`` column) and a ``summary.json`` describing the per-class makeup.
    Returns the summary dict.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    class_counts = _resolve_class_counts(class_counts, class_names, instances_per_class)

    source_dir = Path(source_dir)
    files = sorted(source_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern!r} under {source_dir}")
    source = pads.dataset([str(f) for f in files], format="parquet")

    required = {class_name_col, sample_id_col, image_col, label_col}
    missing_cols = required - set(source.schema.names)
    if missing_cols:
        raise ValueError(f"Source is missing required columns: {sorted(missing_cols)}")

    # Cheap first pass: read only id + classname to plan the per-class sample.
    # The source has a handful of repeated sample_ids; dedupe so each id is
    # sampled at most once and the fixture's sample_ids come out clean.
    idx = source.to_table(columns=[class_name_col, sample_id_col])
    names = idx.column(class_name_col).to_pylist()
    sids = idx.column(sample_id_col).to_pylist()

    by_class: dict[str, set[str]] = {}
    for cn, sid in zip(names, sids):
        if cn in class_counts:
            by_class.setdefault(cn, set()).add(sid)

    rng = Random(seed)
    id2split: dict[str, str] = {}
    per_class: dict[str, dict] = {}
    missing_classes: list[str] = []
    underfilled: dict[str, tuple[int, int]] = {}

    present_classes = [cn for cn in class_counts if by_class.get(cn)]
    class_to_new_id = {cn: i for i, cn in enumerate(present_classes)}

    for class_name, requested in class_counts.items():
        available = by_class.get(class_name, [])
        if not available:
            missing_classes.append(class_name)
            continue
        if len(available) < requested:
            underfilled[class_name] = (requested, len(available))
            if strict:
                continue
            requested = len(available)

        pool = sorted(available)  # stable order (set -> sorted) before the shuffle
        rng.shuffle(pool)
        chosen = pool[:requested]

        if requested == 1:
            if one_instance_policy == "error":
                raise ValueError(
                    f"Class {class_name!r} has only one selected example; it cannot "
                    "appear in both train and val. Request >=2, or set "
                    "one_instance_policy='train'/'val'."
                )
            target = "train" if one_instance_policy == "train" else "val"
            for sid in chosen:
                id2split[sid] = target
            per_class[class_name] = {
                "new_id": class_to_new_id[class_name],
                "total": 1,
                "train": int(target == "train"),
                "val": int(target == "val"),
            }
            continue

        n_train = int(round(requested * train_fraction))
        n_train = max(1, min(n_train, requested - 1))  # both splits get >=1
        for sid in chosen[:n_train]:
            id2split[sid] = "train"
        for sid in chosen[n_train:]:
            id2split[sid] = "val"
        per_class[class_name] = {
            "new_id": class_to_new_id[class_name],
            "total": requested,
            "train": n_train,
            "val": requested - n_train,
        }

    if strict and missing_classes:
        raise ValueError(f"Requested classes not found: {missing_classes}")
    if strict and underfilled:
        detail = {n: {"requested": r, "available": a} for n, (r, a) in underfilled.items()}
        raise ValueError(f"Some requested classes have too few examples: {detail}")

    # Second pass: pull full rows (with image bytes) for only the selected ids.
    # isin may return >1 row for a repeated source id; keep the first per id.
    selected_ids = list(id2split)
    table = source.to_table(filter=pads.field(sample_id_col).isin(selected_ids))
    seen: set[str] = set()
    keep: list[int] = []
    for i, sid in enumerate(table.column(sample_id_col).to_pylist()):
        if sid not in seen:
            seen.add(sid)
            keep.append(i)
    table = table.take(pa.array(keep, type=pa.int64()))
    if table.num_rows != len(selected_ids):
        raise RuntimeError(
            f"Expected {len(selected_ids)} selected rows, got {table.num_rows}"
        )

    # Verify image bytes are actually inlined (parquet_images contract).
    image_struct = table.column(image_col).combine_chunks()
    bytes_idx = table.schema.field(image_col).type.get_field_index("bytes")
    if image_struct.field(bytes_idx).null_count != 0:
        raise RuntimeError("Some selected rows have null image bytes; not parquet_images")

    row_sids = table.column(sample_id_col).to_pylist()
    row_cns = table.column(class_name_col).to_pylist()

    out = table
    if remap_labels:
        new_label = pa.array([class_to_new_id[c] for c in row_cns], type=pa.int64())
        out = out.set_column(out.schema.get_field_index(label_col), label_col, new_label)
        out = out.append_column(
            "original_label", table.column(label_col).cast(pa.int64())
        )
    split_vals = pa.array([id2split[s] for s in row_sids], type=pa.string())
    out = out.append_column(split_col, split_vals)

    # Deterministic interleave of classes/splits within the file.
    order = list(range(out.num_rows))
    rng.shuffle(order)
    out = out.take(pa.array(order, type=pa.int64()))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_dir / "data.parquet")

    n_train = sum(1 for s in id2split.values() if s == "train")
    n_val = sum(1 for s in id2split.values() if s == "val")
    counts = [c["total"] for c in per_class.values()]
    summary = {
        "name": out_dir.name,
        "source": str(Path(source_dir).relative_to(REPO_ROOT)),
        "seed": seed,
        "train_fraction": train_fraction,
        "num_classes": len(per_class),
        "num_samples": len(selected_ids),
        "num_train": n_train,
        "num_val": n_val,
        "imbalance_ratio": (max(counts) / min(counts)) if counts else None,
        "class_names": list(per_class),
        "per_class": per_class,
        "underfilled": {
            n: {"requested": r, "available": a} for n, (r, a) in underfilled.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_data_config(
        out_dir, split_col, sample_id_col, image_col, label_col, class_name_col, remap_labels
    )
    return summary


def _write_data_config(
    out_dir: Path,
    split_col: str,
    sample_id_col: str,
    image_col: str,
    label_col: str,
    class_name_col: str,
    remap_labels: bool,
) -> None:
    """Write an example dojo data config consuming the fixture via split_column."""
    rel = out_dir.relative_to(REPO_ROOT)
    extra = ["classname"] + (["original_label"] if remap_labels else [])
    extra_block = "\n".join(f"  - {c}" for c in extra)
    text = f"""\
# Auto-generated by tests/fixtures/build_fixtures.py -- do not edit by hand.
# Example dojo data config for this committed Tier-1 fixture. The fixture carries
# a canonical `{split_col}` column (train/val), so it is consumed via split_column
# rather than split_from_filename.
backend: parquet_images
manifest_uri: ./{rel.as_posix()}/data.parquet
{split_col}_column: {split_col}
sample_id_column: {sample_id_col}
images:
  column: {image_col}
  bytes_field: bytes
  path_field: path
source_extra_columns:
{extra_block}
targets:
  species:
    label_index_column: {label_col}
    label_name_column: {class_name_col}
    type: multiclass_classification
    missing_policy: error
"""
    (out_dir / "data.yaml").write_text(text)


def _report(summary: dict) -> None:
    print(
        f"[{summary['name']}] {summary['num_classes']} classes, "
        f"{summary['num_samples']} imgs (train={summary['num_train']}, "
        f"val={summary['num_val']}, imbalance={summary['imbalance_ratio']:.1f})"
    )
    for cls, info in summary["underfilled"].items():
        print(f"    ! capped {cls}: requested {info['requested']}, "
              f"used {info['available']} (all that exist)")


def build_all(
    source_dir: str | Path = DEFAULT_SOURCE,
    fixtures_dir: str | Path = FIXTURES_DIR,
    datasets_dir: str | Path = DATASETS_DIR,
) -> None:
    # Toy -> committed Tier-1 fixture. strict: every requested class must fit.
    toy = build_fixture(
        source_dir=source_dir,
        out_dir=Path(fixtures_dir) / "plankton-toyset",
        class_counts=TOYSET_CLASS_COUNTS,
        seed=0,
        strict=True,
    )
    _report(toy)

    # plankton-miniset -> Tier-2 dev dataset under datasets/plankton-miniset (committed via
    # LFS; the one datasets/ entry un-ignored in .gitignore).
    # strict=False so the under-supplied bubble class is capped to what exists
    # rather than aborting the build; the cap is reported above.
    mini = build_fixture(
        source_dir=source_dir,
        out_dir=Path(datasets_dir) / "plankton-miniset",
        class_counts=MINISET_CLASS_COUNTS,
        seed=0,
        strict=False,
    )
    _report(mini)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    ap.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    args = ap.parse_args()
    build_all(args.source_dir, args.fixtures_dir, args.datasets_dir)


if __name__ == "__main__":
    main()
