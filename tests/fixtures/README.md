# Test fixtures

## `plankton-toyset/` — committed Tier-1 fixture

A single `parquet_images` Parquet file with **image bytes inlined** and a
canonical `split` column (`train` / `val`), consumed by the dojo `parquet_images`
backend via `split_column: split`. The `*.parquet` payload is tracked with **git
LFS** (`../../.gitattributes`).

6 visually-distinct classes, **deliberately imbalanced** (~12:1) with a
2-instance class, so the clean baseline already exercises the minimum-stratification
boundary (1 train / 1 val):

| class | total | train | val |
|---|---|---|---|
| bubble | 24 | 19 | 5 |
| fiber | 16 | 13 | 3 |
| Copepod_nauplii | 10 | 8 | 2 |
| Skeletonema | 6 | 5 | 1 |
| Trichodesmium | 4 | 3 | 1 |
| Dinophysis_acuminata | 2 | 1 | 1 |

Labels are remapped to contiguous `0..5`; the original dataset label is kept as
`original_label` and the human-readable name as `classname`. The directory also
holds `data.yaml` (example dojo data config) and `summary.json` (per-class
counts + label map).

## `defects.py` — failure-mode fixtures by injection

Failure-mode datasets are **not** committed. Instead, `defects.py` mutates the
clean `plankton-toyset` table in-process to manufacture exactly one defect at a time,
writing a throwaway `parquet_images` dir (typically under `tmp_path`). See the
module docstring for the rationale and the `conftest.py` wiring snippet.

Available defects (`build_defect(name, dest, **kw)`):

| defect | targets |
|---|---|
| `empty_eval_class` / `empty_train_class` | `empty_eval_classes` / `empty_train_classes` preflight |
| `noncontiguous_labels` | `non_contiguous_class_indices` preflight |
| `single_instance_class` | un-stratifiable class |
| `missing_targets` | `missing_required_targets` / `missing_policy` |
| `all_targets_missing_for_class` | `no_remaining_valid_target_labels` |
| `missing_target_column` | config references a missing column |
| `duplicate_rows` / `near_duplicate_images` | dedup / dataset hashing |
| `corrupt_image_truncated` / `null_image_bytes` | image decode / missing-bytes paths |

`python tests/fixtures/defects.py` smoke-builds every variant and prints the
row-count delta.

## `datasets/plankton-miniset` — Tier-2 dev dataset (committed via LFS)

A larger 30-class long-tail stand-in, written under `datasets/plankton-miniset/`. It is
the one `datasets/` entry un-ignored in `.gitignore` and committed via Git LFS, so
the `p1/plankton-mini_efficientnet` example experiment (and the `data=plankton-miniset`
group override) run on a fresh clone. Built by the same `build_fixtures.py`. Note
`bubble` is capped to the 34 rows that exist in the source (requested 120), so the
set is ~840 images; raise/redistribute counts in `MINISET_CLASS_COUNTS` to taste.

## Regenerating

Fixtures are sampled from a local copy of the Tier-3 dataset
`sbatchelder/NES-plankton-classifier-2022-dataset`, expected (gitignored) at
`./datasets/NES-plankton-classifier-2022-dataset/data`. The builder needs only
`pyarrow` (a base dependency); no `datasets`/`numpy` required. Sampling is seeded
(`seed=0`), so regeneration is reproducible.

```bash
python tests/fixtures/build_fixtures.py   # rebuilds plankton-toyset + datasets/plankton-miniset
```
