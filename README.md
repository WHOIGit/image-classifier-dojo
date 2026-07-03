# image-classifier-dojo

Train computer-vision classifier models, configured through Hydra + Pydantic,
reading Parquet datasets and writing canonical, queryable Parquet results.

> **Status.** This is a clean-break refactor of the `dojo` package. The current
> code implements the **Priority 1 end-to-end thin slice**: one supervised,
> single-head training run wired through every architectural boundary (config →
> data → model → training → storage → results). The full capability roadmap
> (SSL, ensembling, sweeps, export, tabular input, IFCB bins, …) lives in
> [`DESIGN-DOC/`](DESIGN-DOC/13-workplan.md). 

## Installation

The base install is Torch-free — it carries only config, storage, and result
tooling. Training needs the `train` extra (Torch / Lightning / torchvision).

```bash
git clone git@github.com:WHOIGit/image-classifier-dojo.git
cd image-classifier-dojo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Install the Torch build matching your CUDA (or CPU) first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Editable install with the training extra.
pip install -e '.[train]'
```

Optional extras (combine as needed, e.g. `.[train,ssl]`): `train`, `timm`,
`ssl`, `ifcb`, `repr_eval`, `s3`, `onnx`, `all`, and `dev` (test/lint tooling).
The base install alone is enough to compose/validate configs and read result
Parquet without Torch.

The installed console script is `dojo`.

## Commands

The CLI is a [Typer](https://typer.tiangolo.com/) front-end over the Hydra
Compose API. Two commands exist in the P1 slice:

| Command | Purpose |
|---|---|
| `dojo inspect config` | Compose + validate a config, render resolved output paths, and report enabled outputs. |
| `dojo train` | Compose + validate a config and run one supervised training job. |

Both example datasets below are committed (via Git LFS), so these run on a fresh
clone — from the repo root:

```bash
# Dry-run a config: validate and see the resolved run paths / hashes / outputs.
dojo inspect config experiment=p1/plankton-toy

# Train on the bundled toy fixture (6 classes, ~62 images) — the default example.
dojo train experiment=p1/plankton-toy

# Train on the larger plankton-miniset dev set (30 classes, ~842 images).
dojo train experiment=p1/plankton-mini_efficientnet

# JSON report instead of the rich table.
dojo inspect config experiment=p1/plankton-toy --format json
```

### Overriding the data block

`data=<name>` selects a **config group** by name (from
`src/dojo/config_defaults/data/` or the project-root `./configs/data/`) — it is
not a file path. Swap the dataset
on any experiment by selecting a different data group and matching the head's
class count:

```bash
# Run the toy experiment against the plankton-miniset dataset instead.
dojo train experiment=p1/plankton-toy data=plankton-miniset model.heads.species.num_classes=30

# Same idea starting from a config file rather than a packaged experiment.
dojo train --config src/dojo/config_defaults/experiment/p1/plankton-toy.yaml \
  data=plankton-miniset model.heads.species.num_classes=30
```

## Config composition

Configs are composed with the **Hydra Compose API** (but _NOT_ the hydra launcher). 
The rule of thumb: **dash-prefixed tokens are command options;
dash-free `key=value` tokens are config overrides.**

- **Pick an experiment** with the `experiment=` group selector:
  ```bash
  dojo train experiment=p1/plankton-toy
  ```
- **Override any value inline** (dash-free, dotted paths):
  ```bash
  dojo train experiment=p1/plankton-toy runtime.num_workers=0 training.max_epochs=1 output_root=./runs
  ```
- **Point at a file** instead of a packaged experiment, or re-run a previously
  resolved config artifact:
  ```bash
  dojo train --config path/to/root.yaml
  dojo train --resolved-config runs/<experiment>/<run_id>/config/resolved.yaml
  ```

An experiment file is a small root that pulls config *groups* together via its
`defaults:` list — `runtime`, `storage`, `data`, `transforms`, `backbone`,
`optimizer`, `training_outputs`, plus the `model`, `objectives`, and `training`
blocks. Packaged config groups ship inside the package at
[`src/dojo/config_defaults/`](src/dojo/config_defaults). A project-root
`./configs/` directory, if present, **shadows and extends** the packaged groups,
so you can override or add configs without copying the whole tree.

Composition, validation, and path/runtime resolution live in
[`src/dojo/config_loader/`](src/dojo/config_loader): `compositor.py` (Hydra
compose), `resolver.py` (render `run_id`, resolve output paths, derive the
inference pipeline), and `conductor.py` (sequences compose → validate → resolve
into one call). The strict Pydantic contract is in
[`src/dojo/config_schemas/`](src/dojo/config_schemas).

### Run directory

A training run resolves to `<output_root>/<experiment>/<run_id>/` containing:

```
config/        resolved.yaml  (the fully-resolved config artifact)
checkpoints/   best-k *.ckpt + last.ckpt
metrics/       metrics.csv    (the `local` logger sink)
results/       canonical Parquet + _metadata.json
```

## Training results (Parquet format)

Each run writes a single **tall Parquet table** under `<run>/results/`, via
[amplify-db-utils](https://github.com/WHOIGit/amplify-db-utils)
(`DuckDBParquetStore`), hive-partitioned by `record_type` (configurable through
`training_outputs.results.partition_by`). Every row shares one union schema and
is discriminated by its `record_type`; columns not applicable to a row are null.

**Record types (P1):**

- `sample_metadata` — one static row per evaluated sample (native / resize
  dimensions, `source_extra_json`, …).
- `classification_output` — one row per sample per head: `head_name`, `target`,
  `prediction_index`, `prediction_label`, `prediction_confidence`, and the full
  `logits` / `probabilities` vectors (Arrow `list<float32>`, lossless).

**Common provenance columns** on every row include `sample_id`, `uri`, `split`,
`stage` (P1: `train_validation`), `record_type`, `run_id`, `config_hash`,
`dataset_hash`, `checkpoint_hash`, `epoch`, `global_step`, and `schema_version`.
The `*_hash` columns are deterministic and reproducible across runs of the same
config, so results are self-describing and joinable.

**Sidecar.** A `_metadata.json` file sits next to the table with the schema
name/version, `run_id`, and per-record-type semantics — including each
classification head's `target`, the ordered `classes` list, and the
`class_mapping` (index → class name).

**Reading results.** Because the table is written through `amplify-db-utils`, any
reader can open the results directory and filter by `stage` / `record_type`
(equality, `in`, and range filters):

```python
from dojo.results import ResultReader

reader = ResultReader("runs/<experiment>/<run_id>/results")
rows = reader.read({"record_type": "classification_output", "split": "val"})
n = reader.count({"stage": "train_validation"})
```

The full result-schema, hashing, and artifact-layout specification is in
[`DESIGN-DOC/06-results-artifacts-and-metadata.md`](DESIGN-DOC/06-results-artifacts-and-metadata.md).

## amplify dependencies

The dojo builds on two AMPLIfy libraries (installed automatically as base
dependencies):

- **[amplify-db-utils](https://github.com/WHOIGit/amplify-db-utils)** — the
  DuckDB/Parquet result store and schema registry behind the canonical result
  writer and reader.
- **[amplify-storage-utils](https://github.com/WHOIGit/amplify-storage-utils)** —
  the storage backends behind the dojo's URI-based storage interface
  ([`src/dojo/storage/`](src/dojo/storage)): localizing dataset URIs and writing
  run artifacts. P1 uses the local filesystem backend, with an object-store
  (S3) seam. (Path/template resolution of `output_root` / `dir_template` is the
  dojo's own `config_loader/resolver.py`.)

## Example datasets

Two small datasets are committed via **Git LFS** so the examples above run on a
fresh clone (all `*.parquet` are LFS-tracked; `git lfs install` once per machine):

- **`tests/fixtures/plankton-toyset/`** — 6 classes, ~62 images. The default example
  and a test fixture.
- **`datasets/plankton-miniset/`** — 30 classes, ~842 images. A more realistic dev set;
  the one `datasets/` entry un-ignored in `.gitignore`.

The full **NES-plankton-classifier-2022** dataset is *not* committed. You bring it
(e.g. download from HuggingFace into `./datasets/`), and the packaged
[`data/nes_plankton_parquet_images`](src/dojo/config_defaults/data/nes_plankton_parquet_images.yaml)
group is the template pointing at it. Dataset materialization / inspection helpers
(`dojo init`, `dojo inspect dataset`) are on the P2 roadmap
([`DESIGN-DOC/13-workplan.md`](DESIGN-DOC/13-workplan.md)), not yet implemented.

## Development

```bash
pip install -e '.[train,dev]'

pytest                     # fast loop — skips the real-fit integration tests
pytest --run-expensive     # everything, including end-to-end training fits
```

Expensive tests (real Lightning fits / multiprocess dataloaders) are marked
`@pytest.mark.expensive` and skipped by default; pass `--run-expensive` to
include them.
