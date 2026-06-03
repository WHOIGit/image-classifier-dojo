
# 11. Dependencies

## Purpose

Defines the lightweight base install plus optional extras layout.
Distinguishes core dependencies (config / schema / storage / result
reading) from extras gated by functional area.

## Principles

- Base install supports config validation, inspection, storage / result
  access, schema handling, and artifact introspection **without
  requiring Torch**.
- Training, SSL, ONNX, and IFCB support live behind extras. Logging
  sinks other than `local` (Aim, MLflow) are deferred / stubbed at
  runtime; metrics and figures are recorded locally for the foreseeable
  future. The `aim` extra is commented out (undeliverable on current
  Python) and no `mlflow` extra is currently declared.
- `amplify-db-utils` and `amplify-storage-utils` are core dependencies.

## Base dependencies

The two `amplify-*` packages and `ifcbkit` are hosted on the WHOIGit
GitHub organization and installed as direct git references (unpinned,
tracking the default branch); `allow-direct-references = true` is set in
`pyproject.toml`.

```toml
dependencies = [
  "pydantic",
  "pydantic-settings",
  "hydra-core",
  "omegaconf",
  "pyarrow",
  "duckdb",
  "amplify-storage-utils @ git+https://github.com/WHOIGit/amplify-storage-utils.git",
  "amplify-db-utils @ git+https://github.com/WHOIGit/amplify-db-utils.git",
  "typer",
  "rich",
  "coolname",
  "humanize",
  "tqdm",
]
```

## Optional extras

```toml
[project.optional-dependencies]
train = [
  "torch",
  "torchvision",
  "lightning",
  "torchmetrics[visual]",
  "numpy",
  "pandas",
  "pillow",
]

timm = ["timm"]

ssl = ["lightly"]

ifcb = ["ifcbkit @ git+https://github.com/WHOIGit/ifcbkit.git"]

repr_eval = [
  "umap-learn",
  "hdbscan",
  "scikit-learn",
]

# aim extra is defined but its runtime is stubbed and it is currently
# undeliverable on Python 3.13/3.14 (aimrocks has no wheel), so it is
# commented out in pyproject.toml and excluded from `all`.
# aim = ["aim"]
onnx = ["onnx", "onnxruntime-gpu"]

all = [
  "image_classifier_dojo[train,timm,ssl,ifcb,repr_eval,onnx]",
]

dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit",
]
```

Notes on the current `pyproject.toml` state:

- `torchmetrics[visual]` pulls the extra needed for figure/visualization
  metrics.
- `ifcb` installs `ifcbkit` from git (no `[s3]` extra, to avoid its
  conflicting pinned `amplify-storage-utils` reference).
- `onnx` uses `onnxruntime-gpu`.
- The `mlflow` extra and a standalone `s3` extra are **not currently
  declared**. MLflow remains a deferred logger sink (schema present;
  see `appendix-deferred-features.md`); its extra can be re-added when
  the runtime is unstubbed. S3 capability rides along with the git
  `amplify-storage-utils` dependency rather than a separate extra.

### Extra purpose summary

- `train` — Torch stack required to run training / inference.
- `timm` — first-class `model.backbone.source: timm` support. **Not
  deferred**; gated by this extra and raises a clear runtime error when
  missing.
- `ssl` — Lightly only. SSL DINOv2 ViT backbones come from the `timm`
  extra; install both for SSL.
- `ifcb` — `ifcbkit` (from git) for the `ifcb_bins` dataset backend.
  Usable outside SSL (supervised IFCB training, holdout eval, inspect).
  Installed without `ifcbkit`'s own `[s3]` extra to avoid its pinned
  `amplify-storage-utils` reference conflicting with the base git
  dependency; S3 access still comes through the base
  `amplify-storage-utils` install.
- `repr_eval` — UMAP, HDBSCAN, and scikit-learn for representation
  evaluation (projections, clustering, linear / ridge probes, baseline
  metrics). Usable against supervised encoders, not SSL-only.
- `aim` — Aim logger sink (schema present; **runtime stubbed** — see
  `appendix-deferred-features.md`). For the foreseeable future only the
  `local` sink is functional; metrics and figures are recorded locally.
  The `aim` extra is additionally undeliverable on current Python
  (`aimrocks` ships no 3.13/3.14 wheel) and is commented out in
  `pyproject.toml`.
- `mlflow` — MLflow logger sink, deferred (schema present; runtime
  stubbed — see `appendix-deferred-features.md`). **No extra is
  currently declared** in `pyproject.toml`; re-add when unstubbed.
- `onnx` — ONNX export and runtime (`onnxruntime-gpu`).
- *(no standalone `s3` extra)* — S3 storage capability comes through the
  base `amplify-storage-utils` git dependency rather than a separate
  extra.
- `all` — convenience meta-extra that pulls every functional extra
  above (`train,timm,ssl,ifcb,repr_eval,onnx`).
- `dev` — testing and developer tooling.

## Common install recipes

```bash
# Schema / config inspection / result reading only
pip install image_classifier_dojo

# Supervised training
pip install image_classifier_dojo[train]

# Supervised + timm backbones
pip install image_classifier_dojo[train,timm]

# SSL DINOv2 with representation evaluation
pip install image_classifier_dojo[train,timm,ssl,repr_eval]

# IFCB bins (S3 access comes through the base amplify-storage-utils dep)
pip install image_classifier_dojo[train,ifcb]

# All functional extras
pip install image_classifier_dojo[all]

# Local development
pip install -e .[all,dev]
```

## Dropped / removed dependencies

- `torchensemble` — dropped. Bagging / Boosting / Fusion / Adversarial /
  FastGeometric strategies are not ported. See `08-ensembles.md`.
- `pyifcb` — dropped in favor of `ifcbkit`.
- `h5py` / `tables` and the `hdf` extra — dropped. HDF-derived result
  exports are deferred; see `appendix-deferred-features.md`.

## Cross-References

- `01-goals-and-scope.md` — non-goals and deferred features.
- `04-data-and-storage.md` — `ifcb` and `s3` extras.
- `05-models-training-and-heads.md` — `train` and `timm` extras.
- `06-results-artifacts-and-metadata.md` — `aim`, `mlflow` extras.
- `07-ssl-and-representation-eval.md` — `ssl` and `repr_eval` extras.
- `10-export.md` — `onnx` extra.
- `appendix-deferred-features.md` — Aim runtime, MLflow runtime, HDF,
  WebDataset.
