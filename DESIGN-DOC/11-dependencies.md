# 11. Dependencies

## Purpose

Defines the lightweight base install plus optional extras layout.
Distinguishes core dependencies (config / schema / storage / result
reading) from extras gated by functional area.

## Principles

- Base install supports config validation, inspection, storage / result
  access, schema handling, and artifact introspection **without
  requiring Torch**.
- Training, SSL, Aim, MLflow, ONNX, and IFCB support live behind extras.
- `amplify-db-utils` and `amplify-storage-utils` are core dependencies.

## Base dependencies

```toml
dependencies = [
  "pydantic",
  "pydantic-settings",
  "hydra-core",
  "omegaconf",
  "pyarrow",
  "duckdb",
  "amplify-storage-utils",
  "amplify-db-utils",
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
  "torchmetrics",
  "numpy",
  "pandas",
  "pillow",
]

timm = ["timm"]

ssl = ["lightly"]

ifcb = ["ifcbkit[s3]"]

repr_eval = [
  "umap-learn",
  "hdbscan",
  "scikit-learn",
]

aim = ["aim"]
mlflow = ["mlflow"]
onnx = ["onnx", "onnxruntime"]

s3 = ["amplify-storage-utils[s3]"]

all = [
  "image_classifier_dojo[train,timm,ssl,ifcb,repr_eval,aim,onnx,s3]",
]

dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit",
]
```

### Extra purpose summary

- `train` — Torch stack required to run training / inference.
- `timm` — first-class `model.backbone.source: timm` support. **Not
  deferred**; gated by this extra and raises a clear runtime error when
  missing.
- `ssl` — Lightly only. SSL DINOv2 ViT backbones come from the `timm`
  extra; install both for SSL.
- `ifcb` — `ifcbkit` for the `ifcb_bins` dataset backend. Usable
  outside SSL (supervised IFCB training, holdout eval, inspect). Pulls
  `ifcbkit[s3]` so IFCB-via-S3 works when `s3` is also installed.
- `repr_eval` — UMAP, HDBSCAN, and scikit-learn for representation
  evaluation (projections, clustering, linear / ridge probes, baseline
  metrics). Usable against supervised encoders, not SSL-only.
- `aim` — Aim logger sink.
- `mlflow` — MLflow logger sink (schema present; runtime stubbed —
  see `appendix-deferred-features.md`).
- `onnx` — ONNX export and runtime.
- `s3` — S3 capability for storage (`amplify-storage-utils[s3]`).
- `all` — convenience meta-extra that pulls every functional extra
  above.
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

# IFCB bins over S3
pip install image_classifier_dojo[train,ifcb,s3]

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
- `appendix-deferred-features.md` — MLflow runtime, HDF, WebDataset.
