
# Image Classifier Dojo — Refactor Design Doc

## Purpose

Top-level entry point and reading guide for the modular design doc. Each
section below links to its dedicated file. New readers should follow the
sections in order; returning readers should jump directly to the topic
they need.

## Architectural shape

Dojo is a Hydra + Pydantic configuration-first toolkit for training,
evaluating, ensembling, exporting, and inspecting image classification /
regression / ordinal / SSL models. Key shape:

- **Config-first.** Hydra composes; Pydantic validates and is the
  runtime contract.
- **Modular model composition.** `image_input.backbone` (`torchvision` /
  `timm` / `checkpoint`) + optional `tabular_input` encoder + implicit
  image-then-tabular embedding concatenation + optional
  `embedding_adapter` + one-or-more heads. Objectives bind heads to
  losses, metrics, and weights.
- **Canonical results.** Per-row Parquet via `amplify-db-utils` with a
  `_metadata.json` sidecar. Configurable partitioning, dictionary
  encoding, and Arrow list columns for vectors.
- **Three peer output blocks** at the top of the config tree:
  `training_outputs`, `ensemble_outputs`, and `sweep_outputs`, all
  rooted under a single `output_root`.
- **Task selection via `task.type`** (`supervised`, `ssl`,
  `snapshot_ensemble`) — there are no `train supervised` / `train ssl`
  subcommands.
- **Representation evaluation** (`representation_eval`) is a top-level
  block, usable against supervised or SSL encoders, training-integrated
  or standalone.
- **Ensembling** is prediction-space only in the initial implementation:
  explicit candidate discovery, candidate manifests, supported selection
  strategies, and combine modes. Snapshot ensembles share a run
  directory with their training step.
- **Sweeps** normalize into a top-level `sweep:` block. Functional
  `mode: grid` supports ordinary hyperparameter sweeps and
  batch-run-style seed sweeps; `mode: bayesian` is a deferred schema
  slot.
- **Lightweight base install + extras.** Base install supports config /
  schema / storage / result reading without Torch.

## Design principles

- Pydantic schemas are the runtime contract.
- Keep task logic separate from model composition.
- Prefer canonical internal representations (single-head normalizes to
  the multi-head shape).
- Results are first-class artifacts; tall Parquet with a sidecar.
- Avoid generic metadata junk drawers; use explicit columns.
- Keep storage behind a Dojo interface (`amplify-storage-utils`
  underneath).
- Use `ifcbkit` for IFCB raw-bin handling.
- Make embeddings first-class.
- Optimize for external orchestration — entry points return structured
  results.

## File map

Read in order:

- [01. Goals and Scope](01-goals-and-scope.md) — what's in, what's out,
  what's deferred.
- [02. CLI and Task Types](02-cli-and-task-types.md) — canonical command
  surface; `dojo init` / `train` / `infer` / `eval` / `inspect` /
  `ensemble` / `export`; `task.type` semantics.
- [03. Configuration](03-configuration.md) — canonical config tree;
  `output_root` and the three peer `*_outputs` blocks; path-template
  syntax; existing-run-dir policy.
- [04. Data and Storage](04-data-and-storage.md) — supported dataset
  backends (`csv_manifest`, `parquet_manifest`, `parquet_images`,
  `ifcb_bins`); shared sample contract; `dojo inspect dataset`;
  storage resolver.
- [05. Models, Training, and Heads](05-models-training-and-heads.md) —
  transforms, backbones, freeze policies, heads, objectives, supervised
  training, optimizer / scheduler / checkpointing, transfer learning.
- [06. Results, Artifacts, and Metadata](06-results-artifacts-and-metadata.md)
  — canonical result schemas; identifiers and hashes;
  `_metadata.json` sidecar; partitioning; on-disk artifact layout;
  logging and diagnostics.
- [07. SSL and Representation Evaluation](07-ssl-and-representation-eval.md)
  — Lightly DINOv2 (functional); `representation_eval`; probes,
  projections, clustering, diagnostics.
- [08. Ensembles](08-ensembles.md) — prediction-space ensembling;
  candidate discovery; manifests; selection strategies; combine modes;
  `ensemble_outputs:` layout; snapshot-ensemble integration.
- [09. Sweeps and Batch Runs](09-sweeps-and-batch-runs.md) — top-level
  `sweep:` block; grid sweeps; batch-run-style seed sweeps; deferred
  Bayesian schema; `sweep_outputs:` block; sweep-id provenance; sweeps
  as candidate-source feeders.
- [10. Export](10-export.md) — TorchScript and ONNX exports;
  `*_outputs.export` sub-blocks; `dojo export` command; export
  metadata; bucket-aware ONNX.
- [11. Dependencies](11-dependencies.md) — base install plus optional
  extras (`train`, `timm`, `ssl`, `ifcb`, `repr_eval`, `onnx`, `all`,
  `dev`). The `aim` extra is commented out and no `mlflow` / `s3` extra
  is currently declared (Aim/MLflow logging is deferred; S3 rides on the
  base `amplify-storage-utils` dependency).
- [12. Validation, Testing, and Preflight](12-validation-testing-and-preflight.md)
  — layered validation; `runtime.preflight` controls; stub-test policy
  for deferred features; test scope by area; fixture tiers.
- [13. Workplan](13-workplan.md) — priority order for the new
  `src/dojo` implementation; thin-slice gate; deferred backlog.

Appendices and reference:

- [Appendix — Deferred Features](appendix-deferred-features.md) —
  runtime stubs with their `NotImplementedError` test obligations, plus
  schema backlog and cleanup items.
- [Appendix — Repository Structure](appendix-repository-structure.md) —
  proposed `configs/`, `src/dojo/`, and `tests/` tree derived from the
  canonical decisions above.
- [Glossary](glossary.md) — config keys, identifiers, hashes, record
  taxonomies; the single source of truth for terminology.

## Tightly-coupled cross-cutting links

Bookmark these pairs:

- `03-configuration.md` & `06-results-artifacts-and-metadata.md` —
  `output_root`, the `*_outputs` blocks, and path templating connect
  to result writing and the artifact layout.
- `06-results-artifacts-and-metadata.md` & `08-ensembles.md` — cached
  result files feed ensembles; ensemble result rows reuse the standard
  schema with `stage=ensemble_eval`, `ensemble_result_scope`, and the
  `ensemble_member_id` union column for retained member rows.
- `05-models-training-and-heads.md` &
  `07-ssl-and-representation-eval.md` — supervised training can
  schedule representation evaluation; representation eval also runs
  standalone against checkpoints from either task type.
- `09-sweeps-and-batch-runs.md` & `03-configuration.md` /
  `06-results-artifacts-and-metadata.md` / `08-ensembles.md` —
  `sweep_outputs:` schema; `sweep_id` provenance on result rows;
  sweeps that feed ensemble candidate discovery.
- `12-validation-testing-and-preflight.md` &
  `appendix-deferred-features.md` — runtime-stub entries name their
  `NotImplementedError` tests; schema backlog and cleanup items name
  their validation or verification obligations.
