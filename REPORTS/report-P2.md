# P2 Implementation and Experiment Report

## Purpose

Continuous record for completing Priority 2 and running the NES plankton
training experiments requested after P2 is ready.

## 2026-07-03 — Continuation Start

- Starting state: P1 is complete and an initial P2 foundation slice is present
  in the working tree: `dojo init`, `dojo inspect dataset`, `csv_manifest`,
  `parquet_manifest`, `parquet_images`, and P2.5a classification result rows
  with `head_hash`, `target_index`, and `target_name`.
- User direction: continue P2 until all P2 items are complete, then run the
  NES training experiments.
- Baseline experiment target: `efficientnet_b0`, 224 x 224 image input, AdamW,
  cross entropy, horizontal and vertical flip augmentations, and normalization
  values acquired once from `dojo inspect dataset` over the NES training split.
- Normalization decision: `dojo inspect dataset --normalization` is the source
  of deterministic train-split mean/std. Those values are then hard-coded into
  experiment configs rather than dynamically consumed during training.
- Figure scope added to P2: standalone Plotly HTML files for train/val loss,
  first-epoch-normalized losses, val F1 macro/micro, confusion matrix, and
  per-class dropdown metrics.
- Next implementation milestone: stats-cache/normalization hardening and
  class-imbalance training support, then figures/analysis and remaining P2
  contracts.

## 2026-07-03 — Weighted Cross Entropy Milestone

- Implemented `weighted_cross_entropy` as a real strict-schema objective loss
  value.
- Weight source: train-split class counts computed by the dataset builder and
  passed into `SupervisedTaskModule`.
- Supported weighting schemes:
  - `inverse_frequency` (default)
  - `effective_number` with configurable `beta`
- Weights are normalized to mean one over non-empty classes so the loss scale
  stays comparable to plain cross entropy.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/training/test_metrics_and_losses.py tests/unit/config_schemas`
  (`23 passed`).

## 2026-07-03 — Dataset Preflight Milestone

- Implemented train-time dataset preflight checks for:
  - `empty_train_classes`
  - `non_contiguous_class_indices`
  - `imbalance_ratio_gt`
- `error` severity stops `execute_train`; `warn` severity emits a Python
  warning; `ignore` suppresses the issue.
- Class-count source is the deterministic dataset bundle computed before
  training.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/data/test_preflight.py tests/unit/data/test_parquet_images.py
  tests/unit/training/test_metrics_and_losses.py` (`12 passed`).

## 2026-07-03 — Training Figures Milestone

- Added standalone HTML figure generation under `training_outputs.figures.dir`
  when `training_outputs.figures.enabled: true`.
- Implemented available P2 figure outputs:
  - `loss_curves.html`
  - `loss_curves_normalized.html`
  - `val_f1_curves.html`
  - `confusion_matrix.html`
  - `per_class_metrics.html`
- Plot implementation uses embedded JSON data plus Plotly.js from the CDN,
  avoiding a new Python Plotly dependency.
- Inputs:
  - Lightning `metrics/metrics.csv` for epoch line plots.
  - Canonical result Parquet `classification_output` rows for confusion matrix
    and per-class metrics.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/training/test_figures.py tests/unit/training` (`8 passed`).

## 2026-07-03 — Embedding Adapter Milestone

- Implemented `model.embedding_adapter.enabled: true` for supervised models.
- Supported adapter types:
  - `linear`
  - `mlp`
- The adapter transforms the image backbone embedding before all heads; heads
  are built against the adapter output dimension.
- Tabular input remains deferred/not enabled.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/model/test_supervised.py tests/unit/model tests/unit/config_schemas`
  (`31 passed`).

## 2026-07-03 — Compatibility Hash and Checkpoint Contract Milestone

- Resolved supported torchvision `efficientnet_b0` `output_dim: auto` to the
  concrete model shape (`1280`) during config resolution.
- Implemented compatibility hash extractors for the currently supported
  supervised subset:
  - `target_schema_hash`
  - `class_mapping_hash`
  - `model_config_hash`
  - `preprocessing_hash`
- Result `_metadata.json` now carries the compatibility block instead of `null`.
- `SupervisedTaskModule.on_save_checkpoint` embeds
  `checkpoint["dojo_inference_contract"]`.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/config_schemas/test_hashing.py tests/unit/results/test_metadata.py
  tests/unit/training/test_checkpoint.py tests/unit/training/test_metrics_and_losses.py
  tests/unit/model/test_supervised.py` (`28 passed`).

## 2026-07-03 — Aspect Bucket Transform and Sampler Milestone

- Added `transforms.pipeline` support for:
  - `aspect_bucket`
  - `grayscale`
- `aspect_bucket` chooses a configured canvas from native image dimensions and
  resizes via letterboxing.
- Added `training.sampler.type: batch_aspect_buckets`, which groups DataLoader
  batches by deterministic bucket assignment so variable canvas sizes remain
  tensor-stackable.
- Added `aspect_bucket` to the shared sample/batch contract and
  `sample_metadata` result rows.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/data/test_transforms_builder.py
  tests/unit/data/test_aspect_bucket_sampler.py tests/unit/data/test_parquet_images.py
  tests/unit/data/test_manifest_backends.py` (`13 passed`).

## 2026-07-03 — Inference and Holdout Eval Milestone

- Added `src/dojo/inference/` for checkpoint-backed non-training forward
  passes.
- Added minimal `eval_outputs` path resolution.
- Added canonical result support for:
  - `record_type=embedding`
  - `stage=infer`
  - `stage=holdout_eval`
- Added `dojo infer predictions`, `dojo infer embeddings`, and
  `dojo eval holdout` command groups.
- Inference rebuilds the model from `checkpoint["dojo_inference_contract"]`
  instead of the producing run's `resolved.yaml`.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/inference/test_run.py tests/unit/results/test_writer.py
  tests/unit/results/test_reader.py tests/unit/results/test_metadata.py`
  (`10 passed`).

## 2026-07-03 — Inspect Backbone / Checkpoint and Backbone Registry Milestone

- Added `dojo inspect backbone`.
- Added `dojo inspect checkpoint`.
- Broadened backbone schema/building:
  - `architecture.source: torchvision | timm`
  - torchvision architecture name is now open string.
  - `weights.source: none | library | checkpoint`
- `timm` remains optional and raises a clear runtime error if the extra is not
  installed.
- Checkpoint initialization supports Dojo Lightning state dict prefixes.
- Verification: focused compile passed; focused tests passed with
  `tests/unit/model/test_backbone.py tests/integration/test_inspect_config.py`
  (`10 passed`).

## 2026-07-03 — NES Dataset Inspect / Normalization

- Command intent: run `dojo inspect dataset` over
  `datasets/NES-plankton-classifier-2022-dataset` with `--stats
  --normalization` to acquire deterministic train-split normalization for the
  experiment configs.
- Dataset backend: `parquet_images` with split-from-filename.
- Dataset hash: `sha256:b46e9b74e0eef36d012dfb59c8c39b74ecdae7654ad41d55632b79bdb810cf6a`.
- Splits:
  - train: 77,617 rows
  - val: 19,409 rows
- Classes: 155 (`label` index range 0-154).
- Normalization values to hard-code:
  - mean: `[0.6425475478172302, 0.6425475478172302, 0.6425475478172302]`
  - std: `[0.19147475063800812, 0.19147475063800812, 0.19147475063800812]`
- Note: the current stats-cache writer inlines per-sample dimensions, which
  makes `dojo_stats.json` very large for NES. This is acceptable as a local
  generated artifact for this run, but should be changed to the design-doc
  sidecar-Parquet format before treating stats caches as durable committed
  assets.

## 2026-07-03 — NES Experiment Configs

- Added local experiment configs under `configs/experiment/p2/`:
  - `nes_effb0_224_baseline.yaml`
  - `nes_effb0_buckets.yaml`
  - `nes_effb0_weighted.yaml`
  - `nes_effb0_buckets_weighted.yaml`
- Shared baseline choices:
  - EfficientNet-B0 backbone, no library pretraining.
  - Linear embedding adapter (`output_dim: 512`).
  - AdamW optimizer.
  - Cross entropy or weighted cross entropy depending on variant.
  - Horizontal and vertical flip augmentation.
  - NES train-split normalization hard-coded from inspect output.
  - Figures enabled.
- Validation: all four configs passed `dojo inspect config --format json`.

## 2026-07-03 — NES Baseline Smoke Fit and Metrics Cleanup

- Ran a limited CPU smoke fit of `experiment=p2/nes_effb0_224_baseline` with:
  - `training.max_epochs=1`
  - `limit_train_batches=2`
  - `limit_val_batches=2`
  - results/figures disabled for speed
- Outcome: fit completed and wrote a checkpoint.
- Observed expected preflight warning:
  `imbalance_ratio_gt` train ratio about `72.7`, threshold `20`.
- Added post-fit metrics CSV cleanup so Lightning's split train/val epoch rows
  are merged into one row per epoch.
- Verification: `tests/unit/training/test_figures.py` passed (`2 passed`).

## 2026-07-03 — P2 Sampler / Foreground Transform Completion

- Added `foreground_crop` to the transform schema and image transform builder.
- Added `training.sampler.type` values:
  - `class_balanced`
  - `weighted`
- Class-balanced / weighted samplers derive per-sample weights from train-split
  class counts and can compose with `aspect_bucket` by sampling inside each
  bucket so batches remain stackable.
- Non-training paths now choose bucket grouping from the dataset's resolved
  aspect-bucket state and do not resample validation / inference / holdout
  datasets.
- Verification: focused data/schema/training tests passed
  (`34 passed`).

## 2026-07-03 — NES Memory Fix: Lazy Parquet Images / Streaming Results

- Problem found during full NES readiness work: `parquet_images` eagerly read
  the embedded `image.bytes` column into Python lists, and DataLoader workers
  could duplicate that memory. Validation scoring also accumulated all result
  rows, including logits/probabilities, before writing.
- Refactor:
  - `parquet_images` split tables now exclude the embedded image column.
  - Split tables carry lightweight hidden row references:
    `__dojo_file_path`, `__dojo_row_group`, and `__dojo_row_in_group`.
  - `ParquetImagesDataset.__getitem__` reads image bytes lazily from the source
    Parquet row group through a one-row-group reader cache.
  - `dojo inspect dataset` uses the same lazy row references for dimensions and
    normalization.
  - Training result scoring and inference/eval output writing now flush records
    per batch instead of accumulating a full split.
- Regression coverage:
  - Added a unit assertion that `ParquetImagesDataset` has row references and
    no `_image_bytes` materialization.
  - Focused tests passed for parquet images, samplers, inspect dataset,
    figures, inference, and result writer (`16 passed`), then full fast tests
    passed (`110 passed, 8 skipped`).
  - Multiprocess DataLoader integration smoke passed:
    `tests/integration/test_train_supervised.py::test_runs_with_multiprocess_dataloader_workers`.
- NES constructor check:
  - `build_datasets(experiment=p2/nes_effb0_224_baseline)` over the full NES
    files produced splits `train=77617`, `val=19409`.
  - `/usr/bin/time -v` reported max RSS about `1,697,484 KB`.
- GPU note:
  - Requested GPU verification could not run locally because
    `torch.cuda.is_available()` and Lightning CUDA availability both report
    `False` in this session.

## 2026-07-03 02:34 EDT — NES Memory Fix: Materialized Image Cache

- Follow-up diagnosis on a live full NES training process showed RAM remained
  high even after lazy row references:
  - Parent `dojo` process around `2.7 GB` PSS.
  - Eight `pt_data_worker` processes together pushed process-tree PSS to about
    `9.3 GB`.
- Cause: even without retaining row-group tables, PyArrow's Parquet access
  still reads embedded image data at row-group granularity; workers also import
  their own Python/PyTorch/PyArrow stacks under Python 3.14 `forkserver`.
- Implemented `data.image_cache` for `parquet_images`:
  - `enabled`
  - `dir`
  - `progress`
  - `force_rebuild`
- Cache behavior:
  - Computes a deterministic `dataset_content_hash` over source Parquet file
    names and bytes.
  - Uses `<cache_base>/<dataset_content_hash>/manifest.json` as the completion
    marker.
  - Materializes embedded image bytes to normal image files with a Rich progress
    bar.
  - Adds materialized image paths to split tables so DataLoader workers open
    files directly instead of reading Parquet row groups.
  - Reuses the cache when a complete marker with the same content hash and file
    list is present.
- Enabled the image cache in the packaged NES data config.
- Verification:
  - Focused data tests passed (`10 passed`).
  - NES-derived pilot materialization smoke wrote and reused materialized paths
    successfully.
  - Full fast test suite passed (`111 passed, 8 skipped`).

## 2026-07-03 02:43 EDT — Live GPU NES Baseline Cache Check

- User launched:
  `dojo train --config=configs/experiment/p2/nes_effb0_224_baseline.yaml runtime.num_workers=4 training.batch_size=32`.
- Host process check found active parent PID `129340` with four
  `pt_data_worker` processes, matching `runtime.num_workers=4`.
- `nvidia-smi` showed the train process using the GPU:
  - GPU memory for PID `129340`: about `1,832 MiB`.
  - Overall GPU utilization at the sample point: `67%`.
- The materialized cache was complete:
  - Cache directory: `.cache/dojo/materialized_images/02465b0abed4db7f82535fb2a51cc0e52af208afdbd8eb62dd3e63204aab9b93`.
  - Manifest `dataset_content_hash`:
    `sha256:02465b0abed4db7f82535fb2a51cc0e52af208afdbd8eb62dd3e63204aab9b93`.
  - Manifest `image_count`: `97,026`.
  - Disk footprint: about `1.7G`.
- Process-tree PSS at the sample point was about `5.18 GB`:
  - Parent PID `129340`: `2.06 GB` PSS.
  - Forkserver PID `130033`: `0.007 GB` PSS.
  - Worker PIDs `132890`, `132977`, `133065`, `133149`: approximately
    `1.07 GB`, `0.63 GB`, `0.64 GB`, and `1.78 GB` PSS.
- Interpretation: RAM is materially lower than the earlier lazy-Parquet
  worker path (~`9.3 GB` PSS tree), and the active workers are now in the
  file-backed image-loading path. Remaining memory pressure is likely from
  model/framework state, worker imports, DataLoader prefetch, and per-process
  batch/transform state rather than retained embedded Parquet image columns.

## 2026-07-03 02:56 EDT — P2 Data / Eval Contract Cleanup

- Stats-cache cleanup:
  - `dojo inspect dataset --stats` now writes per-sample dimensions to a
    Parquet sidecar instead of embedding dimension rows in the JSON stats
    cache.
  - Full image inspect passes now compute and record a separate
    `dataset_content_hash` over image bytes plus declared tabular feature
    values when present.
  - The JSON report still includes dimension rows for display/inspection, but
    the durable cache references the sidecar.
- Target-label cleanup:
  - Index-only classification targets can now resolve readable class mappings
    from recognized dataset schema metadata, including HuggingFace
    `ClassLabel`-style `names`.
  - Index-only datasets without schema metadata still fall back to index-string
    labels downstream.
- Holdout-eval cleanup:
  - `dojo eval holdout` now reconstructs metrics from the checkpoint's
    `objective_summary` and writes a `metric_summary` into
    `eval_manifest.json`.
- Focused verification passed:
  - Inspect/stats-cache tests.
  - Target-label metadata tests.
  - Inference/holdout-eval tests.

## 2026-07-03 09:47 EDT — Config Compare and Question Queue Review

- Added `dojo inspect config-compare`.
  - Required inputs: `--config-a` and `--config-b`.
  - Accepts authored YAMLs and resolved YAMLs; authored inputs are composed and
    validated but do not have runtime output paths rendered.
  - Output formats: text and JSON.
  - Report sections are organized by hash-source inputs:
    `config_hash`, `target_schema_hash`, `class_mapping_hash`,
    `model_config_hash`, `preprocessing_hash`, per-head `head_hash`, and
    `other` for full-config diffs outside hash inputs.
  - Because the command compares YAMLs only, class-mapping-dependent sections
    use index-string fallback labels and report that assumption.
- Reviewed `QUESTIONS-FOR-SIDNEY.md` against current code.
  - Removed the normalization question because the design doc now explicitly
    defines normalization as a decode-tier statistic and the code matches.
  - Tightened the NES hardware item to reflect the current GPU-enabled run.
  - Kept open questions for materialized-cache hash identity and true
    multi-target dataset scope.
- Focused verification passed:
  - `tests/integration/test_inspect_config.py` (`7 passed`).
  - Focused Ruff checks for compare/inspect files.
  - Full fast suite passed (`115 passed, 8 skipped`).

## 2026-07-03 10:15 EDT — Dataset Inspect Class Mapping Visibility

- Updated `dojo inspect dataset` text output to show:
  - per-head class count total,
  - class index,
  - readable class label,
  - train-split count per class.
- The JSON output already carried the same values under
  `aspects.class_mapping.per_head` and `aspects.class_counts.per_head`; added
  regression assertions for those fields.
- Focused verification passed:
  - `tests/integration/test_inspect_dataset.py` (`2 passed`).
  - Focused Ruff checks for inspect-dataset files.

## 2026-07-03 10:29 EDT — True Multi-Target Multihead Fixture and Runtime Path

- Added a true multi-target toy fixture by extending synthetic manifests with
  `coarse_label` / `coarse_name`, using `artifact` and `organism` labels.
- Updated the supervised data contract:
  - `DecodedSample` and `SampleBatch` keep legacy `target` for the primary
    configured target.
  - They now also expose `targets`, a logical target-name to tensor/index map
    for all configured targets.
- Updated dataset assembly and inspection:
  - `DataBundle` now carries `target_names`, `class_counts_by_target`, and
    `class_mapping_by_target`.
  - `dojo inspect dataset` reports per-head counts and mappings using each
    head's configured target.
- Updated training and inference behavior:
  - Each objective reads labels from the target configured on its head.
  - Weighted loss class counts are selected per head target.
  - Result scoring, checkpoint inference contracts, prediction rows, and
    holdout-eval metrics use per-head target labels and class mappings.
- Added regression tests for:
  - multi-target dataset samples/batches and inspect output,
  - Lightning objective routing where the coarse head would fail if it used the
    primary species labels.
- Decision junction recorded in `QUESTIONS-FOR-SIDNEY.md`: class-balanced /
  weighted DataLoader sampling still defaults to the primary target until we
  decide whether sampler config needs an explicit multi-target policy.
- Verification passed:
  - Focused Ruff checks on touched implementation and tests.
  - Focused tests: `6 passed, 1 warning`.
  - Full suite: `119 passed, 8 skipped, 1 warning`.
