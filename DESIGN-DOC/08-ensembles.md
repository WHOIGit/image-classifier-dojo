# 08. Ensembles

## Purpose

Defines prediction-space ensembling: candidate discovery, candidate
manifests, compatibility checking, selection strategies, combine modes,
the `dojo ensemble` / `dojo ensemble candidates` commands, and the
`ensemble_outputs:` block. Cross-run and snapshot ensembling are
candidate-source patterns, not separate ensemble algorithms.

## Concept

Ensembling is **prediction-space** multi-model output processing:

```text
candidate discovery
candidate compatibility validation
candidate scoring
ensemble selection
ensemble construction
ensemble evaluation
artifact bundling
canonical result export
```

It is not a separate training algorithm. There are no `cross_run_ensemble`
or `snapshot_ensemble` ensemble algorithm types — those are candidate
sources for the one prediction-space ensemble pipeline. Weight-space
ensembles (model soup, SWA, EMA) are deferred — see
`appendix-deferred-features.md`. `torchensemble` is dropped as a
dependency; Bagging / Boosting / Fusion / Adversarial / FastGeometric
strategies are not ported.

## Commands

- `dojo ensemble` — ensemble evaluation / inference against a candidate
  manifest or candidate-discovery sources.
- `dojo ensemble candidates` — artifact-inspection / manifest-writing
  command. Does not initialize experiment logging.

Cross-run ensembling works via `dojo ensemble` with `run_checkpoints`
candidate sources. Snapshot ensembling against a brand-new training run
is `task.type: snapshot_ensemble` (see `02-cli-and-task-types.md`).
Re-running ensembling against a historical training run is a plain
`dojo ensemble` invocation with a `run_checkpoints` source pointing at
that run directory.

## Candidate discovery

Candidate discovery in the initial implementation is **limited to
explicit sources**:

- explicit artifact lists;
- run-directory globs;
- result-URI globs;
- pre-built candidate manifests.

Broad automatic registry-based cross-run discovery is deferred — see
`appendix-deferred-features.md`.

Manifest output is JSON (not Parquet) under
`ensemble_outputs.manifests.dir`, default
`{ensemble_outputs.dir}/ensemble_manifests/`. To write a shared manifest
outside a normal run directory:

```bash
dojo ensemble candidates experiment=ifcb/candidate_search \
  ensemble_outputs.manifests.dir=./shared_manifests
```

### Candidate types

Member sources may be `.ckpt` files, exported `.pt` / `.onnx` models,
run-directory references, or pre-discovered manifest entries. Each
member-level row carries an `ensemble_member_id` union column equal to
the member's `checkpoint_hash` (checkpoint member) or `model_id`
(exported model member). See `06-results-artifacts-and-metadata.md`.

### Discovery example

```yaml
ensemble:
  candidates:
    sources:
      - type: run_dir_glob
        uri_glob: s3://dojo-runs/ifcb_species/*/
      - type: explicit
        artifacts:
          - run_dir: ./runs/resnet50_a
          - result_uri: ./runs/convnext_b/results
    metadata_resolution:
      policy: cascade
      order:
        - resolved_config
        - result_metadata
        - checkpoint
        - exported_model
      drift_check:
        enabled: true
        compare: [resolved_config, checkpoint, result_metadata]
  target:
    split: val
    dataset_id: ifcb_species_v4   # use dataset_hash when dataset does not self-name
  source_policy: inference_as_needed
```

## Compatibility

Hard compatibility requirements for a prediction-space ensemble:

```text
sample identity semantics
required input fields available
target schema
head names
head task types
class mappings
ordinal encoding / decoding rules
regression target units and scaling
output tensor shapes
output tensor meanings
```

Per-head compatibility — a candidate may be excluded from one head and
included for another if multi-head compatibility allows.

Compatibility assessment uses a cascading metadata policy by default:

1. `config/resolved.json`
2. result metadata
3. checkpoint metadata
4. exported model metadata

Explicit per-metadata-source policies and drift checks across multiple
metadata targets are supported. Compatibility hashes used for
fast-equality checks: `target_schema_hash`, `class_mapping_hash`,
`model_config_hash`, `preprocessing_hash`. See
`06-results-artifacts-and-metadata.md` for hash field selection rules and
the human-readable diff path on mismatch.

### Member-specific preprocessing

Candidate models may have different preprocessing requirements (image
size, normalization, crop, tabular feature transforms, etc.). Member
preprocessing metadata travels with each member; the ensemble runner
applies preprocessing per-member when necessary, with an optional
shared-preprocessing fast path when all members agree on input shape /
normalization / etc.

This enables heterogeneous ensembles such as ViT@224 + ConvNeXt@320 +
EfficientNet@384 against the same evaluation dataset.

## Source policy

Controls behavior when candidate-result coverage is partial:

- `strict_no_inference` — refuse to run; only cached results are used.
- `inference_as_needed` — run inference for any missing rows.
- `force_inference` — re-run inference for every member ignoring caches.

Ensemble commands default to using existing result files as inputs when
input dataset and output target match (`inference_as_needed`).

## Selection strategies

Supported in the initial implementation:

- `all` — use every supplied candidate.
- `best_candidate` — select the single best candidate by the configured
  metric (useful as a baseline / control).
- `top_k` — top K candidates by validation metric.
- `greedy_forward_selection` — start with the best single candidate and
  iteratively add the candidate that most improves ensemble validation
  performance.
- `cycle_end_snapshots` — select cycle-end snapshots from a single
  training run (used by `task.type: snapshot_ensemble`).

Deferred selection / weighted strategies: see
`appendix-deferred-features.md`.

### Selection examples

```yaml
ensemble:
  selection:
    strategy: top_k
    k: 5
    metric: val/species/macro_f1
    mode: max
```

```yaml
ensemble:
  selection:
    strategy: greedy_forward_selection
    metric: val/species/macro_f1
    mode: max
    max_members: 8
    stop_if_no_improvement: true
```

## Combine modes

Classification:

- `probabilities_mean`
- `logits_mean`
- `majority_vote`
- `soft_vote`

Regression:

- `prediction_mean`
- `prediction_median`

Ordinal:

- `ordinal_probabilities_mean`
- `ordinal_logits_mean`

### Deferred combine modes

- weighted combine modes (weighted logits / probabilities / vote / mean);
- `prediction_trimmed_mean`.

### Combine internals

`logits_mean` (classification):

```text
ensemble_logits = mean(member_logits)
probabilities = softmax(ensemble_logits)
prediction_index = argmax(probabilities)
prediction_confidence = max(probabilities)
```

`probabilities_mean` (classification):

```text
member_probabilities = [softmax(logits_1), softmax(logits_2), ...]
probabilities = mean(member_probabilities)
prediction_index = argmax(probabilities)
prediction_confidence = max(probabilities)
```

`probabilities_mean` is usually safer when combining heterogeneous models
because it reduces sensitivity to different logit scales.

Regression:

```text
prediction_value = mean(member_predictions)
prediction_uncertainty = std(member_predictions)
```

Ordinal: average ordinal logits (or ordinal probabilities) then decode
through the configured ordinal decoding rule. The ensemble artifact
records the decoding rule used per ordinal head.

## Ensemble run example

```yaml
ensemble:
  candidates:
    manifest_uri: ./shared_manifests/ifcb_candidates.json
  target:
    split: holdout
  source_policy: strict_no_inference
  selection:
    strategy: greedy_forward_selection
    metric: val/species/macro_f1
    mode: max
    max_members: 8
  inference:
    combine:
      classification: probabilities_mean
      regression: prediction_median
      ordinal: ordinal_probabilities_mean

ensemble_outputs:
  dir_template: "{experiment.name}/ensembles/{ensemble_id}"
```

## `ensemble_outputs:` block

Peer to `training_outputs:` and `sweep_outputs:`. Sub-blocks:
`results`, `export`, `metrics`, `figures`, `manifests`, `members`.

Default on-disk sub-directories under the resolved `ensemble_outputs.dir`:

```text
ensemble_outputs.dir/
  config/
  exports/
  metrics/
  ensemble_figures/
  ensemble_results/
  ensemble_manifests/
  ensemble_members/        # optional; only when materialized locally
```

Member materialization to `ensemble_members/` is only allowed for
`dojo ensemble` runs. Local member files may be symlinked; remote member
files may be cached through `storage` config and symlinked. Member files
are never re-uploaded by Dojo.

Ensemble metrics and figures should compare member best/final metrics
against the resulting ensemble model.

For `task.type: snapshot_ensemble`, `ensemble_outputs.dir_template`
defaults to match `training_outputs.dir_template` so training and
ensemble outputs share one run directory. In that shared directory:

- writers namespace per-row data by `stage`
  (`train_validation` vs. `ensemble_eval`) so canonical result Parquet
  files coexist without collision;
- metrics / figures filenames include the producing block when
  namespacing is needed;
- the existing-run-dir overwrite policy is evaluated **once** per
  resolved physical directory at startup so the ensemble step does not
  delete training artifacts.

### Ensemble result records

Ensemble result rows use the standard result schema. Member-level rows
carry `ensemble_member_id` (union column with `checkpoint_hash` or
`model_id`). Ensemble-level rows use `stage=ensemble_eval`.

> **Open item.** The exact partitioning of member-level vs.
> ensemble-level rows (which rows are tagged `stage=ensemble_eval` vs.
> `stage=train_validation`, and how `ensemble_member_id` flows through
> partition keys when training and ensemble outputs share a directory)
> is under-specified. Writers should namespace by `stage`, but the full
> partitioning recipe is not yet pinned down. See
> `06-results-artifacts-and-metadata.md` for the partition-key
> vocabulary.

## Cross-References

- `02-cli-and-task-types.md` — `dojo ensemble`, `dojo ensemble
  candidates`, `task.type: snapshot_ensemble`.
- `03-configuration.md` — placement of `ensemble:` and
  `ensemble_outputs:`, snapshot-ensemble directory sharing.
- `05-models-training-and-heads.md` — snapshot-cycle scheduler and
  checkpointing.
- `06-results-artifacts-and-metadata.md` — `ensemble_member_id`,
  `stage=ensemble_eval` rows, compatibility hashes,
  `ensemble_manifests/` JSON manifests, open item on partitioning.
- `09-sweeps-and-batch-runs.md` — Hydra sweeps over ensemble selection
  / combine axes; sweeps as candidate-source feeders.
- `10-export.md` — `ensemble_outputs.export` and ensemble model
  artifacts.
- `appendix-deferred-features.md` — weight-space ensembles, weighted
  combine modes, `prediction_trimmed_mean`, registry-based candidate
  discovery.
- `glossary.md` — `ensemble_member_id`, `ensemble_id`, candidate /
  selection vocabulary.
