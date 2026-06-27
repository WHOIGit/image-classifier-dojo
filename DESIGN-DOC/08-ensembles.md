
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

Cross-run ensembling works via `dojo ensemble` with `run_dir_glob` or
explicit `run_dir` candidates. Snapshot ensembling against a brand-new
training run is `task.type: snapshot_ensemble` (see
`02-cli-and-task-types.md`). Re-running ensembling against a historical
training run is a plain `dojo ensemble` invocation with a `run_dir`
candidate, or a `run_dir_glob` source that matches one or more run
directories.

## Candidate discovery

Candidate discovery in the initial implementation is **limited to these
explicit source types**:

- `explicit` — inline candidate list.
- `run_dir_glob` — Dojo training-run directories.
- `checkpoint_glob` — raw checkpoint files.
- `result_uri_glob` — cached result / prediction directories.
- `manifest` — pre-built candidate manifest.

Broad automatic registry-based cross-run discovery is deferred — see
`appendix-deferred-features.md`.

Manifest output is JSON (not Parquet) under
`ensemble_outputs.manifests.dir`, default
`{ensemble_outputs.dir}/ensemble_manifests/`. Every `dojo ensemble` run
writes a manifest for provenance, even when member prediction rows are not
materialized into `ensemble_results/`. The manifest records discovered
candidates, compatibility status, exclusion reasons when available,
selected members, assigned `ensemble_member_id` values, source result
selectors / URIs, semantic compatibility status, member-provenance hashes,
combine config, and selection config. Candidate-audit metadata is useful
for inspection, but
`ensemble_hash` is derived from the selected-ensemble identity block
(selected members + combine + selection), not from every discovered
candidate.

To write a shared manifest outside a normal run directory:

```bash
dojo ensemble candidates experiment=ifcb/candidate_search \
  ensemble_outputs.manifests.dir=./shared_manifests
```

### Candidate source types and artifact kinds

`sources[].type` says how Dojo discovers candidates. Candidate `kind`
says what artifact each discovered candidate represents.

Canonical source enum:

```text
explicit
run_dir_glob
checkpoint_glob
result_uri_glob
manifest
```

Candidate artifact kinds:

- `run_dir` — Dojo training-run directory. Preferred for normal
  historical-run ensembling because Dojo can read resolved config,
  checkpoint metadata, validation metrics, and cached result artifacts.
- `checkpoint` — raw model checkpoint file. Requires enough config
  metadata to rebuild the model; Dojo may need to run validation /
  inference to produce comparable per-model metrics.
- `result_uri` — cached predictions / result rows. Useful when model
  files are unavailable or inference should not be rerun.
- `exported_model` — portable `.pt` / `.onnx` export with enough config
  metadata to run inference if needed.

Each member-level row carries an `ensemble_member_id` union column equal
to the member's `checkpoint_hash` (checkpoint member) or `model_id`
(exported model member). Combined ensemble rows have
`ensemble_result_scope=ensemble` and no `ensemble_member_id`. See
`06-results-artifacts-and-metadata.md`.

### Discovery example

```yaml
ensemble:
  candidates:
    sources:
      - type: run_dir_glob
        uri_glob: s3://dojo-runs/ifcb_species/*/
      - type: checkpoint_glob
        uri_glob: ./checkpoints/*.ckpt
        config_uri: ./configs/ensemble_member_base.yaml
      - type: result_uri_glob
        uri_glob: ./cached_predictions/*/results/
      - type: manifest
        manifest_uri: ./shared_manifests/ifcb_candidates.json
      - type: explicit
        candidates:
          - kind: run_dir
            uri: ./runs/resnet50_a
          - kind: checkpoint
            uri: ./checkpoints/convnext_b.ckpt
            config_uri: ./configs/convnext_b.yaml
          - kind: result_uri
            uri: ./runs/convnext_b/results
          - kind: exported_model
            uri: ./exports/efficientnet_d/model.pt
            config_uri: ./exports/efficientnet_d/config/resolved.yaml
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

### Authored config vs. runtime artifacts

Authored config describes candidate sources, target split, source policy,
selection strategy, combine modes, and output-retention policy. It should
not enumerate discovered compatible candidates, selected members, assigned
`ensemble_member_id` values, measured cost fields, or row selectors unless
the user is explicitly providing an `explicit` candidate source.

Resolved config fills defaults and concrete output paths. When
`ensemble_outputs.dir_template` contains `{ensemble_id}`, that path
resolves after candidate discovery, compatibility validation, selection,
and selected-ensemble identity generation.

The ensemble manifest is the execution artifact that records what the run
actually found and selected: all discovered candidates, compatible
candidates, incompatible candidates with reasons when practical, selected
members, source result selectors, hashes, cost metadata when known, and the
selection / combine config snapshot.

### Metadata resolution

Candidate source type controls discovery. `metadata_resolution` controls
where Dojo reads candidate metadata after candidates are discovered.
Metadata includes target schema, class mapping, preprocessing /
transform config, model config, dataset identity, checkpoint provenance,
and cached-result provenance.

Initial `metadata_resolution.policy` values:

- `cascade` — read metadata sources in `order` and use the first
  available authoritative value for each field. **Default.**
- `strict` — require all configured / available metadata sources for a
  candidate to agree on compatibility-critical fields.

Initial metadata source names:

- `resolved_config` — Dojo `config/resolved.yaml` or `resolved.json`
  from a run directory or explicit `config_uri`.
- `result_metadata` — metadata stored alongside cached result /
  prediction files.
- `checkpoint` — metadata embedded in or adjacent to a `.ckpt` member.
- `exported_model` — metadata embedded in or adjacent to a portable
  `.pt` / `.onnx` export.

`drift_check` is a validation pass, not a discovery mechanism. When
enabled, Dojo compares the listed metadata sources if more than one is
available for a candidate. Any mismatch in compatibility-critical fields
is reported according to the policy; for the initial implementation,
drift in target schema, class mapping, preprocessing, or model-output
shape is an error.

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
metadata targets are supported. Cross-member fast-equality checks apply to
semantic output contracts such as `target_schema_hash` and
`class_mapping_hash`. `model_config_hash` and `preprocessing_hash` are
member-provenance hashes: they validate a member's own artifacts and cached
rows, and they support drift checks across metadata sources for the same
candidate, but they are **not** required to match across ensemble members.
See `06-results-artifacts-and-metadata.md` for hash field selection rules
and the human-readable diff path on mismatch.

### Member-specific preprocessing

Candidate models may have different preprocessing requirements (image
size, normalization, crop, tabular feature transforms, etc.). Member
preprocessing metadata travels with each member; the ensemble runner
applies preprocessing per-member when necessary, with an optional
shared-preprocessing fast path when all members agree on input shape /
normalization / etc.

This enables heterogeneous ensembles such as ViT@224 + ConvNeXt@320 +
EfficientNet@384 against the same evaluation dataset.

`preprocessing_hash` equality is only a fast-path signal for shared
preprocessing. A mismatch does not make members incompatible as long as the
evaluation data provides each member's required input fields and the
member's own preprocessing metadata is available. Drift within one
candidate remains an error when multiple metadata sources disagree about
that candidate's preprocessing contract.

## Source policy

Controls behavior when candidate-result coverage is partial:

- `strict_no_inference` — refuse to run; only cached results are used.
- `inference_as_needed` — run inference for any missing rows.
- `force_inference` — re-run inference for every member ignoring caches.

Ensemble commands default to using existing result files as inputs when
input dataset and output target match (`inference_as_needed`).
Cached-result inputs must match the ensemble target by `dataset_hash`, or
by explicit `dataset_id` plus `split` when the dataset self-names and a
content hash is unavailable. Target schema, class mapping, output tensor
shape, and output tensor meaning still govern cached-result compatibility.
`model_config_hash` and `preprocessing_hash` validate cached rows against
their own producing member when present; they do not need to match other
members.

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

Selection config is normal Hydra config and can be swept. For example,
Hydra may sweep `ensemble.selection.strategy`,
`ensemble.selection.max_members`, or combine-mode settings to compare which
candidate subsets contribute most to ensemble quality.

Candidate metadata may include optional cost fields such as inference
latency, parameter count, FLOPs, and peak memory when known. Initial
metrics / figures can compare quality against these costs; cost-aware
selection objectives are an extension on top of the same manifest fields.

## Combine modes

Classification:

- `probabilities_mean`
- `logits_mean`
- `majority_vote`

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

`majority_vote` (classification):

```text
member_votes = [argmax(member_output_i) for each member]
prediction_index = the class with the most member votes
                   # ties broken deterministically by lowest class index
prediction_confidence = vote_count(prediction_index) / num_members
```

Hard voting needs only each member's `prediction_index`, which is why it
is the one classification mode that combines from members that logged a
prediction but no `logits` / `probabilities`. Ties are broken
deterministically by **lowest class index** (no RNG).

Regression:

```text
prediction_value = mean(member_predictions)
prediction_uncertainty = std(member_predictions)
```

Ordinal: average ordinal logits (or ordinal probabilities) then decode
through the configured ordinal decoding rule
(`model.heads.<name>.ordinal.decoding`). The ensemble artifact records the
decoding rule used per ordinal head.

### Using cached results correctly

Cached-result ensembling (`source_policy: strict_no_inference`, or
`inference_as_needed` when all rows are present in cache) reads member
predictions from existing result Parquet files instead of re-running
inference. The combine math is identical; what changes is that each
chosen combine mode imposes a column requirement on every member's
cached results.

Required columns per combine mode (column definitions live in
`06-results-artifacts-and-metadata.md`):

| Combine mode | Required cached columns |
|---|---|
| `logits_mean` (classification) | `logits` |
| `probabilities_mean` (classification) | `probabilities` (or `logits` to derive) |
| `majority_vote` (classification) | `prediction_index` |
| `prediction_mean` / `prediction_median` (regression) | `prediction_value` (use `prediction_value_internal` if combining in transformed space) |
| `ordinal_logits_mean` | `ordinal_logits` |
| `ordinal_probabilities_mean` | `probabilities` (per-bin) |

Every member must also carry the standard provenance columns
(`sample_id`, `head_name`, `target`) and the partition keys needed to
locate its rows; see `06-results-artifacts-and-metadata.md` for the full
schema.

Notes and limitations:

- Cached rows are joined across members by `sample_id` at the configured
  target split. Members missing rows under `strict_no_inference` fail
  the run; under `inference_as_needed` the runner fills gaps by running
  that member's inference.
- A member whose cached results omit a column required by the chosen
  combine mode (e.g. only `prediction_index` logged, but the ensemble
  asks for `logits_mean`) cannot participate from cache; either switch
  combine mode, drop the member, or use `inference_as_needed` /
  `force_inference` to re-derive the needed columns.
- Compatibility checks (`target_schema_hash`, `class_mapping_hash`,
  ordinal encoding rule, regression units) apply identically to cached
  and fresh-inference ensembling.
- Per-member preprocessing differences do not matter for cached-result
  combine — predictions already reflect each member's preprocessing.
- For regression, choose `prediction_value` vs. `prediction_value_internal`
  deliberately and consistently across members; mixing external- and
  internal-space combines is not supported.

## Ensemble run example

```yaml
ensemble:
  candidates:
    sources:
      - type: manifest
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
  results:
    member_results:
      mode: none
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

- training result rows are written under `results/` and ensemble result
  rows are written under `ensemble_results/`, so the two writers do not
  collide even when the top-level output directory is shared;
- row `stage` remains semantic provenance (`train_validation` vs.
  `ensemble_eval`), not the primary collision-avoidance mechanism;
- metrics / figures filenames include the producing block when
  namespacing is needed;
- the existing-run-dir overwrite policy is evaluated **once** per
  resolved physical directory at startup so the ensemble step does not
  delete training artifacts.

### Member result retention

`ensemble.source_policy` controls how member predictions are obtained for
ensemble math. `ensemble_outputs.results.member_results.mode` controls
whether selected member prediction rows are retained under
`ensemble_outputs.results.dir`.

Initial modes:

- `none` — default. Write combined ensemble rows and the always-written
  ensemble manifest only. Do not materialize member-level rows into
  `ensemble_results/`.
- `transcribe` — require source result rows for every selected member,
  validate dataset / split, semantic output compatibility, and
  member-provenance hashes, then rewrite those rows into canonical ensemble
  form with `stage=ensemble_eval`, `ensemble_result_scope=member`,
  `ensemble_id`, `ensemble_hash`, and `ensemble_member_id`. This mode does
  not run inference just to retain rows.
- `inference` — require `ensemble.source_policy: force_inference`; run
  selected member inference and write fresh member-level rows into
  `ensemble_results/`.

Retention applies only to selected ensemble members, not every discovered
compatible candidate. With `mode: none`, member identity is still known at
execution time from the selected-member manifest, but no member-level
result rows or `ensemble_member_id` values are persisted in
`ensemble_results/`.

### Ensemble result records

Ensemble result rows use the standard result schema plus ensemble
provenance columns.

- All rows written by the ensemble pipeline use `stage=ensemble_eval`.
- Combined ensemble rows use `ensemble_result_scope=ensemble` and leave
  `ensemble_member_id` null.
- Retained member rows use `ensemble_result_scope=member` and populate
  `ensemble_member_id` with the member's `checkpoint_hash` or `model_id`.
- Cached source rows that remain in their source result dataset keep their
  original `stage`; if transcribed into `ensemble_results/`, they are
  rewritten as ensemble-produced rows.

Default `ensemble_outputs.results.partition_by` is
`[stage, record_type, ensemble_result_scope]`. `ensemble_member_id` remains
available as a filter column and may be used as an opt-in partition key for
member-only analysis outputs.

## Cross-References

- `02-cli-and-task-types.md` — `dojo ensemble`, `dojo ensemble
  candidates`, `task.type: snapshot_ensemble`.
- `03-configuration.md` — placement of `ensemble:` and
  `ensemble_outputs:`, snapshot-ensemble directory sharing.
- `05-models-training-and-heads.md` — snapshot-cycle scheduler and
  checkpointing.
- `06-results-artifacts-and-metadata.md` — `ensemble_result_scope`,
  `ensemble_member_id`, `stage=ensemble_eval` rows, compatibility and
  member-provenance hashes, `ensemble_manifests/` JSON manifests,
  partitioning.
- `09-sweeps-and-batch-runs.md` — Hydra sweeps over ensemble selection
  / combine axes; sweeps as candidate-source feeders.
- `10-export.md` — `ensemble_outputs.export` and ensemble model
  artifacts.
- `appendix-deferred-features.md` — weight-space ensembles, weighted
  combine modes, `prediction_trimmed_mean`, registry-based candidate
  discovery.
- `glossary.md` — `ensemble_member_id`, `ensemble_id`, candidate /
  selection vocabulary.
