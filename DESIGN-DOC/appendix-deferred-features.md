
# Appendix — Deferred Features

## Purpose

Lists features that are intentionally out of scope for the initial
implementation but preserved as backlog items. This is a **roadmap, not a
contract**: nothing here is reflected in the running schema or code until
it is built. Per `12-validation-testing-and-preflight.md`, deferred
features are simply **absent from the strict (`extra="forbid"`) schema**,
so configuring one fails generic validation at load — there are no
reserved config slots, no runtime `NotImplementedError` stubs, and no
per-feature tests. Because nothing in code references this list, keep it
in sync with reality by hand. Each entry records its future config-token
shape (so the destination is documented) and the obligations that *do*
exist when it is promoted; the P4.8 cleanup milestone names a
verification obligation instead.

## How deferral works

A deferred feature's config token (enum value, discriminated-union tag,
or block) is **not added** to the Pydantic schema. Because the schema is
strict (`extra="forbid"`) and enums list only implemented values,
authoring the token fails generic validation at config load — the same
failure a typo produces. The error names the offending key/value only; it
does not mention this backlog.

Deferred features therefore carry **no per-feature tests**. The class is
covered generically (one test that unknown keys are rejected, one that
out-of-enum values are rejected). When a feature is promoted, its token
is added to the schema and gains real functional tests; there is no stub
test to delete first.

## Deferred features

### P4.1 Bayesian sweeps

- Initial sweeps are config-defined grid sweeps with Dojo-owned
  expansion (grid / explicit value lists). See
  `09-sweeps-and-batch-runs.md`.
- Future config shape: a `bayesian` value for `sweep.mode` plus a
  `sweep.bayesian` block (engine, metric, sampler, trial budget, search
  params). Neither is in the schema initially, so `sweep.mode: bayesian`
  fails as an out-of-enum value and a `sweep.bayesian` block fails as an
  unknown key.
- When promoted: add the `bayesian` mode and `sweep.bayesian` schema,
  then real functional tests for expansion and trial orchestration.

### P4.2 Multilabel support

- Future config shape: `multilabel_classification` as a
  `model.heads.<name>.type` value.
- Meaning: one classifier head emits several binary labels from a single
  target/output vector. This is different from multi-head multiclass,
  where each target has its own `multiclass_classification` head.
- Current status: `multilabel_classification` is **not** a member of the
  head-type union, so authoring it fails as an unknown discriminated-union
  tag. Multi-head multiclass is functional and does not depend on it.
- When promoted: add the head type, its result record type, and real
  functional training / inference tests.

### P4.3 Aim logger sink

- Future config shape: `aim` as a `training_outputs.logging.sinks[].type`
  value. `aim` is **not** a registered sink type, so authoring it fails
  as an unknown discriminated-union tag.
- Rationale: for the foreseeable future the project records metrics and
  figures **locally only** (the `local` sink). Aim was previously
  planned as functional but is deferred; the `aim` dependency also has
  no installable build on current Python (`aimrocks` ships no 3.13/3.14
  wheel), which reinforces deferring it.
- Extra: `aim` (commented out in `pyproject.toml` while undeliverable on
  current Python).
- When promoted: register the sink type and add functional logging tests.

### P4.4 MLflow logger sink

- Future config shape: `mlflow` as a
  `training_outputs.logging.sinks[].type` value. `mlflow` is **not** a
  registered sink type, so authoring it fails as an unknown
  discriminated-union tag.
- Extra: `mlflow` — **not currently declared** in `pyproject.toml`;
  add the extra when the sink is built.
- When promoted: register the sink type and add functional logging tests.

### P4.5 Non-DINOv2 SSL methods

- Future config shape: `simclr`, `vicreg`, `pmsn`, `dino` as `ssl.method`
  values (each with its own method config). None are schema members, so
  authoring any of them fails as an out-of-enum value.
- The functional method is `dino_v2` via Lightly.
- When promoted: add the method and its config, then functional SSL tests.

### P4.6a Weight-space ensembles

- Model soup, greedy soup, uniform soup, SWA, EMA.
- These produce a single exported model artifact (not a prediction-space
  combine), so they need new schema (a different ensemble paradigm) that
  is not present initially; authoring it fails generic validation.
- When promoted: design the schema and add functional tests per strategy.

### P4.6b Weighted ensemble combine modes

- Future combine values: weighted_logits_mean, weighted_probabilities_mean,
  weighted_vote, weighted_prediction_mean, weighted_ordinal_logits_mean,
  weighted_ordinal_probabilities_mean (plus the per-member weight plumbing
  they need). None are schema members, so authoring any fails as an
  out-of-enum value.
- When promoted: add the modes and the weight source, then functional
  tests.

### P4.6c `prediction_trimmed_mean`

- Sort member predictions, drop configured low / high extremes, average
  the rest.
- Future config shape: `prediction_trimmed_mean` as a regression combine
  value. Not a schema member, so authoring it fails as an out-of-enum
  value.
- When promoted: add the mode (and its trim fraction) and a functional
  test.

### P4.7 HDF / HDF5 derived result exports

- `.h5` metrics rollups, `results.h5`. Would re-introduce an `hdf` extra
  and `h5py` / `tables` dependencies (currently dropped).
- Future config shape: an `hdf` export format value. Not a schema member,
  so authoring it fails as an out-of-enum value.
- When promoted: re-add the extra, the format, and functional export
  tests.

### P4.8 Deprecated package removal

- Cleanup milestone, not a deferred feature.
- Remove `src/dojo_deprecated/` once the new `src/dojo` implementation
  covers everything through P4.7 and nothing imports from it, per
  `13-workplan.md`.
- Verification obligation: no imports from `src/dojo_deprecated/`, no
  tests depending on it, and any remaining useful behavior has either
  been reimplemented in `src/dojo` or intentionally dropped.

### P4.9 Distributional and count regression heads

- Future config shape: `distributional_regression` (emits distribution
  parameters, e.g. `gaussian`, `negative_binomial`) and `count_regression`
  (emits count / rate parameters) as `model.heads.<name>.type` values.
- Deferred together because neither has a result record type in the
  initial implementation: `06-results-artifacts-and-metadata.md` defines
  `regression_output` (single value + uncertainty) but no
  `distributional_output` / `count_output` for multi-parameter
  distribution or count / rate heads. Their dedicated losses
  (`gaussian_nll`, `negative_binomial_nll`, `poisson_nll`) are deferred
  with them; plain `regression` (`mse`, `mae`, `huber`, `smooth_l1`,
  `quantile`) is the only functional regression head type.
- Neither is a member of the head-type union, so authoring either fails
  as an unknown discriminated-union tag. When a head is promoted, add its
  result record type, losses, and functional inference tests.

### P4.10 WebDataset backend

- Future config shape: `webdataset` as a `data.backend` value. Not a
  schema member, so authoring it fails as an out-of-enum value.
- When promoted: add the backend and its datamodule, plus functional
  dataset tests.

### P4.11 Registry-based ensemble candidate discovery

- Broad automatic registry-based cross-run candidate discovery.
- Note: cross-run ensembling itself is **functional** via explicit
  candidate sources (`explicit`, `run_dir_glob`, `checkpoint_glob`,
  `result_uri_glob`, `manifest`). Only the registry-driven discovery is
  deferred. The term "cross-run ensemble" is no longer used as a
  distinct mode (see `08-ensembles.md`).
- Future config shape: a `registry` candidate-source type. Not a schema
  member, so authoring it fails as an unknown discriminated-union tag.
- When promoted: add the source type and functional discovery tests.

### P4.12 `majority_vote` probability-mass tie-break

Intentionally has no entry here: it is an enhancement to the functional
lowest-index tie-break, not a deferred config token, so it carries no
obligation. Defined in `13-workplan.md` (P4.12).

### P4.13 Tabular-only model schema

- Initial tabular support is image-backed supervised modeling with optional
  tabular features: an image backbone remains required, and tabular
  features may be concatenated implicitly after image embeddings.
- Tabular-only modeling is deferred because it requires a broader schema
  change: explicit model input enablement, image-free data validation,
  image-free preprocessing / result metadata, export metadata without
  image input shape, and updated preflight rules.
- Because `model.image_input.backbone` is required initially, configuring
  a tabular-only model fails generic validation (a required field is
  missing). When promoted, relax the requirement and add functional
  schema / runtime tests.

### P4.14 Expanded tabular encoder families

- Initial `model.tabular_input.encoder.type` values are intentionally small:
  `identity`, `linear`, and `mlp`.
- Potential future encoder families:
  `tab_transformer`, `ft_transformer`, `tabnet`, `embedding_bag`, and
  `wide_and_deep`.
- These are deferred because each adds nontrivial schema, dependency,
  preprocessing, export, and compatibility-hash surface area.
- None are members of the encoder-type enum, so authoring any fails as an
  out-of-enum value. When an encoder family is promoted, add functional
  tests for its schema, construction, hashing, export metadata, and
  inference behavior.

### P4.15 `dojo init --wizard`

- Interactive project/config questionnaire that writes a local config tree
  and optionally fixture data.
- Deferred because the initial `dojo init` should stay deterministic and
  template-based: copy packaged configs, materialize dependency closures,
  and optionally materialize a small fixture dataset.
- Not a config token but a CLI flag: the `--wizard` option simply does not
  exist initially, so passing it is a standard unknown-option error. When
  promoted, add CLI interaction tests for questionnaire branching,
  generated files, and non-clobber behavior.

### P4.16 Automated sweep execution runners

- Initial sweep execution is manual: `dojo sweep prepare` expands the sweep
  and writes per-run resolved configs; users run concrete jobs explicitly,
  for example with `dojo sweep train SWEEP_DIR --index N` or the lower-level
  `dojo train --resolved-config ...` path.
- Initial `sweep.execution.mode` value: `manual` (the only schema member).
- Deferred execution modes (not yet schema members, so authoring either
  fails as an out-of-enum value):
  - `local_sequential` — run prepared sweep jobs sequentially on the local
    machine.
  - `slurm` — submit prepared sweep jobs to a Slurm / HPC queue.
- Deferred because runner orchestration adds scheduling, retry, log
  collection, cluster-specific configuration, and concurrency behavior
  beyond the initial sweep artifact contract. Initial `dojo sweep status`
  reads live per-run status files and does not persist a sweep-level status
  cache.
- When promoted: add integration tests for local sequential execution and
  unit / contract tests for Slurm script generation, submission dry-runs,
  manifest status updates, and failure recovery behavior.

## Cross-References

- `01-goals-and-scope.md` — scope and non-goals.
- `08-ensembles.md` — explicit candidate sources for the
  not-deferred parts of cross-run ensembling; deferred selection /
  combine modes.
- `09-sweeps-and-batch-runs.md` — config-defined grid sweeps vs.
  Bayesian HPO.
- `11-dependencies.md` — extras list and dropped dependencies.
- `12-validation-testing-and-preflight.md` — strict-schema deferral
  policy in full.
- `glossary.md` — terminology.
