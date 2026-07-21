# Inference and Holdout Evaluation Specification

## Purpose

Checkpoint-backed non-training forward passes and holdout evaluation.
Source design: `DESIGN-DOC/02-cli-and-task-types.md`,
`DESIGN-DOC/06-results-artifacts-and-metadata.md` (workplan P2.6).

## Requirements

### Requirement: Inference commands
`dojo infer predictions` and `dojo infer embeddings` SHALL run
checkpoint-backed forward passes, rebuilding the model from
`checkpoint["dojo_inference_contract"]`, and write canonical result rows
with `stage=infer` (`classification_output` and/or `embedding` record
types) including canonical provenance columns.

#### Scenario: Inference from a bare checkpoint
- **WHEN** inference runs against a checkpoint outside its producing run
  directory
- **THEN** it succeeds without reading the producing run's
  `resolved.yaml`

### Requirement: Holdout evaluation
`dojo eval holdout` SHALL reconstruct scorers from the checkpoint's
`objective_summary` and `target_schema`, write result rows with
`stage=holdout_eval`, and write a `metric_summary` into the
always-written `eval_manifest.json` under resolved `eval_outputs`
handling.

#### Scenario: Metrics without training config
- **WHEN** holdout eval runs from a checkpoint plus a holdout dataset
- **THEN** per-head metrics are computed from the embedded contract and
  summarized in `eval_manifest.json`

### Requirement: Per-head labels in non-training outputs
Prediction rows, inference contracts, and holdout-eval metrics SHALL use
per-head target labels and class mappings (multi-target aware).

#### Scenario: Multi-head checkpoint
- **WHEN** a two-head (species + coarse) checkpoint is evaluated
- **THEN** each head's rows and metrics use that head's class mapping
