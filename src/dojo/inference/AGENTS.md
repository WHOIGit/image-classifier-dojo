# src/dojo/inference — checkpoint-backed inference and holdout eval

## Purpose

Loads finished Dojo checkpoints through their embedded inference contract and
writes canonical prediction / embedding / holdout-eval outputs.

## Ownership

Owns artifact loading and non-training forward passes. It may reuse data,
model, results, and training metric/loss helpers, but it does not own training
or config composition.

## Local Contracts

- Checkpoints must carry `checkpoint["dojo_inference_contract"]`.
- Inference rebuilds the model from the contract rather than the producing
  run's `resolved.yaml`.
- Result rows use `stage=infer` or `stage=holdout_eval`.
- `dojo eval holdout` writes `eval_manifest.json` with checkpoint/dataset
  provenance, record counts, compatibility hashes, and a metric summary
  computed from the checkpoint's `objective_summary`.

## Work Guidance

- Keep command I/O in `cli/`; functions here should return structured results.

## Verification

- `tests/unit/inference/` and integration coverage through CLI commands.

## Child DOX Index
