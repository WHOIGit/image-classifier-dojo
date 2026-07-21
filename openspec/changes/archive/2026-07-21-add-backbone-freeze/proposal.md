## Why

Workplan P3 transfer-learning experiments (`REPORTS/TRAINING-PLANS-P2.md`
groups 07/08) need to hold a pretrained backbone fixed and train only the
heads / embedding adapter. Until now `training.freeze.backbone.policy`
accepted only `none`, so frozen-backbone transfer was impossible and
`apply_freeze_policy` raised `NotImplementedError`.

## What Changes

- Add `training.freeze.backbone.policy: frozen`, which sets
  `requires_grad=False` on every backbone parameter at model build.
- Exclude frozen parameters from the AdamW optimizer so no state is
  allocated for them and they receive no updates.
- `none` remains the default (fully trainable), preserving current behavior.

## Capabilities

### Modified Capabilities

- `model-composition`: the backbone freeze policy now supports `frozen`
  in addition to `none`.
