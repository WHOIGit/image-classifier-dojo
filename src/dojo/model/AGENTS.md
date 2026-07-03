# src/dojo/model — backbone + heads + composition

## Purpose

Modular model composition for supervised classification.

- `backbone.py` — image backbone (torchvision in P1; timm/checkpoint deferred).
- `heads.py` — classification head(s).
- `supervised.py` — assembles backbone + head(s) into the supervised model.

## Ownership

Owns `nn.Module` construction only. Does not own the training loop, loss
weighting, or metrics — those live in `training/`.

## Local Contracts

- Single-head normalizes to the multi-head shape internally (canonical
  representation).
- Head `num_classes` must match the selected data group's class count — this is
  the override users pass (`model.heads.<name>.num_classes=...`).
- Keep task logic out of model composition.

## Verification

- `tests/unit/model/` (`test_backbone.py`, `test_heads.py`, `test_supervised.py`).
</content>
