# src/dojo/config_schemas — the runtime contract

## Purpose

Strict Pydantic schema that every composed config is validated against, plus
deterministic hashing. `root.py` is the `RootConfig` tree; `hashing.py` derives
the reproducible `config_hash` / `dataset_hash` / `checkpoint_hash`.

## Ownership

Owns the authoritative shape of a valid config and the hash definitions. Does
not compose or resolve configs (that is `config_loader/`).

## Local Contracts

- Strict schema: `extra="forbid"`. Deferred features are absent, not stubbed —
  configuring an unimplemented feature must fail generic validation, with no
  reserved slots or runtime `NotImplementedError`.
- Schemas are the single runtime contract; downstream layers trust validated
  models and do not re-validate ad hoc.
- Hashes must be stable across runs of the same config; changing a hash input
  is a breaking, versioned change.

## Verification

- `tests/unit/config_schemas/` (`test_hashing.py`, `test_transforms.py`,
  `test_data_split.py`).
</content>
