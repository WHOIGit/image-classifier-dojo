# src/dojo/config_schemas — the runtime contract

## Purpose

Strict Pydantic schema that every composed config is validated against, plus
deterministic hashing. `root.py` is the `RootConfig` tree; `hashing.py` derives
the reproducible config, head, and compatibility content hashes used by
runtime/result/checkpoint provenance.

## Ownership

Owns the authoritative shape of a valid config and the hash definitions. Does
not compose or resolve configs (that is `config_loader/`).

## Local Contracts

- Strict schema: `extra="forbid"`. Deferred features are absent, not stubbed —
  configuring an unimplemented feature must fail generic validation, with no
  reserved slots or runtime `NotImplementedError`.
- Schemas are the single runtime contract; downstream layers trust validated
  models and do not re-validate ad hoc.
- Hashes must be stable across runs of the same config/content; changing a hash
  input is a breaking, versioned change. `head_hash` and compatibility hashes
  use the same canonical JSON recipe as `config_hash`.
- Class-mapping compatibility hashes are per head. Multi-target datasets must
  hash the class mapping for each head's configured target, while preserving
  single-target behavior through the primary class mapping.

## Verification

- `tests/unit/config_schemas/` (`test_hashing.py`, `test_transforms.py`,
  `test_data_split.py`).
</content>
