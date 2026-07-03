# src/dojo/results — canonical result store

## Purpose

Writes and reads the canonical tall-Parquet result table plus its
`_metadata.json` sidecar, via `amplify-db-utils` (`DuckDBParquetStore`).

- `schemas.py` — the union result schema + `schema_version`.
- `records.py` — row builders for each `record_type`.
- `writer.py` — writes the hive-partitioned table + sidecar.
- `reader.py` — `ResultReader` (equality / `in` / range filters).
- `metadata.py` — the `_metadata.json` sidecar (schema, run_id, per-record-type
  semantics, class maps).

## Ownership

Owns the result schema, hashing-column population, partitioning, and sidecar.
Authoritative spec: `DESIGN-DOC/06-results-artifacts-and-metadata.md`.

## Local Contracts

- One union schema for all rows, discriminated by `record_type`; inapplicable
  columns are null. Implemented types: `sample_metadata`,
  `classification_output`, and `embedding`.
- Vectors (`logits`, `probabilities`) are Arrow `list<float32>`, lossless.
- Common provenance columns (`sample_id`, `uri`, `split`, `stage`, `run_id`,
  `config_hash`, `dataset_hash`, `checkpoint_hash`, `epoch`, `global_step`,
  `schema_version`) appear on every row and must stay reproducible.
- `classification_output` rows carry `head_hash`, `target_index`, and
  `target_name` so rows are self-scoring without joining to the source
  manifest.
- `embedding` rows carry `embedding_kind`, `embedding`, `embedding_dim`, and
  optional `embedding_model_name`; inference/eval rows use `stage=infer` or
  `stage=holdout_eval`.
- Bump `schema_version` on any schema change; keep the sidecar in sync.

## Verification

- `tests/unit/results/` (`test_writer.py`, `test_reader.py`, `test_metadata.py`).
</content>
