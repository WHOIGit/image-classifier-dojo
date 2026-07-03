# src/dojo/data — datasets, sample contract, transforms

## Purpose

Reads Parquet datasets into decoded samples and batches for training.

- `contract.py` — `DecodedSample` / `SampleBatch` TypedDicts: the shared sample
  contract (image tensor, target index, and provenance the result writer needs).
- `parquet_images.py` — the P1 `parquet_images` backend.
- `dataset.py`, `identity.py`, `transforms.py` — dataset assembly, sample
  identity/hashing, and transform application.

## Ownership

Owns decoding, splitting, target-label mapping, and collation. Does not own
transform *schema* (in `config_schemas`) nor result columns (in `results`), but
must emit exactly the provenance fields the sample contract declares.

## Local Contracts

- `DecodedSample` / `SampleBatch` are the boundary contract with `training/`
  and `results/`. Keep them in sync with the `sample_metadata` columns in
  `06-results-artifacts-and-metadata.md`.
- Split is data-driven; the split column must be present in the dataset.

## Verification

- `tests/unit/data/` (parquet images, split-from-filename, target labels,
  transforms builder, identity).
</content>
