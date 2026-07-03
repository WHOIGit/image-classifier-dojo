# src/dojo/storage — URI-based storage seam

## Purpose

The dojo's URI-based storage interface (`io.py`): localize dataset URIs and
write run artifacts. Backed by `amplify-storage-utils`. P1 uses the local
filesystem backend with an object-store (S3) seam.

## Ownership

Owns the storage abstraction only. Path/template resolution of `output_root` /
`dir_template` is NOT here — that is `config_loader/resolver.py`.

## Local Contracts

- Callers address storage by URI; do not hardcode local paths past this seam.
- Keep the object-store seam intact so S3 can drop in without caller changes.

## Verification

- `tests/unit/storage/test_io.py`.
</content>
