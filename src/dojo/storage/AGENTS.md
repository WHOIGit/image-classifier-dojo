# src/dojo/storage — URI-based storage seam

## Purpose

The dojo's URI-based storage interface (`io.py`): localize dataset URIs and
write run artifacts. Backed by `amplify-storage-utils`. P1 uses the local
filesystem backend with an object-store (S3) seam.

`paths.py` holds the platform-dependent path rules that sit above that seam:
what counts as an already-absolute URI (`is_absolute_uri`, POSIX *and* Windows
syntax) and how a config-derived name is made creatable on the running
filesystem (`sanitize_path_component` / `sanitize_relative_path`). `is_windows()`
is the single platform check — patch it, never `os.name`, which pathlib also
reads.

## Ownership

Owns the storage abstraction only. Path/template resolution of `output_root` /
`dir_template` is NOT here — that is `config_loader/resolver.py`.

## Local Contracts

- Callers address storage by URI; do not hardcode local paths past this seam.
- Keep the object-store seam intact so S3 can drop in without caller changes.
- Sanitized names are per-platform and therefore not portable; nothing that
  feeds a `*_hash` may derive from them (see `data/identity.py`).

## Verification

- `tests/unit/storage/test_io.py`, `tests/unit/storage/test_paths.py`.
</content>
