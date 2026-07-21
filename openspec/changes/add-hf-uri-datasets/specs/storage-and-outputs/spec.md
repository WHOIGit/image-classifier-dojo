## MODIFIED Requirements

### Requirement: Storage resolver
Storage access SHALL go through the Dojo storage interface backed by
`amplify-storage-utils`. The interface's `localize()` SHALL return a
local filesystem path for a given URI, resolving local paths directly and
materializing recognized remote dataset URIs (HuggingFace Hub references)
to a cached local path before returning.

#### Scenario: Local filesystem run
- **WHEN** a run uses local paths
- **THEN** reads and writes resolve through the storage interface and
  `localize()` returns the path unchanged

#### Scenario: Remote dataset URI localized
- **WHEN** `localize()` is called with a HuggingFace Hub dataset URI
- **THEN** it returns a local cached path pointing at the downloaded
  dataset files, reusing the cache on subsequent calls
