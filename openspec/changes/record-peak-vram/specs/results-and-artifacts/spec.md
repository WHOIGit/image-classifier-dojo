# results-and-artifacts (delta)

Draft delta — several design points are unresolved; see Open Questions
in proposal.md before implementing.

## ADDED Requirements

### Requirement: Peak VRAM in run metadata

CUDA runs SHALL record the run-scoped peak reserved GPU memory
(`torch.cuda.max_memory_reserved()`, reset at run start) in the
`_metadata.json` sidecar under an optional `resource_stats` block. The
block SHALL be omitted on non-CUDA devices and SHALL NOT contribute to
any identity hash.

#### Scenario: Peak recorded for a CUDA training run

- **GIVEN** a training run on a CUDA device
- **WHEN** the run completes and the `_metadata.json` sidecar is written
- **THEN** `resource_stats.peak_vram_reserved_bytes` is a positive
  integer covering the whole run, and re-running with identical config
  yields identical identity hashes regardless of the recorded value

#### Scenario: Omitted on CPU

- **GIVEN** a training run on a CPU-only host
- **WHEN** the sidecar is written
- **THEN** it contains no `resource_stats` block
