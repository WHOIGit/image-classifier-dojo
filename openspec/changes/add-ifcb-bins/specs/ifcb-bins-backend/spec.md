## ADDED Requirements

### Requirement: IFCB bins dataset backend
The system SHALL provide an `ifcb_bins` dataset backend via `ifcbkit`
(gated by the `ifcb` extra) across train, eval, and infer paths, with
custom dataset and dataloaders producing the shared sample contract and
feeding bin-length values into the frozen stats cache.

#### Scenario: Training over IFCB bins
- **WHEN** a supervised config selects `data.backend: ifcb_bins`
- **THEN** training consumes bin samples through the shared contract and
  results carry canonical provenance
