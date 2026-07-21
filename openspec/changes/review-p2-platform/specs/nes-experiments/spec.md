## ADDED Requirements

### Requirement: Committed NES experiment configs
The four committed P2 NES experiment configs SHALL pass
`dojo inspect config`, and their final reported metrics SHALL come from
GPU runs without pilot overrides, recorded in `REPORTS/report-P2.md`.
The configs are `configs/experiment/p2/nes_effb0_224_baseline.yaml`,
`nes_effb0_buckets.yaml`, `nes_effb0_weighted.yaml`, and
`nes_effb0_buckets_weighted.yaml`.

#### Scenario: Config validity
- **WHEN** `dojo inspect config --format json` runs on each of the four
  configs
- **THEN** all four compose and validate

#### Scenario: Final numbers are GPU numbers
- **WHEN** P2 experiment results are cited
- **THEN** they come from full GPU runs, not the earlier CPU pilot
  timings
