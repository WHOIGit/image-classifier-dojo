## ADDED Requirements

### Requirement: Committed NES experiment configs
The P2 NES experiment configs under `configs/experiment/p2/` SHALL pass
`dojo inspect config`, and their final reported metrics SHALL come from
GPU runs without pilot overrides, recorded in `REPORTS/P2-report.md`. The current experiment matrix is described in
`REPORTS/P2-TRAINING-PLANS.md`.

#### Scenario: Config validity
- **WHEN** `dojo inspect config --format json` runs on each committed
  `configs/experiment/p2/` config
- **THEN** every config composes and validates

#### Scenario: Final numbers are GPU numbers
- **WHEN** P2 experiment results are cited
- **THEN** they come from full GPU runs, not the earlier CPU pilot
  timings
