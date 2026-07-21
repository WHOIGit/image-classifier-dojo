## ADDED Requirements

### Requirement: DINOv2 pretraining
The system SHALL support DINOv2 self-supervised pretraining via Lightly
(gated by the `ssl` extra), producing checkpoints consumable as
`weights.source: checkpoint` backbone initialization for supervised
transfer.

#### Scenario: SSL checkpoint reused for supervised transfer
- **WHEN** a supervised config points its backbone at a DINOv2
  checkpoint
- **THEN** the backbone initializes from the SSL weights
