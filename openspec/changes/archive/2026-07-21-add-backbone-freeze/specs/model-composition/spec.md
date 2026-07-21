## ADDED Requirements

### Requirement: Backbone freeze policy
The backbone freeze policy `training.freeze.backbone.policy` SHALL support
`none` (fully trainable, the default) and `frozen`. When `frozen`, the
system SHALL set `requires_grad=False` on every backbone parameter at model
build and SHALL exclude those parameters from the optimizer, so a
pretrained or checkpoint-initialized backbone stays fixed while heads and
any embedding adapter train.

#### Scenario: Frozen backbone transfer
- **WHEN** `training.freeze.backbone.policy: frozen` is set with a
  checkpoint- or library-initialized backbone
- **THEN** backbone parameters are not updated during training and only the
  heads and embedding adapter (if enabled) receive gradients

#### Scenario: Default trainable backbone
- **WHEN** `training.freeze.backbone.policy` is unset or `none`
- **THEN** all backbone parameters remain trainable
