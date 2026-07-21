# Model Composition Specification

## Purpose

Backbone registry, heads, objectives, multi-head normalization, and the
embedding adapter for supervised models. Source design:
`DESIGN-DOC/05-models-training-and-heads.md` (workplan P2.3).
## Requirements
### Requirement: Backbone registry
The backbone registry SHALL support `architecture.source: torchvision`
(open architecture-name string) and `architecture.source: timm` (gated
by the `timm` extra, raising a clear runtime error when the extra is
absent), with `weights.source: none | library | checkpoint`
initialization. `output_dim: auto` SHALL resolve to the concrete
architecture output dimension at config resolution.

#### Scenario: Checkpoint-initialized backbone
- **WHEN** `weights.source: checkpoint` points at a Dojo Lightning
  checkpoint
- **THEN** state-dict prefixes are handled and the backbone initializes
  from the checkpoint (supervised transfer learning)

#### Scenario: timm without the extra
- **WHEN** a timm backbone is configured but `timm` is not installed
- **THEN** a clear runtime error names the missing extra

### Requirement: Heads and objectives
The system SHALL provide a head registry with target validation, and
objectives that bind heads to losses, metrics, and weights, with
multi-head / multi-objective normalization.

#### Scenario: Per-head target routing
- **WHEN** multiple heads are configured against different targets
- **THEN** each objective reads labels from the target configured on its
  head (a coarse head never trains against the primary species labels)

### Requirement: Embedding adapter
`model.embedding_adapter.enabled: true` SHALL insert a `linear` or `mlp`
adapter that transforms the image backbone embedding before all heads;
heads are built against the adapter output dimension.

#### Scenario: Adapter changes head input dim
- **WHEN** an adapter with `output_dim: 512` is enabled on a 1280-dim
  backbone
- **THEN** all heads are constructed with 512-dim input

### Requirement: Packaged backbone config groups
Packaged config groups SHALL include torchvision defaults and
`backbone/timm/default` / `backbone/timm/efficientnet_b0` (timm
`efficientnet_b0`, `output_dim: auto`, `input_channels: 3`,
`weights.source: library`), selectable via
`/backbone/timm@model.image_input.backbone`.

#### Scenario: Experiment selects packaged timm backbone
- **WHEN** an authored experiment selects the packaged timm default
- **THEN** config composition and validation succeed

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

