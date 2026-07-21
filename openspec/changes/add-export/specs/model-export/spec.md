## ADDED Requirements

### Requirement: TorchScript and ONNX export
The system SHALL export trained checkpoints to TorchScript and ONNX
(ONNX gated by the `onnx` extra), building the model from the embedded
`dojo_inference_contract`, and SHALL write an export metadata artifact
carrying the compatibility hashes, resolved inference pipeline, and
per-head class maps.

#### Scenario: Export from a checkpoint
- **WHEN** `dojo export` runs against a Dojo checkpoint
- **THEN** the exported model plus metadata artifact are written without
  reading the producing run's `resolved.yaml`

### Requirement: Bucket-aware ONNX
ONNX export of aspect-bucket models SHALL preserve per-bucket input
shapes.

#### Scenario: Bucketed model export
- **WHEN** a model trained with `aspect_bucket` transforms is exported
  to ONNX
- **THEN** the export handles the configured bucket canvases rather than
  assuming a single fixed input size
