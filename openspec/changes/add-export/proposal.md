## Why

Workplan P3.1 (`DESIGN-DOC/13-workplan.md`, `DESIGN-DOC/10-export.md`):
downstream serving and ensembling of exported models need TorchScript
and ONNX export. Not started; no `src/dojo/export/` exists.

## What Changes

- TorchScript and ONNX export of trained checkpoints via the embedded
  portable inference contract.
- Export metadata artifact (compatibility hashes, preprocessing state,
  class maps) alongside the exported model.
- Bucket-aware ONNX export for aspect-bucket models.
- ONNX path gated by the existing `onnx` extra.

## Capabilities

### New Capabilities

- `model-export`: TorchScript/ONNX export commands, export metadata, and
  bucket-aware ONNX.

### Modified Capabilities

- `results-and-artifacts`: export metadata joins the artifact taxonomy.

## Impact

- New `src/dojo/export/`; new CLI group (`dojo export`).
- Consumes `dojo_inference_contract`; exercised later by P3.2
  ensembling of exported models.
