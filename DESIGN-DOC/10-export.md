
# 10. Export

## Purpose

Defines portable model export: artifact types, the `*_outputs.export`
sub-block, the `dojo export` command, and export metadata.

## Artifact types

Export artifact `type` values in config:

- `torchscript`
- `onnx`

There is no `pt` type. `.pt` is the **default filename extension** that
TorchScript artifacts use; the `type` field names the artifact format,
not its filename suffix.

State-dict-only export is not an initial-implementation artifact type.
Pickled state dicts are training-internal artifacts produced by Lightning
checkpointing (`.ckpt`); portable exports go through `torchscript` or
`onnx`.

## Configuration

Export is explicit. It is controlled by `training_outputs.export`,
`ensemble_outputs.export`, `sweep_outputs.export`, `eval_outputs.export`,
the `dojo export` command, or task-orchestration config. ONNX export is
**not** a runtime training config item — it lives in the export config.

```yaml
training_outputs:
  export:
    enabled: true
    artifacts:
      - type: torchscript
        name: model.pt
        source: best_checkpoint
      - type: onnx
        name: model.onnx
        source: best_checkpoint
        opset: 18
        dynamic_axes: true
```

The same `export:` sub-block shape applies under `ensemble_outputs:`,
`sweep_outputs:`, and `eval_outputs:`. Each writes into its own `exports/`
sub-directory under the resolved `*_outputs.dir`.

## `dojo export`

Explicit conversion of checkpoints, ensemble manifests, or already-existing
run output into portable artifacts.

`--checkpoint`, `--output`, `--type`, and `--ensemble-manifest` are
command options, not config keys (see `02-cli-and-task-types.md`).

```bash
dojo export \
  --checkpoint s3://bucket/runs/run123/checkpoints/best.ckpt \
  --output s3://bucket/runs/run123/exports/model.pt \
  --type torchscript
```

Ensemble export:

```bash
dojo export \
  --ensemble-manifest s3://bucket/runs/run123/ensemble_manifests/manifest.json \
  --output s3://bucket/runs/run123/exports/snapshot_ensemble.pt \
  --type torchscript
```

`dojo export` does not decide which checkpoints belong in an ensemble.
That decision belongs to `dojo ensemble`.

## Export metadata

Exported artifacts include:

```text
model architecture
backbone architecture source / name
backbone weights provenance
head definitions
ordinal encoding / decoding (per ordinal head)
objective summary
class names
class index mappings
normalization mean / std
image mode
resize policy
bucket definitions
tabular feature names / order
tabular encoder config and resolved input / output dimensions
tabular normalization stats
tabular imputation (per-column strategy + frozen fill values)
tabular missing-indicator columns (when enabled)
model input names and implicit concatenation order
input shape
checkpoint weight URI
config_hash
model_config_hash
target_schema_hash
class_mapping_hash
preprocessing_hash
Dojo version
```

This metadata is the **portable inference contract**
(`06-results-artifacts-and-metadata.md`) — the same schema embedded in
training checkpoints under `checkpoint["dojo_inference_contract"]` — so
`dojo infer` / `dojo eval` consume exports and checkpoints through one path.

If authored config used `weights.name: DEFAULT`, export metadata records
the resolved concrete provider weight identity, not the `DEFAULT` alias.

For ONNX, metadata is also embedded into ONNX metadata properties when
possible, with `metadata.json` written alongside `model.onnx`.

### Aspect / size buckets and ONNX

For bucketed-resize models:

- CNN / ConvNeXt / ResNet exports often support dynamic H / W via
  `dynamic_axes: true`.
- ViT exports are typically more robust as **one ONNX file per bucket
  shape**. The `metadata.json` records bucket → ONNX file mapping:

```yaml
preprocessing:
  resize_policy: bucketed
  buckets:
    - name: square
      size: [224, 224]
      min_aspect: 0.75
      max_aspect: 1.33
      onnx_model: model_224x224.onnx
    - name: wide
      size: [224, 448]
      min_aspect: 1.33
      max_aspect: 3.0
      onnx_model: model_224x448.onnx
```

## Exports directory

Exports always land under `<*_outputs.dir>/exports/`. See
`06-results-artifacts-and-metadata.md` for the per-block artifact
layout. Snapshot-ensemble runs share one `exports/` directory across the
training and ensemble phases; per-artifact `name` values disambiguate.

## Cross-References

- `02-cli-and-task-types.md` — `dojo export` command.
- `03-configuration.md` — `*_outputs.export` placement.
- `05-models-training-and-heads.md` — supervised model composition
  (export source).
- `06-results-artifacts-and-metadata.md` — `exports/` sub-directory,
  `model_id` / `model_hash`, compatibility-hash field selection.
- `08-ensembles.md` — exported ensemble artifacts.
- `11-dependencies.md` — `onnx` optional extra.
- `glossary.md` — `model_id`, `model_hash` definitions.
