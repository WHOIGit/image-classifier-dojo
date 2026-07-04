# configs — project-root config shadow

## Purpose

Optional project-root Hydra config groups that **shadow and extend** the
packaged groups in `src/dojo/config_defaults/`, so datasets/experiments can be
overridden or added without copying the whole packaged tree.

## Ownership

Owns local, uncommitted-friendly config overrides, including data selectors for
`plankton-miniset`, NES, and locally generated cleanup-automorph ROI manifests,
plus local `p1/` and `p2/` experiment roots. Packaged defaults are owned by
`src/dojo/config_defaults/`.

## Local Contracts

- Group names here override same-named packaged groups; keep the schema shape
  identical (must validate against `config_schemas`).
- Mirror the packaged directory layout (`data/`, `experiment/`, …).
- Cleanup-automorph ROI selectors point at gitignored local manifests under
  `datasets/cleanup-automorph-*-rois/`; target/head names preserve capitalized
  morphocode keys (`Z1`, `A1`, …), while helper columns stay lowercase
  (`z1_label`, `a1_label_name`, …).
- Cleanup-automorph multihead morphology experiments keep `Z1` and `T1` /
  `Type` transcribed as data targets but do not enable them as model heads or
  objectives. T1-only experiments may enable `T1` as the sole head.
- The cleanup-automorph denticle experiment uses `drop_sample` on morphology
  targets so only rows complete across all configured heads reach training or
  dataset inspection; `T1` remains `mask_objective`.
</content>
