## 1. Config schema

- [ ] 1.1 Add `RandomErasingStep` Pydantic model to `src/dojo/config_schemas/root.py` with fields: `name: Literal["random_erasing"]`, `enabled: bool = True`, `train_only: bool = True`, `num_patches: int = 1`, `scale: tuple[float, float] = (0.02, 0.33)`, `ratio: tuple[float, float] = (0.3, 3.3)`, `value: float | Literal["random"] = 0.0`, `p: float = 0.5`
- [ ] 1.2 Add `RandomErasingStep` to the `TransformStep` discriminated union in `src/dojo/config_schemas/root.py`
- [ ] 1.3 Verify: `openspec validate --all` passes; schema round-trips a `random_erasing` step via Pydantic

## 2. Transform implementation

- [ ] 2.1 Add `_random_erasing` helper function to `src/dojo/data/transforms.py`: rejection-sample a bounding box per patch (max 10 attempts per patch), fill with `value` float or `torch.rand_like` slice, apply probability gate `p` once before the patch loop
- [ ] 2.2 Add `RandomErasingStep` to the imports from `dojo.config_schemas.root` in `src/dojo/data/transforms.py`
- [ ] 2.3 Add dispatch branch for `RandomErasingStep` in `_CompiledImageTransform.__call__` after the flip branches
- [ ] 2.4 Verify: `tests/` — run unit tests covering `_random_erasing`: no-op at p=0, single black patch, multi-patch random fill, graceful skip when rejection sampling exhausted

## 3. Experiment config

- [ ] 3.1 Fill in `configs/experiment/p2/06_augmentation/nes_cutouts.yaml`: replace the placeholder comment with a concrete `random_erasing` pipeline step (`num_patches: 1`, `scale: [0.02, 0.33]`, `ratio: [0.3, 3.3]`, `value: 0.0`, `p: 0.5`, `train_only: true`) after the `resize` step and before `normalize`
- [ ] 3.2 Update the experiment name in the yaml from `p2_aug_cutouts` (already set) — no change needed; verify `dojo cfg show experiment=p2/06_augmentation/nes_cutouts` resolves without error

## 4. Spec sync

- [ ] 4.1 Merge `random_erasing` into the `transforms.pipeline` step list in `openspec/specs/transforms-and-sampling/spec.md` (add it to the enumerated step names in the Transform pipeline builder requirement)
- [ ] 4.2 Verify: `openspec validate --all` passes with the updated main spec
