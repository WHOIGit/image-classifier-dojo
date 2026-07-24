## MODIFIED Requirements

### Requirement: Transform pipeline builder
`transforms.pipeline` SHALL support `resize` (direct to `(height, width)`),
`letterbox` (aspect-preserving padded), `aspect_bucket`, `foreground_crop`,
`grayscale`, `random_erasing`, flips, and `normalize`, with a per-step
`train_only` flag. A resolved-only `inference_pipeline` (train-only steps
stripped) SHALL be derived for non-train stages, export, and
`preprocessing_hash`.

#### Scenario: train_only steps excluded from inference
- **WHEN** a pipeline contains train-only augmentations (e.g. flips, random_erasing)
- **THEN** the derived `inference_pipeline` omits them
