from dojo.config_schemas import RootConfig
from dojo.config_loader import compose_config
from dojo.config_loader import resolve_runtime_and_paths


def test_p1_transform_pipeline_derives_inference_pipeline():
    composed = compose_config(
        overrides=["experiment=p1/plankton-toy"]
    )
    cfg = RootConfig.model_validate(composed.config)
    resolved = resolve_runtime_and_paths(cfg).config

    assert resolved.transforms.image_mode == "rgb"
    # The authored pipeline interleaves train-only augmentation between the
    # deterministic resize and normalize steps.
    assert [step.name for step in resolved.transforms.pipeline] == [
        "letterbox",
        "rotate",
        "horizontal_flip",
        "normalize",
    ]
    # inference_pipeline drops the train_only augmentations.
    assert resolved.transforms.inference_pipeline is not None
    assert [step.name for step in resolved.transforms.inference_pipeline] == [
        "letterbox",
        "normalize",
    ]
