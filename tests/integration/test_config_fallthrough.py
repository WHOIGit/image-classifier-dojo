"""Search-path fall-through between packaged and project-root ./configs.

A packaged experiment (src/dojo/configs) and a local experiment (./configs) must
both compose: the packaged one resolves entirely from the package even though
./configs is searched first, and the local one resolves its packaged config
groups (runtime, backbone, …) from the searchpath while its own experiment file
and local `data` group are found first under ./configs.
"""

from omegaconf import OmegaConf

from dojo.config_schemas import RootConfig
from dojo.config_loader import compose_config


def _resolve(name: str) -> RootConfig:
    composed = compose_config(overrides=[f"experiment=p1/{name}"])
    return RootConfig.model_validate(OmegaConf.to_container(composed.config, resolve=True))


def test_packaged_experiment_composes_as_primary_from_package():
    # Regression: the packaged primary config must load even though it lives
    # only in the packaged dir (on the searchpath), not in ./configs.
    cfg = _resolve("plankton-toy")
    assert cfg.training.max_epochs == 3
    assert cfg.checkpointing.monitor == "val/species/f1_macro"
    assert cfg.checkpointing.mode == "max"
    assert list(cfg.objectives["species"].metrics) == ["f1_macro"]


def test_local_experiment_resolves_packaged_groups_via_fallthrough():
    # The local plankton-miniset experiment lives under ./configs and selects the
    # local `data=plankton-miniset` group, but pulls every other group from the
    # packaged searchpath.
    cfg = _resolve("plankton-mini_efficientnet")
    # From the local experiment file / local data group:
    assert cfg.training.max_epochs == 6
    assert cfg.model.heads["species"].num_classes == 30
    assert cfg.data.manifest_uri == "./datasets/plankton-miniset/data.parquet"
    assert cfg.data.split_column == "split"
    # Inherited from packaged groups via the searchpath:
    assert cfg.model.image_input.backbone.architecture.name == "efficientnet_b0"
