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


def test_authored_experiment_can_select_packaged_timm_default(tmp_path):
    config_file = tmp_path / "timm-toy.yaml"
    config_file.write_text(
        """
# @package _global_
defaults:
  - /runtime: default
  - /storage: local_only
  - /data: plankton-toyset
  - /transforms: supervised_default
  - /backbone/timm@model.image_input.backbone: default
  - /optimizer: adamw
  - /training_outputs: p1_local
  - _self_

experiment:
  name: timm_default_smoke

task:
  type: supervised

model:
  image_input:
    name: image
    backbone:
      weights:
        source: none
  tabular_input:
    enabled: false
    name: tabular
  embedding_adapter:
    enabled: false
  heads:
    species:
      type: multiclass_classification
      target: species
      num_classes: 6
      network:
        type: linear

objectives:
  species:
    head: species
    loss: cross_entropy
    metrics: [f1_macro]
    weight: 1.0

training:
  max_epochs: 1
  batch_size: 16
  freeze:
    backbone:
      policy: none

checkpointing:
  monitor: val/species/f1_macro
  mode: max
  save_top_k: 1
  save_last: true

output_root: ./runs
""",
        encoding="utf-8",
    )

    composed = compose_config(config_file=config_file)
    cfg = RootConfig.model_validate(OmegaConf.to_container(composed.config, resolve=True))

    assert cfg.model.image_input.backbone.architecture.source == "timm"
    assert cfg.model.image_input.backbone.architecture.name == "efficientnet_b0"
    assert cfg.model.image_input.backbone.weights.source == "none"
