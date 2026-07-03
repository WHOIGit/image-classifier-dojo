"""Hydra composition wrapper for Dojo configs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


class ConfigCompositionError(RuntimeError):
    """Raised when a config root cannot be selected or composed."""


@dataclass(frozen=True)
class ComposedConfig:
    config: DictConfig
    config_name: str
    config_dirs: tuple[Path, ...]
    overrides: tuple[str, ...]


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def default_config_dirs(extra_config_dirs: Iterable[Path] = ()) -> list[Path]:
    """Return explicit dirs, then local ./configs, then packaged defaults."""

    candidates: list[Path] = [Path(path) for path in extra_config_dirs]
    local_configs = Path.cwd() / "configs"
    if local_configs.exists():
        candidates.append(local_configs)

    with as_file(files("dojo").joinpath("config_defaults")) as packaged_defaults:
        candidates.append(Path(packaged_defaults))
        return _dedupe_paths(candidates)


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _relative_config_name(config_file: Path, config_dirs: Iterable[Path]) -> tuple[str, Path] | None:
    config_file = config_file.resolve()
    for config_dir in config_dirs:
        config_dir = config_dir.resolve()
        try:
            relative = config_file.relative_to(config_dir)
        except ValueError:
            continue
        if relative.suffix not in {".yaml", ".yml"}:
            continue
        return relative.with_suffix("").as_posix(), config_dir
    return None


def _split_experiment_selector(overrides: list[str]) -> tuple[str | None, list[str]]:
    remaining: list[str] = []
    experiment_name: str | None = None
    for override in overrides:
        if experiment_name is None and override.startswith("experiment="):
            experiment_name = override.split("=", 1)[1]
            continue
        remaining.append(override)
    if experiment_name is None:
        return None, remaining
    if not experiment_name:
        raise ConfigCompositionError("experiment= selector cannot be empty")

    experiment_file_prefixes = ("configs/experiment/", "./configs/experiment/")
    suggested_group = None
    for prefix in experiment_file_prefixes:
        if experiment_name.startswith(prefix):
            suggested_group = experiment_name.removeprefix(prefix)
            suggested_group = suggested_group.removesuffix(".yaml").removesuffix(".yml")
            break

    if experiment_name.endswith((".yaml", ".yml")) or suggested_group is not None:
        group_hint = suggested_group or "p1/plankton-toy"
        config_hint = (
            experiment_name
            if experiment_name.endswith((".yaml", ".yml"))
            else f"{experiment_name}.yaml"
        )
        raise ConfigCompositionError(
            "experiment= selects a Hydra experiment group, not a YAML file. "
            "Use --config "
            f"{config_hint} for a concrete file, or use "
            f"experiment={group_hint} for this config group."
        )
    return f"experiment/{experiment_name}", remaining


def _owning_dir_first(config_name: str, config_dirs: list[Path]) -> list[Path]:
    """Reorder so the dir that contains ``config_name`` is primary.

    Hydra cannot load a *primary* config that lives only on the runtime
    ``hydra.searchpath``; the primary config must be in the dir passed to
    ``initialize_config_dir``. The remaining dirs stay on the searchpath so
    group defaults still fall through (e.g. a local experiment that extends a
    packaged one resolves the packaged base from the searchpath).
    """

    candidates = [f"{config_name}.yaml", f"{config_name}.yml"]
    for index, config_dir in enumerate(config_dirs):
        if any((config_dir / candidate).exists() for candidate in candidates):
            rest = [d for position, d in enumerate(config_dirs) if position != index]
            return [config_dir, *rest]
    return list(config_dirs)


def _compose_from_search_path(
    *,
    config_name: str,
    config_dirs: list[Path],
    overrides: list[str],
) -> ComposedConfig:
    if not config_dirs:
        raise ConfigCompositionError("no config directories are available")
    config_dirs = _owning_dir_first(config_name, config_dirs)
    primary = config_dirs[0]
    fallback_dirs = config_dirs[1:]
    hydra_overrides = list(overrides)
    if fallback_dirs:
        searchpath = ",".join(_file_uri(path) for path in fallback_dirs)
        hydra_overrides.append(f"hydra.searchpath=[{searchpath}]")

    with initialize_config_dir(version_base=None, config_dir=str(primary)):
        cfg = compose(
            config_name=config_name,
            overrides=hydra_overrides,
            return_hydra_config=False,
        )
    OmegaConf.resolve(cfg)
    return ComposedConfig(
        config=cfg,
        config_name=config_name,
        config_dirs=tuple(config_dirs),
        overrides=tuple(overrides),
    )


def compose_config(
    *,
    overrides: Iterable[str] = (),
    config_file: Path | None = None,
    resolved_config_file: Path | None = None,
    config_dirs: Iterable[Path] = (),
) -> ComposedConfig:
    """Compose or load a Dojo root config.

    P1 supports direct experiment roots, e.g.
    ``experiment=p1/plankton-toy``, and direct files under a
    config root. Local ``./configs`` shadows packaged defaults, so local roots
    can reference those defaults without copying the whole tree.
    """

    override_list = list(overrides)
    roots_selected = sum(
        source is not None for source in (config_file, resolved_config_file)
    )
    if roots_selected > 1:
        raise ConfigCompositionError("--config and --resolved-config are mutually exclusive")

    search_dirs = default_config_dirs(config_dirs)

    if resolved_config_file is not None:
        path = resolved_config_file.resolve()
        if not path.exists():
            raise ConfigCompositionError(f"resolved config does not exist: {path}")
        cfg = OmegaConf.load(path)
        if override_list:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_list))
        OmegaConf.resolve(cfg)
        return ComposedConfig(
            config=cfg,
            config_name=str(path),
            config_dirs=tuple(search_dirs),
            overrides=tuple(override_list),
        )

    if config_file is not None:
        path = config_file.resolve()
        if not path.exists():
            raise ConfigCompositionError(f"config file does not exist: {path}")
        relative = _relative_config_name(path, search_dirs)
        if relative is None:
            config_name = path.with_suffix("").name
            compose_dirs = _dedupe_paths([path.parent, *search_dirs])
        else:
            config_name, owning_dir = relative
            compose_dirs = _dedupe_paths([owning_dir, *search_dirs])
        return _compose_from_search_path(
            config_name=config_name,
            config_dirs=compose_dirs,
            overrides=override_list,
        )

    config_name, remaining_overrides = _split_experiment_selector(override_list)
    if config_name is None:
        for config_dir in search_dirs:
            if (config_dir / "config.yaml").exists():
                config_name = "config"
                break
    if config_name is None:
        raise ConfigCompositionError(
            "select a config root, for example "
            "experiment=p1/plankton-toy or --config PATH"
        )

    return _compose_from_search_path(
        config_name=config_name,
        config_dirs=search_dirs,
        overrides=remaining_overrides,
    )
