"""Config loader: authored YAML / overrides -> validated, resolved ``RootConfig``.

The subsystem that turns user input into a ready config root, in three steps:

- :mod:`compositor` -- Hydra Compose API wrapper; merges defaults + overrides
  into a raw, untyped ``DictConfig``.
- :mod:`resolver` -- given a validated config, render ``run_id``, resolve output
  paths, and derive the inference pipeline.
- :mod:`conductor` -- sequences compose -> validate -> resolve into one call.

The :func:`compose_and_resolve` conductor is the single entry point callers
(the CLI, tests, a Prefect flow) use; the rest is its delegated machinery.
"""

from dojo.config_loader.compositor import (
    ComposedConfig,
    ConfigCompositionError,
    compose_config,
    default_config_dirs,
)
from dojo.config_loader.compare import compare_config_files
from dojo.config_loader.conductor import (
    compose_and_resolve,
    validate_authored_only_fields,
)
from dojo.config_loader.resolver import (
    ResolutionResult,
    render_template,
    resolve_runtime_and_paths,
)

__all__ = [
    "compose_config",
    "ComposedConfig",
    "ConfigCompositionError",
    "default_config_dirs",
    "compare_config_files",
    "resolve_runtime_and_paths",
    "ResolutionResult",
    "render_template",
    "compose_and_resolve",
    "validate_authored_only_fields",
]
