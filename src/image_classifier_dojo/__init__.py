# src/image_classifier_dojo/__init__.py
import sys
import importlib
from typing import TYPE_CHECKING, Any

import dojo as _dojo
__all__ = getattr(_dojo, "__all__", [])

SUBMODULES = ("schemas", "multiclass", "utils", "patches")

for submodule in SUBMODULES:
    m = importlib.import_module(f"dojo.{submodule}")
    sys.modules[f"{__name__}.{submodule}"] = m
    globals()[submodule] = m

def __dir__():
    return sorted(set(globals().keys()) | set(dir(_dojo)))

def __getattr__(name: str) -> Any:  # PEP 562: forward unknown attrs to dojo
    return getattr(_dojo, name)

if TYPE_CHECKING:
    # Re-exporting dojo’s public API and submodules for ide introspection
    from dojo import *  # noqa: F401,F403
    from dojo import schemas as schemas
    from dojo import multiclass as multiclass
    from dojo import utils as utils
    from dojo import patches as patches