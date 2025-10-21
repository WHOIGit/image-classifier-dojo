# src/image_classifier_dojo/__init__.py
import importlib
import sys

import dojo as _dojo
__all__ = getattr(_dojo, "__all__", [])

SUBMODULES = ("multiclass", "utils", "patches")

for submodule in SUBMODULES:
    m = importlib.import_module(f"dojo.{submodule}")
    sys.modules[f"{__name__}.{submodule}"] = m
    globals()[submodule] = m

def __dir__():
    return sorted(set(globals().keys()) | set(dir(_dojo)))


