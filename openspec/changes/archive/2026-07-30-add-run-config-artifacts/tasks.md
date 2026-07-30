## 1. Carry composition provenance through resolution

- [x] 1.1 Add a provenance dataclass to `src/dojo/config_loader/resolver.py` (or a
      sibling module) holding the composed `DictConfig`, the overrides tuple, and the
      optional invoked command
- [x] 1.2 Extend `ResolutionResult` with an optional provenance field, defaulted so
      existing construction sites stay valid
- [x] 1.3 Thread the `ComposedConfig` returned by `compose_config` into the
      `ResolutionResult` built by `compose_and_resolve`
      (`src/dojo/config_loader/conductor.py:37-57`)
- [x] 1.4 Add an `invoked_command` parameter to `compose_and_resolve` so the CLI can
      supply the real invocation; default `None`
- [x] 1.5 Export any new public names from `src/dojo/config_loader/__init__.py`
- [x] 1.6 Confirm the other `compose_and_resolve` callers (`src/dojo/cli/eval.py:34`,
      `src/dojo/cli/infer.py:25`, `src/dojo/cli/inspect.py:262,362,418`) still work
      unchanged
- [x] 1.7 Verify: `pytest tests/integration/test_config_fallthrough.py
      tests/integration/test_inspect_config.py`

## 2. Shared run-config-artifact writer

- [x] 2.1 Create the writer module outside `dojo.training` exposing one function that
      takes a `config/` directory, a resolved `RootConfig`, and optional provenance
- [x] 2.2 Write `resolved.yaml` and `resolved.json` from a single
      `cfg.model_dump(mode="json", exclude_none=True)` container so they cannot drift
- [x] 2.3 Write `composed.yaml` from the composed `DictConfig` via OmegaConf when
      provenance is present
- [x] 2.4 Write `overrides.txt` (one override per line) and `cli.txt` (the invoked
      command) when provenance supplies them; trailing newline on both
- [x] 2.5 Skip the three provenance-derived files entirely when provenance is absent
      — no reconstruction from the resolved config
- [x] 2.6 Add unit tests: full set with provenance, resolved-only pair without it,
      `resolved.yaml` and `resolved.json` parsing equal, overrides recorded verbatim
- [x] 2.7 Verify: `pytest tests/unit/config_loader` (new directory for the writer's
      tests)

## 3. Wire into the training run

- [x] 3.1 Replace `_write_resolved_config` (`src/dojo/training/run.py:73-78`) with a
      call to the shared writer
- [x] 3.2 Add an optional provenance parameter to `execute_train`
      (`src/dojo/training/run.py:281`), defaulting to `None`
- [x] 3.3 Capture the invocation in `dojo train` (`src/dojo/cli/train.py`) and pass
      provenance from the `ResolutionResult` into `execute_train`
- [x] 3.4 Update `src/dojo/training/AGENTS.md:27-29` to describe the optional
      provenance argument alongside the existing resolved-config-only note
- [x] 3.5 Extend the fixture end-to-end test to assert `config/` holds all five files
      for a composed run, and only the resolved pair for a direct `execute_train`
      call (`tests/integration/test_train_supervised.py`)
- [x] 3.6 Verify: `pytest tests/integration/test_train_supervised.py
      tests/integration/test_training_fit.py`

## 4. Documentation and change validation

- [x] 4.1 Confirm `src/dojo/cli/render.py:72` still resolves `config/resolved.yaml`
      against a freshly written run directory
- [x] 4.2 Update `src/dojo/config_loader/AGENTS.md` to note that
      `compose_and_resolve` now surfaces composition provenance
- [x] 4.3 Verify: `openspec validate --all` is green and
      `pytest tests/unit tests/integration` passes
