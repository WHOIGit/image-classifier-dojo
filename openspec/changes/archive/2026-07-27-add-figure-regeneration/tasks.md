## 1. `dojo render` command group

- [x] 1.1 Add a new top-level `render` Typer sub-app and register it in
      `cli/main.py` alongside `eval`/`infer`/`inspect`
- [x] 1.2 Add `dojo render run <run_dir>` taking the run's top-level directory
      as a positional argument; abort with a clear message if `metrics/` or
      `results/` are missing

## 2. Regenerate figures from a completed run

- [x] 2.1 Resolve `<run_dir>/metrics/metrics.csv` and the `<run_dir>/results/`
      `classification_output` rows for the shared figure entry point
- [x] 2.2 Load `<run_dir>/config/resolved.yaml` via the existing resolved-config
      load path and rebuild `objective_to_head`
      (`{name: obj.head or name for enabled objectives}`); fall back to identity
      when the snapshot is absent
- [x] 2.3 Call `write_training_figures()` with those inputs so regenerated
      output is byte-identical to the training-time figure set

## 3. Output handling

- [x] 3.1 Overwrite `<run_dir>/figures/` in place by default
- [x] 3.2 Add `--backup`: copy an existing `figures/` to the next
      numeric-incremented sibling (`figures.1/`, `figures.2/`, …) before
      overwriting

## 4. Verification

- [x] 4.1 Unit test: point `render run` at a fixture run dir (metrics + results
      + `config/resolved.yaml`) and assert the full HTML figure set is written
- [x] 4.2 Unit test: `--backup` preserves the prior `figures/` under an
      incremented sibling and still overwrites `figures/`
- [x] 4.3 Unit test: identity fallback when `config/resolved.yaml` is absent
