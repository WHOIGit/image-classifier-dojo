## 1. Figure regeneration command

- [ ] 1.1 Add `dojo eval training-figures` accepting a run directory and/or
      `--resolved-config`, resolving `metrics/metrics.csv` and the
      `classification_output` result rows
- [ ] 1.2 Refactor the training-time figure builders into a shared entry point
      callable from both the trainer and the new command
- [ ] 1.3 Write the full HTML figure set to `training_outputs.figures.dir`
      (or an explicit output override) with `--clobber` handling
- [ ] 1.4 Verification: run a short training job, delete `figures/`, regenerate
      via the command, and confirm the HTML set matches a fresh training run's
      figures for the same inputs
