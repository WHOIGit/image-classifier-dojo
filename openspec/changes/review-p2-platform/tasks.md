## 1. NES GPU experiment campaign

- [ ] 1.1 Run the `00_baseline` config on the GPU host, no pilot overrides
- [ ] 1.2 Run the `04_imbalance` group (label-smoothing, focal, weighted-CE,
      class-balanced sampler, weighted sampler)
- [ ] 1.3 Run the `05_input-fit` group (letterbox, aspect-buckets)
- [ ] 1.4 Run the remaining single-head groups per `REPORTS/P2-TRAINING-PLANS.md`
      (`01_architecture`, `02_provider`, `03_pretraining-and-normalization`,
      `06_augmentation`)
- [ ] 1.5 Record final metrics/figures in `REPORTS/P2-report.md`
      and close the hardware item in `QUESTIONS-FOR-SIDNEY.md`
- [ ] 1.6 Verification: compare the runs' `metrics/metrics.csv` and figures;
      confirm result Parquet loads via the results reader

## 2. Multihead first-runs review

- [ ] 2.1 Review commit `02bc75e` (multihead first runs): configs, result
      rows, per-head metrics, and figures
- [ ] 2.2 Confirm sampler head selection behaved as specified
      (most-imbalanced classification head when unset)
- [ ] 2.3 Verification: `pytest tests/unit/model tests/unit/training
      tests/integration/test_train_supervised.py`

## 3. Contract confirmations

- [ ] 3.1 Confirm stats-cache Parquet sidecar is the durable format;
      regenerate the NES stats cache in that format
- [ ] 3.2 Re-check process-tree PSS on a full NES GPU run with the
      materialized image cache
- [ ] 3.3 Decide on P4.17 local `./configs` shadow rename (schedule or drop)
- [ ] 3.4 Sync any clarified behavior into `openspec/specs/` and
      `DESIGN-DOC/`; verification: `openspec validate --specs`
