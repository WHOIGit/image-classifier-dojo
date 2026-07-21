## 1. NES GPU experiment campaign

- [ ] 1.1 Run `nes_effb0_224_baseline` on the GPU host, no pilot overrides
- [ ] 1.2 Run `nes_effb0_buckets`
- [ ] 1.3 Run `nes_effb0_weighted`
- [ ] 1.4 Run `nes_effb0_buckets_weighted`
- [ ] 1.5 Record final metrics/figures in `REPORTS/report-P2.md` and close
      the hardware item in `QUESTIONS-FOR-SIDNEY.md`
- [ ] 1.6 Verification: compare the four runs' `metrics/metrics.csv` and
      figures; confirm result Parquet loads via the results reader

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
