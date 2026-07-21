## Why

P2 (core supervised platform) is implemented per `REPORTS/report-P2.md`
and `DESIGN-DOC/13-workplan.md`, but several items still need human
review, real-hardware verification, or a decision from Sidney before P2
can be declared closed and the NES experiment campaign judged. This
change collects that verification debt so it is tracked in one place
instead of scattered across `QUESTIONS-FOR-SIDNEY.md` and report notes.

## What Changes

No new capabilities — this is a review/verification change:

- Run the four committed NES experiment configs
  (`configs/experiment/p2/nes_effb0_*.yaml`) on the GPU host without
  pilot overrides and record final numbers (open item in
  `QUESTIONS-FOR-SIDNEY.md`).
- Review the `multihead first runs` work (commit `02bc75e`) — the
  multi-target multihead runtime path is the newest and least-reviewed
  P2 surface; confirm its results and figures look correct.
- Confirm the stats-cache Parquet-sidecar format is the durable
  committed-asset shape (the earlier inline-dimensions JSON was flagged
  as unacceptable for NES-scale caches).
- Confirm memory behavior of the materialized image cache on a full NES
  GPU run remains acceptable (~5 GB process-tree PSS at last sample).
- Decide whether the P4.17 `./configs` → `./config_defaults` local
  shadow rename should be scheduled or dropped.
- Sync outcomes back into `DESIGN-DOC/` and the main specs under
  `openspec/specs/` where behavior was clarified.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

(none expected; if review uncovers contract changes, add delta specs
here — likely candidates: `dataset-backends`, `supervised-training`)

## Impact

- `configs/experiment/p2/*`, `REPORTS/report-P2.md`,
  `QUESTIONS-FOR-SIDNEY.md`
- No planned code changes unless review finds defects.
