## Why

P2 (core supervised platform) is implemented — its behavior is captured
across `openspec/specs/` and `DESIGN-DOC/13-workplan.md` — but several
items still need human review, real-hardware verification, or a decision
from Sidney before P2 can be declared closed and the NES experiment
campaign judged. This change collects that verification debt so it is
tracked in one place instead of scattered across `QUESTIONS-FOR-SIDNEY.md`.

## What Changes

No new capabilities — this is a review/verification change:

- Run the NES experiment configs under `configs/experiment/p2/` on the
  GPU host without pilot overrides and record final numbers (open item in
  `QUESTIONS-FOR-SIDNEY.md`). See `REPORTS/P2-TRAINING-PLANS.md` for the
  current experiment matrix.
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

- `configs/experiment/p2/*`, `REPORTS/P2-TRAINING-PLANS.md`,
  `QUESTIONS-FOR-SIDNEY.md`
- No planned code changes unless review finds defects.
