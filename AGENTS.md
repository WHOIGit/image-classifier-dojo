# image-classifier-dojo — root AGENTS.md

## Project Purpose

Train computer-vision classifier models configured through Hydra + Pydantic,
reading manifest-backed image datasets and writing canonical, queryable Parquet
results. The current code is the completed **Priority-1 end-to-end thin slice**
plus the implemented **Priority-2 core supervised platform** surface. The full
roadmap lives in `DESIGN-DOC/`.

## Ownership

- `src/dojo/` — the package (implemented P1 slice). `src/dojo_deprecated/` is
  legacy reference code, outside the DOX contract; do not extend it.
- `flake.nix` — reproducible Nix development shell, including Git and Git LFS
  for fetching LFS-tracked dataset assets.
- `DESIGN-DOC/` — authoritative spec and roadmap.
- `tests/`, `configs/`, `datasets/` — see their child docs.
- `REPORTS/` — durable implementation and experiment reports.
- `QUESTIONS-FOR-SIDNEY.md` — review queue for decision junctions made while
  continuing through blockers or underspecified P2 scope.
- `amplify-db-utils/` — a **nested sibling git repo** (its own `.git`), vendored
  here for local development; governed by its own repo, not this DOX tree.
- `.agents/`, `.codex/`, `MY-NOTES/` — currently empty scratch dirs; no contract.

## Local Contracts

- Pydantic schemas (`src/dojo/config_schemas/`) are the runtime contract; strict
  schema (`extra="forbid"`), deferred features absent not stubbed.
- Functional supervised dataset backends are `csv_manifest`, `parquet_manifest`,
  and `parquet_images`.
- `*_hash` columns and `run_id` are deterministic/reproducible per config.
- Dash-prefixed CLI tokens are options; dash-free `key=value` are Hydra overrides.
- The installed console script is `dojo`; base install is Torch-free for
  config/init/storage/result operations and lightweight inspect paths. Dataset
  inspection, model inspection, training, inference, and eval use the vision
  stack.

## Verification

- `pytest` (fast loop, skips expensive); `pytest --run-expensive` (full,
  including real Lightning fits). Run from repo root.

## Communications from the SATI Agent

SATI (`~/Projects/sibert-2026/sati`) is a sibling project: slice-aware object-detection
training/inference over very large images (100+ MP), built on the same input methodology
as dojo — Hydra Compose + strict Pydantic schemas + a compose→validate→resolve
`config_loader` (SATI's compositor/resolver/conductor split is copied from dojo's), with
deterministic `*_hash` provenance columns and Parquet result ledgers. Sidney (the end
user) explicitly requests that design notes transfer between SATI and dojo in both
directions; entries below are notes from the SATI agent that may be worth adopting here.

- **Deterministic `config_hash` for file-backed weights (2026-07-02).** If a weights/
  checkpoint config field points at a real local file, hash the file *bytes* (SATI's
  `checkpoint_hash`) and substitute that digest for the path string in the config-hash
  source; hub asset names (not local files) hash verbatim. This keeps `config_hash`
  machine- and location-independent without erasing model identity. Dojo's
  `config_schemas/hashing.py` may want the same rule for backbone/checkpoint paths.
- **Hydra primary-root constraint (2026-07-02).** SATI hit the same constraint dojo's
  compositor documents (primary config can't live on the runtime searchpath) and adopted
  a hybrid: packaged `config.yaml` root for selector-less runs, dojo-style
  `experiment=name` promotion otherwise. No action needed in dojo; recorded for parity.
- **Timestamps in reports and questions (2026-07-02 23:15 EDT, from Sidney).** Sidney asks
  the dojo agent to carry full timestamps — `YYYY-MM-DD HH:MM TZ` — on every new
  `REPORTS/` entry heading and every new `QUESTIONS-FOR-SIDNEY.md` entry, not just dates.
  SATI adopted the same rule in its `REPORTS/AGENTS.md`; record it in dojo's
  `REPORTS/AGENTS.md` Local Contracts on the next DOX pass there.

---

# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

- [src/dojo/](src/dojo/AGENTS.md) — the `dojo` package; its own index covers
  `cli`, `config_loader`, `config_schemas`, `configs`, `data`, `model`,
  `training`, `storage`, `results`, and `inference`.
- [tests/](tests/AGENTS.md) — unit + integration suite and committed fixtures.
- [DESIGN-DOC/](DESIGN-DOC/AGENTS.md) — authoritative design spec and roadmap.
- [configs/](configs/AGENTS.md) — project-root config groups shadowing the
  packaged ones.
- [datasets/](datasets/AGENTS.md) — committed (LFS) and bring-your-own datasets.
- [REPORTS/](REPORTS/AGENTS.md) — durable implementation and experiment reports.

Not indexed (no local DOX contract): `src/dojo_deprecated/` (legacy reference),
`runs/` (generated run artifacts), `amplify-db-utils/` (nested sibling repo,
self-governed), `.agents/` / `.codex/` / `MY-NOTES/` (empty scratch).
