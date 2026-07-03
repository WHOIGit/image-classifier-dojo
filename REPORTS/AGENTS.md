# REPORTS — durable work reports

## Purpose

Durable implementation and experiment reports for project work that needs a
continuous record of steps, decisions, verification, and findings.

## Ownership

Owns human-readable Markdown reports under `REPORTS/`. Reports may summarize
commands, outputs, decisions, and links to produced artifacts, but they do not
own runtime code contracts.

## Local Contracts

- Keep reports chronological and factual.
- Use full timestamps on new report entry headings:
  `YYYY-MM-DD HH:MM TZ`.
- Record decision junctions in `QUESTIONS-FOR-SIDNEY.md` when work continues
  through ambiguity; link or mention them from the report when relevant.
- Do not store generated model artifacts, Parquet results, or large figures in
  `REPORTS/`; link to run directories instead.

## Work Guidance

- Prefer concise timestamped entries with command intent, result, and next
  action.

## Verification

- Reports are verified by review and by the referenced command/test outcomes.

## Child DOX Index
