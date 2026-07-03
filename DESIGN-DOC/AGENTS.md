# DESIGN-DOC — authoritative design specification

## Purpose

The full capability roadmap and canonical design for the dojo refactor. The
implemented code is the P1 thin slice; this tree specifies the whole system
(SSL, ensembles, sweeps, export, tabular input, IFCB bins).

## Ownership

Owns the design spec. `00-OVERVIEW.md` is the entry point and reading order;
`glossary.md` is the single source of truth for terminology (config keys,
identifiers, hashes, record taxonomies).

## Local Contracts

- This spec is authoritative over code intent; when code and spec diverge,
  reconcile explicitly (update the doc or the code, not neither).
- Deferred features live here and in `appendix-deferred-features.md`; they are
  absent from the strict schema, not stubbed in code.
- `13-workplan.md` is the implementation priority order and thin-slice gate.
- The PDF is a compiled artifact of the Markdown (`DESIGN-DOC-as-pdf.sh`); edit
  the Markdown, not the PDF.
</content>
