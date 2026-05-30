# Reports Index

This directory is organized by benchmark and report purpose.

## Layout

```text
reports/
  README.md
  alfworld/
  meetings/
  virtualhome/
    data-processing/
    audits/
    validation/
    experiments/
  archive/
    ipynb_checkpoints/
```

## VirtualHome

### Living survey (start here)

```text
reports/virtualhome/wormi-data-construction-failure-survey-2026-05-30.md
```

Chronological "attempt → problem encountered" survey of every VirtualHome
data-construction attempt against the paper Table 1 target, plus ranked
root-cause diagnosis and open questions. **Living document — update it after
every new data-build / training / eval experiment.**

### Data-validity acceptance standard (the ex-ante bar — MUST pass before training)

```text
reports/virtualhome/wormi-data-validity-acceptance-standard-2026-05-31.md
```

Permanent, versioned acceptance criteria with explicit thresholds. Core principle:
data is valid **iff the EVAL pipeline (not the build pipeline) can replay the
expert to ~100%**. HARD gates (NO-GO if any fails) vs SOFT (logged). Enforced by
`tools/expert_replay_vh.py` (A-class) + `tools/validate_virtualhome_dataset.py`.

### `virtualhome/data-processing/`

Data construction protocols, split definitions, implementation notes, and
current processing documentation.

Most relevant current files:

```text
reports/virtualhome/data-processing/vh-current-data-processing-2026-05-28.md
reports/virtualhome/data-processing/vh-current-data-processing-2026-05-28-zh.md
```

Other files here include older paper-like specifications, TMoW compact
processing notes, house/config notes, and correction plans.

### `virtualhome/audits/`

Semantic and quality audits of data construction choices.

Use this directory for:

- semantic validity analysis;
- source/object ambiguity analysis;
- external-claim checks;
- data construction risk reviews.

### `virtualhome/validation/`

Machine-readable validation outputs, mostly JSON.

Use this directory for:

- loader compatibility checks;
- replay checks;
- leakage checks;
- chat-template supervision checks;
- manifest-like validation artifacts.

### `virtualhome/experiments/`

Training, evaluation, and failure-analysis notes.

Use this directory for:

- reproduction progress logs;
- stage-2 diagnosis;
- meta-learning/threaded-vs-sequential analysis;
- final or partial evaluation summaries.

Current summary table:

```text
reports/virtualhome/experiments/vh-historical-results-summary-2026-05-29.md
```

## ALFWorld

```text
reports/alfworld/
```

Contains ALFWorld data processing and protocol notes.

## Meetings

```text
reports/meetings/
```

Contains meeting notes and discussion summaries.

## Archive

```text
reports/archive/
```

Contains non-primary artifacts such as old Jupyter checkpoint files. These are
kept for traceability but should not be treated as canonical reports.
