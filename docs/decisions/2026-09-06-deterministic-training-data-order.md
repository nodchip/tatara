# Bounded deterministic training-data order

Status: accepted

## Context

Repeated runs with different experiment labels do not establish different input
orders. Teacher-score calibration comparisons need identical record order within
each raw/calibrated pair and a deliberate, reproducible change between series.
Copying or globally shuffling a large PSV file requires substantial disk space.

## Decision

Provide a versioned, seeded two-level permutation: permute fixed-size blocks,
read each selected block contiguously, then permute its records in memory.
The relative permutation depends on seed, epoch, and the selected record count,
not on record contents or the range's absolute starting offset. Preserve every
40-byte record and exclude held-out ranges.
Use fixed integer arithmetic and bounded allocations; reject oversized block
tables rather than silently fall back to sequential order.

## Consequences

This is a block permutation, not a uniform permutation of the complete dataset.
It needs no derived dataset on disk and avoids random seeks for individual records.
Distinct configured seeds must be verified through the actual training path,
not inferred from series names. Parallel prefetch completion order must not undo
the specified permutation. The default sequential path must remain unchanged.

`--data-order-seed` selects the permutation in both trainer architectures.
The training loop restores batch sequence after parallel decoding, and
`experiment.json` records the seed, algorithm, block size, and cursor policy.
Every invocation starts from epoch zero, including an optimizer resume. The data
cursor is not checkpointed: restarting is not uninterrupted-training equivalence.
The option does not change initialization seeds or promise bitwise GPU results.

Calibration and score filtering run after the permutation. For a paired comparison,
use the same record range, batch size, seed, training amount, and retained positions;
different score-based filters can otherwise make the consumed streams diverge.
External experiment controllers must propagate and verify these settings. Labels
alone, or a reader unit test, do not prove that a production run used the policy.
