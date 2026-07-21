# Empty `--resume` Compatibility

## Goal

Make `nnue-train --resume ""` behave exactly like omitting `--resume`, matching
the command-line compatibility expected by scripts that always emit the option.
Non-empty checkpoint paths must retain the existing resume behavior.

## Behavior

- An exactly empty `--resume` value is normalized to `None`.
- A non-empty value remains a checkpoint path and is loaded as before.
- Whitespace-only values are not trimmed or treated as empty.
- The behavior applies regardless of whether the global option appears before or
  after the architecture subcommand.

## Implementation

Keep the existing `Option<PathBuf>` CLI field. Override Clap's default
`PathBufValueParser`, which rejects empty values before parsing completes, with
an `OsStringValueParser` mapped to `PathBuf`. Add a small normalization method on
`Cli` and call it immediately after Clap parses the process arguments. This keeps
non-UTF-8 path support and every downstream consumer consistent:
mutual-exclusion validation, checkpoint loading, start-superbatch selection,
experiment naming, and lineage all continue to use `None` for a fresh run
without adding local empty-path checks.

Update the `--resume` help text to state that an empty value means no resume.

## Error Handling

Only the exact empty path is suppressed. Missing, unreadable, or malformed
non-empty checkpoint paths continue to produce the existing errors.

## Tests

Add GPU-independent CLI regression tests that verify:

1. `--resume ""` becomes `None` after normalization.
2. A non-empty checkpoint path remains unchanged.
3. Empty `--resume` works when the global option is placed after the subcommand.
4. A whitespace-only checkpoint path remains unchanged.

Run the focused `nnue-trainer` CLI tests, then the repository's required local
CI before reporting completion.

## Non-goals

- Do not change empty-value handling for `--init-from` or other path options.
- Do not trim whitespace from paths.
- Do not alter checkpoint formats or resume semantics for non-empty paths.
