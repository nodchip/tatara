# Empty `--resume` Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an exactly empty `--resume` value behave like an omitted option while preserving every non-empty checkpoint path.

**Architecture:** Keep `Cli::resume` as `Option<PathBuf>`, but override Clap's empty-rejecting `PathBufValueParser` with `OsStringValueParser` mapped to `PathBuf`. Normalize the parsed field once before command dispatch so downstream validation, checkpoint loading, experiment naming, and lineage continue to consume the existing field without scattered empty-path checks.

**Tech Stack:** Rust 2024, Clap derive, Cargo unit tests, PowerShell/Visual Studio native CUDA build environment.

## Global Constraints

- Treat only the exact empty value as absent; do not trim whitespace.
- Preserve existing behavior and errors for every non-empty checkpoint path.
- Do not change `--init-from` or any other path option.
- Keep the existing `Option<PathBuf>` field and downstream resume logic.
- Execute inline with `superpowers:executing-plans`; do not dispatch subagents.

---

### Task 1: Normalize empty `--resume` values at the CLI boundary

**Files:**
- Modify: `bins/nnue_train/src/cli.rs:243-252`
- Modify: `bins/nnue_train/src/main.rs:55-80`
- Test: `bins/nnue_train/src/tests/cli_tests.rs`
- Include: `docs/superpowers/plans/2026-07-21-empty-resume.md`
- Update: `docs/superpowers/specs/2026-07-21-empty-resume-design.md`

**Interfaces:**
- Consumes: `Cli::resume: Option<PathBuf>` populated by Clap.
- Parser boundary: `OsStringValueParser` accepts empty and non-UTF-8 OS strings, then maps them to `PathBuf`.
- Produces: `Cli::normalize_empty_resume(&mut self)`, which changes only `Some(path)` where `path.as_os_str().is_empty()` to `None`.
- Runtime contract: `main` calls the method exactly once, immediately after `Cli::parse()` and before matching on `cli.arch`.

- [ ] **Step 1: Write the failing CLI regression tests**

Add the following GPU-independent tests to `bins/nnue_train/src/tests/cli_tests.rs`:

```rust
#[test]
fn empty_resume_is_normalized_to_absent_before_or_after_subcommand() {
    for argv in [
        vec!["nnue-train", "--resume", "", "layerstack"],
        vec!["nnue-train", "layerstack", "--resume", ""],
    ] {
        let mut cli = Cli::try_parse_from(argv).expect("empty resume should parse");
        cli.normalize_empty_resume();
        assert!(cli.resume.is_none());
    }
}

#[test]
fn non_empty_resume_path_is_preserved() {
    let mut cli = Cli::try_parse_from([
        "nnue-train",
        "--resume",
        "checkpoints/run-20.ckpt",
        "layerstack",
    ])
    .expect("non-empty resume should parse");

    cli.normalize_empty_resume();

    assert_eq!(
        cli.resume.as_deref(),
        Some(std::path::Path::new("checkpoints/run-20.ckpt"))
    );
}

#[test]
fn whitespace_only_resume_path_is_preserved() {
    let mut cli = Cli::try_parse_from(["nnue-train", "--resume", " ", "layerstack"])
        .expect("whitespace resume should parse");

    cli.normalize_empty_resume();

    assert_eq!(cli.resume.as_deref(), Some(std::path::Path::new(" ")));
}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run from a Visual Studio Developer environment:

```powershell
cargo test -p nnue-trainer --no-default-features --features native-cuda-host `
  --release tests::cli_tests::empty_resume -- --nocapture
```

Expected: compilation fails because `Cli::normalize_empty_resume` does not yet exist. Confirm that the failure names that missing method rather than a test syntax or environment problem.

- [ ] **Step 3: Implement the minimal normalization and runtime call**

In `bins/nnue_train/src/cli.rs`, import the parser types, configure `--resume` to accept an empty OS string, extend the help text, and add the normalization method:

```rust
use clap::{
    Args, Parser, Subcommand,
    builder::{OsStringValueParser, TypedValueParser},
};

#[arg(
    long,
    global = true,
    value_parser = OsStringValueParser::new().map(PathBuf::from)
)]
pub(crate) resume: Option<PathBuf>,
```

```rust
/// Normalize compatibility values that represent an omitted CLI option.
pub(crate) fn normalize_empty_resume(&mut self) {
    if self
        .resume
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        self.resume = None;
    }
}
```

The help text must include this user-visible sentence:

```text
An empty value is treated as if `--resume` was omitted.
```

In `bins/nnue_train/src/main.rs`, normalize immediately after parsing:

```rust
let mut cli = Cli::parse();
cli.normalize_empty_resume();
```

- [ ] **Step 4: Run all CLI tests and verify GREEN**

Run:

```powershell
cargo test -p nnue-trainer --no-default-features --features native-cuda-host `
  --release tests::cli_tests -- --nocapture
```

Expected: all `tests::cli_tests` tests pass, including the empty and non-empty resume cases.

- [ ] **Step 5: Verify user-visible help and the real empty-resume launch path**

Run:

```powershell
cargo run -p nnue-trainer --no-default-features --features native-cuda-host `
  --release -- --help
cargo run -p nnue-trainer --no-default-features --features native-cuda-host `
  --release -- simple --resume ""
```

Expected: help contains `An empty value is treated as if --resume was omitted`; the second command runs the fresh-run Simple GPU smoke path and ends with `[smoke/simple] PASSED` instead of trying to open an empty checkpoint path.

- [ ] **Step 6: Run repository verification and inspect the complete diff**

Run:

```powershell
bash scripts/local-ci.sh
git diff --check
git diff -- bins/nnue_train/src/cli.rs bins/nnue_train/src/main.rs `
  bins/nnue_train/src/tests/cli_tests.rs docs/superpowers/plans/2026-07-21-empty-resume.md `
  docs/superpowers/specs/2026-07-21-empty-resume-design.md
git status --short
```

Expected: local CI exits 0 with `PASS`; `git diff --check` is silent; only the intended CLI, entrypoint, test, plan, and design files are changed.

- [ ] **Step 7: Commit the verified implementation**

Stage only the task files, verify the staged scope, and commit:

```powershell
git add bins/nnue_train/src/cli.rs bins/nnue_train/src/main.rs `
  bins/nnue_train/src/tests/cli_tests.rs docs/superpowers/plans/2026-07-21-empty-resume.md `
  docs/superpowers/specs/2026-07-21-empty-resume-design.md
git diff --cached --name-only
git diff --cached --check
git commit -m "nnue_train: 空のresume引数を未指定として扱う"
```

Expected staged files are exactly the five paths listed above. Do not push.
