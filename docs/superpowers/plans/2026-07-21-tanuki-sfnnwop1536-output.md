# Tanuki SFNNwoP1536 Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--output-format tanuki-sfnnwop1536`, produce the eight-stack progress-routed SFNN file consumed by the current `hakubishin-private`, and verify a bounded training export in that engine.

**Architecture:** Keep the existing Tatara and KingRank9 YaneuraOu formats intact. Refactor the SFNNWithoutPsqt writer around a private format profile, expose a dedicated Tanuki entry point, and validate semantic routing constraints at the CLI boundary before trainer allocation. Network dimensions remain variable; only the end-to-end check uses the current engine's 768-8-32 shape.

**Tech Stack:** Rust 2024, Clap derive, `nnue-format`, `nnue-train`, native CUDA/MSVC, PowerShell, YaneuraOu USI.

## Global Constraints

- Execute inline with `superpowers:executing-plans`; do not dispatch subagents.
- Do not change the bytes or accepted configurations of `tatara` or `yaneuraou` output.
- Tanuki output requires plain LayerStack, `halfka-hm-merged`, `progress8kpabs`, exactly 8 buckets, and an explicit `progress.bin`.
- Do not impose 768-8-32 as a format restriction; preserve normal LayerStack dimension validation.
- Allow the FT factorizer because export folds it into inference weights.
- Reject PSQT, threat, and effect-bucket extensions for Tanuki output.
- Do not modify or commit files in `C:\home\nodchip\hakubishin-private`.
- Do not push.

---

### Task 1: Add the reusable eight-stack Tanuki serializer

**Files:**
- Modify: `crates/nnue-format/src/yaneuraou.rs`
- Modify: `crates/nnue-format/src/lib.rs`
- Include: `docs/superpowers/plans/2026-07-21-tanuki-sfnnwop1536-output.md`

**Interfaces:**
- Consumes: `LayerStackWeights` with a plain `FeatureSet::HalfKaHmMerged` shape.
- Produces: `pub const TANUKI_SFNNWOP1536_LAYER_STACKS: usize = 8` and `pub fn save_tanuki_sfnnwop1536<W: Write>(writer: &mut W, weights: &LayerStackWeights) -> io::Result<()>`.
- Preserves: `save_yaneuraou` remains 9-stack KingRank9 only.

- [ ] **Step 1: Write failing serializer tests**

Add tests in `crates/nnue-format/src/yaneuraou.rs` that call the not-yet-defined public function:

```rust
#[test]
fn tanuki_profile_writes_expected_header_and_exactly_eight_networks() {
    use std::io::{Cursor, Read};

    let weights = LayerStackWeights::zeroed(
        FeatureSet::HalfKaHmMerged.spec(),
        128,
        4,
        3,
        TANUKI_SFNNWOP1536_LAYER_STACKS,
    );
    let mut bytes = Vec::new();
    save_tanuki_sfnnwop1536(&mut bytes, &weights).unwrap();

    let mut cursor = Cursor::new(bytes.as_slice());
    assert_eq!(read_u32_test(&mut cursor), YO_VERSION);
    assert_eq!(read_u32_test(&mut cursor), YO_TOP_HASH);
    let arch_len = read_u32_test(&mut cursor) as usize;
    let mut arch = vec![0; arch_len];
    cursor.read_exact(&mut arch).unwrap();
    assert_eq!(
        std::str::from_utf8(&arch).unwrap(),
        "Network trained with https://github.com/official-stockfish/nnue-pytorch"
    );
    assert_eq!(read_u32_test(&mut cursor), YO_FT_HASH);
    crate::layerstack_weights::read_leb128_tensor_i16(&mut cursor, Some(128)).unwrap();
    crate::layerstack_weights::read_leb128_tensor_i16(
        &mut cursor,
        Some(FeatureSet::HalfKaHmMerged.spec().ft_in() * 128),
    )
    .unwrap();

    let l1_out = 4usize;
    let l2_out = 3usize;
    let dense_bytes = l1_out * 4
        + l1_out * 128usize.div_ceil(32) * 32
        + l2_out * 4
        + l2_out * ((l1_out - 1) * 2).div_ceil(32) * 32
        + 4
        + l2_out.div_ceil(32) * 32;
    for _ in 0..TANUKI_SFNNWOP1536_LAYER_STACKS {
        assert_eq!(read_u32_test(&mut cursor), YO_NETWORK_HASH);
        cursor.set_position(cursor.position() + dense_bytes as u64);
    }
    assert_eq!(cursor.position() as usize, bytes.len());
}

#[test]
fn tanuki_profile_accepts_variable_valid_dimensions() {
    for (ft_out, l1_out, l2_out) in [(128, 2, 2), (256, 7, 16), (768, 8, 32)] {
        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            ft_out,
            l1_out,
            l2_out,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
        );
        save_tanuki_sfnnwop1536(&mut Vec::new(), &weights).unwrap();
    }
}

#[test]
fn tanuki_profile_rejects_wrong_feature_or_bucket_count() {
    let wrong_feature = LayerStackWeights::zeroed(
        FeatureSet::HalfKp.spec(),
        128,
        4,
        3,
        TANUKI_SFNNWOP1536_LAYER_STACKS,
    );
    assert!(
        save_tanuki_sfnnwop1536(&mut Vec::new(), &wrong_feature)
            .unwrap_err()
            .to_string()
            .contains("HalfKaHmMerged")
    );

    let wrong_buckets = LayerStackWeights::zeroed(
        FeatureSet::HalfKaHmMerged.spec(),
        128,
        4,
        3,
        YANEURAOU_LAYER_STACKS,
    );
    assert!(
        save_tanuki_sfnnwop1536(&mut Vec::new(), &wrong_buckets)
            .unwrap_err()
            .to_string()
            .contains("8 LayerStacks")
    );
}
```

Add this test helper inside the test module:

```rust
fn read_u32_test(reader: &mut impl std::io::Read) -> u32 {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
cargo test -p nnue-format yaneuraou::tests::tanuki_profile -- --nocapture
```

Expected: compilation fails because `save_tanuki_sfnnwop1536` and `TANUKI_SFNNWOP1536_LAYER_STACKS` do not exist.

- [ ] **Step 3: Implement the profile-based serializer**

In `crates/nnue-format/src/yaneuraou.rs`, introduce a private profile and route both public writers through one implementation:

```rust
pub const YANEURAOU_LAYER_STACKS: usize = 9;
pub const TANUKI_SFNNWOP1536_LAYER_STACKS: usize = 8;

#[derive(Clone, Copy)]
enum SfnnProfile {
    Yaneuraou,
    TanukiSfnnwoP1536,
}

impl SfnnProfile {
    fn layer_stacks(self) -> usize {
        match self {
            Self::Yaneuraou => YANEURAOU_LAYER_STACKS,
            Self::TanukiSfnnwoP1536 => TANUKI_SFNNWOP1536_LAYER_STACKS,
        }
    }

    fn architecture_string(self, arch: &Architecture) -> String {
        match self {
            Self::Yaneuraou => yaneuraou_arch_string(arch),
            Self::TanukiSfnnwoP1536 =>
                "Network trained with https://github.com/official-stockfish/nnue-pytorch"
                    .to_string(),
        }
    }
}

pub fn save_yaneuraou<W: Write>(writer: &mut W, weights: &LayerStackWeights) -> io::Result<()> {
    save_sfnn(writer, weights, SfnnProfile::Yaneuraou)
}

pub fn save_tanuki_sfnnwop1536<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
) -> io::Result<()> {
    save_sfnn(writer, weights, SfnnProfile::TanukiSfnnwoP1536)
}
```

Rename the current `arch_string` to `yaneuraou_arch_string`. Change `architecture`, `validate_weights`, the dense loop, and all shape-length checks to use `profile.layer_stacks()` instead of the YaneuraOu constant. Before writing the Tanuki profile, enforce:

```rust
if matches!(profile, SfnnProfile::TanukiSfnnwoP1536)
    && arch.feature_set != FeatureSet::HalfKaHmMerged
{
    return invalid_input("Tanuki SFNNwoP1536 requires HalfKaHmMerged features");
}
```

Export the new public items from `crates/nnue-format/src/lib.rs`:

```rust
pub use yaneuraou::{
    TANUKI_SFNNWOP1536_LAYER_STACKS, YANEURAOU_LAYER_STACKS,
    save_tanuki_sfnnwop1536, save_yaneuraou,
};
```

- [ ] **Step 4: Run serializer tests and existing format tests**

Run:

```powershell
cargo test -p nnue-format yaneuraou::tests -- --nocapture
cargo test -p nnue-format --lib
```

Expected: all tests pass, including existing 9-stack YaneuraOu tests.

- [ ] **Step 5: Commit the serializer**

```powershell
git add crates/nnue-format/src/yaneuraou.rs crates/nnue-format/src/lib.rs `
  docs/superpowers/plans/2026-07-21-tanuki-sfnnwop1536-output.md
git diff --cached --check
git commit -m "nnue_format: Tanuki SFNNwoP1536出力を追加"
```

---

### Task 2: Add CLI parsing and semantic validation

**Files:**
- Modify: `bins/nnue_train/src/cli.rs`
- Modify: `bins/nnue_train/src/training.rs`
- Modify: `bins/nnue_train/src/tests/cli_tests.rs`
- Modify: `crates/nnue-train/src/trainer.rs`

**Interfaces:**
- Consumes: `OutputFormatArg`, `LayerstackArgs`, `FeatureSetSpec`, and `BucketMode`.
- Produces: `OutputFormatArg::TanukiSfnnwoP1536`, `OutputFormat::TanukiSfnnwoP1536`, `validate_tanuki_output_config`, and `validate_plain_sfnn_extensions`.

- [ ] **Step 1: Write failing CLI and validation tests**

Add tests to `bins/nnue_train/src/tests/cli_tests.rs`:

```rust
fn validate_tanuki_argv(
    layerstack_args: &[&str],
    feature_set: FeatureSet,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut argv = vec![
        "nnue-train",
        "--output-format",
        "tanuki-sfnnwop1536",
        "--feature-set",
        feature_set.canonical_name(),
        "layerstack",
    ];
    argv.extend_from_slice(layerstack_args);
    let cli = Cli::try_parse_from(argv).expect("Tanuki argv should parse");
    let ArchCommand::LayerStack(args) = &cli.arch else {
        unreachable!("layerstack subcommand was requested")
    };
    validate_output_format(cli.output_format, validate_bucket_mode(args)?)?;
    validate_tanuki_output_config(&cli, args, feature_set.spec())
}

#[test]
fn tanuki_output_format_parses_and_simple_rejects_it() {
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--output-format",
        "tanuki-sfnnwop1536",
        "layerstack",
    ])
    .expect("Tanuki LayerStack output should parse");
    assert_eq!(cli.output_format, OutputFormatArg::TanukiSfnnwoP1536);

    let simple = simple_cli(&["--output-format", "tanuki-sfnnwop1536"]);
    let error = reject_simple_unsupported_flags(&simple).unwrap_err();
    assert!(error.to_string().contains("only with the layerstack"));
}

#[test]
fn tanuki_output_requires_progress8kpabs_eight_buckets_and_coefficients() {
    validate_tanuki_argv(
        &[
        "--bucket-mode", "progress8kpabs",
        "--num-buckets", "8",
        "--progress-coeff", "progress.bin",
        ],
        FeatureSet::HalfKaHmMerged,
    )
    .expect("valid Tanuki output config");

    for args in [
        vec!["--bucket-mode", "kingrank9", "--num-buckets", "9", "--progress-coeff", "progress.bin"],
        vec!["--bucket-mode", "progress8kpabs", "--num-buckets", "9", "--progress-coeff", "progress.bin"],
        vec!["--bucket-mode", "progress8kpabs", "--num-buckets", "8"],
    ] {
        let error = validate_tanuki_argv(&args, FeatureSet::HalfKaHmMerged).unwrap_err();
        assert!(!error.to_string().is_empty());
    }

    let error = validate_tanuki_argv(
        &[
            "--bucket-mode", "progress8kpabs",
            "--num-buckets", "8",
            "--progress-coeff", "progress.bin",
        ],
        FeatureSet::HalfKp,
    )
    .unwrap_err();
    assert!(error.to_string().contains("halfka-hm-merged"));
}

#[test]
fn sfnn_outputs_reject_non_plain_layerstack_extensions() {
    for output_format in [
        OutputFormatArg::Yaneuraou,
        OutputFormatArg::TanukiSfnnwoP1536,
    ] {
        for (psqt, threat, effect) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let error = validate_plain_sfnn_extensions(
                output_format,
                psqt,
                threat,
                effect,
            )
            .unwrap_err();
            assert!(error.to_string().contains("plain LayerStack"));
        }
    }
}
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
cargo test -p nnue-trainer --no-default-features --features native-cuda-host `
  --release tests::cli_tests::tanuki_output -- --nocapture
```

Expected: compilation fails because the Tanuki variants and validation function do not exist.

- [ ] **Step 3: Implement CLI and validation**

Add the explicitly named Clap value:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum OutputFormatArg {
    #[default]
    Tatara,
    Yaneuraou,
    #[value(name = "tanuki-sfnnwop1536")]
    TanukiSfnnwoP1536,
}
```

Map it to the new library enum and add `OutputFormat::TanukiSfnnwoP1536` in `crates/nnue-train/src/trainer.rs`.

Extend `validate_output_format` without weakening the existing YaneuraOu rule:

```rust
match output_format {
    OutputFormatArg::Yaneuraou if !matches!(bucket_mode, BucketMode::KingRank9) => Err(
        "--output-format yaneuraou requires LayerStack --bucket-mode kingrank9; progress8kpabs routing is not representable in YaneuraOu SFNN".into(),
    ),
    OutputFormatArg::TanukiSfnnwoP1536
        if !matches!(bucket_mode, BucketMode::Progress8KpAbs) =>
    {
        Err("--output-format tanuki-sfnnwop1536 requires LayerStack --bucket-mode progress8kpabs".into())
    }
    _ => Ok(()),
}
```

Add a validation function in `bins/nnue_train/src/training.rs`:

```rust
pub(crate) fn validate_tanuki_output_config(
    cli: &Cli,
    args: &LayerstackArgs,
    feature_set: FeatureSetSpec,
) -> Result<(), Box<dyn std::error::Error>> {
    if cli.output_format != OutputFormatArg::TanukiSfnnwoP1536 {
        return Ok(());
    }
    if feature_set != FeatureSet::HalfKaHmMerged.spec() {
        return Err("--output-format tanuki-sfnnwop1536 requires --feature-set halfka-hm-merged".into());
    }
    if args.bucket_mode != "progress8kpabs" {
        return Err("--output-format tanuki-sfnnwop1536 requires --bucket-mode progress8kpabs".into());
    }
    if args.num_buckets != nnue_format::TANUKI_SFNNWOP1536_LAYER_STACKS {
        return Err("--output-format tanuki-sfnnwop1536 requires --num-buckets 8".into());
    }
    if args.progress_coeff.is_none() {
        return Err("--output-format tanuki-sfnnwop1536 requires --progress-coeff <progress.bin>".into());
    }
    Ok(())
}
```

Call it immediately after shared feature-set and bucket-mode resolution. Generalize the existing plain-SFNN extension check to both non-Tatara outputs:

```rust
pub(crate) fn validate_plain_sfnn_extensions(
    output_format: OutputFormatArg,
    psqt: bool,
    threat: bool,
    effect_bucket: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if output_format != OutputFormatArg::Tatara && (psqt || threat || effect_bucket) {
        return Err(format!(
            "--output-format {} supports plain LayerStack only; PSQT, threat-profile, and effect-bucket models are not representable",
            output_format.as_str()
        )
        .into());
    }
    Ok(())
}
```

Add an exhaustive `OutputFormatArg::as_str()` method returning `tatara`, `yaneuraou`, or `tanuki-sfnnwop1536`. Call `validate_plain_sfnn_extensions` with `layerstack.psqt`, `threat_profile.is_some()`, and `effect_bucket_config.is_some()`. Update `reject_simple_unsupported_flags` to reject both SFNN formats with `cli.output_format.as_str()` in the message.

- [ ] **Step 4: Run all CLI tests**

Run:

```powershell
cargo test -p nnue-trainer --no-default-features --features native-cuda-host `
  --release tests::cli_tests -- --nocapture
```

Expected: all CLI tests pass; existing YaneuraOu validation remains green.

- [ ] **Step 5: Commit CLI and validation**

```powershell
git add bins/nnue_train/src/cli.rs bins/nnue_train/src/training.rs `
  bins/nnue_train/src/tests/cli_tests.rs crates/nnue-train/src/trainer.rs
git diff --cached --check
git commit -m "nnue_train: Tanuki出力のCLI制約を追加"
```

---

### Task 3: Wire checkpoint export and document the training command

**Files:**
- Modify: `bins/nnue_train/src/trainer_common.rs`
- Modify: `docs/net-to-yaneuraou.md`
- Modify: `docs/training-quickstart.ja.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `OutputFormat::TanukiSfnnwoP1536` and `save_tanuki_sfnnwop1536`.
- Produces: inference `.bin` files directly loadable by a matching Tanuki/Hakubishin engine build.

- [ ] **Step 1: Add the failing export-dispatch expectation**

Update exhaustive matches in `trainer_common.rs` only after compiling once to confirm the new enum produces the expected non-exhaustive-match failure. The LayerStack arm must call the dedicated writer; the Simple arm must return a LayerStack-only error.

- [ ] **Step 2: Run `cargo check` and verify RED**

Run:

```powershell
cargo check -p nnue-trainer --no-default-features --features native-cuda-host
```

Expected: non-exhaustive match errors identify `OutputFormat::TanukiSfnnwoP1536` in `trainer_common.rs`.

- [ ] **Step 3: Wire the save dispatch**

Implement the LayerStack match:

```rust
nnue_train::trainer::OutputFormat::TanukiSfnnwoP1536 => {
    nnue_format::save_tanuki_sfnnwop1536(writer, self)
}
```

Implement the Simple rejection:

```rust
nnue_train::trainer::OutputFormat::TanukiSfnnwoP1536 => Err(std::io::Error::new(
    std::io::ErrorKind::InvalidInput,
    "--output-format tanuki-sfnnwop1536 is supported only for the LayerStack architecture",
)),
```

- [ ] **Step 4: Document the exact compatibility contract**

Add this command shape to `docs/net-to-yaneuraou.md` and link it from the quickstart and README:

```powershell
target\release\nnue-train.exe `
  --data D:\training_data\shuffled.bin `
  --output D:\nnue\tanuki-run --net-id tanuki-run `
  --output-format tanuki-sfnnwop1536 `
  --feature-set halfka-hm-merged `
  --superbatches 400 --threads 16 `
  layerstack `
  --bucket-mode progress8kpabs --num-buckets 8 `
  --progress-coeff C:\home\nodchip\hakubishin-private\source\progress.bin `
  --ft-out 768 --l1 8 --l2 32
```

State that the dimensions are examples matching the current default engine, not format restrictions; custom values require a matching engine build. State that the same `progress.bin` and an explicit engine `FV_SCALE` value are required.

- [ ] **Step 5: Run format, compile, and targeted tests**

Run:

```powershell
cargo fmt --all -- --check
cargo check -p nnue-trainer --no-default-features --features native-cuda-host
cargo test -p nnue-format --lib
cargo test -p nnue-trainer --no-default-features --features native-cuda-host `
  --release tests::cli_tests -- --nocapture
```

Expected: every command exits 0.

- [ ] **Step 6: Commit dispatch and documentation**

```powershell
git add bins/nnue_train/src/trainer_common.rs docs/net-to-yaneuraou.md `
  docs/training-quickstart.ja.md README.md
git diff --cached --check
git commit -m "docs: Tanuki互換評価関数の学習手順を追加"
```

---

### Task 4: Verify bounded training and real Hakubishin loading

**Files:**
- Verify only: `crates/shogi-format/tests/data/sample.psv`
- Verify only: `C:\home\nodchip\hakubishin-private\source\progress.bin`
- Verify only: `C:\home\nodchip\hakubishin-private\build\NNUE\YaneuraOu-NNUE.exe`
- Temporary output only: `C:\tmp\tatara-tanuki-sfnnwop1536-smoke`

**Interfaces:**
- Consumes: the new CLI format, the 100-record PSV fixture, the current progress coefficients, and current engine binary.
- Produces: evidence that a trained 768-8-32 checkpoint reaches YaneuraOu `readyok` without loader errors.

- [ ] **Step 1: Confirm fixture and engine prerequisites**

Run:

```powershell
(Get-Item crates/shogi-format/tests/data/sample.psv).Length % 40
(Get-Item C:\home\nodchip\hakubishin-private\source\progress.bin).Length
Test-Path C:\home\nodchip\hakubishin-private\build\NNUE\YaneuraOu-NNUE.exe
```

Expected: `0`, `1003104`, and `True`.

- [ ] **Step 2: Run one bounded training update**

From a Visual Studio Developer environment, run:

```powershell
cargo run -p nnue-trainer --no-default-features --features native-cuda-host `
  --release -- `
  --data crates/shogi-format/tests/data/sample.psv `
  --output C:\tmp\tatara-tanuki-sfnnwop1536-smoke\checkpoints `
  --net-id smoke `
  --output-format tanuki-sfnnwop1536 `
  --feature-set halfka-hm-merged `
  --batch-size 16 --batches-per-superbatch 1 --superbatches 1 `
  --save-rate 1 --threads 1 --optimizer adamw --no-ft-factorize `
  layerstack `
  --bucket-mode progress8kpabs --num-buckets 8 `
  --progress-coeff C:\home\nodchip\hakubishin-private\source\progress.bin `
  --ft-out 768 --l1 8 --l2 32
```

Expected: one batch trains, and `smoke-1.bin` plus the resume checkpoint are written. The run must not allocate or process more than one 16-position batch.

- [ ] **Step 3: Load the generated file in the current engine**

Copy the engine and generated `.bin` under the temporary root so no file in `hakubishin-private` is modified:

```powershell
$smokeRoot = 'C:\tmp\tatara-tanuki-sfnnwop1536-smoke'
New-Item -ItemType Directory -Force "$smokeRoot\engine\eval" | Out-Null
Copy-Item 'C:\home\nodchip\hakubishin-private\build\NNUE\YaneuraOu-NNUE.exe' `
  "$smokeRoot\engine\YaneuraOu-NNUE.exe"
Copy-Item "$smokeRoot\checkpoints\smoke-1.bin" "$smokeRoot\engine\eval\nn.bin"
@(
  'usi'
  'setoption name EvalDir value eval'
  'setoption name ProgressFilePath value C:\home\nodchip\hakubishin-private\source\progress.bin'
  'setoption name FV_SCALE value 24'
  'isready'
  'quit'
) -join "`n" | & "$smokeRoot\engine\YaneuraOu-NNUE.exe"
```

Expected output includes both evaluation/progress load messages and `readyok`; it contains no `NNUE header version mismatch`, `NNUE hash mismatch`, `failed to read`, or `Error!`.

- [ ] **Step 4: Run repository verification**

Run:

```powershell
cargo fmt --all -- --check
cargo test -p nnue-format --lib
cargo test -p nnue-trainer --no-default-features --features native-cuda-host `
  --release tests::cli_tests -- --nocapture
bash scripts/local-ci.sh
git diff --check
git status --short --branch
```

Expected: targeted commands pass. `local-ci.sh` should print `PASS`; if the known local `cargo-oxide` prerequisite is still absent, record the exact environmental stop after confirming fmt, workspace clippy, and native CUDA compile coverage separately. Do not push.

- [ ] **Step 5: Clean temporary artifacts safely**

Resolve the path and verify it is strictly under `C:\tmp` before removal:

```powershell
$smokeRoot = (Resolve-Path 'C:\tmp\tatara-tanuki-sfnnwop1536-smoke').Path
if (-not $smokeRoot.StartsWith('C:\tmp\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "refusing to remove unexpected path: $smokeRoot"
}
Remove-Item -LiteralPath $smokeRoot -Recurse -Force
```

- [ ] **Step 6: Inspect final history and report the production command**

Run:

```powershell
git status --short --branch
git log -5 --oneline
```

Report the exact commits, verification outcomes, output filename pattern, matching-dimension requirement, shared `progress.bin` requirement, and engine-side `FV_SCALE` setting. Do not push.
