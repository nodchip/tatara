#!/usr/bin/env bash
# 本番 trainer と同じ native-cuda-host 経路の必須 CI。cuda-oxide の導入状態に
# 依存せず、CPU crate と NVCC-built kernel / portable host runtime を検査する。
# cuda-oxide の kernel build と backend parity は
# `scripts/cuda-oxide-parity-ci.sh` で任意に実行する。
#
# GPU test は `--release` で実行する: 数値同等性テストが debug build の f32 fma
# off で tolerance を満たさず fail するため (release は本番経路と同じ codegen)。
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== cargo fmt --all -- --check =="
cargo fmt --all -- --check

echo "== CPU workspace clippy =="
cargo clippy --workspace --all-targets \
  --exclude gpu-runtime \
  --exclude progress-kpabs-train \
  --exclude nnue-trainer \
  -- -D warnings

echo "== native CUDA clippy =="
cargo clippy -p cuda-native-runtime --all-targets --features native-cuda -- -D warnings
cargo clippy -p gpu-runtime --all-targets \
  --no-default-features --features native-cuda -- -D warnings
cargo clippy -p nnue-trainer --all-targets \
  --no-default-features --features native-cuda-host -- -D warnings

echo "== CPU workspace tests (release) =="
cargo test --workspace --release \
  --exclude gpu-runtime \
  --exclude progress-kpabs-train \
  --exclude nnue-trainer

echo "== native CUDA tests (release) =="
cargo test -p cuda-native-runtime --features native-cuda --release
cargo test -p gpu-runtime --no-default-features --features native-cuda --release
cargo test -p nnue-trainer \
  --no-default-features --features native-cuda-host --release

echo "PASS (native-cuda-host production CI)"
