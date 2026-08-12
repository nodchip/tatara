#!/usr/bin/env bash
# cuda-oxide backend の補助検査。これは本番受入条件ではなく、cuda-oxide を
# セットアップした Linux / WSL 環境で Rust device kernel と native CUDA C++
# kernel の同期、数値同等性、cuda-oxide 側の workspace test を確認する。
set -euo pipefail

cd "$(dirname "$0")/.."

# CUDA_OXIDE_TARGET は build-kernels.sh の GPU auto-detect に任せる。特定 target
# を試す場合だけ呼び出し側で設定する。
: "${LLVM_LINK_BIN:=/usr/bin/llvm-link-22}"
: "${OPT_BIN:=/usr/bin/opt-22}"
: "${LLC_BIN:=/usr/bin/llc-22}"
export LLVM_LINK_BIN OPT_BIN LLC_BIN

echo "== cuda-oxide workspace clippy =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== bash scripts/build-kernels.sh (cuda-oxide artifacts) =="
bash scripts/build-kernels.sh

echo "== bash scripts/check-native-cuda-parity.sh =="
bash scripts/check-native-cuda-parity.sh

echo "== cuda-oxide workspace tests (release) =="
cargo test --workspace --release

echo "PASS (optional cuda-oxide parity CI)"
