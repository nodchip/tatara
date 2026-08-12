# 本番必須CIをnative-cuda-hostに限定する

## Status

Accepted

## Context

本番trainerは`--no-default-features --features native-cuda-host`でbuildし、NVCCが
生成したCUDA C++ fat binaryとportable Rust host runtimeを使う。この経路は
cuda-oxideのRust compiler codegen backendを使わない。

一方、従来の`local-ci.sh`は本番経路の検査に加えて、cuda-oxide backendのbuild、
Rust device kernel artifactの生成、native CUDA C++ backendとのparityまで必須に
していた。cuda-oxideはLinuxを対象とする実験的なcompiler backendであり、対応
toolchainを構築できないnative Windows環境では、本番binaryをbuild・testできても
CI全体が失敗する。

## Decision

- `scripts/local-ci.sh`をproduction acceptanceの必須CIとする。
- 必須CIはCPU共通crateと`native-cuda-host`構成をfmt、clippy、release testで検査する。
- 必須CIからcuda-oxide、LLVM device codegen、PTX artifact生成への依存を除く。
- cuda-oxide kernel build、workspace test、native backendとのparityは
  `scripts/cuda-oxide-parity-ci.sh`へ分離し、任意の補助CIとする。
- cuda-oxide featureとRust device kernel実装は比較用backendとして維持する。

## Consequences

native Windowsの本番変更は、cuda-oxideを構築できなくても本番と同じ経路で完結して
検証できる。cuda-oxide固有の退行やRust/CUDA C++ kernel間の乖離は必須CIでは検出
されないため、device kernelまたはbackend共通ABIを変更するときは、対応する
Linux / WSL環境で補助CIを実行する。
