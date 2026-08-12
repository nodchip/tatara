# Tanuki LayerStack 配布ネットワークの pretrained import

- Status: Accepted
- Date: 2026-08-12

## Context

進行度 8 分割の LayerStack 学習は、推論エンジンへ渡す Tanuki SFNNwoP1536 形式を
直接出力できる。この形式は量子化済み weight を持つ一方、Tatara の内部 checkpoint
とは header と最終層の量子化 scale が異なる。そのため、配布済みの最良ネットワークを
起点に weight-only の追加学習を行うと、`--init-from` が内部 checkpoint の version
不一致として reject していた。

外部形式には optimizer state、lookahead state、global step、LR schedule、共有 L1
factorizer が含まれない。層次元と bucket 数も自己記述されず、architecture string は
固定文字列である。

## Decision

LayerStack の `--init-from` は file 先頭の version と top hash が Tanuki
SFNNwoP1536 の固定値に一致した場合、その外部形式として読み込む。それ以外は Tatara
内部 checkpoint の loader に渡す。外部 loader は次を要求・検証する。

- feature set は `HalfKaHmMerged`、bucket 数は 8
- caller が指定した FT/L1/L2 次元と正の有限 `--scale`
- 固定 header、architecture string、FT hash、各 LayerStack の network hash
- affine weight の 32-byte padding が全て 0 であること、および file 終端

量子化値は出力時と同じ scale で f32 に戻す。共有 L1 factorizer は配布時に各 bucket
の L1 へ fold 済みなので、import 後は factorizer buffer を 0 とする。これにより同じ
scale で再出力した byte 列は入力と一致する。

## Consequences

- 配布済み Tanuki SFNNwoP1536 ネットワークを直接 pretrained start に利用できる。
- `--init-from` の既存契約どおり optimizer、lookahead、global step、LR schedule は
  reset される。これらを継承する場合は raw `.ckpt` と `--resume` を使う。
- 外部形式にない次元は CLI 契約が担うため、出力時と異なる層次元や scale の指定は
  header だけでは検出できない。誤った次元は構造検証または終端検証で reject されるが、
  同じ byte 長になる別構成を避ける責任は caller に残る。
