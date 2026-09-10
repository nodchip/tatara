# LayerStack の明示的な共有特徴による追加学習

**Status**: Proposed

**Date**: 2026-09-10

量子化済みの推論用ファイルには、学習中の FT factorizer の仮想行が保存されない。
追加学習でこの共有成分を使う場合、元の分解を復元する必要はない。読み込んだ実行を
保持し、仮想行をゼロにすれば、初期の評価関数を保ったまま共有成分を学習できる。
この変換は Stockfish の
[feature conversion](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/features.html)
と同じ初期化原理を用いる。

既存の追加学習との比較を維持するため、`--init-from` では共有特徴を既定で無効にする。
LayerStack で `--ft-factorize` を明示したときだけ、読み込み後に仮想行をゼロで拡張する。
Simple の追加学習、raw checkpoint の形状一致条件、推論用ファイルの形式は変更しない。

入力の feature set と全層の次元は GPU buffer を変更する前に照合する。FT と任意の
PSQT の重み、moment、勾配、lookahead は学習用の全行数で確保する。moment と勾配は
ゼロ、lookahead は読み込んだ重みと同値にする。forward 用の畳み込み済み buffer も
読み込み直後に同期する。出力時には既存の畳み込み処理で仮想行を実行へ加える。

この選択により、学習前の量子化出力を保存したまま、玉位置をまたぐ勾配共有を検証できる。
共有成分が学習を改善する保証はなく、追加学習そのものの効果との区別には同条件の対照が
必要である。GPU の回帰検査では、初期出力、export bytes、optimizer state、仮想行の更新、
不適合入力に対する無変更の拒否を確認する。
