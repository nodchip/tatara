//! YaneuraOu SFNNWithoutPsqt evaluation-file serialization.

use std::io::{self, Read, Write};

use shogi_features::{FeatureSet, FeatureSetSpec};

use crate::LayerStackWeights;
use crate::layerstack_weights::{QA, QB, read_leb128_tensor_i16, write_leb128_tensor_i16};

const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;
const TANUKI_ARCHITECTURE: &str =
    "Network trained with https://github.com/official-stockfish/nnue-pytorch";

/// YaneuraOu SFNN が要求する KingRank9 LayerStack 数。
pub const YANEURAOU_LAYER_STACKS: usize = 9;
/// Tanuki SFNNwoP1536 が要求する進行度別 LayerStack 数。
pub const TANUKI_SFNNWOP1536_LAYER_STACKS: usize = 8;

const MAX_FT_OUT: usize = 8192;
const MAX_HIDDEN_DIM: usize = 4096;

struct YoFeature {
    feature_set: FeatureSet,
    yo_name: &'static str,
    gen_key: &'static str,
}

const YO_FEATURES: [YoFeature; 5] = [
    YoFeature {
        feature_set: FeatureSet::HalfKp,
        yo_name: "HalfKP",
        gen_key: "halfkp",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaSplit,
        yo_name: "HalfKA1",
        gen_key: "halfka1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaMerged,
        yo_name: "HalfKA2",
        gen_key: "halfka2",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmSplit,
        yo_name: "HalfKA_hm1",
        gen_key: "halfkahm1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmMerged,
        yo_name: "HalfKA_hm2",
        gen_key: "halfkahm2",
    },
];

/// LayerStack weights を YaneuraOu SFNNWithoutPsqt 形式で書き出す。
///
/// feature set と各層次元は weights の shape から決定する。YaneuraOu SFNN が
/// 表現できない拡張 feature、PSQT、KingRank9 以外の bucket 数は reject する。
/// bucket routing mode 自体は weights に含まれないため、caller は学習 config 等から
/// KingRank9 であることを確認してから呼ぶ必要がある。
pub fn save_yaneuraou<W: Write>(writer: &mut W, weights: &LayerStackWeights) -> io::Result<()> {
    save_sfnn(writer, weights, SfnnProfile::Yaneuraou)
}

/// LayerStack weights を Tanuki SFNNwoP1536 互換形式で書き出す。
///
/// 現行 Tanuki/Hakubishin 構成に合わせて HalfKA_hm merged feature と 8 個の
/// LayerStack を要求する。各層の次元はファイル形式では固定しない。`eval_scale` は
/// 学習時の score scale で、nnue-pytorch 互換の最終層量子化に使用する。
pub fn save_tanuki_sfnnwop1536<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
    eval_scale: f32,
) -> io::Result<()> {
    save_sfnn(
        writer,
        weights,
        SfnnProfile::TanukiSfnnwoP1536 {
            eval_scale: f64::from(eval_scale),
        },
    )
}

/// `bytes` が Tanuki SFNNwoP1536 evaluation file の固定 header で始まるかを返す。
///
/// これは format dispatch 用の軽量判定であり、file 全体の検証は
/// [`load_tanuki_sfnnwop1536`] が行う。
pub fn is_tanuki_sfnnwop1536_header(bytes: &[u8]) -> bool {
    bytes.get(..4) == Some(YO_VERSION.to_le_bytes().as_slice())
        && bytes.get(4..8) == Some(YO_TOP_HASH.to_le_bytes().as_slice())
}

/// Tanuki SFNNwoP1536 evaluation file を学習用 LayerStack weights として読み込む。
///
/// 外部形式は層次元や bucket 数を自己記述しないため、caller が要求する値を渡す。
/// header、architecture、各 layer hash、padding、終端を検証し、契約と異なる file は
/// `InvalidData` で reject する。外部形式では共有 L1 factorizer が各 bucket の L1 に
/// fold 済みなので、読み込み後の `l1f_w` / `l1f_b` は 0 になる。
pub fn load_tanuki_sfnnwop1536<R: Read>(
    reader: &mut R,
    expected: FeatureSetSpec,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
    eval_scale: f32,
) -> io::Result<LayerStackWeights> {
    validate_tanuki_import_config(expected, ft_out, l1_out, l2_out, num_buckets, eval_scale)?;

    require_u32(reader, YO_VERSION, "version")?;
    require_u32(reader, YO_TOP_HASH, "top hash")?;
    let arch_len = read_u32(reader)? as usize;
    if arch_len > 4096 {
        return invalid_data(format!(
            "Tanuki SFNNwoP1536 architecture string is too long: {arch_len} bytes"
        ));
    }
    let mut arch = vec![0; arch_len];
    reader.read_exact(&mut arch)?;
    if arch != TANUKI_ARCHITECTURE.as_bytes() {
        return invalid_data("Tanuki SFNNwoP1536 architecture string differs");
    }
    require_u32(reader, YO_FT_HASH, "feature transformer hash")?;

    let mut weights = LayerStackWeights::zeroed(
        expected,
        ft_out,
        l1_out,
        l2_out,
        TANUKI_SFNNWOP1536_LAYER_STACKS,
    );
    weights.ft_b = dequantize_i16(read_leb128_tensor_i16(reader, Some(ft_out))?, f64::from(QA));
    weights.ft_w = dequantize_i16(
        read_leb128_tensor_i16(reader, Some(expected.ft_in() * ft_out))?,
        f64::from(QA),
    );

    let l2_in = (l1_out - 1) * 2;
    let (output_bias_scale, output_weight_scale) = SfnnProfile::TanukiSfnnwoP1536 {
        eval_scale: f64::from(eval_scale),
    }
    .output_quantisation_scales();
    for bucket in 0..TANUKI_SFNNWOP1536_LAYER_STACKS {
        require_u32(reader, YO_NETWORK_HASH, "network hash")?;

        let (biases, dense) =
            read_affine(reader, ft_out, l1_out, f64::from(QA * QB), f64::from(QB))?;
        weights.l1_b[bucket * l1_out..(bucket + 1) * l1_out].copy_from_slice(&biases);
        weights.l1_w[bucket * l1_out * ft_out..(bucket + 1) * l1_out * ft_out]
            .copy_from_slice(&dense);

        let (biases, dense) =
            read_affine(reader, l2_in, l2_out, f64::from(QA * QB), f64::from(QB))?;
        weights.l2_b[bucket * l2_out..(bucket + 1) * l2_out].copy_from_slice(&biases);
        weights.l2_w[bucket * l2_out * l2_in..(bucket + 1) * l2_out * l2_in]
            .copy_from_slice(&dense);

        let (biases, dense) =
            read_affine(reader, l2_out, 1, output_bias_scale, output_weight_scale)?;
        weights.l3_b[bucket] = biases[0];
        weights.l3_w[bucket * l2_out..(bucket + 1) * l2_out].copy_from_slice(&dense);
    }

    let mut trailing = [0];
    if reader.read(&mut trailing)? != 0 {
        return invalid_data("Tanuki SFNNwoP1536 file has trailing data");
    }
    Ok(weights)
}

#[derive(Clone, Copy)]
enum SfnnProfile {
    Yaneuraou,
    TanukiSfnnwoP1536 { eval_scale: f64 },
}

impl SfnnProfile {
    fn layer_stacks(self) -> usize {
        match self {
            Self::Yaneuraou => YANEURAOU_LAYER_STACKS,
            Self::TanukiSfnnwoP1536 { .. } => TANUKI_SFNNWOP1536_LAYER_STACKS,
        }
    }

    fn architecture_string(self, arch: &Architecture) -> String {
        match self {
            Self::Yaneuraou => yaneuraou_arch_string(arch),
            Self::TanukiSfnnwoP1536 { .. } => TANUKI_ARCHITECTURE.to_string(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Yaneuraou => "YaneuraOu SFNN",
            Self::TanukiSfnnwoP1536 { .. } => "Tanuki SFNNwoP1536",
        }
    }

    fn output_quantisation_scales(self) -> (f64, f64) {
        match self {
            Self::Yaneuraou => (f64::from(QA * QB), f64::from(QB)),
            Self::TanukiSfnnwoP1536 { eval_scale } => {
                let bias_scale = eval_scale * 16.0;
                (bias_scale, bias_scale / f64::from(QA))
            }
        }
    }
}

fn save_sfnn<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
    profile: SfnnProfile,
) -> io::Result<()> {
    let arch = architecture(weights, profile)?;
    validate_weights(&arch, weights, profile)?;

    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;

    write_u32(writer, YO_VERSION)?;
    write_u32(writer, YO_TOP_HASH)?;
    let arch_string = profile.architecture_string(&arch);
    write_u32(
        writer,
        u32::try_from(arch_string.len()).expect("architecture string length fits in u32"),
    )?;
    writer.write_all(arch_string.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_b, QA as f64))?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_w, QA as f64))?;

    for bucket in 0..profile.layer_stacks() {
        write_u32(writer, YO_NETWORK_HASH)?;

        // factorizer 共有項は通常 export 前に L1 へ fold 済み。未 fold の weights を
        // caller が渡した場合にも同じ推論 weight になるよう加算する。
        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1f_b[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out + output * ft_out + input]
                    + weights.l1f_w[input * l1_out + output]
            })
        });
        write_affine(writer, l1_biases, l1_weights, ft_out, l1_out)?;

        let l2_biases = (0..l2_out).map(|output| weights.l2_b[bucket * l2_out + output]);
        let l2_weights = (0..l2_out).flat_map(|output| {
            (0..l2_in)
                .map(move |input| weights.l2_w[bucket * l2_out * l2_in + output * l2_in + input])
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, l2_out)?;

        let (output_bias_scale, output_weight_scale) = profile.output_quantisation_scales();
        write_affine_scaled(
            writer,
            std::iter::once(weights.l3_b[bucket]),
            (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]),
            l2_out,
            1,
            output_bias_scale,
            output_weight_scale,
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct Architecture {
    feature_set: FeatureSet,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
}

fn architecture(weights: &LayerStackWeights, profile: SfnnProfile) -> io::Result<Architecture> {
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|feature_set| feature_set.spec() == weights.feature_set)
        .ok_or_else(|| invalid_input_err("feature set is not representable in YaneuraOu SFNN"))?;
    let ft_out = weights.ft_b.len();
    let l1_out = weights.l1f_b.len();
    let num_buckets = weights.num_buckets;
    let layer_stacks = profile.layer_stacks();
    if num_buckets != layer_stacks {
        return invalid_input(format!(
            "{} requires {layer_stacks} LayerStacks{}, but weights have {num_buckets} buckets",
            profile.name(),
            if matches!(profile, SfnnProfile::Yaneuraou) {
                " (KingRank9)"
            } else {
                ""
            }
        ));
    }
    let l2_out = weights.l2_b.len().checked_div(num_buckets).unwrap_or(0);
    Ok(Architecture {
        feature_set,
        ft_out,
        l1_out,
        l2_out,
    })
}

fn yaneuraou_arch_string(arch: &Architecture) -> String {
    let feature = YO_FEATURES
        .iter()
        .find(|feature| feature.feature_set == arch.feature_set)
        .expect("every FeatureSet has a YaneuraOu mapping");
    let input_size = arch.feature_set.spec().ft_in();
    let h1 = arch.l1_out - 1;
    let network = if arch.feature_set == FeatureSet::HalfKaHmMerged
        && arch.ft_out == 1536
        && arch.l1_out == 16
        && arch.l2_out == 32
    {
        "SFNN-1536".to_string()
    } else {
        format!(
            "SFNN_{}_{}_{}_{}_k3k3",
            feature.gen_key, arch.ft_out, h1, arch.l2_out
        )
        .to_ascii_uppercase()
    };
    format!(
        "ModelType=SFNNWithoutPsqt;Features={}(Friend)[{input_size}->{}x2],Network={network}{{LayerStack={YANEURAOU_LAYER_STACKS}}}",
        feature.yo_name, arch.ft_out
    )
}

fn validate_weights(
    arch: &Architecture,
    weights: &LayerStackWeights,
    profile: SfnnProfile,
) -> io::Result<()> {
    if let SfnnProfile::TanukiSfnnwoP1536 { eval_scale } = profile
        && (!eval_scale.is_finite() || eval_scale <= 0.0)
    {
        return invalid_input("Tanuki SFNNwoP1536 requires a finite positive eval scale");
    }
    if matches!(profile, SfnnProfile::TanukiSfnnwoP1536 { .. })
        && arch.feature_set != FeatureSet::HalfKaHmMerged
    {
        return invalid_input("Tanuki SFNNwoP1536 requires HalfKaHmMerged features");
    }
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not representable in YaneuraOu SFNN");
    }
    if arch.ft_out == 0 || arch.ft_out > MAX_FT_OUT || !arch.ft_out.is_multiple_of(32) {
        return invalid_input(format!(
            "unsupported FT output dimension {} (expected a positive multiple of 32 up to {MAX_FT_OUT})",
            arch.ft_out
        ));
    }
    if arch.l1_out < 2 || arch.l1_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L1 output dimension {} (expected 2..={MAX_HIDDEN_DIM})",
            arch.l1_out
        ));
    }
    if arch.l2_out == 0 || arch.l2_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 output dimension {} (expected 1..={MAX_HIDDEN_DIM})",
            arch.l2_out
        ));
    }
    let l2_in = (arch.l1_out - 1) * 2;
    let spec = arch.feature_set.spec();
    let layer_stacks = profile.layer_stacks();
    let lengths = [
        ("ft_b", weights.ft_b.len(), arch.ft_out),
        ("ft_w", weights.ft_w.len(), spec.ft_in() * arch.ft_out),
        ("l1_b", weights.l1_b.len(), layer_stacks * arch.l1_out),
        (
            "l1_w",
            weights.l1_w.len(),
            layer_stacks * arch.l1_out * arch.ft_out,
        ),
        ("l1f_b", weights.l1f_b.len(), arch.l1_out),
        ("l1f_w", weights.l1f_w.len(), arch.ft_out * arch.l1_out),
        ("l2_b", weights.l2_b.len(), layer_stacks * arch.l2_out),
        ("l3_b", weights.l3_b.len(), layer_stacks),
        (
            "l2_w",
            weights.l2_w.len(),
            layer_stacks * arch.l2_out * l2_in,
        ),
        ("l3_w", weights.l3_w.len(), layer_stacks * arch.l2_out),
    ];
    for (name, actual, expected) in lengths {
        if actual != expected {
            return invalid_input(format!(
                "{name} length mismatch: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

fn write_affine<W, B, V>(
    writer: &mut W,
    biases: B,
    weights: V,
    input_dimensions: usize,
    output_dimensions: usize,
) -> io::Result<()>
where
    W: Write,
    B: IntoIterator<Item = f32>,
    V: IntoIterator<Item = f32>,
{
    write_affine_scaled(
        writer,
        biases,
        weights,
        input_dimensions,
        output_dimensions,
        f64::from(QA * QB),
        f64::from(QB),
    )
}

fn write_affine_scaled<W, B, V>(
    writer: &mut W,
    biases: B,
    weights: V,
    input_dimensions: usize,
    output_dimensions: usize,
    bias_scale: f64,
    weight_scale: f64,
) -> io::Result<()>
where
    W: Write,
    B: IntoIterator<Item = f32>,
    V: IntoIterator<Item = f32>,
{
    for bias in biases {
        writer.write_all(&quantize_i32(bias, bias_scale).to_le_bytes())?;
    }
    let padded_input = input_dimensions.div_ceil(32) * 32;
    let mut weights = weights.into_iter();
    for _ in 0..output_dimensions {
        for input in 0..padded_input {
            let value = if input < input_dimensions {
                weights
                    .next()
                    .ok_or_else(|| invalid_input_err("affine weight iterator is short"))?
            } else {
                0.0
            };
            writer.write_all(&[quantize_i8(value, weight_scale) as u8])?;
        }
    }
    if weights.next().is_some() {
        return invalid_input("affine weight iterator has extra values");
    }
    Ok(())
}

fn validate_tanuki_import_config(
    expected: FeatureSetSpec,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
    eval_scale: f32,
) -> io::Result<()> {
    if expected != FeatureSet::HalfKaHmMerged.spec() {
        return invalid_input("Tanuki SFNNwoP1536 requires HalfKaHmMerged features");
    }
    if num_buckets != TANUKI_SFNNWOP1536_LAYER_STACKS {
        return invalid_input(format!(
            "Tanuki SFNNwoP1536 requires {TANUKI_SFNNWOP1536_LAYER_STACKS} LayerStacks, but {num_buckets} were requested"
        ));
    }
    if !eval_scale.is_finite() || eval_scale <= 0.0 {
        return invalid_input("Tanuki SFNNwoP1536 requires a finite positive eval scale");
    }
    if ft_out == 0 || ft_out > MAX_FT_OUT || !ft_out.is_multiple_of(32) {
        return invalid_input(format!(
            "unsupported FT output dimension {ft_out} (expected a positive multiple of 32 up to {MAX_FT_OUT})"
        ));
    }
    if !(2..=MAX_HIDDEN_DIM).contains(&l1_out) {
        return invalid_input(format!(
            "unsupported L1 output dimension {l1_out} (expected 2..={MAX_HIDDEN_DIM})"
        ));
    }
    if l2_out == 0 || l2_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 output dimension {l2_out} (expected 1..={MAX_HIDDEN_DIM})"
        ));
    }
    Ok(())
}

fn read_affine<R: Read>(
    reader: &mut R,
    input_dimensions: usize,
    output_dimensions: usize,
    bias_scale: f64,
    weight_scale: f64,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let mut biases = Vec::with_capacity(output_dimensions);
    for _ in 0..output_dimensions {
        biases.push(read_i32(reader)? as f32 / bias_scale as f32);
    }

    let padded_input = input_dimensions.div_ceil(32) * 32;
    let mut weights = Vec::with_capacity(input_dimensions * output_dimensions);
    let mut row = vec![0; padded_input];
    for _ in 0..output_dimensions {
        reader.read_exact(&mut row)?;
        if row[input_dimensions..].iter().any(|&value| value != 0) {
            return invalid_data("Tanuki SFNNwoP1536 affine padding is non-zero");
        }
        weights.extend(
            row[..input_dimensions]
                .iter()
                .map(|&value| (value as i8) as f32 / weight_scale as f32),
        );
    }
    Ok((biases, weights))
}

fn dequantize_i16(values: Vec<i16>, scale: f64) -> Vec<f32> {
    values
        .into_iter()
        .map(|value| value as f32 / scale as f32)
        .collect()
}

fn quantize_i16(values: &[f32], scale: f64) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            (value as f64 * scale)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

fn quantize_i32(value: f32, scale: f64) -> i32 {
    (value as f64 * scale)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn quantize_i8(value: f32, scale: f64) -> i8 {
    (value as f64 * scale)
        .round()
        .clamp(i8::MIN as f64, i8::MAX as f64) as i8
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn require_u32<R: Read>(reader: &mut R, expected: u32, field: &str) -> io::Result<()> {
    let actual = read_u32(reader)?;
    if actual != expected {
        return invalid_data(format!(
            "Tanuki SFNNwoP1536 {field} differs: expected 0x{expected:08x}, got 0x{actual:08x}"
        ));
    }
    Ok(())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_input_err(message))
}

fn invalid_input_err(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32_test(reader: &mut impl std::io::Read) -> u32 {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

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
        save_tanuki_sfnnwop1536(&mut bytes, &weights, 600.0).unwrap();

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
    fn tanuki_profile_loads_and_reexports_byte_for_byte() {
        let mut weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            32,
            3,
            2,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
        );
        weights.ft_b[0] = 4.0 / QA as f32;
        weights.ft_w[17] = -5.0 / QA as f32;
        weights.l1_b[2] = 6.0 / (QA * QB) as f32;
        weights.l1_w[31] = -7.0 / QB as f32;
        weights.l1f_b[1] = 8.0 / (QA * QB) as f32;
        weights.l1f_w[7] = -9.0 / QB as f32;
        weights.l2_b[4] = 10.0 / (QA * QB) as f32;
        weights.l2_w[9] = -11.0 / QB as f32;
        weights.l3_b[3] = 12.0 / (600.0 * 16.0);
        weights.l3_w[5] = -13.0 / ((600.0 * 16.0) / QA as f32);

        let mut direct = Vec::new();
        save_tanuki_sfnnwop1536(&mut direct, &weights, 600.0).unwrap();
        assert!(is_tanuki_sfnnwop1536_header(&direct));

        let loaded = load_tanuki_sfnnwop1536(
            &mut direct.as_slice(),
            FeatureSet::HalfKaHmMerged.spec(),
            32,
            3,
            2,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
            600.0,
        )
        .unwrap();
        assert!(loaded.l1f_w.iter().all(|&value| value == 0.0));
        assert!(loaded.l1f_b.iter().all(|&value| value == 0.0));

        let mut reexported = Vec::new();
        save_tanuki_sfnnwop1536(&mut reexported, &loaded, 600.0).unwrap();
        assert_eq!(direct, reexported);
    }

    #[test]
    fn tanuki_profile_loader_rejects_header_and_trailing_data() {
        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            32,
            2,
            2,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
        );
        let mut bytes = Vec::new();
        save_tanuki_sfnnwop1536(&mut bytes, &weights, 600.0).unwrap();

        let mut wrong_header = bytes.clone();
        wrong_header[4] ^= 1;
        let error = load_tanuki_sfnnwop1536(
            &mut wrong_header.as_slice(),
            FeatureSet::HalfKaHmMerged.spec(),
            32,
            2,
            2,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
            600.0,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("top hash"));

        bytes.push(0);
        let error = load_tanuki_sfnnwop1536(
            &mut bytes.as_slice(),
            FeatureSet::HalfKaHmMerged.spec(),
            32,
            2,
            2,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
            600.0,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("trailing data"));
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
            save_tanuki_sfnnwop1536(&mut Vec::new(), &weights, 600.0).unwrap();
        }
    }

    #[test]
    fn tanuki_profile_quantizes_output_layer_like_nnue_pytorch() {
        use std::io::{Cursor, Read};

        let mut weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            2,
            2,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
        );
        weights.l3_b[0] = 0.25;
        weights.l3_w[0] = 0.25;

        let mut bytes = Vec::new();
        save_tanuki_sfnnwop1536(&mut bytes, &weights, 600.0).unwrap();
        let mut cursor = Cursor::new(bytes.as_slice());
        cursor.set_position(8);
        let arch_len = read_u32_test(&mut cursor) as u64;
        cursor.set_position(cursor.position() + arch_len);
        assert_eq!(read_u32_test(&mut cursor), YO_FT_HASH);
        crate::layerstack_weights::read_leb128_tensor_i16(&mut cursor, Some(128)).unwrap();
        crate::layerstack_weights::read_leb128_tensor_i16(
            &mut cursor,
            Some(FeatureSet::HalfKaHmMerged.spec().ft_in() * 128),
        )
        .unwrap();

        assert_eq!(read_u32_test(&mut cursor), YO_NETWORK_HASH);
        let l1_bytes = 2 * 4 + 2 * 128;
        let l2_bytes = 2 * 4 + 2 * 32;
        cursor.set_position(cursor.position() + (l1_bytes + l2_bytes) as u64);

        let mut bias = [0; 4];
        cursor.read_exact(&mut bias).unwrap();
        assert_eq!(i32::from_le_bytes(bias), 2_400);
        let mut first_weight = [0];
        cursor.read_exact(&mut first_weight).unwrap();
        assert_eq!(first_weight[0] as i8, 19);
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
            save_tanuki_sfnnwop1536(&mut Vec::new(), &wrong_feature, 600.0)
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
            save_tanuki_sfnnwop1536(&mut Vec::new(), &wrong_buckets, 600.0)
                .unwrap_err()
                .to_string()
                .contains("8 LayerStacks")
        );

        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            TANUKI_SFNNWOP1536_LAYER_STACKS,
        );
        assert!(
            save_tanuki_sfnnwop1536(&mut Vec::new(), &weights, 0.0)
                .unwrap_err()
                .to_string()
                .contains("positive")
        );
    }

    #[test]
    fn baseline_architecture_string_matches_yaneuraou() {
        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            1536,
            16,
            32,
            YANEURAOU_LAYER_STACKS,
        );
        assert_eq!(
            yaneuraou_arch_string(&architecture(&weights, SfnnProfile::Yaneuraou).unwrap()),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}"
        );
    }

    #[test]
    fn generated_architecture_names_match_yaneuraou_loader_contract() {
        let cases = [
            (
                FeatureSet::HalfKaHmMerged,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaHmMerged,
                512,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_HALFKAHM2_512_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKp,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->1536x2],Network=SFNN_HALFKP_1536_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaSplit,
                768,
                8,
                16,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA1(Friend)[138510->768x2],Network=SFNN_HALFKA1_768_7_16_K3K3{LayerStack=9}",
            ),
        ];

        for (feature_set, ft_out, l1_out, l2_out, expected) in cases {
            let weights = LayerStackWeights::zeroed(
                feature_set.spec(),
                ft_out,
                l1_out,
                l2_out,
                YANEURAOU_LAYER_STACKS,
            );
            assert_eq!(
                yaneuraou_arch_string(&architecture(&weights, SfnnProfile::Yaneuraou).unwrap()),
                expected
            );
        }
    }

    #[test]
    fn rejects_non_kingrank9_shape() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 8);
        let error = save_yaneuraou(&mut Vec::new(), &weights).unwrap_err();
        assert!(error.to_string().contains("KingRank9"), "{error}");
    }

    #[test]
    fn direct_export_matches_tatara_reload_export_byte_for_byte() {
        let mut weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        );
        weights.ft_b[0] = 0.25;
        weights.ft_w[17] = -0.5;
        weights.l1_b[2] = 0.75;
        weights.l1_w[31] = -0.25;
        weights.l1f_b[1] = 0.25;
        weights.l1f_w[7] = -0.5;
        weights.l2_b[4] = 0.5;
        weights.l2_w[9] = -0.75;
        weights.l3_b[3] = 0.125;
        weights.l3_w[5] = -0.125;

        let mut direct = Vec::new();
        save_yaneuraou(&mut direct, &weights).unwrap();

        let mut tatara = Vec::new();
        weights.save_quantised(&mut tatara, Some(28)).unwrap();
        let reloaded = LayerStackWeights::load_quantised(
            &mut tatara.as_slice(),
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        )
        .unwrap();
        let mut post_hoc = Vec::new();
        save_yaneuraou(&mut post_hoc, &reloaded).unwrap();

        assert_eq!(direct, post_hoc);
    }

    #[test]
    fn affine_weights_are_row_major_and_padded() {
        let mut output = Vec::new();
        write_affine(
            &mut output,
            [1.0, -1.0],
            [
                1.0 / 64.0,
                2.0 / 64.0,
                3.0 / 64.0,
                -1.0 / 64.0,
                -2.0 / 64.0,
                -3.0 / 64.0,
            ],
            3,
            2,
        )
        .unwrap();
        assert_eq!(
            i32::from_le_bytes(output[0..4].try_into().unwrap()),
            QA * QB
        );
        assert_eq!(
            i32::from_le_bytes(output[4..8].try_into().unwrap()),
            -(QA * QB)
        );
        assert_eq!(&output[8..11], &[1, 2, 3]);
        assert!(output[11..40].iter().all(|&byte| byte == 0));
        assert_eq!(&output[40..43], &[255, 254, 253]);
        assert!(output[43..72].iter().all(|&byte| byte == 0));
    }
}
