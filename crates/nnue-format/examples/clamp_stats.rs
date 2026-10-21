//! Read-only aggregate weight statistics for generic LayerStack and native exports.
//! Signed endpoint occupancy does not establish pre-export clipping or playing strength.
//! Native exports require an explicit evaluation scale for output weight reconstruction.
//!
//! 使い方:
//!   cargo run --release -p nnue-format --example clamp_stats -- \
//!     <net.bin> <feature-set> <ft_out> <l1> <l2> <num_buckets> [--eval-scale <scale>]

use std::io::{self, BufRead};

use nnue_format::layerstack_weights::{LayerStackWeights, QA, QB};
use nnue_format::{is_tanuki_sfnnwop1536_header, load_tanuki_sfnnwop1536};
use shogi_features::FeatureSet;

#[derive(Debug, PartialEq)]
struct Statistics {
    negative_endpoint: usize,
    positive_endpoint: usize,
    near_endpoint: usize,
    max_abs: f32,
    p99_abs: f32,
    p999_abs: f32,
}

fn summarize(w: &[f32], scale: f64, minimum: i32, maximum: i32) -> io::Result<Option<Statistics>> {
    if !scale.is_finite() || scale <= 0.0 || minimum >= 0 || maximum <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid quantisation bounds",
        ));
    }
    if w.iter().any(|v| !v.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "non-finite weight",
        ));
    }
    if w.is_empty() {
        return Ok(None);
    }
    let mut abs: Vec<f32> = w.iter().map(|v| v.abs()).collect();
    abs.sort_by(f32::total_cmp);
    let pct = |q: f64| abs[((abs.len() - 1) as f64 * q) as usize];
    let mut result = Statistics {
        negative_endpoint: 0,
        positive_endpoint: 0,
        near_endpoint: 0,
        max_abs: abs[abs.len() - 1],
        p99_abs: pct(0.99),
        p999_abs: pct(0.999),
    };
    for value in w {
        // Reconstruct wire integers; signed quantisation ranges are asymmetric.
        let integer = (f64::from(*value) * scale).round();
        if integer < f64::from(minimum) || integer > f64::from(maximum) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "weight outside quantised range",
            ));
        }
        result.negative_endpoint += usize::from(integer == f64::from(minimum));
        result.positive_endpoint += usize::from(integer == f64::from(maximum));
        result.near_endpoint +=
            usize::from(integer <= f64::from(minimum) * 0.9 || integer >= f64::from(maximum) * 0.9);
    }
    Ok(Some(result))
}

fn stats(name: &str, w: &[f32], scale: f64, minimum: i32, maximum: i32) -> io::Result<()> {
    if w.is_empty() {
        println!("{name:>8}: (empty)");
        return Ok(());
    }
    let s = summarize(w, scale, minimum, maximum)?.expect("nonempty weights");
    println!(
        "{name}: n={} max_abs={:.9} p99_abs={:.9} p999_abs={:.9} negative_endpoint={} positive_endpoint={} near_endpoint={} scale={scale:.9} range=[{minimum},{maximum}]",
        w.len(),
        s.max_abs,
        s.p99_abs,
        s.p999_abs,
        s.negative_endpoint,
        s.positive_endpoint,
        s.near_endpoint,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnue_format::save_tanuki_sfnnwop1536;

    fn feature() -> FeatureSet {
        FeatureSet::from_canonical_name("halfka-hm-merged").unwrap()
    }

    fn fixture(native: bool) -> Vec<u8> {
        let mut weights = LayerStackWeights::zeroed(feature().spec(), 32, 3, 2, 8);
        let scale = if native {
            320.0 * 16.0 / QA as f32
        } else {
            QB as f32
        };
        weights.l3_w[0] = -128.0 / scale;
        weights.l3_w[1] = 127.0 / scale;
        weights.ft_w[0] = -32768.0 / QA as f32;
        weights.ft_w[1] = 32767.0 / QA as f32;
        let mut bytes = Vec::new();
        if native {
            save_tanuki_sfnnwop1536(&mut bytes, &weights, 320.0).unwrap();
        } else {
            weights.save_quantised(&mut bytes, None).unwrap();
        }
        bytes
    }

    #[test]
    fn signed_endpoints_exclude_adjacent_integers() {
        let values = [-128.0, -127.0, 0.0, 126.0, 127.0].map(|v| v / 64.0);
        let s = summarize(&values, 64.0, -128, 127).unwrap().unwrap();
        assert_eq!(
            (s.negative_endpoint, s.positive_endpoint, s.near_endpoint),
            (1, 1, 4)
        );
        assert_eq!(s.max_abs, 2.0);
        assert_eq!(s.p99_abs, 127.0 / 64.0);
        assert_eq!(s.p999_abs, 127.0 / 64.0);
    }

    #[test]
    fn empty_singleton_and_invalid_statistics() {
        assert_eq!(summarize(&[], 64.0, -128, 127).unwrap(), None);
        let s = summarize(&[0.0], 64.0, -128, 127).unwrap().unwrap();
        assert_eq!(
            (s.max_abs, s.p99_abs, s.p999_abs, s.near_endpoint),
            (0.0, 0.0, 0.0, 0)
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0] {
            assert!(summarize(&[value], 64.0, -128, 127).is_err());
        }
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(summarize(&[0.0], scale, -128, 127).is_err());
        }
    }

    #[test]
    fn both_formats_use_correct_scale_and_preserve_input() {
        for native in [false, true] {
            let bytes = fixture(native);
            let original = bytes.clone();
            let scale_arg = native.then_some(320.0);
            let (weights, scale) = load(
                &mut io::Cursor::new(bytes.as_slice()),
                feature(),
                [32, 3, 2, 8],
                scale_arg,
            )
            .unwrap();
            assert_eq!(
                scale,
                if native {
                    320.0 * 16.0 / f64::from(QA)
                } else {
                    f64::from(QB)
                }
            );
            let s = summarize(&weights.l3_w, scale, -128, 127).unwrap().unwrap();
            assert_eq!((s.negative_endpoint, s.positive_endpoint), (1, 1));
            let s = summarize(&weights.ft_w, f64::from(QA), -32768, 32767)
                .unwrap()
                .unwrap();
            assert_eq!((s.negative_endpoint, s.positive_endpoint), (1, 1));
            assert!(
                weights
                    .l1f_w
                    .iter()
                    .chain(&weights.l1f_b)
                    .all(|v| *v == 0.0)
            );
            assert_eq!(bytes, original);
        }
    }

    #[test]
    fn rejects_missing_invalid_and_inapplicable_scale() {
        let bytes = fixture(true);
        for scale in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::MAX),
        ] {
            assert!(
                load(
                    &mut io::Cursor::new(bytes.as_slice()),
                    feature(),
                    [32, 3, 2, 8],
                    scale
                )
                .is_err()
            );
        }
        assert!(
            load(
                &mut io::Cursor::new(fixture(false)),
                feature(),
                [32, 3, 2, 8],
                Some(320.0)
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_truncated_trailing_and_wrong_dimensions() {
        for native in [false, true] {
            let bytes = fixture(native);
            let scale = native.then_some(320.0);
            assert!(
                load(
                    &mut io::Cursor::new(&bytes[..bytes.len() - 1]),
                    feature(),
                    [32, 3, 2, 8],
                    scale
                )
                .is_err()
            );
            let mut trailing = bytes.clone();
            trailing.push(0);
            assert!(
                load(
                    &mut io::Cursor::new(trailing),
                    feature(),
                    [32, 3, 2, 8],
                    scale
                )
                .is_err()
            );
            assert!(load(&mut io::Cursor::new(bytes), feature(), [64, 3, 2, 8], scale).is_err());
        }
    }

    #[test]
    fn rejects_zero_or_underflowing_dimensions_before_reading() {
        for dims in [
            [0, 3, 2, 8],
            [32, 0, 2, 8],
            [32, 1, 2, 8],
            [32, 3, 0, 8],
            [32, 3, 2, 0],
        ] {
            let error = load(&mut io::Cursor::new([]), feature(), dims, None).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
    }
}

fn load<R: BufRead>(
    reader: &mut R,
    fs: FeatureSet,
    dims: [usize; 4],
    eval_scale: Option<f32>,
) -> io::Result<(LayerStackWeights, f64)> {
    let [ft, l1, l2, buckets] = dims;
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
    if ft == 0 || l1 < 2 || l2 == 0 || buckets == 0 {
        return Err(invalid("invalid layer dimensions"));
    }
    if is_tanuki_sfnnwop1536_header(reader.fill_buf()?) {
        let scale = eval_scale.ok_or_else(|| invalid("native export requires --eval-scale"))?;
        if !scale.is_finite() || scale <= 0.0 || !(scale * 16.0).is_finite() {
            return Err(invalid("invalid evaluation scale"));
        }
        let weights = load_tanuki_sfnnwop1536(reader, fs.spec(), ft, l1, l2, buckets, scale)?;
        Ok((weights, f64::from(scale) * 16.0 / f64::from(QA)))
    } else {
        if eval_scale.is_some() {
            return Err(invalid("--eval-scale only applies to native exports"));
        }
        let weights = LayerStackWeights::load_quantised(reader, fs.spec(), ft, l1, l2, buckets)?;
        if reader.read(&mut [0])? != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing export data",
            ));
        }
        Ok((weights, f64::from(QB)))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 && args.len() != 9 {
        return Err(
            "usage: clamp_stats <net.bin> <feature-set> <ft_out> <l1> <l2> <num_buckets> [--eval-scale <positive-number>]".into(),
        );
    }
    let path = &args[1];
    let fs = FeatureSet::from_canonical_name(&args[2])
        .ok_or_else(|| format!("unknown feature set: {}", args[2]))?;
    let ft_out: usize = args[3].parse()?;
    let l1: usize = args[4].parse()?;
    let l2: usize = args[5].parse()?;
    let num_buckets: usize = args[6].parse()?;

    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let eval_scale = if args.len() == 9 {
        if args[7] != "--eval-scale" {
            return Err("unknown option".into());
        }
        Some(args[8].parse()?)
    } else {
        None
    };
    let (w, output_scale) = load(&mut reader, fs, [ft_out, l1, l2, num_buckets], eval_scale)?;
    println!(
        "aggregate quantised weights; endpoint occupancy does not establish pre-export clipping"
    );
    stats("l1_w", &w.l1_w, f64::from(QB), -128, 127)?;
    stats("l2_w", &w.l2_w, f64::from(QB), -128, 127)?;
    stats("l3_w", &w.l3_w, output_scale, -128, 127)?;
    println!("l1f_w/l1f_b: folded into l1; not independently stored");
    stats("ft_w", &w.ft_w, f64::from(QA), -32768, 32767)?;
    stats("ft_b", &w.ft_b, f64::from(QA), -32768, 32767)?;
    Ok(())
}
