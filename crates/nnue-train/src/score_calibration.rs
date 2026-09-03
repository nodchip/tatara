//! Train-onlyで固定した単調写像を、教師scoreへ読み込み時に適用する。

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

const SCORE_VALUES: usize = 1 << 16;
const MAX_MAP_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileFingerprint {
    path: String,
    bytes: u64,
    records: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetFingerprint {
    files: Vec<FileFingerprint>,
    records: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationMap {
    schema_version: u32,
    method: String,
    rounding: String,
    unseen_score_rule: String,
    source_train: DatasetFingerprint,
    reference_train: DatasetFingerprint,
    mapping: Vec<[i16; 2]>,
}

/// 検証済みの全i16入力に対する較正先lookup table。
#[derive(Debug)]
pub struct ScoreCalibration {
    lookup: Box<[i16; SCORE_VALUES]>,
}

impl ScoreCalibration {
    /// rshogi `calibrate_psv_scores fit`形式のmapを読み、identityと単調性を検証する。
    pub fn load(path: &Path, expected_sha256: &str) -> io::Result<Self> {
        if expected_sha256.len() != 64
            || !expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || expected_sha256
                .bytes()
                .any(|byte| byte.is_ascii_uppercase())
        {
            return Err(invalid_data(
                "score calibration SHA-256 must be lowercase hex",
            ));
        }
        let bytes = fs::read(path)?;
        if !(2..=MAX_MAP_BYTES).contains(&bytes.len()) {
            return Err(invalid_data(
                "score calibration map size is outside the contract",
            ));
        }
        if sha256_hex(&bytes) != expected_sha256 {
            return Err(invalid_data("score calibration map SHA-256 differs"));
        }
        let map: CalibrationMap = serde_json::from_slice(&bytes)
            .map_err(|error| invalid_data(format!("score calibration map is invalid: {error}")))?;
        validate_map(&map)?;
        Ok(Self {
            lookup: build_lookup_table(&map.mapping)?,
        })
    }

    /// 1件のraw teacher scoreを固定写像へ通す。
    #[inline]
    pub fn apply(&self, score: i16) -> i16 {
        self.lookup[score_index(score)]
    }

    #[cfg(test)]
    pub(crate) fn from_mapping(mapping: &[[i16; 2]]) -> io::Result<Self> {
        Ok(Self {
            lookup: build_lookup_table(mapping)?,
        })
    }
}

fn validate_map(map: &CalibrationMap) -> io::Result<()> {
    if map.schema_version != 1
        || map.method != "tie_preserving_quantile"
        || map.rounding != "round_ties_even"
        || map.unseen_score_rule != "linear_interpolation_with_endpoint_clamp"
    {
        return Err(invalid_data("score calibration map contract differs"));
    }
    validate_dataset(&map.source_train)?;
    validate_dataset(&map.reference_train)?;
    if map.source_train.records == 0 || map.source_train.records != map.reference_train.records {
        return Err(invalid_data("score calibration train identities differ"));
    }
    validate_mapping(&map.mapping)
}

fn validate_dataset(dataset: &DatasetFingerprint) -> io::Result<()> {
    if dataset.files.is_empty() {
        return Err(invalid_data("score calibration dataset is empty"));
    }
    let mut records = 0_u64;
    for file in &dataset.files {
        if file.path.is_empty()
            || file.bytes == 0
            || !file.bytes.is_multiple_of(40)
            || file.records != file.bytes / 40
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(invalid_data(
                "score calibration dataset identity is invalid",
            ));
        }
        records = records
            .checked_add(file.records)
            .ok_or_else(|| invalid_data("score calibration record count overflow"))?;
    }
    if records != dataset.records {
        return Err(invalid_data(
            "score calibration dataset record count differs",
        ));
    }
    Ok(())
}

fn validate_mapping(mapping: &[[i16; 2]]) -> io::Result<()> {
    if mapping.is_empty() {
        return Err(invalid_data("score calibration mapping is empty"));
    }
    for pair in mapping.windows(2) {
        if pair[0][0] >= pair[1][0] || pair[0][1] > pair[1][1] {
            return Err(invalid_data("score calibration mapping is not monotone"));
        }
    }
    Ok(())
}

fn build_lookup_table(mapping: &[[i16; 2]]) -> io::Result<Box<[i16; SCORE_VALUES]>> {
    validate_mapping(mapping)?;
    let mut lookup = Box::new([0_i16; SCORE_VALUES]);
    let first = mapping[0];
    let last = *mapping.last().expect("mapping is nonempty");
    let mut segment = 0_usize;
    for (index, slot) in lookup.iter_mut().enumerate() {
        let score = score_from_index(index);
        *slot = if score <= first[0] {
            first[1]
        } else if score >= last[0] {
            last[1]
        } else {
            while mapping[segment + 1][0] < score {
                segment += 1;
            }
            let [x0, y0] = mapping[segment];
            let [x1, y1] = mapping[segment + 1];
            if score == x0 {
                y0
            } else if score == x1 {
                y1
            } else {
                let width = i128::from(i32::from(x1) - i32::from(x0));
                let offset = i128::from(i32::from(score) - i32::from(x0));
                let delta = i128::from(i32::from(y1) - i32::from(y0));
                let numerator = i128::from(y0) * width + offset * delta;
                round_ratio_ties_even(numerator, width as u64)?
            }
        };
    }
    Ok(lookup)
}

fn round_ratio_ties_even(numerator: i128, denominator: u64) -> io::Result<i16> {
    if denominator == 0 {
        return Err(invalid_data(
            "score calibration rounding denominator is zero",
        ));
    }
    let negative = numerator < 0;
    let magnitude = numerator.abs();
    let denominator = i128::from(denominator);
    let quotient = magnitude / denominator;
    let remainder = magnitude % denominator;
    let rounded = match (remainder * 2).cmp(&denominator) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal => quotient + (quotient & 1),
    };
    let signed = if negative { -rounded } else { rounded };
    i16::try_from(signed).map_err(|_| invalid_data("score calibration result exceeds i16"))
}

fn score_index(score: i16) -> usize {
    (i32::from(score) - i32::from(i16::MIN)) as usize
}

fn score_from_index(index: usize) -> i16 {
    (index as i32 + i32::from(i16::MIN)) as i16
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

// SHA-256はmap identity gateだけで使う。外部crateを増やさずCPU-only CIを維持する。
fn sha256_hex(input: &[u8]) -> String {
    let mut state = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        sha256_compress(&mut state, chunk);
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

fn sha256_compress(state: &mut [u32; 8], chunk: &[u8]) {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut schedule = [0_u32; 64];
    for (index, bytes) in chunk.chunks_exact(4).take(16).enumerate() {
        schedule[index] = u32::from_be_bytes(bytes.try_into().expect("four-byte word"));
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let mut work = *state;
    for index in 0..64 {
        let sigma1 = work[4].rotate_right(6) ^ work[4].rotate_right(11) ^ work[4].rotate_right(25);
        let choose = (work[4] & work[5]) ^ (!work[4] & work[6]);
        let t1 = work[7]
            .wrapping_add(sigma1)
            .wrapping_add(choose)
            .wrapping_add(K[index])
            .wrapping_add(schedule[index]);
        let sigma0 = work[0].rotate_right(2) ^ work[0].rotate_right(13) ^ work[0].rotate_right(22);
        let majority = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]);
        let t2 = sigma0.wrapping_add(majority);
        work = [
            t1.wrapping_add(t2),
            work[0],
            work[1],
            work[2],
            work[3].wrapping_add(t1),
            work[4],
            work[5],
            work[6],
        ];
    }
    for (slot, value) in state.iter_mut().zip(work) {
        *slot = slot.wrapping_add(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn interpolation_clamps_and_uses_ties_to_even() {
        let lookup = build_lookup_table(&[[0, 0], [4, 10]]).unwrap();
        assert_eq!(lookup[score_index(i16::MIN)], 0);
        assert_eq!(lookup[score_index(1)], 2);
        assert_eq!(lookup[score_index(3)], 8);
        assert_eq!(lookup[score_index(i16::MAX)], 10);
    }

    #[test]
    fn rejects_non_monotone_mapping() {
        assert!(build_lookup_table(&[[0, 0], [1, -1]]).is_err());
        assert!(build_lookup_table(&[[0, 0], [0, 1]]).is_err());
    }

    #[test]
    #[ignore = "requires an external audited score calibration map"]
    fn loads_external_audited_map() {
        let path = std::env::var_os("TATARA_SCORE_CALIBRATION_MAP")
            .map(std::path::PathBuf::from)
            .expect("set TATARA_SCORE_CALIBRATION_MAP");
        let sha256 = std::env::var("TATARA_SCORE_CALIBRATION_MAP_SHA256")
            .expect("set TATARA_SCORE_CALIBRATION_MAP_SHA256");
        let calibration = ScoreCalibration::load(&path, &sha256).expect("load audited map");
        for score in i16::MIN..i16::MAX {
            assert!(calibration.apply(score) <= calibration.apply(score + 1));
        }
    }
}
