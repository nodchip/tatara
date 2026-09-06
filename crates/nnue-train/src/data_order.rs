//! Deterministic, bounded-memory permutations of fixed-width PSV records.
//!
//! Blocks are permuted first, then records within each block. This is not a
//! uniform permutation of the entire dataset: records from one block remain
//! adjacent. The order depends only on seed, epoch, and record count, never on
//! teacher scores. Contiguous block reads avoid one disk seek per record.

use std::io::{self, Read, Seek, SeekFrom};

use shogi_format::PackedSfenValue;

use crate::dataloader::PSV_RECORD_BYTES;

/// Version the ordering algorithm independently of a human-readable series name.
pub const DATA_ORDER_ALGORITHM: &str = "psv-block-shuffle-splitmix64-v1";
pub const DATA_ORDER_BLOCK_RECORDS: u64 = 262_144;
const MAX_BLOCKS: u64 = 1_048_576;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        // Rejection removes modulo bias without changing the accepted values.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % bound;
            }
        }
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            values.swap(i, j);
        }
    }
}

/// One epoch over `[start, end)`, excluding bytes outside that training range.
/// Memory is bounded by one block, its record permutation, and at most
/// `MAX_BLOCKS` block indices. No output dataset is written.
pub struct ShuffledPsvReader<R> {
    reader: R,
    start: u64,
    records: u64,
    block_records: u64,
    blocks: Vec<u64>,
    next_block: usize,
    buffer: Vec<u8>,
    record_order: Vec<usize>,
    next_record: usize,
    rng: SplitMix64,
}

impl<R: Read + Seek> ShuffledPsvReader<R> {
    pub fn new(reader: R, start: u64, end: u64, seed: u64, epoch: u64) -> io::Result<Self> {
        Self::with_block_records(reader, start, end, seed, epoch, DATA_ORDER_BLOCK_RECORDS)
    }

    fn with_block_records(
        mut reader: R,
        start: u64,
        end: u64,
        seed: u64,
        epoch: u64,
        block_records: u64,
    ) -> io::Result<Self> {
        let file_size = reader.seek(SeekFrom::End(0))?;
        if start > end
            || end > file_size
            || !start.is_multiple_of(PSV_RECORD_BYTES)
            || !end.is_multiple_of(PSV_RECORD_BYTES)
            || block_records == 0
            || block_records > DATA_ORDER_BLOCK_RECORDS
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid PSV shuffle range",
            ));
        }
        let records = (end - start) / PSV_RECORD_BYTES;
        let block_count = records.div_ceil(block_records);
        if block_count > MAX_BLOCKS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "PSV shuffle block limit exceeded",
            ));
        }
        let mut rng = SplitMix64(seed);
        let mut epoch_rng = SplitMix64(epoch);
        rng.0 = rng.next() ^ epoch_rng.next().rotate_left(17);
        let mut blocks: Vec<u64> = (0..block_count).collect();
        rng.shuffle(&mut blocks);
        Ok(Self {
            reader,
            start,
            records,
            block_records,
            blocks,
            next_block: 0,
            buffer: Vec::new(),
            record_order: Vec::new(),
            next_record: 0,
            rng,
        })
    }

    pub fn next_psv(&mut self) -> io::Result<Option<PackedSfenValue>> {
        if self.next_record == self.record_order.len() {
            let Some(&block) = self.blocks.get(self.next_block) else {
                return Ok(None);
            };
            let first = block * self.block_records;
            let count = (self.records - first).min(self.block_records) as usize;
            self.reader
                .seek(SeekFrom::Start(self.start + first * PSV_RECORD_BYTES))?;
            self.buffer.resize(count * PSV_RECORD_BYTES as usize, 0);
            self.reader.read_exact(&mut self.buffer)?;
            self.record_order.clear();
            self.record_order.extend(0..count);
            self.rng.shuffle(&mut self.record_order);
            self.next_record = 0;
            self.next_block += 1;
        }
        let offset = self.record_order[self.next_record] * PSV_RECORD_BYTES as usize;
        let mut record = PackedSfenValue::default();
        record
            .as_bytes_mut()
            .copy_from_slice(&self.buffer[offset..offset + PSV_RECORD_BYTES as usize]);
        self.next_record += 1;
        Ok(Some(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture(count: u64) -> Vec<u8> {
        (0..count)
            .flat_map(|index| {
                let mut bytes = [0u8; PSV_RECORD_BYTES as usize];
                bytes[..8].copy_from_slice(&index.to_le_bytes());
                bytes
            })
            .collect()
    }

    fn indices(bytes: Vec<u8>, start: u64, end: u64, seed: u64, epoch: u64) -> Vec<u64> {
        let mut reader = ShuffledPsvReader::with_block_records(
            Cursor::new(bytes),
            start * 40,
            end * 40,
            seed,
            epoch,
            4,
        )
        .unwrap();
        let mut result = Vec::new();
        while let Some(record) = reader.next_psv().unwrap() {
            result.push(u64::from_le_bytes(
                record.as_bytes()[..8].try_into().unwrap(),
            ));
        }
        assert!(reader.next_psv().unwrap().is_none());
        result
    }

    #[test]
    fn splitmix_known_vector() {
        assert_eq!(SplitMix64(0).next(), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn each_record_once_with_short_last_block_and_heldout_edges() {
        let mut order = indices(fixture(31), 3, 26, 123, 0);
        order.sort_unstable();
        assert_eq!(order, (3..26).collect::<Vec<_>>());
    }

    #[test]
    fn reproducible_and_varies_by_seed_and_epoch() {
        let order = indices(fixture(31), 0, 31, 123, 0);
        assert_eq!(
            order,
            [
                26, 24, 27, 25, 19, 17, 18, 16, 2, 0, 1, 3, 11, 10, 8, 9, 20, 22, 21, 23, 6, 5, 4,
                7, 12, 14, 15, 13, 28, 30, 29
            ]
        );
        assert_eq!(order, indices(fixture(31), 0, 31, 123, 0));
        assert_ne!(order, indices(fixture(31), 0, 31, 124, 0));
        assert_ne!(order, indices(fixture(31), 0, 31, 123, 1));
    }

    #[test]
    fn scores_do_not_affect_order_and_bytes_are_preserved() {
        let mut changed = fixture(31);
        for record in changed.chunks_exact_mut(40) {
            record[32..34].copy_from_slice(&1234i16.to_le_bytes());
        }
        assert_eq!(
            indices(fixture(31), 0, 31, 123, 0),
            indices(changed.clone(), 0, 31, 123, 0)
        );
        let mut reader =
            ShuffledPsvReader::new(Cursor::new(changed.clone()), 0, 31 * 40, 1, 0).unwrap();
        while let Some(record) = reader.next_psv().unwrap() {
            let index = u64::from_le_bytes(record.as_bytes()[..8].try_into().unwrap()) as usize;
            assert_eq!(record.as_bytes(), &changed[index * 40..(index + 1) * 40]);
        }
    }

    #[test]
    fn empty_and_single_record_ranges() {
        assert!(indices(fixture(2), 1, 1, 1, 0).is_empty());
        assert_eq!(indices(fixture(2), 1, 2, 1, 0), [1]);
    }

    #[test]
    fn production_block_boundary_preserves_every_record() {
        let count = DATA_ORDER_BLOCK_RECORDS + 3;
        let mut reader =
            ShuffledPsvReader::new(Cursor::new(fixture(count)), 0, count * 40, 0, 0).unwrap();
        let mut seen = vec![false; count as usize];
        while let Some(record) = reader.next_psv().unwrap() {
            let index = u64::from_le_bytes(record.as_bytes()[..8].try_into().unwrap()) as usize;
            assert!(!seen[index]);
            seen[index] = true;
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn truncated_source_is_an_error_not_clean_eof() {
        struct ShortReader(Cursor<Vec<u8>>);
        impl Read for ShortReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.0.read(buf)
            }
        }
        impl Seek for ShortReader {
            fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
                if matches!(from, SeekFrom::End(0)) {
                    Ok(80)
                } else {
                    self.0.seek(from)
                }
            }
        }
        let mut reader =
            ShuffledPsvReader::new(ShortReader(Cursor::new(fixture(1))), 0, 80, 0, 0).unwrap();
        assert_eq!(
            reader.next_psv().err().unwrap().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn rejects_invalid_ranges_and_bounded_memory_violation() {
        for (start, end) in [(1, 40), (0, 41), (80, 40), (0, 120)] {
            assert!(ShuffledPsvReader::new(Cursor::new(fixture(2)), start, end, 0, 0).is_err());
        }
        let end = (MAX_BLOCKS + 1) * DATA_ORDER_BLOCK_RECORDS * 40;
        struct VirtualFile(u64);
        impl Read for VirtualFile {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                panic!("must reject before reading")
            }
        }
        impl Seek for VirtualFile {
            fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
                Ok(self.0)
            }
        }
        assert!(ShuffledPsvReader::new(VirtualFile(end), 0, end, 0, 0).is_err());
    }
}
