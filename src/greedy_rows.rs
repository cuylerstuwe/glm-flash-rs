//! Checked BF16 row addressing and the serial numeric argmax oracle.
use anyhow::{Context, Result, ensure};
pub(super) const LOCAL: usize = 38720;
pub(super) const ROW_BYTES: usize = LOCAL * 2;
pub(super) fn address(ptr: u64, bytes: usize, offset: usize, rank: usize) -> Result<u64> {
    ensure!(
        rank < 4 && ptr != 0 && ptr.is_multiple_of(2) && offset.is_multiple_of(2),
        "greedy row alignment/rank"
    );
    ensure!(
        offset
            .checked_add(ROW_BYTES)
            .is_some_and(|end| end <= bytes),
        "greedy row storage bounds"
    );
    let start = ptr
        .checked_add(offset as u64)
        .context("greedy row address overflow")?;
    start
        .checked_add(ROW_BYTES as u64)
        .context("greedy row end overflow")?;
    Ok(start)
}
pub fn reference(bytes: &[u8]) -> Result<usize> {
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(2),
        "BF16 logits required"
    );
    let mut best = (0, f32::NEG_INFINITY);
    for (id, b) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let x = f32::from_bits((u16::from_le_bytes(*b) as u32) << 16);
        ensure!(x.is_finite(), "nonfinite reference logit");
        if x > best.1 {
            best = (id, x);
        }
    }
    Ok(best.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_bounds_reject_alignment_short_storage_and_overflow() {
        for row in 0..16 {
            for rank in 0..4 {
                assert_eq!(
                    address(4096, 16 * ROW_BYTES, row * ROW_BYTES, rank).unwrap(),
                    4096 + (row * ROW_BYTES) as u64
                );
            }
        }
        for (ptr, bytes, offset, rank) in [
            (0, ROW_BYTES, 0, 0),
            (4097, ROW_BYTES, 0, 0),
            (4096, ROW_BYTES + 2, 1, 0),
            (4096, ROW_BYTES - 1, 0, 0),
            (4096, 16 * ROW_BYTES, 16 * ROW_BYTES, 0),
            (4096, ROW_BYTES, 0, 4),
            (4096, usize::MAX, usize::MAX - 1, 0),
            (u64::MAX - 1, ROW_BYTES, 0, 0),
        ] {
            assert!(address(ptr, bytes, offset, rank).is_err());
        }
    }
    #[test]
    fn numeric_zeros_ties_extrema_and_nonfinite_match_serial_contract() {
        let bytes = |values: &[u16]| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        for values in [
            &[0x8000, 0][..],
            &[0, 0x8000],
            &[0xbf80, 0xbf80],
            &[0x7f7f, 0x7f7f],
        ] {
            assert_eq!(reference(&bytes(values)).unwrap(), 0);
        }
        assert_eq!(
            reference(&bytes(&[0xff7f, 0xbf80, 0x8000, 0, 0x3f80, 0x7f7f])).unwrap(),
            5
        );
        for bad in [0x7f80, 0xff80, 0x7fc1, 0xffc1] {
            assert!(reference(&bytes(&[0x3f80, bad])).is_err());
        }
        assert!(reference(&[]).is_err());
        assert!(reference(&[0]).is_err());
    }
}
