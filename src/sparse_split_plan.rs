//! Tensor-parallel sparse row ownership with a fixed selected-ID format.
use anyhow::{Result, ensure};
pub(crate) const WORLD: usize = 4;
pub(crate) const GRID: usize = 64;
pub(crate) const MAX_ROWS: usize = 1024;
pub(crate) const POOLS: usize = 512;
pub(crate) const ROW_BYTES: usize = POOLS * 8;
pub(crate) const SEND_BYTES: usize = MAX_ROWS / WORLD * ROW_BYTES;
pub(crate) const RECV_BYTES: usize = WORLD * SEND_BYTES;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) rows: usize,
    pub(crate) bounds: [usize; WORLD + 1],
    pub(crate) stride: usize,
}
pub(crate) fn enabled(value: &str) -> Result<bool> {
    match value.trim() {
        "" | "0" => Ok(false),
        "1" => Ok(true),
        _ => anyhow::bail!("ALASKA_SELECT_SPLIT accepts only 0 or retained policy 1"),
    }
}
impl Plan {
    pub(crate) fn new(on: bool, fast: bool, position: usize, rows: usize) -> Result<Option<Self>> {
        let end = position
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("selection position overflow"))?;
        ensure!(end <= 1048576, "selection total context bound");
        if !on || !fast || rows < 256 || end / 4 < 4096 {
            return Ok(None);
        }
        ensure!(rows <= MAX_ROWS, "selection attention slab bound");
        let mut bounds = [0; WORLD + 1];
        for (rank, bound) in bounds.iter_mut().enumerate().take(WORLD).skip(1) {
            *bound = (rank * rows).div_ceil(WORLD).div_ceil(GRID) * GRID;
            *bound = (*bound).min(rows);
        }
        bounds[WORLD] = rows;
        let owned = bounds.windows(2).map(|b| b[1] - b[0]).max().unwrap();
        let stride = owned.div_ceil(GRID) * GRID * ROW_BYTES;
        ensure!(stride <= SEND_BYTES, "selection packed stride bound");
        Ok(Some(Self {
            rows,
            bounds,
            stride,
        }))
    }
    pub(crate) fn own(self, rank: usize) -> Result<std::ops::Range<usize>> {
        ensure!(rank < WORLD, "selection rank bound");
        Ok(self.bounds[rank]..self.bounds[rank + 1])
    }
    /// A global 64-row attention block is always wholly owned by one rank.
    /// Return only live pool-ID bytes; received padding is never addressable.
    pub(crate) fn region(self, offset: usize, rows: usize) -> Result<(usize, usize)> {
        ensure!(
            offset.is_multiple_of(GRID) && rows > 0 && rows <= GRID,
            "selection block alignment"
        );
        let end = offset
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("selection row overflow"))?;
        ensure!(end <= self.rows, "selection live row bound");
        let rank = (0..WORLD)
            .find(|&r| self.bounds[r] <= offset && offset < self.bounds[r + 1])
            .ok_or_else(|| anyhow::anyhow!("selection row has no owner"))?;
        ensure!(
            end <= self.bounds[rank + 1],
            "selection block crosses owner"
        );
        let local = (offset - self.bounds[rank]) * ROW_BYTES;
        let bytes = rows * ROW_BYTES;
        ensure!(
            local + bytes <= self.stride && rank * self.stride + local + bytes <= RECV_BYTES,
            "selection pool region bound"
        );
        Ok((rank * self.stride + local, bytes))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_threshold_and_ownership() {
        assert!(!enabled("").unwrap());
        assert!(!enabled("0").unwrap());
        assert!(enabled("1").unwrap());
        assert!(enabled("4096").is_err());
        assert!(Plan::new(true, true, 16127, 256).unwrap().is_none());
        assert!(Plan::new(true, true, 16128, 256).unwrap().is_some());
        assert!(Plan::new(true, true, 20000, 255).unwrap().is_none());
        assert!(Plan::new(true, false, 20000, 512).unwrap().is_none());
        assert!(Plan::new(false, true, 20000, 512).unwrap().is_none());
        assert_eq!(
            Plan::new(true, true, 20000, 257).unwrap().unwrap().bounds,
            [0, 128, 192, 256, 257]
        );
        assert!(Plan::new(true, true, 20000, 1025).is_err());
        assert!(Plan::new(true, true, 1048576, 1).is_err());
    }
    #[test]
    fn every_tail_owns_exactly_once_and_never_reads_poisoned_padding() {
        for rows in 256..=MAX_ROWS {
            let p = Plan::new(true, true, 20000, rows).unwrap().unwrap();
            let mut seen = vec![false; rows];
            let mut packed = vec![usize::MAX; p.stride / ROW_BYTES * WORLD];
            for rank in 0..WORLD {
                let own = p.own(rank).unwrap();
                for row in own.clone() {
                    assert!(!seen[row]);
                    seen[row] = true;
                    packed[rank * p.stride / ROW_BYTES + row - own.start] = row;
                }
            }
            assert!(seen.into_iter().all(|v| v));
            for offset in (0..rows).step_by(GRID) {
                let n = (rows - offset).min(GRID);
                let (at, bytes) = p.region(offset, n).unwrap();
                assert_eq!(bytes, n * ROW_BYTES);
                assert_eq!(
                    &packed[at / ROW_BYTES..at / ROW_BYTES + n],
                    &(offset..offset + n).collect::<Vec<_>>()
                );
            }
            assert!(p.region(rows, 1).is_err());
            assert!(p.region(1, 1).is_err());
            assert!(p.own(4).is_err());
        }
        assert_eq!(
            Plan::new(true, true, 20000, 256).unwrap().unwrap().stride,
            256 * 1024
        );
        assert_eq!(
            Plan::new(true, true, 20000, 1024).unwrap().unwrap().stride,
            SEND_BYTES
        );
    }
}
