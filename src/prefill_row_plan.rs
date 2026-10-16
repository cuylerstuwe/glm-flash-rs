//! Exact TP4 ownership of independent HC rows, on the frozen 64-row grid.
use anyhow::{Result, ensure};

pub(crate) const WORLD: usize = 4;
pub(crate) const WIDTH: usize = 4096;
pub(crate) const MAX_ROWS: usize = 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Policy(bool);
impl Policy {
    pub(crate) fn parse(value: Option<&str>) -> Result<Self> {
        ensure!(
            matches!(value, None | Some("0") | Some("1")),
            "ALASKA_PREFILL_ROW_SPLIT must be 0/1"
        );
        Ok(Self(value == Some("1")))
    }
    pub(crate) fn environment() -> Result<Self> {
        match std::env::var("ALASKA_PREFILL_ROW_SPLIT") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(error.into()),
        }
    }
    pub(crate) fn identity(self) -> [u8; 16] {
        let mut out = *b"ALPFROW1\0\0\0\0\0\0\0\0";
        out[8] = u8::from(self.0);
        out
    }
    pub(crate) fn enabled(self) -> bool {
        self.0
    }
    pub(crate) fn applies(
        self,
        rows: usize,
        piece: usize,
        overlap: bool,
        hc_pipeline: bool,
    ) -> Result<bool> {
        ensure!(
            rows > 0 && rows <= 8192 && piece > 0 && piece <= MAX_ROWS && piece.is_multiple_of(64),
            "row split geometry"
        );
        Ok(self.0 && overlap && hc_pipeline && rows > piece)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rows {
    pub first: usize,
    pub count: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TapSpan {
    pub first: usize,
    pub count: usize,
    pub ring: usize,
}
/// At most two contiguous spans for the retained part of one original piece.
pub(crate) fn retained_spans(
    position: usize,
    total: usize,
    offset: usize,
    rows: usize,
    cap: usize,
) -> Result<[Option<TapSpan>; 2]> {
    ensure!(
        matches!(cap, 8 | 16 | 2048)
            && total > 0
            && total <= 8192
            && offset < total
            && rows > 0
            && rows <= total - offset
            && rows <= MAX_ROWS,
        "retained tap piece geometry"
    );
    ensure!(
        position <= 1048576 && total <= 1048576 - position,
        "retained tap context bound"
    );
    let end = offset + rows;
    let first = offset.max(total.saturating_sub(cap)).min(end);
    if first == end {
        return Ok([None, None]);
    }
    let ring = (position + first) % cap;
    let count = (end - first).min(cap - ring);
    Ok([
        Some(TapSpan { first, count, ring }),
        (first + count < end).then_some(TapSpan {
            first: first + count,
            count: end - first - count,
            ring: 0,
        }),
    ])
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Transfer {
    pub peer: usize,
    pub send_offset: usize,
    pub send_bytes: usize,
    pub receive_offset: usize,
    pub receive_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    rows: usize,
    rank: usize,
    bounds: [usize; 5],
}
impl Plan {
    pub(crate) fn new(rows: usize, rank: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && rank < WORLD,
            "row split rows/rank"
        );
        let bounds = std::array::from_fn(|k| {
            if k == WORLD {
                rows
            } else {
                (k * rows).div_ceil(WORLD).next_multiple_of(64).min(rows)
            }
        });
        Ok(Self { rows, rank, bounds })
    }
    pub(crate) fn own(self) -> Rows {
        self.owned(self.rank)
    }
    fn owned(self, rank: usize) -> Rows {
        Rows {
            first: self.bounds[rank],
            count: self.bounds[rank + 1] - self.bounds[rank],
        }
    }
    fn peer(self, index: usize) -> usize {
        if index < self.rank { index } else { index + 1 }
    }
    pub(crate) fn partials(self, capacity: usize) -> Result<[Transfer; 3]> {
        ensure!(
            capacity >= self.rows && capacity <= MAX_ROWS,
            "row split partial capacity"
        );
        let own = self.own();
        Ok(std::array::from_fn(|index| {
            let peer = self.peer(index);
            let theirs = self.owned(peer);
            Transfer {
                peer,
                send_offset: (self.rank * capacity + theirs.first) * WIDTH * 2,
                send_bytes: theirs.count * WIDTH * 2,
                receive_offset: (peer * capacity + own.first) * WIDTH * 2,
                receive_bytes: own.count * WIDTH * 2,
            }
        }))
    }
    pub(crate) fn share(
        self,
        row_bytes: usize,
        base: usize,
        capacity_bytes: usize,
    ) -> Result<[Transfer; 3]> {
        self.share_clipped(row_bytes, base, capacity_bytes, 0, self.rows)
    }
    pub(crate) fn share_clipped(
        self,
        row_bytes: usize,
        base: usize,
        capacity_bytes: usize,
        first: usize,
        count: usize,
    ) -> Result<[Transfer; 3]> {
        ensure!(
            row_bytes > 0 && row_bytes <= 65536 && base <= capacity_bytes,
            "row split share width/base"
        );
        ensure!(
            count > 0 && first < self.rows && count <= self.rows - first,
            "row split share window"
        );
        let length = count
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("row split share size"))?;
        ensure!(length <= capacity_bytes - base, "row split share capacity");
        let clipped = |owned: Rows| {
            let begin = owned.first.max(first).min(first + count);
            let end = (owned.first + owned.count).min(first + count).max(begin);
            Rows {
                first: begin - first,
                count: end - begin,
            }
        };
        let own = clipped(self.own());
        Ok(std::array::from_fn(|index| {
            let peer = self.peer(index);
            let theirs = clipped(self.owned(peer));
            Transfer {
                peer,
                send_offset: base + own.first * row_bytes,
                send_bytes: own.count * row_bytes,
                receive_offset: base + theirs.first * row_bytes,
                receive_bytes: theirs.count * row_bytes,
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_tail_partitions_once_and_cross_rank_bytes_match() -> Result<()> {
        for rows in 1..=MAX_ROWS {
            let all = (0..WORLD)
                .map(|rank| Plan::new(rows, rank))
                .collect::<Result<Vec<_>>>()?;
            let mut ownership = vec![0; rows];
            for plan in &all {
                let own = plan.own();
                assert!(own.first == rows || own.first.is_multiple_of(64));
                for n in &mut ownership[own.first..own.first + own.count] {
                    *n += 1;
                }
                for transfer in plan.partials(MAX_ROWS)? {
                    let reverse = all[transfer.peer]
                        .partials(MAX_ROWS)?
                        .into_iter()
                        .find(|p| p.peer == plan.rank)
                        .unwrap();
                    assert_eq!(transfer.send_bytes, reverse.receive_bytes);
                    assert_eq!(transfer.receive_bytes, reverse.send_bytes);
                    // The logical rank identity remains the outermost stride.
                    assert_eq!(transfer.send_offset / (MAX_ROWS * WIDTH * 2), plan.rank);
                    if transfer.receive_bytes > 0 {
                        assert_eq!(
                            transfer.receive_offset / (MAX_ROWS * WIDTH * 2),
                            transfer.peer
                        );
                    }
                }
                for transfer in plan.share(256, 37, rows * 256 + 37)? {
                    let reverse = all[transfer.peer]
                        .share(256, 37, rows * 256 + 37)?
                        .into_iter()
                        .find(|p| p.peer == plan.rank)
                        .unwrap();
                    assert_eq!(
                        (transfer.send_offset, transfer.send_bytes),
                        (reverse.receive_offset, reverse.receive_bytes)
                    );
                }
            }
            assert!(ownership.iter().all(|n| *n == 1));
        }
        Ok(())
    }
    #[test]
    fn copied_rank_and_row_patterns_reconstruct_only_owned_partials_and_all_shared_rows()
    -> Result<()> {
        for rows in [1, 63, 64, 65, 127, 128, 129, 255, 257, 511, 513, 1023, 1024] {
            let plans = (0..WORLD)
                .map(|rank| Plan::new(rows, rank))
                .collect::<Result<Vec<_>>>()?;
            // Interpret the exact byte offsets without allocating a full-width tensor.
            let mut partials = vec![vec![usize::MAX; WORLD * MAX_ROWS]; WORLD];
            let mut shared = vec![vec![usize::MAX; rows]; WORLD];
            for rank in 0..WORLD {
                for row in 0..rows {
                    partials[rank][rank * MAX_ROWS + row] = rank * MAX_ROWS + row;
                }
                let own = plans[rank].own();
                for (row, value) in shared[rank]
                    .iter_mut()
                    .enumerate()
                    .skip(own.first)
                    .take(own.count)
                {
                    *value = rank * MAX_ROWS + row;
                }
            }
            let source = partials.clone();
            let share_source = shared.clone();
            for rank in 0..WORLD {
                for t in plans[rank].partials(MAX_ROWS)? {
                    let reverse = plans[t.peer]
                        .partials(MAX_ROWS)?
                        .into_iter()
                        .find(|p| p.peer == rank)
                        .unwrap();
                    for n in 0..t.receive_bytes / (WIDTH * 2) {
                        partials[rank][t.receive_offset / (WIDTH * 2) + n] =
                            source[t.peer][reverse.send_offset / (WIDTH * 2) + n];
                    }
                }
                for t in plans[rank].share(1, 0, rows)? {
                    shared[rank][t.receive_offset..t.receive_offset + t.receive_bytes]
                        .copy_from_slice(
                            &share_source[t.peer]
                                [t.receive_offset..t.receive_offset + t.receive_bytes],
                        );
                }
                let own = plans[rank].own();
                for peer in 0..WORLD {
                    for row in 0..MAX_ROWS {
                        let valid = (peer == rank && row < rows)
                            || (own.first..own.first + own.count).contains(&row);
                        assert_eq!(
                            partials[rank][peer * MAX_ROWS + row],
                            if valid {
                                peer * MAX_ROWS + row
                            } else {
                                usize::MAX
                            }
                        );
                    }
                }
                assert!(shared[rank].iter().all(|x| *x != usize::MAX));
            }
            assert!(shared.windows(2).all(|p| p[0] == p[1]));
        }
        Ok(())
    }
    #[test]
    fn default_off_and_invalid_geometry_never_yield_work() -> Result<()> {
        let off = Policy::parse(None)?;
        let on = Policy::parse(Some("1"))?;
        assert!(!off.enabled());
        assert_ne!(off.identity(), on.identity());
        assert!(!off.applies(8192, 512, true, true)?);
        assert!(on.applies(513, 512, true, true)?);
        for (rows, overlap, pipeline) in
            [(512, true, true), (8192, false, true), (8192, true, false)]
        {
            assert!(!on.applies(rows, 512, overlap, pipeline)?);
        }
        for invalid in ["", "true", "2", " 1", "1 "] {
            assert!(Policy::parse(Some(invalid)).is_err());
        }
        for (rows, rank) in [(0, 0), (1025, 0), (1, 4)] {
            assert!(Plan::new(rows, rank).is_err());
        }
        let plan = Plan::new(65, 1)?;
        assert_eq!(
            plan.own(),
            Rows {
                first: 64,
                count: 0
            }
        );
        assert!(plan.partials(64).is_err());
        assert!(plan.partials(1025).is_err());
        for (width, base, capacity) in [
            (0, 0, 1),
            (65537, 0, usize::MAX),
            (1, usize::MAX, usize::MAX),
            (1, 0, 64),
        ] {
            assert!(plan.share(width, base, capacity).is_err());
        }
        Ok(())
    }
    #[test]
    fn clipped_shares_preserve_owner_and_ring_boundaries() -> Result<()> {
        for rows in [1, 63, 64, 65, 129, 257, 511, 512, 513, 1023, 1024] {
            for (first, count) in [(0, rows), (rows - 1, 1), (rows / 2, rows - rows / 2)] {
                for rank in 0..WORLD {
                    let p = Plan::new(rows, rank)?;
                    for t in p.share_clipped(40960, 17, 17 + count * 40960, first, count)? {
                        let reverse = Plan::new(rows, t.peer)?
                            .share_clipped(40960, 17, 17 + count * 40960, first, count)?
                            .into_iter()
                            .find(|r| r.peer == rank)
                            .unwrap();
                        assert_eq!(
                            (t.send_offset, t.send_bytes),
                            (reverse.receive_offset, reverse.receive_bytes)
                        );
                        assert!(t.send_offset + t.send_bytes <= 17 + count * 40960);
                        assert!(t.receive_offset + t.receive_bytes <= 17 + count * 40960);
                    }
                }
            }
        }
        for cap in [8, 16, 2048] {
            for position in [0, 64, 1984, 4096] {
                for total in [1025, 2048, 2049, 8191, 8192] {
                    let mut seen = vec![0; cap];
                    let mut covered = 0;
                    for offset in (0..total).step_by(512) {
                        for s in
                            retained_spans(position, total, offset, (total - offset).min(512), cap)?
                                .into_iter()
                                .flatten()
                        {
                            assert!(s.first >= total.saturating_sub(cap));
                            assert!(s.ring + s.count <= cap);
                            for i in 0..s.count {
                                assert_eq!((position + s.first + i) % cap, s.ring + i);
                                seen[s.ring + i] += 1;
                                covered += 1;
                            }
                        }
                    }
                    assert_eq!(covered, total.min(cap));
                    assert!(seen.iter().all(|n| *n <= 1));
                }
            }
        }
        assert!(retained_spans(1048576, 1, 0, 1, 2048).is_err());
        assert!(retained_spans(0, 64, 63, 2, 2048).is_err());
        let p = Plan::new(64, 0)?;
        for (first, count) in [(64, 1), (63, 2), (0, 0)] {
            assert!(p.share_clipped(2, 0, 128, first, count).is_err());
        }
        Ok(())
    }
}
