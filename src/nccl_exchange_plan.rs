//! Bounded byte-copy plan for sequence-parallel prefill. No CUDA dependency.
use anyhow::{Result, ensure};
pub(super) const MAX_TRANSFERS: usize = 64;
const MAX_GROUP_BYTES: usize = 128 * 1024 * 1024;
#[derive(Clone, Copy, Default)]
pub(super) struct Region {
    pub address: u64,
    pub capacity: usize,
    pub offset: usize,
    pub bytes: usize,
}
#[derive(Clone, Copy, Default)]
pub(super) struct Transfer {
    pub peer: usize,
    pub send: Region,
    pub receive: Region,
}
impl Region {
    fn range(self) -> Result<(u64, u64)> {
        ensure!(
            self.address != 0
                && self.offset <= self.capacity
                && self.bytes <= self.capacity - self.offset,
            "NCCL exchange buffer range"
        );
        let begin = self.address.checked_add(self.offset as u64);
        let end = begin.and_then(|v| v.checked_add(self.bytes as u64));
        ensure!(
            begin.is_some() && end.is_some(),
            "NCCL exchange address overflow"
        );
        Ok((begin.unwrap(), end.unwrap()))
    }
}
fn overlaps(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 != a.1 && b.0 != b.1 && a.0 < b.1 && b.0 < a.1
}
pub(super) fn validate(transfers: &[Transfer], rank: usize, world: usize) -> Result<()> {
    ensure!(world == 4 && rank < world, "NCCL exchange ranks");
    ensure!(
        !transfers.is_empty() && transfers.len() <= MAX_TRANSFERS,
        "NCCL exchange descriptor bound"
    );
    let mut bytes = 0usize;
    for (i, transfer) in transfers.iter().enumerate() {
        ensure!(
            transfer.peer < world && transfer.peer != rank,
            "NCCL exchange peer"
        );
        let send = transfer.send.range()?;
        let receive = transfer.receive.range()?;
        bytes = bytes
            .checked_add(transfer.send.bytes)
            .and_then(|v| v.checked_add(transfer.receive.bytes))
            .unwrap_or(usize::MAX);
        ensure!(bytes <= MAX_GROUP_BYTES, "NCCL exchange byte bound");
        // NCCL may execute operations concurrently. A receive must not destroy
        // any send operand or overlap another receive, including aliases.
        ensure!(!overlaps(send, receive), "NCCL exchange aliases its send");
        for other in &transfers[..i] {
            ensure!(
                !overlaps(receive, other.receive.range()?),
                "NCCL exchange overlapping receives"
            );
            ensure!(
                !overlaps(receive, other.send.range()?) && !overlaps(send, other.receive.range()?),
                "NCCL exchange send/receive overlap"
            );
        }
    }
    ensure!(bytes > 0, "NCCL exchange empty group");
    Ok(())
}
/// Once started, drain all native descriptors and end the group even after a
/// native call reports an error. No fallible Rust work may occur in `operation`.
pub(super) fn grouped(
    calls: usize,
    start: impl FnOnce() -> i32,
    mut operation: impl FnMut(usize) -> i32,
    end: impl FnOnce() -> i32,
) -> i32 {
    let mut status = start();
    if status != 0 {
        return status;
    }
    for i in 0..calls {
        let current = operation(i);
        if status == 0 {
            status = current;
        }
    }
    let finished = end();
    if status == 0 { finished } else { status }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn region(address: u64, offset: usize, bytes: usize) -> Region {
        Region {
            address,
            capacity: 4096,
            offset,
            bytes,
        }
    }
    fn plan() -> Vec<Transfer> {
        (1..4)
            .map(|peer| Transfer {
                peer,
                send: region(10000, peer * 128, 128),
                receive: region(20000, peer * 256, 64),
            })
            .collect()
    }
    #[test]
    fn unequal_peer_slices_and_empty_owner_ranges_are_valid() {
        validate(&plan(), 0, 4).unwrap();
        let mut p = plan();
        p[1].receive.bytes = 0;
        p[2].send.bytes = 0;
        validate(&p, 0, 4).unwrap();
        // Sharing own rows from one buffer to each peer is legal; the peers'
        // disjoint rows are received into that same buffer in place.
        for (i, entry) in p.iter_mut().enumerate() {
            entry.send = region(10000, 0, 128);
            entry.receive = region(10000, (i + 1) * 128, 128);
        }
        validate(&p, 0, 4).unwrap();
    }
    #[test]
    fn bad_bounds_aliases_peers_and_excessive_groups_reject_before_native_work() {
        let changes: [fn(&mut Vec<Transfer>); 8] = [
            |p| p[0].send.offset = 4097,
            |p| p[0].receive.bytes = 4096,
            |p| p[0].send.address = u64::MAX,
            |p| p[0].peer = 0,
            |p| p[0].peer = 4,
            |p| p[1].receive = p[0].receive,
            |p| p[1].receive = p[2].send,
            |p| p[0].receive = p[0].send,
        ];
        for change in changes {
            let mut p = plan();
            change(&mut p);
            assert!(validate(&p, 0, 4).is_err());
        }
        assert!(validate(&[], 0, 4).is_err());
        assert!(validate(&vec![plan()[0]; MAX_TRANSFERS + 1], 0, 4).is_err());
        let mut p = plan();
        for x in &mut p {
            x.send.bytes = 0;
            x.receive.bytes = 0;
        }
        assert!(validate(&p, 0, 4).is_err());
        let mut p = plan();
        p[0].send.capacity = MAX_GROUP_BYTES + 1;
        p[0].send.offset = 0;
        p[0].send.bytes = MAX_GROUP_BYTES + 1;
        assert!(validate(&p, 0, 4).is_err());
    }
    #[test]
    fn native_errors_still_close_started_groups_and_keep_first_failure() {
        use std::cell::RefCell;
        for (start, fail_at, finish, expected) in [
            (0, None, 0, 0),
            (7, None, 0, 7),
            (0, Some(2), 9, 3),
            (0, None, 9, 9),
        ] {
            let trace = RefCell::new(Vec::new());
            let status = grouped(
                6,
                || {
                    trace.borrow_mut().push(100);
                    start
                },
                |i| {
                    trace.borrow_mut().push(i);
                    if Some(i) == fail_at { 3 } else { 0 }
                },
                || {
                    trace.borrow_mut().push(200);
                    finish
                },
            );
            assert_eq!(status, expected);
            assert_eq!(
                *trace.borrow(),
                if start == 0 {
                    vec![100, 0, 1, 2, 3, 4, 5, 200]
                } else {
                    vec![100]
                }
            );
        }
    }
}
