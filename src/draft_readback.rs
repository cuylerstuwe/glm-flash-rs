//! Optional bounded, reusable pinned staging for the drafter's two small
//! result buffers. The same bytes are consumed after the same stream boundary;
//! no sampling, graph, target arithmetic or precision changes.
use crate::cuda::{Buffer, Device, Pinned, Stream};
use anyhow::{Result, ensure};
use std::rc::Rc;

pub(crate) fn enabled() -> Result<bool> {
    match std::env::var("ALASKA_DRAFT_READBACK") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        _ => anyhow::bail!("ALASKA_DRAFT_READBACK must be 0/1"),
    }
}

pub(crate) struct Pair {
    device: Rc<Device>,
    host: Pinned,
    first_bytes: usize,
    second_bytes: usize,
    in_flight: bool,
}
impl Pair {
    pub(crate) fn new(device: &Rc<Device>, first: usize, second: usize) -> Result<Self> {
        ensure!(
            first > 0 && second > 0 && first.checked_add(second).is_some_and(|n| n <= 65536),
            "draft readback bound"
        );
        Ok(Self {
            device: device.clone(),
            host: Pinned::new(device, first + second)?,
            first_bytes: first,
            second_bytes: second,
            in_flight: false,
        })
    }
    /// Copies are ordered after the producer on this stream. No CPU access or
    /// reuse occurs until it drains. A failed second enqueue still owns the
    /// first destination until Drop drains it.
    pub(crate) fn copy(
        &mut self,
        stream: &Stream,
        first: &Buffer,
        second: &Buffer,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        ensure!(!self.in_flight, "failed draft readback cannot be reused");
        ensure!(
            first.bytes >= self.first_bytes && second.bytes == self.second_bytes,
            "draft readback geometry"
        );
        self.in_flight = true;
        unsafe {
            self.host
                .download_from(stream, 0, first, 0, self.first_bytes)?;
            self.host
                .download_from(stream, self.first_bytes, second, 0, self.second_bytes)?;
        }
        stream.sync()?;
        self.in_flight = false;
        let bytes = self.host.bytes_mut();
        Ok((
            bytes[..self.first_bytes].to_vec(),
            bytes[self.first_bytes..].to_vec(),
        ))
    }
}
impl Drop for Pair {
    fn drop(&mut self) {
        if self.in_flight {
            let _ = self.device.sync();
        }
    }
}

pub(crate) fn probe() -> Result<serde_json::Value> {
    let device = Device::new()?;
    let stream = device.stream()?;
    let mut cases = Vec::new();
    for (a, b) in [
        (1, 1),
        (4096, 3584),
        (4 * 7 * crate::candidates::payload_bytes(16)?, 7 * 256 * 2),
    ] {
        let first = device.alloc(a + 32)?;
        let second = device.alloc(b)?;
        let mut pair = Pair::new(&device, a, b)?;
        for turn in 0..8usize {
            let x: Vec<u8> = (0..a + 32)
                .map(|i| (i.wrapping_mul(31) + turn * 17) as u8)
                .collect();
            let y: Vec<u8> = (0..b)
                .map(|i| (i.wrapping_mul(19) + turn * 37) as u8)
                .collect();
            first.write(0, &x)?;
            second.write(0, &y)?;
            if turn % 2 == 0 {
                unsafe {
                    first.zero_prefix(&stream, a)?;
                }
            }
            let got = pair.copy(&stream, &first, &second)?;
            let expected = if turn % 2 == 0 {
                vec![0; a]
            } else {
                x[..a].to_vec()
            };
            ensure!(
                got.0 == expected && got.1 == y,
                "pinned draft copy mismatch"
            );
            ensure!(first.read(a, 32)? == x[a..], "source tail changed");
        }
        cases.push(serde_json::json!({"first_bytes":a,"second_bytes":b,"reuses":8,"exact":true,"producer_ordered":true}));
    }
    let other = Device::new()?;
    let first = device.alloc(256)?;
    let second = other.alloc(128)?;
    let mut failed = Pair::new(&device, 256, 128)?;
    ensure!(
        failed.copy(&stream, &first, &second).is_err() && failed.in_flight,
        "partial enqueue must retain destination"
    );
    ensure!(
        failed.copy(&stream, &first, &second).is_err(),
        "failed owner reused"
    );
    drop(failed);
    stream.sync()?;
    ensure!(
        Pair::new(&device, usize::MAX, 2).is_err(),
        "readback bound overflow"
    );
    Ok(
        serde_json::json!({"passed":true,"cases":cases,"partial_enqueue_drained":true,"failed_owner_rejected":true,"scope":"pinned pair component only; model and performance qualification separate"}),
    )
}
