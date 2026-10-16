//! Optional graph-captured shared/routed expert concurrency. Each branch keeps
//! its existing kernels, buffers and arithmetic; only their launch dependency
//! changes. The caller must supply disjoint writable scratch and join before
//! reusing any input or consuming either result.
use crate::cuda::{Device, Fence, Stream};
use anyhow::{Result, ensure};
use std::rc::Rc;

pub(crate) fn enabled() -> Result<bool> {
    match std::env::var("ALASKA_SHARED_EXPERT_OVERLAP") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        _ => anyhow::bail!("ALASKA_SHARED_EXPERT_OVERLAP must be 0/1"),
    }
}

pub(crate) struct Branch {
    // Stream drains before the fences or caller-owned scratch can be freed.
    stream: Stream,
    fork: Fence,
    joined: Fence,
}
impl Branch {
    pub(crate) fn configured(
        device: &Rc<Device>,
        comm: &crate::comm::Nccl,
        main: &Stream,
    ) -> Result<Option<Self>> {
        let parsed = enabled();
        let mut record = [0u8; 16];
        record[..8].copy_from_slice(b"ALSHRD01");
        record[8] = u8::from(parsed.is_ok());
        record[9] = u8::from(*parsed.as_ref().unwrap_or(&false));
        let all = comm.exchange_host(device, main, &record)?;
        ensure!(
            record[8] == 1 && all.iter().all(|r| r == &record),
            "shared expert overlap policy differs/invalid on a rank"
        );
        let owner = if parsed? {
            (|| {
                Ok(Some(Self {
                    stream: device.stream()?,
                    fork: device.fence()?,
                    joined: device.fence()?,
                }))
            })()
        } else {
            Ok(None)
        };
        record[10] = u8::from(owner.is_ok());
        let all = comm.exchange_host(device, main, &record)?;
        ensure!(
            record[10] == 1 && all.iter().all(|r| r == &record),
            "shared expert overlap setup failed on a rank"
        );
        owner
    }

    pub(crate) fn run(
        &mut self,
        main: &Stream,
        shared: impl FnOnce(&Stream) -> Result<()>,
        routed: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        self.fork.record(main)?;
        self.fork.wait_on(&self.stream)?;
        let work = shared(&self.stream).and_then(|_| routed());
        // Join even after a partially submitted branch fails. During capture
        // this closes the fork before capture cleanup; eager teardown drains
        // the stream before caller scratch is released.
        let joined = self
            .joined
            .record(&self.stream)
            .and_then(|_| self.joined.wait_on(main));
        work?;
        joined
    }
}
