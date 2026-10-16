//! Native-owner progress, observed by the supervisor outside CUDA ownership.
use anyhow::{Context, Result, ensure};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::{Duration, Instant},
};
pub(crate) struct Watch {
    last: Duration,
    stamp: Option<u64>,
}
impl Watch {
    pub fn new() -> Self {
        Self {
            last: Duration::ZERO,
            stamp: None,
        }
    }
    pub fn observe(&mut self, now: Duration, stamp: Option<u64>) -> bool {
        if let Some(stamp) = stamp
            && self.stamp.is_none_or(|old| stamp > old)
        {
            self.stamp = Some(stamp);
            self.last = now;
        }
        now.saturating_sub(self.last)
            >= Duration::from_secs(if self.stamp.is_some() { 60 } else { 180 })
    }
}
pub(crate) fn heartbeat(address: SocketAddr, key: Option<&str>) -> Result<u64> {
    let mut socket = TcpStream::connect_timeout(&address, Duration::from_secs(1))?;
    socket.set_read_timeout(Some(Duration::from_secs(1)))?;
    socket.set_write_timeout(Some(Duration::from_secs(1)))?;
    let auth = key.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
    write!(
        socket,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n"
    )?;
    let mut bytes = Vec::new();
    let end = Instant::now() + Duration::from_secs(1);
    let mut chunk = [0; 8192];
    loop {
        let left = end.saturating_duration_since(Instant::now());
        ensure!(!left.is_zero(), "health read deadline");
        socket.set_read_timeout(Some(left))?;
        let n = socket.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        ensure!(bytes.len() + n <= 131072, "service health byte bound");
        bytes.extend_from_slice(&chunk[..n]);
    }
    ensure!(bytes.len() <= 131072, "service health byte bound");
    let split = bytes
        .windows(4)
        .position(|v| v == b"\r\n\r\n")
        .context("health HTTP header")?;
    ensure!(bytes.starts_with(b"HTTP/1.1 200 "), "service not ready");
    let v: serde_json::Value = serde_json::from_slice(&bytes[split + 4..])?;
    ensure!(v["broker"]["available"] == true, "native owner unavailable");
    v["broker"]["native"]["observed_ns"]
        .as_u64()
        .context("native heartbeat missing")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_holds_can_progress_but_stalls_expire() {
        let mut w = Watch::new();
        assert!(!w.observe(Duration::from_secs(179), None));
        assert!(w.observe(Duration::from_secs(180), None));
        assert!(!w.observe(Duration::from_secs(181), Some(1)));
        assert!(!w.observe(Duration::from_secs(240), Some(1)));
        assert!(w.observe(Duration::from_secs(241), Some(1)));
        assert!(!w.observe(Duration::from_secs(242), Some(2)));
        assert!(!w.observe(Duration::from_secs(280), None));
    }
}
