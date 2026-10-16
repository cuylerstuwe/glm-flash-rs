//! Bounded HTTPS image retrieval. Rust owns redirects, checked DNS addresses,
//! source limits and cancellation; ureq/rustls own HTTP framing and verified TLS.
use crate::{
    vision_fetch_policy::{Policy, Target, public},
    vision_request::Control,
};
use anyhow::{Context, Result, ensure};
use std::{
    io::Read,
    net::{SocketAddr, ToSocketAddrs},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use ureq::{
    Agent,
    tls::{RootCerts, TlsConfig},
    unversioned::{
        resolver::{ResolvedSocketAddrs, Resolver},
        transport::{DefaultConnector, NextTimeout},
    },
};
const DNS_SLOTS: usize = 4;
static RESOLVERS: OnceLock<Arc<AtomicUsize>> = OnceLock::new();
const POLL: Duration = Duration::from_millis(10);
const MEDIA: [&str; 4] = ["image/jpeg", "image/png", "image/webp", "image/gif"];
fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|v| !v.is_zero())
        .context("image download timed out")
}
fn cancelled(control: &Control) -> Result<()> {
    ensure!(!control.status()?.2, "image preparation cancelled");
    Ok(())
}

struct DnsCredit(Arc<AtomicUsize>);
impl Drop for DnsCredit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
#[derive(Clone)]
pub(crate) struct Client {
    slots: Arc<AtomicUsize>,
    policy: Policy,
    timeout: Duration,
    #[cfg(test)]
    roots: Option<RootCerts>,
}
impl Client {
    pub(crate) fn new() -> Self {
        Self {
            slots: RESOLVERS
                .get_or_init(|| Arc::new(AtomicUsize::new(0)))
                .clone(),
            policy: Policy::default(),
            timeout: Duration::from_secs(10),
            #[cfg(test)]
            roots: None,
        }
    }
    #[cfg(test)]
    pub(crate) fn local_test(timeout: Duration) -> Self {
        assert!(!timeout.is_zero() && timeout <= Duration::from_secs(10));
        let mut client = Self::new();
        client.policy = Policy {
            http: true,
            private: true,
        };
        client.timeout = timeout;
        client
    }
    fn reserve_dns(&self, deadline: Instant, control: &Control) -> Result<DnsCredit> {
        loop {
            cancelled(control)?;
            let time = remaining(deadline)?;
            let n = self.slots.load(Ordering::Acquire);
            if n < DNS_SLOTS
                && self
                    .slots
                    .compare_exchange(n, n + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                return Ok(DnsCredit(self.slots.clone()));
            }
            thread::sleep(time.min(POLL));
        }
    }
    fn resolve(&self, target: &Target, deadline: Instant, control: &Control) -> Result<SocketAddr> {
        let credit = self.reserve_dns(deadline, control)?;
        let host = target.host.clone();
        let port = target.port;
        let (tx, rx) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("alaska-image-dns".into())
            .spawn(move || {
                let _credit = credit;
                // A blocked libc lookup retains one of four global credits until
                // its real thread exits. It owns only this bounded host string and
                // address receipt, never image buffers or a model request lease.
                let result = (host.as_str(), port)
                    .to_socket_addrs()
                    .map_err(|_| ())
                    .and_then(|iter| {
                        let addrs = iter.take(257).collect::<Vec<_>>();
                        if addrs.len() > 256 {
                            Err(())
                        } else {
                            Ok(addrs)
                        }
                    });
                let _ = tx.send(result);
            })
            .map_err(|_| anyhow::anyhow!("image DNS worker unavailable"))?;
        loop {
            cancelled(control)?;
            match rx.recv_timeout(remaining(deadline)?.min(POLL)) {
                Ok(Ok(addrs)) => return checked_addresses(&addrs, self.policy),
                Ok(Err(())) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("image host could not be resolved")
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
    /// One already-admitted fetch may finish its bounded network operation
    /// while held. Holds are observed before/after the operation, retaining the
    /// same job/result. Cancellation is checked between DNS waits/body chunks;
    /// blocked connect/handshake/header/read work has the global socket timeout.
    pub(crate) fn fetch(
        &self,
        url: &str,
        max_bytes: usize,
        deadline: Instant,
        control: &Control,
    ) -> Result<Vec<u8>> {
        ensure!(
            max_bytes <= crate::vision_decode::MAX_ENCODED_BYTES,
            "image byte policy"
        );
        // The frozen ten-second per-image budget includes DNS and every
        // redirect, within the caller's shared thirty-second request budget.
        let deadline = deadline.min(Instant::now() + self.timeout);
        let mut target = Target::parse(url, self.policy)?;
        for redirects in 0..=3 {
            cancelled(control)?;
            let address = self.resolve(&target, deadline, control)?;
            let timeout = remaining(deadline)?.min(Duration::from_secs(10));
            let roots = RootCerts::PlatformVerifier;
            #[cfg(test)]
            let roots = self.roots.clone().unwrap_or(roots);
            let config = Agent::config_builder()
                .proxy(None)
                .max_redirects(0)
                .http_status_as_error(false)
                .https_only(!self.policy.http)
                .timeout_global(Some(timeout))
                .max_response_header_size(128 << 10)
                .input_buffer_size(64 << 10)
                .output_buffer_size(16 << 10)
                .tls_config(TlsConfig::builder().root_certs(roots).build())
                .user_agent("tensorfold-glm53-vision/2")
                .build();
            let agent = Agent::with_parts(
                config,
                DefaultConnector::default(),
                Pinned {
                    host: target.host.clone(),
                    port: target.port,
                    address,
                },
            );
            // Deliberately fresh agent and explicit headers on every hop: no
            // cookies, authentication, proxy environment or origin credentials
            // can leak across redirects. TLS still verifies the original host.
            let mut response = agent
                .get(&target.uri)
                .header("Accept", MEDIA.join(", "))
                .header("Accept-Encoding", "identity")
                .call()
                .map_err(|_| anyhow::anyhow!("image download failed or timed out"))?;
            cancelled(control)?;
            remaining(deadline)?;
            let status = response.status().as_u16();
            if [301, 302, 303, 307, 308].contains(&status) {
                ensure!(redirects < 3, "image download has too many redirects");
                let location = single_header(response.headers(), "location")?
                    .context("image redirect has no destination")?;
                ensure!(!location.is_empty(), "image redirect has no destination");
                target = target.redirect(location, self.policy)?;
                continue;
            }
            ensure!(
                status == 200,
                "image download returned a non-success status"
            );
            validate_headers(response.headers(), max_bytes)?;
            let mut result = Vec::new();
            result
                .try_reserve_exact(max_bytes)
                .context("image download allocation")?;
            let mut buffer = [0u8; 64 << 10];
            let mut reader = response.body_mut().as_reader();
            loop {
                cancelled(control)?;
                remaining(deadline)?;
                let allowed = (max_bytes + 1 - result.len()).min(buffer.len());
                let n = reader
                    .read(&mut buffer[..allowed])
                    .map_err(|_| anyhow::anyhow!("image body failed or timed out"))?;
                if n == 0 {
                    break;
                }
                ensure!(n <= max_bytes - result.len(), "image download byte bound");
                result.extend_from_slice(&buffer[..n]);
            }
            cancelled(control)?;
            remaining(deadline)?;
            return Ok(result);
        }
        anyhow::bail!("image download failed")
    }
}
fn checked_addresses(addrs: &[SocketAddr], policy: Policy) -> Result<SocketAddr> {
    ensure!(
        !addrs.is_empty() && addrs.len() <= 256,
        "image address count bound"
    );
    ensure!(
        policy.private || addrs.iter().all(|a| public(a.ip())),
        "image URLs must resolve only to public internet addresses"
    );
    Ok(addrs[0]) // Frozen path uses the first checked address, without another DNS lookup.
}
fn single_header<'a>(headers: &'a ureq::http::HeaderMap, name: &str) -> Result<Option<&'a str>> {
    let mut values = headers.get_all(name).iter();
    let first = values.next();
    ensure!(values.next().is_none(), "duplicate image response header");
    first
        .map(|v| {
            v.to_str()
                .map_err(|_| anyhow::anyhow!("invalid image response header"))
        })
        .transpose()
}
fn validate_headers(headers: &ureq::http::HeaderMap, max_bytes: usize) -> Result<()> {
    ensure!(
        single_header(headers, "content-encoding")?
            .unwrap_or("identity")
            .eq_ignore_ascii_case("identity"),
        "compressed HTTP image responses are unsupported"
    );
    let media = single_header(headers, "content-type")?
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    ensure!(
        MEDIA.contains(&media.as_str()),
        "image content type must be JPEG, PNG, WebP or GIF"
    );
    if let Some(length) = single_header(headers, "content-length")? {
        ensure!(
            !length.is_empty() && length.bytes().all(|b| b.is_ascii_digit()),
            "image content length syntax"
        );
        ensure!(
            length.parse::<usize>().is_ok_and(|n| n <= max_bytes),
            "image content length bound"
        );
    }
    Ok(())
}
struct Pinned {
    host: String,
    port: u16,
    address: SocketAddr,
}
impl std::fmt::Debug for Pinned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PinnedImageAddress")
    }
}
impl Resolver for Pinned {
    fn resolve(
        &self,
        uri: &ureq::http::Uri,
        _: &ureq::config::Config,
        _: NextTimeout,
    ) -> std::result::Result<ResolvedSocketAddrs, ureq::Error> {
        let host = uri
            .host()
            .unwrap_or("")
            .trim_start_matches('[')
            .trim_end_matches(']');
        if host != self.host || uri.port_u16().unwrap_or(443) != self.port {
            return Err(ureq::Error::HostNotFound);
        }
        let mut addrs = self.empty();
        addrs.push(self.address);
        Ok(addrs)
    }
}

#[cfg(test)]
#[path = "vision_fetch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "vision_fetch_tls_tests.rs"]
mod tls_tests;
