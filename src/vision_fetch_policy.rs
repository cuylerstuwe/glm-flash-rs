//! Public image URL policy. Keep source/query bytes independent of the URL
//! parser's path normalization: signed URLs may depend on their exact spelling.
use anyhow::{Context, Result, ensure};
use std::net::{IpAddr, Ipv6Addr};

#[derive(Clone, Copy, Default)]
pub(crate) struct Policy {
    pub(crate) http: bool,
    pub(crate) private: bool,
}
pub(crate) struct Target {
    pub(crate) uri: String,
    pub(crate) host: String,
    pub(crate) port: u16,
    origin: String,
    resource: String,
}
fn quote(value: &str, query: bool) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric()
            || b"/%:@!$&'()*+,;=-._~".contains(&byte)
            || (query && byte == b'?')
        {
            out.push(byte as char);
        } else {
            use std::fmt::Write;
            write!(out, "%{byte:02X}").expect("write string");
        }
    }
    out
}
impl Target {
    pub(crate) fn parse(value: &str, policy: Policy) -> Result<Self> {
        ensure!(
            value.chars().count() <= 4096
                && !value.chars().any(|c| c <= ' ' || c == '\u{7f}')
                && !value.contains('\\'),
            "image URL length/control policy"
        );
        let (scheme, rest) = value.split_once("://").context("image URL scheme")?;
        let scheme = scheme.to_ascii_lowercase();
        ensure!(
            scheme == "https" || (policy.http && scheme == "http"),
            "image URL scheme policy"
        );
        let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..end];
        ensure!(
            !authority.is_empty() && !authority.contains(['@', '%']),
            "image URL authority policy"
        );
        let (raw_host, explicit_port) = if let Some(ipv6) = authority.strip_prefix('[') {
            let (host, tail) = ipv6.split_once(']').context("image IPv6 authority")?;
            let parsed = host
                .parse::<Ipv6Addr>()
                .map_err(|_| anyhow::anyhow!("image IPv6 authority"))?;
            (
                parsed.to_string(),
                if tail.is_empty() {
                    None
                } else {
                    Some(tail.strip_prefix(':').context("image authority suffix")?)
                },
            )
        } else {
            let (host, port) = authority
                .rsplit_once(':')
                .map_or((authority, None), |(h, p)| (h, Some(p)));
            ensure!(!host.contains(':') && !host.is_empty(), "image host policy");
            // Match IDNA2003's commonly used sharp-s/final-sigma mappings. The
            // vetted UTS46 implementation additionally rejects disallowed host
            // scalars; this difference is explicit in VISION-FETCH.md.
            #[allow(deprecated)]
            let ascii = idna::Config::default()
                .transitional_processing(true)
                .to_ascii(host)
                .map_err(|_| anyhow::anyhow!("image IDNA host policy"))?;
            (ascii.to_ascii_lowercase(), port)
        };
        let default = if scheme == "https" { 443 } else { 80 };
        let port = match explicit_port {
            None | Some("") => default,
            Some(p) if p.bytes().all(|b| b.is_ascii_digit()) => p
                .parse::<u16>()
                .map_err(|_| anyhow::anyhow!("image port range"))?,
            _ => anyhow::bail!("image port syntax"),
        };
        ensure!(policy.http || port == 443, "image HTTPS port policy");
        ensure!(
            policy.private
                || !matches!(
                    raw_host.trim_end_matches('.'),
                    "localhost" | "metadata.google.internal" | "instance-data" | "metadata"
                ),
            "image host must be public"
        );
        let suffix = &rest[end..];
        let suffix = if let Some((head, fragment)) = suffix.split_once('#') {
            ensure!(fragment.is_empty(), "image URL fragment policy");
            head
        } else {
            suffix
        };
        let (raw_path, query) = suffix
            .split_once('?')
            .map_or((suffix, None), |(p, q)| (p, Some(q)));
        let path = quote(if raw_path.is_empty() { "/" } else { raw_path }, false);
        let authority = if raw_host.contains(':') {
            format!("[{raw_host}]")
        } else {
            raw_host.clone()
        };
        let origin = format!("{scheme}://{authority}:{port}");
        let mut uri = format!("{origin}{path}");
        if let Some(q) = query.filter(|q| !q.is_empty()) {
            uri.push('?');
            uri.push_str(&quote(q, true));
        }
        // The HTTP parser validates the final authority and request target. We
        // intentionally do not round-trip the path through WHATWG normalization.
        let _: ureq::http::Uri = uri
            .parse()
            .map_err(|_| anyhow::anyhow!("image request URI policy"))?;
        Ok(Self {
            uri,
            host: raw_host,
            port,
            origin,
            resource: suffix.to_owned(),
        })
    }
    pub(crate) fn redirect(&self, location: &str, policy: Policy) -> Result<Self> {
        ensure!(
            location.chars().count() <= 4096
                && !location.chars().any(|c| c <= ' ' || c == '\u{7f}')
                && !location.contains('\\'),
            "image redirect length/control policy"
        );
        let scheme = self.origin.split(':').next().expect("validated scheme");
        let mut location = location;
        if let Some((candidate, suffix)) = location.split_once(':')
            && candidate
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphabetic)
            && candidate
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
        {
            // Only a leading syntactic scheme is absolute. A URL embedded in a
            // relative query (next=https://...) must remain a query string.
            if suffix.starts_with("//") {
                return Self::parse(location, policy);
            }
            ensure!(
                candidate.eq_ignore_ascii_case(scheme),
                "image redirect scheme policy"
            );
            location = suffix;
        }
        if location.starts_with("//") {
            return Self::parse(&format!("{scheme}:{location}"), policy);
        }
        let location = if let Some((head, fragment)) = location.split_once('#') {
            ensure!(fragment.is_empty(), "image redirect fragment policy");
            head
        } else {
            location
        };
        let (base, old_query) = self
            .resource
            .split_once('?')
            .map_or((self.resource.as_str(), None), |(p, q)| (p, Some(q)));
        let (relative, query) = location
            .split_once('?')
            .map_or((location, None), |(p, q)| (p, Some(q)));
        let (path, params) = split_params(relative);
        let (path, query) = if path.is_empty() && params.is_none() {
            (
                base.to_owned(),
                query.filter(|q| !q.is_empty()).or(old_query),
            )
        } else {
            // Use RFC-style literal dot-segment handling. WHATWG URL joining
            // would additionally normalize %2e and apostrophes in query
            // strings, which can invalidate signed targets.
            let (base, _) = split_params(base);
            let mut segments = if path.starts_with('/') {
                path.split('/').collect::<Vec<_>>()
            } else {
                let mut parts = base.split('/').collect::<Vec<_>>();
                if parts.last().is_some_and(|p| !p.is_empty()) {
                    parts.pop();
                }
                parts.extend(path.split('/'));
                let last = parts.len().saturating_sub(1);
                parts
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, p)| (i == 0 || i == last || !p.is_empty()).then_some(p))
                    .collect()
            };
            let tail_dot = segments.last().is_some_and(|p| *p == "." || *p == "..");
            let mut resolved = Vec::new();
            for segment in segments.drain(..) {
                match segment {
                    ".." => {
                        resolved.pop();
                    }
                    "." => {}
                    p => resolved.push(p),
                }
            }
            if tail_dot {
                resolved.push("");
            }
            let mut path = resolved.join("/");
            if path.is_empty() {
                path.push('/');
            }
            if let Some(params) = params {
                path.push(';');
                path.push_str(params);
            }
            (path, query)
        };
        let slash = if path.starts_with('/') { "" } else { "/" };
        let mut joined = format!("{}{slash}{path}", self.origin);
        if let Some(q) = query.filter(|q| !q.is_empty()) {
            joined.push('?');
            joined.push_str(q);
        }
        Self::parse(&joined, policy)
    }
}
fn split_params(path: &str) -> (&str, Option<&str>) {
    let start = path.rfind('/').map_or(0, |i| i + 1);
    path[start..].find(';').map_or((path, None), |i| {
        let i = start + i;
        (&path[..i], Some(&path[i + 1..]))
    })
}

fn in6(ip: u128, base: [u16; 8], prefix: u32) -> bool {
    let base = u128::from(Ipv6Addr::from(base));
    ip >> (128 - prefix) == base >> (128 - prefix)
}
pub(crate) fn public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => {
            let n = u32::from(v);
            // Reject reserved/private ranges, CGNAT, multicast and metadata endpoints.
            ![
                (0x00000000, 8),
                (0x0a000000, 8),
                (0x64400000, 10),
                (0x7f000000, 8),
                (0xa9fe0000, 16),
                (0xac100000, 12),
                (0xc0000000, 24),
                (0xc0000200, 24),
                (0xc0a80000, 16),
                (0xc6120000, 15),
                (0xc6336400, 24),
                (0xcb007100, 24),
                (0xe0000000, 4),
                (0xf0000000, 4),
                (0xa83f8110, 32),
            ]
            .into_iter()
            .any(|(base, p)| n >> (32 - p) == base >> (32 - p))
        }
        IpAddr::V6(v) => {
            let n = u128::from(v);
            // Reject addresses outside global unicast 2000::/3, including deprecated
            // site-local fec0::/10, plus the disallowed subranges below.
            if !in6(n, [0x2000, 0, 0, 0, 0, 0, 0, 0], 3)
                || in6(n, [0x2002, 0, 0, 0, 0, 0, 0, 0], 16)
                || in6(n, [0x2001, 0xdb8, 0, 0, 0, 0, 0, 0], 32)
            {
                return false;
            }
            if !in6(n, [0x2001, 0, 0, 0, 0, 0, 0, 0], 23) {
                return true;
            }
            [
                ([0x2001, 1, 0, 0, 0, 0, 0, 1], 128),
                ([0x2001, 1, 0, 0, 0, 0, 0, 2], 128),
                ([0x2001, 3, 0, 0, 0, 0, 0, 0], 32),
                ([0x2001, 4, 0x112, 0, 0, 0, 0, 0], 48),
                ([0x2001, 0x20, 0, 0, 0, 0, 0, 0], 28),
                ([0x2001, 0x30, 0, 0, 0, 0, 0, 0], 28),
            ]
            .into_iter()
            .any(|(base, p)| in6(n, base, p))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_targets_and_unicode_host_mapping_are_preserved() {
        let t = Target::parse(
            "HTTPS://faß.de:443/a/../%2e%2e/é.png?sig=a'b%2F?c#",
            Policy::default(),
        )
        .unwrap();
        assert_eq!(t.host, "fass.de");
        assert_eq!(
            t.uri,
            "https://fass.de:443/a/../%2e%2e/%C3%A9.png?sig=a'b%2F?c"
        );
        assert_eq!(
            Target::parse("https://example.com:?", Policy::default())
                .unwrap()
                .uri,
            "https://example.com:443/"
        );
        let t = Target::parse("https://example.com/a/b", Policy::default()).unwrap();
        assert_eq!(
            t.redirect("../c.png", Policy::default()).unwrap().uri,
            "https://example.com:443/c.png"
        );
    }
    #[test]
    fn credentials_local_names_controls_schemes_and_redirects_are_rejected() {
        for url in [
            "https://@example.com/x",
            "https://u:secret@example.com/x",
            "http://example.com/x",
            "https://example.com:444/x",
            "https://example.com/#private",
            "https://example.com/a\\b",
            "https://example.com/a\nb",
            "https://localhost./",
            "https://metadata.google.internal/",
            "https://[fe80::1%25eth0]/",
        ] {
            let e = Target::parse(url, Policy::default())
                .err()
                .unwrap()
                .to_string();
            assert!(!e.contains("secret") && !e.contains("example.com"));
        }
        let t = Target::parse("https://example.com/a", Policy::default()).unwrap();
        assert!(
            t.redirect("//u:secret@other.example/x", Policy::default())
                .is_err()
        );
        assert!(
            t.redirect("http://example.com/x", Policy::default())
                .is_err()
        );
    }
    #[test]
    fn relative_redirects_preserve_frozen_query_and_literal_path_rules() {
        let t = Target::parse("https://example.com/a/b;old?x='y'", Policy::default()).unwrap();
        for (relative, expected) in [
            ("../%2e%2e/c?sig=a'b%2F", "/%2e%2e/c?sig=a'b%2F"),
            ("?", "/a/b;old?x='y'"),
            ("#", "/a/b;old?x='y'"),
            (";new", "/a/;new"),
            ("https:../c", "/c"),
            ("c//d", "/a/c/d"),
            ("/c//d", "/c//d"),
            ("../../../../../d", "/d"),
            (".", "/a/"),
            ("..", "/"),
            (
                "?next=https://other.example/a",
                "/a/b;old?next=https://other.example/a",
            ),
        ] {
            assert_eq!(
                t.redirect(relative, Policy::default()).unwrap().uri,
                format!("https://example.com:443{expected}"),
                "{relative}"
            );
        }
    }
    #[test]
    fn exact_frozen_public_ranges_and_explicit_site_local_correction() {
        for ip in [
            "8.8.8.8",
            "192.0.1.1",
            "168.63.129.15",
            "2606:4700:4700::1111",
            "2001:1::1",
            "2001:3::1",
            "2001:4:112::1",
            "2001:20::1",
            "2001:30::1",
        ] {
            assert!(public(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "0.0.0.0",
            "10.0.0.1",
            "127.1.2.3",
            "169.254.169.254",
            "100.100.100.200",
            "192.0.0.9",
            "192.0.2.1",
            "198.18.0.1",
            "224.0.0.1",
            "240.1.1.1",
            "168.63.129.16",
            "::1",
            "::ffff:8.8.8.8",
            "64:ff9b::808:808",
            "64:ff9b:1::1",
            "2001::1",
            "2001:db8::1",
            "2002::1",
            "fe80::1",
            "fc00::1",
            "fec0::1",
            "ff00::1",
        ] {
            assert!(!public(ip.parse().unwrap()), "{ip}");
        }
    }
}
