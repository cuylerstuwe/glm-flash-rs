//! Bounded image bytes under a retained native preparation. Errors never expose
//! URLs. An admitted network operation completes once, retaining its result on hold.
use anyhow::{Context, Result, ensure};
use base64::{
    Engine, alphabet,
    engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig},
};

pub(crate) fn load(
    url: &str,
    fetch: Option<&crate::vision_fetch::Client>,
    deadline: std::time::Instant,
    control: &crate::vision_request::Control,
) -> Result<Vec<u8>> {
    control.check()?;
    let bytes = if url.starts_with("data:") {
        inline(url, crate::vision_decode::MAX_ENCODED_BYTES)
    } else {
        fetch.context("image URL fetching is disabled")?.fetch(
            url,
            crate::vision_decode::MAX_ENCODED_BYTES,
            deadline,
            control,
        )
    };
    // Do not return even an error through a native hold: retain that exact
    // result until release/cancellation, without reconnecting or refetching.
    control.check()?;
    bytes
}

/// Decode data URLs with permissive ASCII base64 and
/// percent decoding. The immutable source string remains caller-owned.
pub(crate) fn inline(url: &str, max_bytes: usize) -> Result<Vec<u8>> {
    ensure!(
        max_bytes <= crate::vision_decode::MAX_ENCODED_BYTES,
        "inline image byte policy"
    );
    ensure!(url.starts_with("data:"), "expected inline image data URL");
    let (head, data) = url
        .split_once(',')
        .context("image data URL missing comma")?;
    if head.contains(";base64") {
        decode_base64(data, max_bytes)
    } else {
        decode_percent(data, max_bytes)
    }
}

fn decode_base64(data: &str, max_bytes: usize) -> Result<Vec<u8>> {
    ensure!(data.is_ascii(), "image base64 must be ASCII");
    ensure!(
        data.len() <= max_bytes * 4 / 3 + 8,
        "inline image encoded byte bound"
    );
    let mut normalized = Vec::new();
    normalized
        .try_reserve_exact(data.len())
        .context("inline image allocation")?;
    let mut position = 0usize;
    let mut pads = 0;
    // Python's validate=False discards non-alphabet bytes, ignores premature
    // pads, accepts nonzero trailing bits, and ignores data after completed
    // padding. Normalize only that syntax; the vetted crate performs decoding.
    for byte in data.bytes() {
        if byte == b'=' {
            if position >= 2 {
                pads += 1;
                if position + pads >= 4 {
                    normalized.extend(std::iter::repeat_n(b'=', 4 - position));
                    position = 0;
                    break;
                }
            }
        } else if byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/' {
            normalized.push(byte);
            position = (position + 1) % 4;
            pads = 0;
        }
    }
    ensure!(position == 0, "invalid image base64 padding");
    let engine = GeneralPurpose::new(
        &alphabet::STANDARD,
        GeneralPurposeConfig::new()
            .with_decode_padding_mode(DecodePaddingMode::RequireCanonical)
            .with_decode_allow_trailing_bits(true),
    );
    let raw = engine.decode(&normalized).context("invalid image base64")?;
    ensure!(raw.len() <= max_bytes, "inline image decoded byte bound");
    Ok(raw)
}

fn decode_percent(data: &str, max_bytes: usize) -> Result<Vec<u8>> {
    ensure!(
        data.len() <= max_bytes * 3,
        "inline image encoded byte bound"
    );
    let mut raw = Vec::new();
    raw.try_reserve_exact(data.len().min(max_bytes))
        .context("inline image allocation")?;
    let bytes = data.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        ensure!(raw.len() < max_bytes, "inline image decoded byte bound");
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(high), Some(low)) = (hex(bytes[i + 1]), hex(bytes[i + 2]))
        {
            raw.push(high * 16 + low);
            i += 3;
        } else {
            raw.push(bytes[i]);
            i += 1;
        }
    }
    Ok(raw)
}
fn hex(v: u8) -> Option<u8> {
    match v {
        b'0'..=b'9' => Some(v - b'0'),
        b'a'..=b'f' => Some(v - b'a' + 10),
        b'A'..=b'F' => Some(v - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serving_source_keeps_completed_fetch_during_hold_without_refetch() {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            time::{Duration, Instant},
        };
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/image", server.local_addr().unwrap());
        let control = crate::vision_request::Control::new();
        let worker_control = control.clone();
        let worker = std::thread::spawn(move || {
            let client = crate::vision_fetch::Client::local_test(Duration::from_secs(2));
            load(
                &url,
                Some(&client),
                Instant::now() + Duration::from_secs(3),
                &worker_control,
            )
        });
        let (mut socket, _) = server.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut headers = Vec::new();
        while !headers.ends_with(b"\r\n\r\n") {
            let mut b = [0];
            socket.read_exact(&mut b).unwrap();
            headers.extend(b);
        }
        control.hold(true).unwrap();
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 3\r\n\r\nabc",
            )
            .unwrap();
        let end = Instant::now() + Duration::from_secs(2);
        while !control.status().unwrap().1 {
            assert!(Instant::now() < end);
            std::thread::yield_now();
        }
        assert!(!worker.is_finished());
        control.hold(false).unwrap();
        assert_eq!(worker.join().unwrap().unwrap(), b"abc");
        server.set_nonblocking(true).unwrap();
        assert!(server.accept().is_err());
    }
    #[test]
    fn disabled_fetch_still_allows_inline_and_cancellation_rejects_before_work() {
        let control = crate::vision_request::Control::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        assert_eq!(load("data:,abc", None, deadline, &control).unwrap(), b"abc");
        let e = load("https://secret.invalid/private", None, deadline, &control).unwrap_err();
        assert_eq!(e.to_string(), "image URL fetching is disabled");
        control.cancel();
        assert!(load("data:,abc", None, deadline, &control).is_err());
    }
    #[test]
    fn permissive_ascii_base64_preserves_frozen_padding_and_trailing_rules() {
        for (encoded, expected) in [
            ("", &b""[..]),
            ("Zg==", &b"f"[..]),
            ("Zg==ignored", &b"f"[..]),
            ("====Zg====", &b"f"[..]),
            ("Z?m_9-v!", &b"foo"[..]),
            ("Zg= =", &b"f"[..]),
            ("Zg=9v", &b"f\x0fo"[..]),
            ("Zh==", &b"f"[..]),
            ("Zm9=", &b"fo"[..]),
            ("Zm9v===", &b"foo"[..]),
        ] {
            assert_eq!(
                inline(&format!("data:opaque;base64,{encoded}"), 64).unwrap(),
                expected,
                "{encoded}"
            );
        }
        for encoded in ["Z", "Zg", "Zg=", "Zm9", "Zg=Z", "Zg==é"] {
            assert!(
                inline(&format!("data:;base64,{encoded}"), 64).is_err(),
                "{encoded}"
            );
        }
    }
    #[test]
    fn percent_utf8_invalid_escapes_and_plus_follow_unquote_to_bytes() {
        assert_eq!(
            inline("data:any,%00%fF+%gg%4é", 64).unwrap(),
            b"\0\xff+%gg%4\xc3\xa9"
        );
        // The base64 marker is case-sensitive and does not require a MIME type.
        assert_eq!(inline("data:;BASE64,Zg==", 64).unwrap(), b"Zg==");
        assert_eq!(inline("data:,", 0).unwrap(), b"");
    }
    #[test]
    fn bounds_and_errors_do_not_disclose_reference_or_encoded_payload() {
        assert!(inline("data:;base64,Zm9v", 2).is_err());
        assert!(inline("data:,%61%62%63", 2).is_err());
        assert!(inline("data:,abc", 2).is_err());
        for url in [
            "https://secret.invalid/token",
            "data:private",
            "data:;base64,private",
        ] {
            let error = inline(url, 2).unwrap_err().to_string();
            assert!(!error.contains("secret") && !error.contains("private"));
        }
    }
}
