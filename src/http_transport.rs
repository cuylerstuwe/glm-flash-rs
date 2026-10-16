//! Strict bounded HTTP/1.1, one request per connection. Header reads do not
//! consume body bytes; inference admission must precede those reads.
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{Shutdown, TcpStream},
    os::fd::AsRawFd,
    time::{Duration, Instant},
};
pub(crate) const HEADER_BYTES: usize = 16 << 10;
pub(crate) struct Head {
    pub method: String,
    pub path: String,
    pub headers: BTreeMap<String, String>,
    pub length: usize,
}
impl Head {
    pub fn header(&self, name: &str) -> &str {
        self.headers.get(name).map(String::as_str).unwrap_or("")
    }
}
pub(crate) fn parse(bytes: &[u8]) -> Result<Head> {
    ensure!(bytes.len() <= HEADER_BYTES, "header byte limit");
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut req = httparse::Request::new(&mut headers);
    ensure!(
        req.parse(bytes)?.is_complete(),
        "incomplete request headers"
    );
    ensure!(req.version == Some(1), "HTTP/1.1 required");
    let method = req.method.context("missing method")?.to_owned();
    let path = req.path.context("missing path")?.to_owned();
    ensure!(
        path.starts_with('/') && path.len() <= 2048 && !path.contains(['?', '#']),
        "unsupported request target"
    );
    let mut map = BTreeMap::new();
    for h in req.headers.iter() {
        let name = h.name.to_ascii_lowercase();
        let value = std::str::from_utf8(h.value)?.trim().to_owned();
        ensure!(map.insert(name, value).is_none(), "duplicate header");
    }
    ensure!(map.contains_key("host"), "Host required");
    ensure!(
        !map.contains_key("transfer-encoding"),
        "transfer encoding unsupported"
    );
    ensure!(!map.contains_key("expect"), "Expect unsupported");
    let length = if let Some(n) = map.get("content-length") {
        ensure!(
            !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()),
            "invalid Content-Length"
        );
        n.parse::<usize>()?
    } else {
        ensure!(method != "POST", "Content-Length required");
        0
    };
    ensure!(length <= crate::chat_json::BODY_BYTES, "body byte limit");
    ensure!(
        method == "POST" || length == 0,
        "body unsupported for method"
    );
    Ok(Head {
        method,
        path: path.trim_end_matches('/').to_owned(),
        headers: map,
        length,
    })
}
fn remaining(end: Instant) -> Result<Duration> {
    let left = end.saturating_duration_since(Instant::now());
    ensure!(!left.is_zero(), "network read deadline");
    Ok(left)
}
pub(crate) fn head(socket: &mut TcpStream, timeout: Duration) -> Result<Head> {
    let end = Instant::now() + timeout;
    let mut bytes = Vec::with_capacity(HEADER_BYTES);
    let mut scratch = [0u8; 4096];
    loop {
        socket.set_read_timeout(Some(remaining(end)?))?;
        let n = socket.peek(&mut scratch)?;
        ensure!(n != 0, "disconnected during headers");
        let old = bytes.len();
        let take = n.min(HEADER_BYTES - old);
        ensure!(take > 0, "header byte limit");
        bytes.extend_from_slice(&scratch[..take]);
        let marker = bytes[old.saturating_sub(3)..]
            .windows(4)
            .position(|b| b == b"\r\n\r\n")
            .map(|n| old.saturating_sub(3) + n + 4);
        if let Some(n) = marker {
            bytes.truncate(n);
        }
        socket.read_exact(&mut bytes[old..])?;
        if marker.is_some() {
            return parse(&bytes);
        }
    }
}
pub(crate) fn body(socket: &mut TcpStream, length: usize, timeout: Duration) -> Result<Vec<u8>> {
    ensure!(length <= crate::chat_json::BODY_BYTES, "body byte limit");
    let end = Instant::now() + timeout;
    let mut bytes = vec![0; length];
    let mut at = 0;
    while at < length {
        socket.set_read_timeout(Some(remaining(end)?))?;
        let n = socket.read(&mut bytes[at..])?;
        ensure!(n > 0, "disconnected during body");
        at += n;
    }
    Ok(bytes)
}
/// Nonblocking peek without changing flags/timeouts shared by cloned sockets.
pub(crate) fn gone(socket: &TcpStream) -> bool {
    let mut byte = 0u8;
    let n = unsafe {
        libc::recv(
            socket.as_raw_fd(),
            (&mut byte as *mut u8).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    n == 0
        || (n < 0
            && !matches!(
                std::io::Error::last_os_error().kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ))
}
pub(crate) fn json(socket: &mut TcpStream, status: u16, value: &serde_json::Value) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    write!(
        socket,
        "HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    )?;
    socket.write_all(&bytes)?;
    // This is a terminal, one-response connection. Push queued response bytes
    // and FIN before the socket owner closes. Early rejects may still have
    // unread request bytes: a bare close can reset TCP and discard a body held
    // by Nagle after the headers. Never wait for an unauthorized/oversized body.
    socket.shutdown(Shutdown::Write)?;
    Ok(())
}
pub(crate) fn event(socket: &mut TcpStream, value: &serde_json::Value) -> Result<()> {
    socket.write_all(b"data: ")?;
    serde_json::to_writer(&mut *socket, value)?;
    socket.write_all(b"\n\n")?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_framing_rejects_ambiguity() {
        for h in [
            "Content-Length: 1\r\nContent-Length: 1\r\n",
            "Content-Length: +1\r\n",
            "Content-Length: 999999999999999999999999\r\n",
            "Transfer-Encoding: chunked\r\n",
            "Content-Length: 16777217\r\n",
            "Content-Length: 0\r\nExpect: 100-continue\r\n",
        ] {
            assert!(
                parse(
                    format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{h}\r\n")
                        .as_bytes()
                )
                .is_err()
            );
        }
        assert_eq!(parse(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 9\r\n\r\n").unwrap().length,9);
    }
    #[test]
    fn header_reader_leaves_body_unconsumed() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut peer = TcpStream::connect(listener.local_addr()?)?;
        let (mut server, _) = listener.accept()?;
        peer.write_all(b"POST /x HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\n\r\nBODY")?;
        assert_eq!(head(&mut server, Duration::from_secs(1))?.length, 4);
        let mut b = [0; 4];
        assert_eq!(server.peek(&mut b)?, 4);
        assert_eq!(&b, b"BODY");
        Ok(())
    }
    #[test]
    fn terminal_json_survives_close_with_unread_request() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let value =
            serde_json::json!({"error":{"message":"connection capacity","type":"server_error"}});
        let body = serde_json::to_vec(&value)?;
        let mut expected = format!("HTTP/1.1 503 Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).into_bytes();
        expected.extend(&body);
        for _ in 0..128 {
            let mut peer = TcpStream::connect(listener.local_addr()?)?;
            peer.set_read_timeout(Some(Duration::from_secs(1)))?;
            // Deliberately retain Nagle and never consume the inbound headers.
            peer.write_all(
                b"GET /health HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
            )?;
            let (mut server, _) = listener.accept()?;
            json(&mut server, 503, &value)?;
            drop(server);
            let mut got = vec![0; expected.len()];
            peer.read_exact(&mut got)?;
            assert_eq!(got, expected);
        }
        Ok(())
    }
}
