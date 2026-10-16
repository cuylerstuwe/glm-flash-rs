//! Real socket/TLS checks of the production fetch implementation. Private roots
//! and loopback permission exist only in cfg(test); no system trust is changed.
use super::*;
use rustls::{ServerConfig, ServerConnection, StreamOwned, pki_types::PrivateKeyDer};
use std::{fs, io::Write, net::TcpListener, path::PathBuf, process::Command};

struct Certificates {
    dir: PathBuf,
    root: Vec<u8>,
    valid: Arc<ServerConfig>,
    wrong_host: Arc<ServerConfig>,
}
impl Drop for Certificates {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
impl Certificates {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "alaska-test-only-tls-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let run = |args: &[&str]| {
            let out = Command::new("openssl")
                .current_dir(&dir)
                .args(args)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        run(&[
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-keyout",
            "root.key",
            "-out",
            "root.pem",
            "-days",
            "2",
            "-subj",
            "/CN=Alaska isolated test CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        ]);
        let mut configs = Vec::new();
        for (name, host) in [("valid", "localhost"), ("wrong", "wrong.invalid")] {
            let key = format!("{name}.key");
            let csr = format!("{name}.csr");
            let cert = format!("{name}.der");
            let pkcs = format!("{name}-key.der");
            fs::write(dir.join("extensions"), format!(
                "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:{host}\n"
            )).unwrap();
            run(&[
                "req",
                "-new",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-nodes",
                "-keyout",
                &key,
                "-out",
                &csr,
                "-subj",
                "/CN=Alaska isolated test server",
            ]);
            run(&[
                "x509",
                "-req",
                "-in",
                &csr,
                "-CA",
                "root.pem",
                "-CAkey",
                "root.key",
                "-CAcreateserial",
                "-out",
                &cert,
                "-outform",
                "DER",
                "-days",
                "1",
                "-extfile",
                "extensions",
            ]);
            run(&[
                "pkcs8", "-topk8", "-nocrypt", "-in", &key, "-out", &pkcs, "-outform", "DER",
            ]);
            let cfg = ServerConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(
                vec![fs::read(dir.join(cert)).unwrap().into()],
                PrivateKeyDer::Pkcs8(fs::read(dir.join(pkcs)).unwrap().into()),
            )
            .unwrap();
            configs.push(Arc::new(cfg));
        }
        Self {
            root: fs::read(dir.join("root.pem")).unwrap(),
            dir,
            valid: configs.remove(0),
            wrong_host: configs.remove(0),
        }
    }
    fn client(&self, trusted: bool) -> Client {
        let mut client = Client::local_test(Duration::from_secs(2));
        if trusted {
            client.roots = Some(RootCerts::new_with_certs(&[
                ureq::tls::Certificate::from_pem(&self.root).unwrap(),
            ]));
        }
        client
    }
}
fn serve(config: Arc<ServerConfig>) -> (String, thread::JoinHandle<(bool, String, String)>) {
    // Match the first OS address, as the production client deliberately pins
    // the first checked DNS answer. localhost may prefer IPv6 on either host.
    let address = ("localhost", 0).to_socket_addrs().unwrap().next().unwrap();
    let listener = TcpListener::bind(address).unwrap();
    let port = listener.local_addr().unwrap().port();
    let worker = thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "TLS client never connected");
                    thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("TLS listener: {e}"),
            }
        };
        socket.set_nonblocking(false).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut conn = ServerConnection::new(config).unwrap();
        while conn.is_handshaking() {
            if conn.complete_io(&mut socket).is_err() {
                return (false, String::new(), String::new());
            }
        }
        let host = conn.server_name().unwrap_or("").to_string();
        let mut stream = StreamOwned::new(conn, socket);
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            if stream.read_exact(&mut byte).is_err() {
                return (false, host, String::new());
            }
            request.push(byte[0]);
            assert!(request.len() < 8192);
        }
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc").unwrap();
        stream.flush().unwrap();
        stream.conn.send_close_notify();
        let _ = stream.flush();
        (true, host, String::from_utf8(request).unwrap())
    });
    (
        format!("https://localhost:{port}/image?sig=private-test-value"),
        worker,
    )
}

#[test]
fn silent_tls_peer_obeys_the_same_fetch_deadline() {
    let address = ("localhost", 0).to_socket_addrs().unwrap().next().unwrap();
    let listener = TcpListener::bind(address).unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let worker = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        let socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("TLS listener: {e}"),
            }
        };
        // Retain the accepted peer through the client's complete deadline,
        // without sending any TLS bytes or relying on a connection close.
        thread::sleep(Duration::from_millis(500));
        drop(socket);
    });
    let start = Instant::now();
    let error = Client::local_test(Duration::from_millis(100))
        .fetch(
            &format!("https://localhost:{port}/image"),
            3,
            start + Duration::from_secs(1),
            &Control::new(),
        )
        .unwrap_err();
    assert_eq!(error.to_string(), "image download failed or timed out");
    assert!(start.elapsed() >= Duration::from_millis(80));
    assert!(start.elapsed() < Duration::from_millis(450));
    worker.join().unwrap();
}
#[test]
fn verified_tls_retains_hostname_and_rejects_unknown_ca_and_wrong_host() {
    let certs = Certificates::new();
    for (trusted, valid) in [(true, true), (false, true), (true, false)] {
        let (url, worker) = serve(if valid {
            certs.valid.clone()
        } else {
            certs.wrong_host.clone()
        });
        let result = certs.client(trusted).fetch(
            &url,
            3,
            Instant::now() + Duration::from_secs(3),
            &Control::new(),
        );
        let (served, sni, request) = worker.join().unwrap();
        if trusted && valid {
            assert_eq!(result.unwrap(), b"abc");
            assert!(served);
            assert_eq!(sni, "localhost");
            assert!(request.starts_with("GET /image?sig=private-test-value HTTP/1.1\r\n"));
            assert!(request.to_ascii_lowercase().contains("host: localhost:"));
        } else {
            assert!(!served);
            let error = result.unwrap_err().to_string();
            assert_eq!(error, "image download failed or timed out");
            assert!(!error.contains("private-test-value"));
        }
    }
}
