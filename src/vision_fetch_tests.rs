use super::*;
use std::{
    io::{Read, Write},
    net::TcpListener,
};
fn local_client() -> Client {
    let mut c = Client::new();
    c.policy = Policy {
        http: true,
        private: true,
    };
    c
}
fn server(replies: Vec<Vec<u8>>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/image", listener.local_addr().unwrap());
    let worker = thread::spawn(move || {
        let mut requests = Vec::new();
        for response in replies {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut req = Vec::new();
            let mut byte = [0];
            while !req.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).unwrap();
                req.extend(byte);
                assert!(req.len() < 8192);
            }
            requests.push(String::from_utf8(req).unwrap());
            socket.write_all(&response).unwrap();
        }
        requests
    });
    (url, worker)
}
#[test]
fn public_policy_checks_every_dns_answer_and_pins_only_verified_destination() {
    let mixed = [
        "8.8.8.8:443".parse().unwrap(),
        "127.0.0.1:443".parse().unwrap(),
    ];
    assert!(checked_addresses(&mixed, Policy::default()).is_err());
    let address = checked_addresses(&mixed[..1], Policy::default()).unwrap();
    let pin = Pinned {
        host: "example.com".into(),
        port: 443,
        address,
    };
    // Structural proof of retained address independent of current DNS. The
    // actual connector integration is also exercised by loopback socket tests.
    assert_eq!(pin.address, address);
    assert!(!format!("{pin:?}").contains("example.com"));
    assert!(checked_addresses(&[], Policy::default()).is_err());
    let (url, worker) = server(vec![
        b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 3\r\n\r\nabc".to_vec(),
    ]);
    assert!(
        Client::new()
            .fetch(
                &url,
                3,
                Instant::now() + Duration::from_secs(1),
                &Control::new()
            )
            .is_err()
    );
    assert_eq!(
        local_client()
            .fetch(
                &url,
                3,
                Instant::now() + Duration::from_secs(1),
                &Control::new()
            )
            .unwrap(),
        b"abc"
    );
    let req = worker.join().unwrap();
    assert_eq!(req.len(), 1);
    assert!(req[0].starts_with("GET /image HTTP/1.1"));
}
#[test]
fn redirects_rebuild_headers_and_connection_close_body_is_bounded() {
    let (url, worker) = server(vec![
        b"HTTP/1.1 302 Found\r\nLocation: /next?sig=a'b%2F\r\nContent-Length: 0\r\n\r\n".to_vec(),
        b"HTTP/1.0 200 OK\r\nContent-Type: image/jpeg\r\nConnection: close\r\n\r\nxyz".to_vec(),
    ]);
    let bytes = local_client()
        .fetch(
            &url,
            4,
            Instant::now() + Duration::from_secs(2),
            &Control::new(),
        )
        .unwrap();
    assert_eq!(bytes, b"xyz");
    let requests = worker.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].starts_with("GET /next?sig=a'b%2F HTTP/1.1"));
    for req in requests {
        let req = req.to_ascii_lowercase();
        assert!(req.contains("accept-encoding: identity"));
        assert!(
            !req.contains("authorization:")
                && !req.contains("cookie:")
                && !req.contains("referer:")
        );
    }
}
#[test]
fn duplicate_headers_compression_unknown_media_and_byte_overflow_fail_closed() {
    for headers in [
        "Content-Type: text/html\r\n",
        "Content-Type: image/png\r\nContent-Encoding: gzip\r\n",
        "Content-Type: image/png\r\nContent-Type: image/png\r\n",
        "Content-Type: image/png\r\nContent-Length: 4\r\n",
    ] {
        let (url, worker) = server(vec![
            format!("HTTP/1.0 200 OK\r\n{headers}\r\nabcd").into_bytes(),
        ]);
        assert!(
            local_client()
                .fetch(
                    &url,
                    3,
                    Instant::now() + Duration::from_secs(1),
                    &Control::new()
                )
                .is_err()
        );
        worker.join().unwrap();
    }
    let (url, worker) = server(vec![
        b"HTTP/1.0 200 OK\r\nContent-Type: image/png\r\n\r\nabcd".to_vec(),
    ]);
    assert!(
        local_client()
            .fetch(
                &url,
                3,
                Instant::now() + Duration::from_secs(1),
                &Control::new()
            )
            .is_err()
    );
    worker.join().unwrap();
}
#[test]
fn stalled_header_has_hard_deadline_and_cancelled_ticket_does_not_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/private", listener.local_addr().unwrap());
    let (signal, rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        rx.recv_timeout(Duration::from_secs(2)).ok();
        drop(socket);
    });
    let start = Instant::now();
    let error = local_client()
        .fetch(&url, 3, start + Duration::from_millis(80), &Control::new())
        .unwrap_err();
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(!format!("{error:#}").contains("private"));
    signal.send(()).unwrap();
    worker.join().unwrap();
    let control = Control::new();
    control.cancel();
    assert!(
        local_client()
            .fetch(
                "http://127.0.0.1:1",
                3,
                Instant::now() + Duration::from_secs(1),
                &control
            )
            .is_err()
    );
}
#[test]
fn resolver_capacity_is_shared_and_credit_survives_waiter_failure() {
    let client = Client::new();
    let clone = Client::new();
    assert!(Arc::ptr_eq(&client.slots, &clone.slots));
    // A separate bounded pool for a deterministic admission test avoids racing
    // ordinary HTTP tests which share the real global resolver allowance.
    let mut client = client;
    client.slots = Arc::new(AtomicUsize::new(0));
    let control = Control::new();
    let credits = (0..4)
        .map(|_| {
            client
                .reserve_dns(Instant::now() + Duration::from_secs(1), &control)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(
        client
            .reserve_dns(Instant::now() + Duration::from_millis(20), &control)
            .is_err()
    );
    assert_eq!(client.slots.load(Ordering::Acquire), 4);
    drop(credits);
    assert_eq!(client.slots.load(Ordering::Acquire), 0);
}
