use super::*;
use crate::{vision_markup::Markup, vision_request::Budget};
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::mpsc,
};
const PNG: &[u8] = include_bytes!("../native/vision-decode/fixtures/color2-depth8-plain.png");

fn refs(urls: &[String]) -> Vec<Reference> {
    let mut messages = serde_json::json!([{"role":"user","content": urls.iter().map(|url|
        serde_json::json!({"type":"image_url","image_url":url,"detail":"low"})).collect::<Vec<_>>() }]);
    let marked = Markup::mark(&mut messages).unwrap();
    let rendered = messages[0]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["text"].as_str().unwrap())
        .collect::<String>();
    marked.splice(rendered).unwrap().references
}
fn start(sources: Sources) -> (Arc<Budget>, Job) {
    let limits = Limits {
        encoded: sources.encoded_charge(),
        scratch: 384 << 20,
        retained: 2 << 20,
    };
    let budget = Budget::new(limits.total().unwrap()).unwrap();
    let lease = budget.reserve(31, limits.total().unwrap()).unwrap();
    let job = Job::spawn_sources(lease, sources, super::tests::settings(), None, limits).unwrap();
    (budget, job)
}
fn await_state(job: &mut Job, held: bool) -> Result<Option<Ready>> {
    super::tests::poll_until(job, held)
}
// One real request, gated after part of its body. The listener never accepts a
// replacement connection and its request receipt proves the original URL only.
fn gated() -> (
    String,
    mpsc::Receiver<()>,
    mpsc::Sender<bool>,
    JoinHandle<String>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!(
        "http://{}/signed?private=fixture",
        listener.local_addr().unwrap()
    );
    let (sent, partial) = mpsc::channel();
    let (finish, release) = mpsc::channel();
    let worker = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            socket.read_exact(&mut byte).unwrap();
            request.extend(byte);
            assert!(request.len() < 8192);
        }
        write!(
            socket,
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\r\n",
            PNG.len()
        )
        .unwrap();
        socket.write_all(&PNG[..PNG.len() / 2]).unwrap();
        sent.send(()).unwrap();
        if release.recv_timeout(Duration::from_secs(3)).unwrap() {
            socket.write_all(&PNG[PNG.len() / 2..]).unwrap();
        }
        String::from_utf8(request).unwrap()
    });
    (url, partial, finish, worker)
}
#[test]
fn source_job_hold_during_http_retains_one_fetch_and_exact_image() {
    let (url, partial, release, server) = gated();
    let mut sources = Sources::new(refs(&[url]), 64, 16, true).unwrap();
    sources.client = Some(Client::local_test(Duration::from_secs(2)));
    let (budget, mut job) = start(sources);
    partial.recv_timeout(Duration::from_secs(2)).unwrap();
    job.hold(true).unwrap();
    let charged = budget.used().unwrap();
    release.send(true).unwrap();
    assert!(await_state(&mut job, true).unwrap().is_none());
    assert_eq!(budget.used().unwrap(), charged);
    assert!(job.control.status().unwrap().1); // worker stopped after fetched bytes
    thread::sleep(Duration::from_millis(30));
    job.hold(false).unwrap();
    let ready = await_state(&mut job, false).unwrap().unwrap();
    assert_eq!(ready.ticket(), 31);
    assert_eq!(ready.images[0].max_tokens, 16);
    assert_eq!(
        ready.images[0].encoded_sha256,
        <[u8; 32]>::from(Sha256::digest(PNG))
    );
    let rgb = vision_decode::png(PNG, 384 << 20, || Ok(())).unwrap();
    let cold = Prepared::from_rgb(
        Plan::new(rgb.height, rgb.width, 16).unwrap(),
        rgb.rgb,
        &super::tests::settings(),
    )
    .unwrap();
    assert_eq!(ready.images[0].prepared.key(), cold.key());
    assert!(
        server
            .join()
            .unwrap()
            .starts_with("GET /signed?private=fixture HTTP/1.1")
    );
    drop(ready);
    drop(job);
    assert_eq!(budget.used().unwrap(), 0);
}
#[test]
fn remote_timeout_while_held_is_retained_error_without_cancel_or_refetch() {
    let (url, partial, release, server) = gated();
    let mut sources = Sources::new(refs(&[url]), 64, 16, true).unwrap();
    sources.client = Some(Client::local_test(Duration::from_millis(100)));
    let (budget, mut job) = start(sources);
    partial.recv_timeout(Duration::from_secs(2)).unwrap();
    job.hold(true).unwrap();
    let charged = budget.used().unwrap();
    assert!(await_state(&mut job, true).unwrap().is_none());
    assert!(!job.control.status().unwrap().2);
    assert!(job.thread.is_none()); // failed network operation actually exited
    assert_eq!(budget.used().unwrap(), charged);
    release.send(false).unwrap();
    server.join().unwrap();
    job.hold(false).unwrap();
    let error = job.poll().err().unwrap();
    assert!(!format!("{error:#}").contains("private"));
    assert!(!format!("{error:#}").contains("cancel"));
    assert_eq!(budget.used().unwrap(), 0);
}
#[test]
fn cancellation_during_blocked_read_drains_before_reusing_reservation() {
    let (url, partial, release, server) = gated();
    let mut sources = Sources::new(refs(&[url]), 64, 16, true).unwrap();
    sources.client = Some(Client::local_test(Duration::from_millis(150)));
    let (budget, mut job) = start(sources);
    partial.recv_timeout(Duration::from_secs(2)).unwrap();
    let charged = budget.used().unwrap();
    job.cancel();
    assert!(matches!(job.poll().unwrap(), Poll::Pending));
    assert_eq!(budget.used().unwrap(), charged);
    assert!(budget.reserve(32, 1).is_err());
    assert!(await_state(&mut job, false).is_err());
    assert_eq!(budget.used().unwrap(), 0);
    release.send(false).unwrap();
    server.join().unwrap();
}
#[test]
fn inline_source_and_disabled_network_keep_typed_budget_and_capacity_guards() {
    use base64::Engine;
    let data = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(PNG)
    );
    let sources = Sources::new(refs(&[data]), 0, 16, false).unwrap();
    let (budget, mut job) = start(sources);
    let ready = await_state(&mut job, false).unwrap().unwrap();
    assert_eq!(ready.images[0].max_tokens, 16);
    drop(ready);
    drop(job);
    assert_eq!(budget.used().unwrap(), 0);
    let sources = Sources::new(
        refs(&["https://example.invalid/private".into()]),
        0,
        0,
        false,
    )
    .unwrap();
    let (budget, mut job) = start(sources);
    assert!(await_state(&mut job, false).is_err());
    assert_eq!(budget.used().unwrap(), 0);
    let mut bytes = Vec::with_capacity(vision_decode::MAX_ENCODED_BYTES + 1);
    bytes.push(1);
    assert!(Encoded::new(bytes, 16).is_err());
    let mut oversized = String::with_capacity(crate::chat_json::BODY_BYTES + 1);
    oversized.push_str("data:,A");
    let mut references = refs(&["data:,A".into()]);
    references[0].url = oversized;
    assert!(Sources::new(references, 0, 0, false).is_err());
}
