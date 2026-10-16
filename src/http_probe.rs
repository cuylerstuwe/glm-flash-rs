//! Real loopback socket qualification, reusable with the simulated CPU owner or
//! the actual four-rank CUDA owner. Test keys are public fixtures, never live keys.
use crate::{
    chat::{Policy, Preparer},
    http_api::{Config, Server},
    output::Stream,
    output_decode::Table,
    request_broker::{Action, Broker, Command, Receipt},
    request_queue::End,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::Path,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
const API: &str = "test-http-api-key-000000";
const CONTROL: &str = "test-control-key-000000";
pub(crate) fn start(path: &Path, broker: Broker, table: Arc<Table>) -> Result<Server> {
    let mut config = Config::new(Some(API), CONTROL)?;
    config.body_timeout = Duration::from_millis(300);
    config.header_timeout = Duration::from_secs(1);
    config.request_timeout = Duration::from_secs(60);
    Server::start(
        config,
        broker,
        Preparer::new(
            path,
            Policy {
                thinking: false,
                effort: Some("high"),
                clear: Some(false),
            },
        )?,
        table,
    )
}
pub(crate) fn open(
    addr: SocketAddr,
    method: &str,
    path: &str,
    extra: &str,
    bytes: &[u8],
) -> Result<TcpStream> {
    let mut c = TcpStream::connect(addr)?;
    c.set_read_timeout(Some(Duration::from_secs(30)))?;
    c.set_write_timeout(Some(Duration::from_secs(2)))?;
    write!(
        c,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n{extra}\r\n",
        bytes.len()
    )?;
    c.write_all(bytes)?;
    Ok(c)
}
fn read(mut c: TcpStream) -> Result<(u16, String)> {
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        c.read_exact(&mut byte)
            .with_context(|| format!("response header after {} bytes", bytes.len()))?;
        bytes.push(byte[0]);
        ensure!(bytes.len() <= 16384, "response headers too large");
    }
    let head = String::from_utf8(bytes)?;
    let code = head.split_whitespace().nth(1).context("status")?.parse()?;
    let length = head
        .lines()
        .find_map(|l| l.strip_prefix("Content-Length: "))
        .map(str::parse::<usize>)
        .transpose()?;
    let mut body = Vec::new();
    if let Some(length) = length {
        ensure!(length <= 64 << 20, "response byte limit");
        body.resize(length, 0);
        c.read_exact(&mut body)
            .context("response Content-Length body")?;
    } else {
        c.take(64 << 20)
            .read_to_end(&mut body)
            .context("response until EOF")?;
    }
    Ok((code, String::from_utf8(body)?))
}
pub(crate) fn call(
    addr: SocketAddr,
    method: &str,
    path: &str,
    extra: &str,
    value: &Value,
) -> Result<(u16, String)> {
    let bytes = if method == "POST" {
        serde_json::to_vec(value)?
    } else {
        vec![]
    };
    let socket = open(addr, method, path, extra, &bytes)
        .with_context(|| format!("probe opening {method} {path} body={value}"))?;
    read(socket).with_context(|| format!("probe {method} {path} body={value}"))
}
pub(crate) fn auth() -> String {
    format!("Authorization: Bearer {API}\r\n")
}
pub(crate) fn chat() -> Value {
    json!({"messages":[{"role":"user","content":"What is 2 + 2? Answer with one numeral."}],"temperature":0,"max_tokens":8,"return_token_ids":true})
}
pub(crate) fn control(addr: SocketAddr, epoch: u64, held: bool) -> Result<Value> {
    let (status, text) = call(
        addr,
        "POST",
        "/v1/scheduler",
        &format!("X-Scheduler-Token: {CONTROL}\r\n"),
        &json!({"owner":"http-probe","epoch":epoch,"foreground":held,"lease_seconds":30}),
    )?;
    ensure!(status == 200, "control status {status}: {text}");
    let v: Value = serde_json::from_str(&text)?;
    ensure!(
        v["foreground_active"] == held && v["epoch"] == epoch,
        "native control receipt"
    );
    Ok(v)
}
pub(crate) fn empty(broker: &Broker) -> Result<()> {
    let end = Instant::now() + Duration::from_secs(5);
    while broker.status()["clients"] != 0 {
        ensure!(Instant::now() < end, "HTTP client/ticket did not retire");
        thread::sleep(Duration::from_millis(10));
    }
    Ok(())
}
pub(crate) fn role(c: &mut TcpStream) -> Result<()> {
    let mut bytes = Vec::new();
    loop {
        let mut b = [0];
        c.read_exact(&mut b).context("SSE initial role")?;
        bytes.push(b[0]);
        ensure!(bytes.len() < 4096, "role event byte bound");
        if bytes.ends_with(b"\n\n") && bytes.windows(9).any(|s| s == b"assistant") {
            return Ok(());
        }
    }
}
pub(crate) fn events(text: &str) -> Result<Vec<Value>> {
    ensure!(text.ends_with("data: [DONE]\n\n"), "SSE did not terminate");
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|s| *s != "[DONE]")
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
pub(crate) fn exercise(server: Server, broker: &Broker, tools: bool) -> Result<Value> {
    let addr = server.address;
    let mut stalled = Vec::new();
    for _ in 0..32 {
        let mut c = TcpStream::connect(addr)?;
        c.write_all(b"GET /health HTTP/1.1\r\nHo")?;
        stalled.push(c);
    }
    let end = Instant::now() + Duration::from_millis(500);
    while server.connections() < 32 {
        ensure!(Instant::now() < end, "connection cap fixture did not fill");
        thread::sleep(Duration::from_millis(5));
    }
    ensure!(
        call(addr, "GET", "/health", &auth(), &Value::Null)?.0 == 503,
        "connection capacity not enforced"
    );
    drop(stalled);
    let end = Instant::now() + Duration::from_secs(3);
    while server.connections() != 0 {
        ensure!(Instant::now() < end, "stalled headers did not retire");
        thread::sleep(Duration::from_millis(5));
    }
    let headers = auth();
    let mut verified = vec!["bounded_connections"];
    ensure!(
        call(addr, "GET", "/health", "", &Value::Null)?.0 == 401,
        "unauthenticated health"
    );
    ensure!(
        call(addr, "GET", "/v1/models", &headers, &Value::Null)?.0 == 200,
        "models unavailable"
    );
    ensure!(
        call(addr, "GET", "/metrics", "", &Value::Null)?.0 == 401,
        "unauthenticated metrics"
    );
    let (code, metrics) = call(addr, "GET", "/metrics", &headers, &Value::Null)?;
    ensure!(
        code == 200
            && metrics.contains("# TYPE tensorfold_requests_total counter")
            && metrics.contains("tensorfold_completion_tokens_total"),
        "metrics unavailable"
    );
    let (code, health) = call(addr, "GET", "/health", &headers, &Value::Null)?;
    let health: Value = serde_json::from_str(&health)?;
    ensure!(
        code == 200
            && health["backend"] == "alaska-rust"
            && health["requests_total"].is_number()
            && health["context_length"] == 1_048_576,
        "health accounting unavailable"
    );
    verified.push("authenticated_health_and_metrics");
    ensure!(
        call(
            addr,
            "POST",
            "/v1/scheduler",
            &headers,
            &json!({"owner":"x","epoch":1,"foreground":true})
        )?
        .0 == 401,
        "wrong control credential accepted"
    );
    for epoch in [json!(true), json!(-1), json!(1.5), json!("3")] {
        ensure!(
            call(
                addr,
                "POST",
                "/v1/scheduler",
                &format!("X-Scheduler-Token: {CONTROL}\r\n"),
                &json!({"owner":"x","epoch":epoch,"foreground":true})
            )?
            .0 == 400,
            "invalid epoch accepted"
        );
    }
    for modify in [
        json!({"temperature":true}),
        json!({"response_format":{"type":"json_object"}}),
        json!({"max_tokens":131073}),
    ] {
        let mut b = chat();
        for (k, v) in modify.as_object().unwrap() {
            b[k] = v.clone();
        }
        ensure!(
            call(addr, "POST", "/v1/chat/completions", &headers, &b)?.0 == 400,
            "unsupported request accepted"
        );
    }
    empty(broker)?;
    verified.push("auth_and_unsupported_fields");
    let raw_text =
        "[gMASK]<sop><|user|>What is 2 + 2? Answer with one numeral.<|assistant|><think></think>";
    let (status, text) = call(
        addr,
        "POST",
        "/v1/tokenize",
        &headers,
        &json!({"prompt":raw_text}),
    )?;
    ensure!(status == 200, "raw tokenize status");
    let tokenized: Value = serde_json::from_str(&text)?;
    let (status, text) = call(addr, "POST", "/tokenize", &headers, &chat())?;
    ensure!(status == 200, "chat tokenize status");
    let chat_tokens: Value = serde_json::from_str(&text)?;
    ensure!(
        tokenized == chat_tokens && tokenized["count"].as_u64().unwrap() > 0,
        "chat/raw tokenization differs"
    );
    for prompt in [json!(raw_text), tokenized["tokens"].clone()] {
        let body = json!({"prompt":prompt,"temperature":0,"max_tokens":8,"return_token_ids":true,"n":1,"draft":false});
        let (status, text) = call(addr, "POST", "/v1/completions", &headers, &body)?;
        ensure!(status == 200, "raw completion status: {text}");
        let reply: Value = serde_json::from_str(&text)?;
        ensure!(
            reply["object"] == "text_completion"
                && reply["choices"][0]["text"] == "4"
                && reply["tensorfold"]["token_ids"] == json!([19, 154827]),
            "raw completion output"
        );
        let mut stream = body;
        stream["stream"] = json!(true);
        stream["stream_options"] = json!({"include_usage":true});
        let (status, text) = call(addr, "POST", "/v1/completions", &headers, &stream)?;
        ensure!(status == 200, "raw SSE status");
        let ev = events(&text)?;
        let content: String = ev
            .iter()
            .filter_map(|v| v["choices"][0]["text"].as_str())
            .collect();
        ensure!(
            content == "4"
                && ev.iter().all(|e| e["object"] == "text_completion")
                && ev.last().unwrap()["usage"]["completion_tokens"] == 2,
            "raw SSE output"
        );
    }
    for prompt in [
        json!([]),
        json!([true]),
        json!([19.0]),
        json!([-1]),
        json!([154879]),
        json!([[19]]),
    ] {
        ensure!(
            call(
                addr,
                "POST",
                "/v1/completions",
                &headers,
                &json!({"prompt":prompt,"max_tokens":8})
            )?
            .0 == 400,
            "bad raw prompt accepted"
        );
    }
    empty(broker)?;
    verified.push("raw_string_ids_json_sse_and_tokenize");
    let mut b = chat();
    b["_localai_mode"] = json!("background");
    b["_localai_request_id"] = json!("forged");
    let (status, text) = call(
        addr,
        "POST",
        "/v1/chat/completions",
        &(headers.clone() + "X-LocalAI-Mode: foreground\r\nX-Request-Id: authoritative\r\n"),
        &b,
    )?;
    ensure!(status == 200, "chat status {status}: {text}");
    let reply: Value = serde_json::from_str(&text)?;
    ensure!(
        reply["choices"][0]["message"]["content"] == "4"
            && reply["choices"][0]["finish_reason"] == "stop"
            && reply["usage"]["completion_tokens"] == 2
            && reply["tensorfold"]["token_ids"] == json!([19, 154827]),
        "chat reply changed"
    );
    ensure!(
        reply["tensorfold"]["mode"] == "foreground"
            && reply["tensorfold"]["request_id"] == "authoritative"
            && reply["usage"]["prompt_tokens_details"]["cached_tokens"] == 0,
        "forged identity/cache usage"
    );
    let mut b = chat();
    b["stream"] = json!(true);
    b["stream_options"] = json!({"include_usage":true});
    let (status, text) = call(addr, "POST", "/v1/chat/completions", &headers, &b)?;
    ensure!(status == 200, "SSE status");
    let chunks = events(&text)?;
    let end = chunks.last().unwrap();
    let content: String = chunks
        .iter()
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
        .collect();
    ensure!(
        content == "4"
            && chunks[0]["choices"][0]["delta"]["role"] == "assistant"
            && end["choices"][0]["finish_reason"] == "stop"
            && end["usage"]["completion_tokens"] == 2,
        "SSE reply"
    );
    ensure!(
        chunks
            .iter()
            .all(|c| c["id"] == chunks[0]["id"] && c["created"] == chunks[0]["created"]),
        "SSE identity changed"
    );
    verified.push("json_sse_identity_usage");
    for (stop, limit, used, finish, content) in [
        (json!("4"), 8, 1, "stop", Value::Null),
        (Value::Null, 1, 1, "length", json!("4")),
    ] {
        let mut b = chat();
        b["stop"] = stop;
        b["max_tokens"] = json!(limit);
        let (status, text) = call(addr, "POST", "/v1/chat/completions", &headers, &b)?;
        ensure!(status == 200, "terminal HTTP status");
        let v: Value = serde_json::from_str(&text)?;
        ensure!(
            v["choices"][0]["message"]["content"] == content
                && v["choices"][0]["finish_reason"] == finish
                && v["usage"]["completion_tokens"] == used,
            "terminal HTTP outcome"
        );
    }
    verified.push("native_stop_and_length");
    // Native lease receipt arrives before held request submission/streaming.
    empty(broker)?;
    control(addr, 1, true)?;
    let mut b = chat();
    b["stream"] = json!(true);
    let mut background = open(
        addr,
        "POST",
        "/v1/chat/completions",
        &(headers.clone() + "X-LocalAI-Mode: background\r\nX-Request-Id: duplicate\r\n"),
        &serde_json::to_vec(&b)?,
    )?;
    role(&mut background)?;
    ensure!(
        call(
            addr,
            "POST",
            "/v1/chat/completions",
            &(headers.clone() + "X-Request-Id: duplicate\r\n"),
            &chat()
        )?
        .0 == 409,
        "duplicate active ID accepted"
    );
    ensure!(
        call(addr, "POST", "/v1/chat/completions", &headers, &chat())?.0 == 200,
        "foreground held by background"
    );
    drop(background);
    empty(broker)?;
    control(addr, 2, false)?;
    ensure!(
        call(
            addr,
            "POST",
            "/v1/scheduler",
            &format!("X-Scheduler-Token: {CONTROL}\r\n"),
            &json!({"owner":"http-probe","epoch":1,"foreground":true})
        )?
        .0 == 400,
        "stale native epoch not rejected"
    );
    ensure!(
        call(
            addr,
            "POST",
            "/v1/chat/completions",
            &(headers.clone() + "X-LocalAI-Mode: background\r\n"),
            &chat()
        )?
        .0 == 200,
        "background failed after release"
    );
    empty(broker)?;
    verified.push("native_control_foreground_and_disconnect");
    // Invalid framing, no body before auth/admission, and total body/header deadlines.
    let mut raw = TcpStream::connect(addr)?;
    raw.set_read_timeout(Some(Duration::from_secs(3)))?;
    write!(
        raw,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 16777216\r\n\r\n"
    )?;
    ensure!(read(raw)?.0 == 401, "unauthorized request waited for body");
    let mut raw = TcpStream::connect(addr)?;
    raw.set_read_timeout(Some(Duration::from_secs(3)))?;
    write!(
        raw,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n{headers}\r\n{{"
    )?;
    ensure!(read(raw)?.0 == 400, "partial body did not expire");
    let mut raw = TcpStream::connect(addr)?;
    raw.set_read_timeout(Some(Duration::from_secs(3)))?;
    raw.write_all(b"GET /health HTTP/1.1\r\nHo")?;
    ensure!(read(raw)?.0 == 400, "partial headers did not expire");
    empty(broker)?;
    verified.push("framing_and_read_deadlines");
    if tools {
        let mut b = chat();
        b["tools"] = json!([{"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{"n":{"type":"integer"}}}}}]);
        b["stream"] = json!(true);
        b["max_tokens"] = json!(64);
        let (status, text) = call(addr, "POST", "/v1/chat/completions", &headers, &b)?;
        ensure!(status == 200, "tool SSE status");
        let chunks = events(&text)?;
        let calls: Vec<&Value> = chunks
            .iter()
            .filter_map(|v| v["choices"][0]["delta"]["tool_calls"].as_array())
            .flatten()
            .collect();
        ensure!(
            calls.len() == 1
                && calls[0]["function"]["name"] == "run"
                && calls[0]["function"]["arguments"] == "{\"n\":5}"
                && calls[0]["id"].as_str().unwrap().len() == 29
                && chunks.last().unwrap()["choices"][0]["finish_reason"] == "tool_calls",
            "tool frame differs"
        );
        verified.push("tool_framing_scripted_tokens");
    }
    empty(broker)?;
    control(addr, 3, true)?;
    let mut b = chat();
    b["stream"] = json!(true);
    let mut pending = open(
        addr,
        "POST",
        "/v1/chat/completions",
        &(headers + "X-LocalAI-Mode: background\r\n"),
        &serde_json::to_vec(&b)?,
    )?;
    role(&mut pending)?;
    drop(server);
    drop(pending);
    empty(broker)?;
    let receipt = broker
        .lease("http-probe", 4, false, 30.0)?
        .recv_timeout(Duration::from_secs(25))?
        .map_err(anyhow::Error::msg)?;
    ensure!(!receipt.held, "probe cleanup lease still held");
    verified.push("server_shutdown_cancels_owned_clients");
    Ok(
        json!({"checks":verified,"socket":"ephemeral_loopback_only","incarnation":reply["tensorfold"]["incarnation"],"reply":reply,"production_ready":false,"legacy_status_schema":false}),
    )
}
pub(crate) fn probe(path: &Path) -> Result<Value> {
    let tok = crate::tokenizer::load(path)?;
    let table = Arc::new(Table::new(&tok)?);
    let tool = tok
        .encode(
            "<tool_call>run<arg_key>n</arg_key><arg_value>5</arg_value></tool_call>",
            false,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let (broker, owner) = Broker::new();
    let server = start(path, broker.clone(), table.clone())?;
    let join = thread::spawn(move || -> Result<()> {
        let mut jobs: BTreeMap<u64, (bool, Vec<u32>, usize, Stream)> = BTreeMap::new();
        let mut serial = 0;
        let mut gate = crate::priority::Gate::default();
        loop {
            let action = owner.next(Duration::from_millis(1));
            let mut prepared = None;
            let mut submitted = None;
            let mut cancelled = None;
            let mut rejected = None;
            let stop = matches!(action, Action::Stop);
            match action.command() {
                Command::Prepare { background } => {
                    if !background {
                        serial += 1;
                        prepared = Some(serial);
                    }
                }
                Command::Submit {
                    output,
                    background,
                    policy,
                    ..
                } => {
                    serial += 1;
                    submitted = Some(serial);
                    let p = policy.context("HTTP policy missing")?;
                    let mut tokens = if p.tools() { tool.clone() } else { vec![19] };
                    tokens.push(if p.tools() { 154829 } else { 154827 });
                    jobs.insert(
                        serial,
                        (
                            background,
                            tokens,
                            0,
                            Stream::new(
                                table.clone(),
                                p.thinking(),
                                p.tools(),
                                &p.stops(),
                                output,
                            )?,
                        ),
                    );
                }
                Command::Lease {
                    owner: name,
                    epoch,
                    foreground,
                    seconds,
                } => {
                    if let Err(e) = gate.update(&name, epoch, foreground, seconds, owner.now()) {
                        rejected = Some(e.to_string());
                    }
                }
                Command::Cancel(id) => {
                    jobs.remove(&id);
                    cancelled = Some(id);
                }
                _ => {}
            }
            let snapshot = gate.snapshot(owner.now(), false)?;
            let held = snapshot.hold;
            let revision = snapshot.revision;
            gate.acknowledge(revision)?;
            owner.applied(
                action,
                prepared,
                submitted,
                Receipt {
                    applied: revision,
                    held,
                },
                rejected,
            )?;
            if let Some(id) = cancelled {
                owner.publish(id, None, Some(End::Cancelled))?;
            }
            if stop {
                break;
            }
            if let Some((&id, job)) = jobs.iter_mut().find(|(_, j)| !j.0 || !held) {
                let token = job.1[job.2];
                job.2 += 1;
                let ended = job.3.push(token)?.1;
                owner.publish(id, Some(token as i32), ended.then_some(End::Complete))?;
                if ended {
                    jobs.remove(&id);
                }
            }
        }
        Ok(())
    });
    let result = exercise(server, &broker, true);
    broker.shutdown();
    join.join()
        .map_err(|_| anyhow::anyhow!("CPU owner panic"))??;
    let result = result?;
    Ok(
        json!({"milestone":"bounded_loopback_http_component","model_executed":false,"python_executed":false,"result":result}),
    )
}
