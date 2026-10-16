//! Authenticated HTTP serving boundary with configured listener and native scheduling.
use crate::{
    chat::{PreparedText, Preparer},
    http_transport as net,
    output::Delta,
    output_decode::Table,
    request_broker::{Broker, Cancellation},
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
const CONNECTIONS: usize = 32;
struct Auth([u8; 32]);
impl Auth {
    fn new(key: &str) -> Result<Self> {
        ensure!((16..=512).contains(&key.len()), "authentication key bounds");
        Ok(Self(Sha256::digest(key.as_bytes()).into()))
    }
    fn matches(&self, value: &str) -> bool {
        let hash = Sha256::digest(value.as_bytes());
        let mut diff = 0u8;
        for (a, b) in self.0.iter().zip(hash) {
            diff |= std::hint::black_box(*a ^ b);
        }
        std::hint::black_box(diff) == 0
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Reasoning {
    Both,
    Content,
    Reason,
}
impl Reasoning {
    pub fn parse(mode: &str) -> Result<Self> {
        Ok(match mode {
            "both" => Self::Both,
            "reasoning_content" => Self::Content,
            "reasoning" => Self::Reason,
            _ => anyhow::bail!("reasoning field mode"),
        })
    }
    fn insert(self, value: &mut Value, text: &str) {
        if !text.is_empty() {
            if matches!(self, Self::Both | Self::Content) {
                value["reasoning_content"] = json!(text);
            }
            if matches!(self, Self::Both | Self::Reason) {
                value["reasoning"] = json!(text);
            }
        }
    }
}
pub(crate) struct Config {
    /// None: inference routes are unauthenticated, as on the production
    /// compatible service; scheduler control always requires its token.
    api: Option<Auth>,
    scheduler: Auth,
    pub served: String,
    pub listen: SocketAddr,
    pub reasoning: Reasoning,
    pub request_timeout: Duration,
    pub header_timeout: Duration,
    pub body_timeout: Duration,
    pub write_timeout: Duration,
    /// The vision encoder sidecar's socket; images are refused without it.
    pub vision: Option<std::path::PathBuf>,
    pub image_fetch: bool,
    /// Allow a non-loopback listener (the production endpoint).
    pub public: bool,
    /// Set only after service configuration validates the pinned gate report.
    pub qualified: bool,
}
impl Config {
    pub fn new(api: Option<&str>, scheduler: &str) -> Result<Self> {
        ensure!(
            api != Some(scheduler),
            "separate API and scheduler keys required"
        );
        Ok(Self {
            api: api.map(Auth::new).transpose()?,
            scheduler: Auth::new(scheduler)?,
            served: "alaska-rust-experimental".into(),
            listen: "127.0.0.1:0".parse().unwrap(),
            // The compatible service sends reasoning in `reasoning` only;
            // clients that read both fields would otherwise see it twice.
            reasoning: Reasoning::parse("reasoning")?,
            request_timeout: Duration::from_secs(3600),
            header_timeout: Duration::from_secs(5),
            body_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(5),
            vision: None,
            image_fetch: true,
            public: false,
            qualified: false,
        })
    }
}
fn random() -> Result<String> {
    let mut bytes = [0u8; 12];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
struct Connection {
    socket: TcpStream,
    cancel: Option<Cancellation>,
    external: Option<String>,
    expires: Instant,
}
struct Shared {
    connections: Mutex<BTreeMap<u64, Connection>>,
    stop: AtomicBool,
    incarnation: String,
}
struct Lifetime {
    shared: Arc<Shared>,
    id: u64,
}
impl Drop for Lifetime {
    fn drop(&mut self) {
        self.shared.connections.lock().unwrap().remove(&self.id);
    }
}
struct ContextState {
    config: Config,
    broker: Broker,
    preparation: Mutex<Preparer>,
    table: Arc<Table>,
    shared: Arc<Shared>,
    image_memory: Arc<crate::vision_request::Budget>,
    vision: Option<(
        crate::vision_service::Client,
        Mutex<crate::vision_input::Registry>,
    )>,
}
pub(crate) struct Server {
    pub address: SocketAddr,
    shared: Arc<Shared>,
    join: Option<JoinHandle<()>>,
}
impl Server {
    pub fn connections(&self) -> usize {
        self.shared.connections.lock().unwrap().len()
    }
    pub fn start(
        config: Config,
        broker: Broker,
        preparation: Preparer,
        table: Arc<Table>,
    ) -> Result<Self> {
        ensure!(
            !config.request_timeout.is_zero()
                && config.request_timeout <= Duration::from_secs(86400),
            "request timeout bounds"
        );
        for time in [
            config.header_timeout,
            config.body_timeout,
            config.write_timeout,
        ] {
            ensure!(
                !time.is_zero() && time <= Duration::from_secs(60),
                "network timeout bounds"
            );
        }
        ensure!(
            !config.served.is_empty() && config.served.len() <= 160,
            "served model bounds"
        );
        ensure!(
            config.listen.ip().is_loopback() || config.public,
            "a non-loopback listener must be configured as public"
        );
        let listener = TcpListener::bind(config.listen)?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let shared = Arc::new(Shared {
            connections: Mutex::new(BTreeMap::new()),
            stop: AtomicBool::new(false),
            incarnation: random()?,
        });
        let vision = match &config.vision {
            Some(socket) => Some((
                crate::vision_service::Client::new(socket),
                Mutex::new(crate::vision_input::Registry::new(65536)?),
            )),
            None => None,
        };
        let state = Arc::new(ContextState {
            config,
            broker,
            preparation: Mutex::new(preparation),
            table,
            shared: shared.clone(),
            image_memory: crate::vision_request::Budget::new(2 << 30)?,
            vision,
        });
        let owner = shared.clone();
        let join = thread::spawn(move || {
            crate::cpu_pin::http_thread();
            let mut jobs: Vec<JoinHandle<()>> = Vec::new();
            let mut serial = 0u64;
            while !owner.stop.load(Ordering::Acquire) {
                {
                    let all = owner.connections.lock().unwrap();
                    for c in all.values() {
                        if net::gone(&c.socket) || Instant::now() >= c.expires {
                            if let Some(cancel) = &c.cancel {
                                cancel.cancel();
                            }
                            let _ = c.socket.shutdown(Shutdown::Both);
                        }
                    }
                }
                let mut i = 0;
                while i < jobs.len() {
                    if jobs[i].is_finished() {
                        let _ = jobs.swap_remove(i).join();
                    } else {
                        i += 1;
                    }
                }
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        if owner.connections.lock().unwrap().len() >= CONNECTIONS {
                            let _ = socket.set_write_timeout(Some(Duration::from_millis(100)));
                            let _ = net::json(&mut socket, 503, &error(503, "connection capacity"));
                            continue;
                        }
                        let Ok(watch) = socket.try_clone() else {
                            continue;
                        };
                        let Some(id) = serial.checked_add(1) else {
                            break;
                        };
                        serial = id;
                        owner.connections.lock().unwrap().insert(
                            id,
                            Connection {
                                socket: watch,
                                cancel: None,
                                external: None,
                                expires: Instant::now() + state.config.request_timeout,
                            },
                        );
                        let state = state.clone();
                        jobs.push(thread::spawn(move || {
                            let _lifetime = Lifetime {
                                shared: state.shared.clone(),
                                id,
                            };
                            let _ = socket.set_nodelay(true);
                            let _ = socket.set_write_timeout(Some(state.config.write_timeout));
                            let _ = handle(&mut socket, &state, id);
                        }));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10))
                    }
                    Err(_) => break,
                }
            }
            {
                let all = owner.connections.lock().unwrap();
                for c in all.values() {
                    if let Some(cancel) = &c.cancel {
                        cancel.cancel();
                    }
                    let _ = c.socket.shutdown(Shutdown::Both);
                }
            }
            // A requested listener drop cancels only its clients; the owner
            // may serve another adapter. An unexpected accept-loop exit must
            // stop the native owner so a dead service cannot remain healthy.
            if !owner.stop.load(Ordering::Acquire) {
                state.broker.shutdown();
            }
            for job in jobs {
                let _ = job.join();
            }
        });
        Ok(Self {
            address,
            shared,
            join: Some(join),
        })
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
fn has_images(body: &Value) -> bool {
    body["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|m| {
            m["content"].as_array().is_some_and(|parts| {
                parts.iter().any(|p| {
                    matches!(
                        p["type"].as_str(),
                        Some("image_url" | "input_image" | "image")
                    )
                })
            })
        })
    })
}
/// A chat request with images: each image is encoded by the sidecar, gets its
/// content-derived virtual row IDs and expands its placeholder in the prompt.
fn prepare_vision(
    s: &ContextState,
    body: &Value,
    requested: Option<usize>,
    client: &crate::request_broker::Client,
) -> Result<(PreparedText, usize, crate::vision_service::Table)> {
    use crate::{
        cache_pages::{CONTEXT, MAX_OUTPUT},
        vision_service::{ImageRejected, ROW_BYTES, Table},
    };
    let (encoder, registry) = s
        .vision
        .as_ref()
        .ok_or_else(|| ImageRejected("image input is not enabled on this server".into()))?;
    let images = {
        let mut prep = s
            .preparation
            .lock()
            .map_err(|_| anyhow::anyhow!("preparation failed"))?;
        prep.prepare_images(
            &serde_json::to_vec(body)?,
            requested.unwrap_or(MAX_OUTPUT).min(MAX_OUTPUT),
        )?
    };
    if images.references.len() > 8 {
        return Err(ImageRejected("at most 8 images a request".into()).into());
    }
    let preparation = client.prepare_image()?;
    let job = preparation.job.as_ref().unwrap();
    // Reserve before source loading, for all encoded rows plus assembled table,
    // bounded source/base64 copies and row indexes. Retain until final Table Arc.
    let image_rows = images
        .references
        .iter()
        .map(|r| r.token_budget(0, 0))
        .sum::<usize>();
    let memory = s
        .image_memory
        .reserve(job.id(), (100 << 20) + 2 * image_rows * ROW_BYTES)
        .map_err(|_| crate::vision_service::ImageCapacity)?;
    let fetch = s.config.image_fetch.then(crate::vision_fetch::Client::new);
    let clock = crate::vision_request::ActiveDeadline::new(&job.control, Duration::from_secs(30))?;
    let mut rows = Vec::new();
    let mut encoded = Vec::new();
    for reference in &images.references {
        job.control.check()?;
        let bytes = crate::vision_source::load(
            &reference.url,
            fetch.as_ref(),
            clock.deadline(&job.control)?,
            &job.control,
        )
        .map_err(|e| ImageRejected(e.to_string()))?;
        let permit = job.encode()?;
        let result = encoder.encode(&bytes, reference.token_budget(0, 0));
        if result.is_ok()
            || result.as_ref().is_err_and(|e| {
                e.downcast_ref::<ImageRejected>().is_some()
                    || e.downcast_ref::<crate::vision_service::EncoderNotStarted>()
                        .is_some()
            })
        {
            permit.drained();
        } else {
            drop(permit);
        }
        let image = result?;
        job.control.check()?;
        rows.push(
            registry
                .lock()
                .map_err(|_| anyhow::anyhow!("image registry failed"))?
                .register(image.key, image.tokens)?,
        );
        encoded.push(image);
    }
    let prompt = crate::vision_input::ImagePrompt::expand(&images.text.ids, &rows)?;
    let tokens: Arc<[i32]> = prompt.tokens().into();
    if tokens.len().saturating_add(requested.unwrap_or(1)) > CONTEXT {
        return Err(crate::chat::ContextExceeded {
            prompt: tokens.len(),
            requested,
            chat: true,
        }
        .into());
    }
    let output = requested
        .unwrap_or(MAX_OUTPUT)
        .min(MAX_OUTPUT)
        .min(CONTEXT - tokens.len());
    let table = Table::new(
        rows.iter()
            .zip(&encoded)
            .flat_map(|(r, e)| r.ids().iter().copied().zip(e.rows.chunks(ROW_BYTES)))
            .collect(),
    )?
    .retain(memory);
    let text = PreparedText {
        rendered: images.text.rendered,
        ids: tokens,
        reused_ids: 0,
    };
    Ok((text, output, table))
}

fn tokenize_vision(
    s: &ContextState,
    body: &Value,
    client: &crate::request_broker::Client,
) -> Result<Arc<[i32]>> {
    use crate::vision_service::{EncoderNotStarted, ImageRejected};
    let (encoder, _) = s
        .vision
        .as_ref()
        .ok_or_else(|| ImageRejected("image input is not enabled on this server".into()))?;
    let images = s
        .preparation
        .lock()
        .map_err(|_| anyhow::anyhow!("preparation failed"))?
        .tokenize_images(&serde_json::to_vec(body)?)?;
    let preparation = client.prepare_image()?;
    let job = preparation.job.as_ref().unwrap();
    let fetch = s.config.image_fetch.then(crate::vision_fetch::Client::new);
    let _memory = s
        .image_memory
        .reserve(job.id(), 100 << 20)
        .map_err(|_| crate::vision_service::ImageCapacity)?;
    let clock = crate::vision_request::ActiveDeadline::new(&job.control, Duration::from_secs(30))?;
    let mut counts = Vec::with_capacity(images.references.len());
    for reference in &images.references {
        let bytes = crate::vision_source::load(
            &reference.url,
            fetch.as_ref(),
            clock.deadline(&job.control)?,
            &job.control,
        )
        .map_err(|e| ImageRejected(e.to_string()))?;
        // The single sidecar is shared with GPU encodes. Retain the same native
        // admission/drain ownership even though this operation only decodes/counts.
        let permit = job.encode()?;
        let result = encoder.count(&bytes, reference.token_budget(0, 0));
        if result.is_ok()
            || result.as_ref().is_err_and(|e| {
                e.downcast_ref::<ImageRejected>().is_some()
                    || e.downcast_ref::<EncoderNotStarted>().is_some()
            })
        {
            permit.drained();
        } else {
            drop(permit);
        }
        counts.push(result?);
        job.control.check()?;
    }
    crate::vision_input::counted(&images.text.ids, &counts)
}
fn error(status: u16, message: &str) -> Value {
    json!({"error":{"message":message,"type":if status>=500{"server_error"}else{"invalid_request_error"}}})
}
fn refuse(socket: &mut TcpStream, status: u16, message: &str) -> Result<()> {
    net::json(socket, status, &error(status, message))
}
fn json_body(socket: &mut TcpStream, h: &net::Head, timeout: Duration) -> Result<Value> {
    crate::chat_json::parse(&net::body(socket, h.length, timeout)?)
}
fn lease(body: &Value) -> Result<(&str, u64, bool, f64)> {
    let owner = body["owner"].as_str().context("owner required")?;
    let epoch = body["epoch"]
        .as_u64()
        .context("epoch must be an unsigned integer")?;
    let fg = body["foreground"]
        .as_bool()
        .context("foreground boolean required")?;
    let seconds = match body.get("lease_seconds") {
        None => 30.0,
        Some(v) => v.as_f64().context("lease_seconds must be numeric")?,
    };
    ensure!(
        !owner.is_empty()
            && owner.chars().count() <= 160
            && seconds.is_finite()
            && (1.0..=300.0).contains(&seconds),
        "lease bounds"
    );
    Ok((owner, epoch, fg, seconds))
}
fn request(body: &Value, chat: bool) -> Result<(crate::http_numbers::Options, bool, bool)> {
    ensure!(body.is_object(), "request must be an object");
    // Unhandled fields (logprobs, user, metadata, penalties, ...) are ignored.
    // Any string model alias resolves to the configured served model.
    ensure!(
        body.get("model")
            .is_none_or(|m| m.is_null() || m.is_string()),
        "model must be a string"
    );
    // Validate grammar selection and conflicts here. Check native availability
    // after the actual prompt determines whether grammar is active.
    crate::grammar_request::parse(body)?;
    crate::chat::raw_flags(body, chat)?;
    if let Some(n) = body.get("n").filter(|v| !v.is_null() && *v != 1) {
        anyhow::bail!("n must be 1 on this server (got {n}): it returns one choice a request");
    }
    ensure!(
        body.get("draft").is_none_or(Value::is_boolean),
        "draft boolean required"
    );
    let numbers = crate::http_numbers::Options::parse(body)?;
    for key in ["stream", "return_token_ids", "parallel_tool_calls"] {
        ensure!(
            body.get(key).is_none_or(Value::is_boolean),
            "request boolean required"
        );
    }
    let usage = if let Some(options) = body.get("stream_options") {
        ensure!(
            options.is_object()
                && options
                    .as_object()
                    .unwrap()
                    .keys()
                    .all(|k| k == "include_usage"),
            "stream_options fields"
        );
        ensure!(
            options.get("include_usage").is_none_or(Value::is_boolean),
            "include_usage boolean required"
        );
        options["include_usage"] == true
    } else {
        false
    };
    crate::output_policy::Policy::new(false, false, &body["stop"])?;
    Ok((numbers, body["stream"] == true, usage))
}
fn delta(d: &Delta, mode: Reasoning) -> Value {
    let mut v = json!({});
    if !d.content.is_empty() {
        v["content"] = json!(d.content);
    }
    mode.insert(&mut v, &d.reasoning);
    v
}
fn chunk(rid: &str, created: u64, served: &str, delta: Value, finish: Value) -> Value {
    if rid.starts_with("cmpl-") {
        return json!({"id":rid,"object":"text_completion","created":created,"model":served,"choices":[{"index":0,"text":delta.get("content").and_then(Value::as_str).unwrap_or(""),"finish_reason":finish}]});
    }
    json!({"id":rid,"object":"chat.completion.chunk","created":created,"model":served,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
}
fn handle(socket: &mut TcpStream, s: &ContextState, id: u64) -> Result<()> {
    let h = match net::head(socket, s.config.header_timeout) {
        Ok(h) => h,
        Err(_) => return refuse(socket, 400, "invalid or expired HTTP headers"),
    };
    let scheduler = h.path == "/v1/scheduler";
    let auth = if scheduler {
        s.config.scheduler.matches(h.header("x-scheduler-token"))
    } else {
        s.config.api.as_ref().is_none_or(|api| {
            api.matches(
                h.header("authorization")
                    .strip_prefix("Bearer ")
                    .unwrap_or(""),
            )
        })
    };
    if !auth {
        return refuse(socket, 401, "authentication required");
    }
    if scheduler {
        if h.method == "GET" {
            return net::json(
                socket,
                200,
                &json!({"engine":"alaska-rust-experimental","incarnation":s.shared.incarnation,"broker":s.broker.status(),"legacy_status_schema":false}),
            );
        }
        if h.method != "POST" {
            return refuse(socket, 405, "method unsupported");
        }
        let body = match json_body(socket, &h, s.config.body_timeout) {
            Ok(b) => b,
            Err(_) => return refuse(socket, 400, "invalid scheduler JSON"),
        };
        let (owner, epoch, fg, seconds) = match lease(&body) {
            Ok(v) => v,
            Err(_) => return refuse(socket, 400, "invalid scheduler lease"),
        };
        let rx = match s.broker.lease(owner, epoch, fg, seconds) {
            Ok(rx) => rx,
            Err(_) => return refuse(socket, 503, "native control unavailable"),
        };
        return match rx.recv_timeout(Duration::from_secs(26)) {
            Ok(Ok(r)) => net::json(
                socket,
                200,
                &json!({"owner":owner,"epoch":epoch,"foreground":fg,"applied_revision":r.applied,"foreground_active":r.held,"incarnation":s.shared.incarnation}),
            ),
            Ok(Err(e)) if e.rejected => refuse(socket, 400, "native scheduler rejected lease"),
            _ => refuse(socket, 503, "native scheduler control failed"),
        };
    }
    if h.method == "GET" {
        return match h.path.as_str() {
            "/health" | "/v1/health" => {
                let status = s.broker.status();
                net::json(
                    socket,
                    if status["available"] == true {
                        200
                    } else {
                        503
                    },
                    &{
                        let mut h = crate::http_metrics::health(status, &s.shared.incarnation);
                        h["experimental"] = (!s.config.qualified).into();
                        h["production_ready"] = (s.config.qualified && h["ok"] == true).into();
                        h["qualification"] = if s.config.qualified {
                            "pinned_deployment_report"
                        } else {
                            "unqualified"
                        }
                        .into();
                        h["image_memory_reserved_bytes"] = s.image_memory.used()?.into();
                        h["image_memory_limit_bytes"] = (2u64 << 30).into();
                        h
                    },
                )
            }
            "/metrics" => {
                let text = crate::http_metrics::metrics(&s.broker.status(), &s.config.served);
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    text.len()
                )?;
                socket.write_all(text.as_bytes())?;
                socket.shutdown(Shutdown::Write)?;
                Ok(())
            }
            "/v1/models" | "/models" => net::json(
                socket,
                200,
                &json!({"object":"list","data":[{"id":s.config.served,"object":"model","owned_by":"alaska-rust","max_model_len":crate::cache_pages::CONTEXT,"context_length":crate::cache_pages::CONTEXT,"experimental":!s.config.qualified}]}),
            ),
            _ => refuse(socket, 404, "not found"),
        };
    }
    if h.method == "POST" && matches!(h.path.as_str(), "/tokenize" | "/v1/tokenize") {
        let value = match json_body(socket, &h, s.config.body_timeout) {
            Ok(v) => v,
            Err(_) => return refuse(socket, 400, "invalid tokenize JSON"),
        };
        let ids = if has_images(&value) {
            let background = match h
                .header("x-localai-mode")
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "" | "foreground" => false,
                "background" => true,
                _ => return refuse(socket, 400, "invalid X-LocalAI-Mode"),
            };
            let client = match s.broker.begin(background, s.config.request_timeout) {
                Ok(c) => c,
                Err(_) => return refuse(socket, 503, "native preparation unavailable"),
            };
            s.shared
                .connections
                .lock()
                .unwrap()
                .get_mut(&id)
                .context("retired connection")?
                .cancel = Some(client.cancellation());
            if client.prepared(Duration::from_secs(25)).is_err() {
                return refuse(socket, 503, "native preparation unavailable");
            }
            tokenize_vision(s, &value, &client)
        } else {
            s.preparation
                .lock()
                .map_err(|_| anyhow::anyhow!("preparation failed"))?
                .tokenize(&serde_json::to_vec(&value)?)
        };
        return match ids {
            Ok(ids) => net::json(
                socket,
                200,
                &json!({"count":ids.len(),"max_model_len":crate::cache_pages::CONTEXT,"tokens":ids.as_ref(),"token_strs":Value::Null}),
            ),
            Err(e)
                if e.downcast_ref::<crate::vision_service::EncoderNotStarted>()
                    .is_some()
                    || e.downcast_ref::<crate::vision_service::ImageCapacity>()
                        .is_some() =>
            {
                refuse(socket, 503, "image preparation unavailable")
            }
            Err(_) => refuse(socket, 400, "tokenization rejected"),
        };
    }
    let chat = h.path == "/v1/chat/completions";
    if h.method != "POST" || !(chat || h.path == "/v1/completions") {
        return refuse(socket, 404, "not found");
    }
    let mode = h.header("x-localai-mode").trim().to_ascii_lowercase();
    let background = match mode.as_str() {
        "" | "foreground" => false,
        "background" => true,
        _ => return refuse(socket, 400, "invalid X-LocalAI-Mode"),
    };
    let external = if h.header("x-request-id").is_empty() {
        format!("native-{}", random()?)
    } else {
        h.header("x-request-id").to_owned()
    };
    if external.len() > 160
        || !external
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
    {
        return refuse(socket, 400, "invalid X-Request-Id");
    }
    let duplicate = {
        let mut all = s.shared.connections.lock().unwrap();
        let duplicate = all.values().any(|c| c.external.as_ref() == Some(&external));
        if !duplicate {
            all.get_mut(&id).context("retired connection")?.external = Some(external.clone());
        }
        duplicate
    };
    if duplicate {
        return refuse(socket, 409, "request ID is active");
    }
    let client = match s.broker.begin(background, s.config.request_timeout) {
        Ok(c) => c,
        Err(_) => return refuse(socket, 503, "native request capacity unavailable"),
    };
    s.shared
        .connections
        .lock()
        .unwrap()
        .get_mut(&id)
        .context("retired connection")?
        .cancel = Some(client.cancellation());
    if client.prepared(Duration::from_secs(25)).is_err() {
        return refuse(socket, 503, "native preparation unavailable");
    }
    let mut body = match json_body(socket, &h, s.config.body_timeout) {
        Ok(b) => b,
        Err(_) => return refuse(socket, 400, "invalid or expired request body"),
    };
    let (numbers, stream, include_usage) = match request(&body, chat) {
        Ok(v) => v,
        Err(e) => return refuse(socket, 400, &e.to_string()),
    };
    let sampling = match numbers.resolve() {
        Ok(sampling) => sampling,
        Err(_) => return refuse(socket, 503, "request seed unavailable"),
    };
    body["_localai_mode"] = json!(if background {
        "background"
    } else {
        "foreground"
    });
    body["_localai_request_id"] = json!(external);
    let mut vision_table = None;
    let prepared_result = if chat && has_images(&body) {
        prepare_vision(s, &body, numbers.requested, &client).map(|(prepared, output, table)| {
            vision_table = Some(table);
            (prepared, output)
        })
    } else {
        let mut prep = s
            .preparation
            .lock()
            .map_err(|_| anyhow::anyhow!("preparation failed"))?;
        let bytes = serde_json::to_vec(&body)?;
        prep.prepare_request(&bytes, numbers.requested, chat)
    };
    let (prepared, output): (PreparedText, usize) = match prepared_result {
        Ok(p) => p,
        Err(e) => {
            if let Some(c) = e.downcast_ref::<crate::chat::ContextExceeded>() {
                return net::json(
                    socket,
                    400,
                    &json!({"error":{"message":c.to_string(),"type":"invalid_request_error",
                        "code":"context_length_exceeded","param":c.param()}}),
                );
            }
            if let Some(r) = e.downcast_ref::<crate::vision_service::ImageRejected>() {
                return refuse(socket, 400, &r.0);
            }
            if e.downcast_ref::<crate::vision_service::ImageCapacity>()
                .is_some()
            {
                return refuse(socket, 503, "image preparation capacity unavailable");
            }
            if vision_table.is_none() && e.to_string().contains("vision encoder") {
                return refuse(socket, 503, "image preparation unavailable");
            }
            return refuse(socket, 400, "chat preparation rejected");
        }
    };
    let thinking_budget = match crate::thinking_budget::parse(
        &prepared.rendered.body,
        chat,
        prepared.rendered.thinking,
    ) {
        Ok(budget) => budget,
        Err(_) => return refuse(socket, 400, "thinking budget rejected"),
    };
    let mut policy = crate::output_policy::Policy::new(
        prepared.rendered.thinking,
        prepared.rendered.body["tools"]
            .as_array()
            .is_some_and(|a| !a.is_empty()),
        &prepared.rendered.body["stop"],
    )?
    .with_thinking_budget(thinking_budget)?
    .with_drafts(prepared.rendered.body.get("draft") != Some(&Value::Bool(false)));
    let grammar =
        match crate::grammar_bound::Bound::from_request(&prepared.rendered.body, &prepared.ids) {
            Ok(grammar) => grammar,
            Err(e) => return refuse(socket, 400, &e.to_string()),
        };
    policy = policy
        .with_grammar(grammar)
        .with_ignore_eos(prepared.rendered.body["ignore_eos"] == true)?;
    // The broker's deferred grammar phase accepts only after every rank has
    // compiled the same constraint; unavailable native managers reject here
    // through the submission receipt, before JSON/SSE response headers.
    if let Some(sampling) = sampling {
        policy = policy.with_sampling(sampling)?;
    }
    if let Some(table) = vision_table
        && client.attach_vision(Arc::new(table)).is_err()
    {
        return refuse(socket, 503, "native submission unavailable");
    }
    if client
        .submit_policy(prepared.ids.clone(), output, Some(policy))
        .is_err()
    {
        return refuse(socket, 503, "native submission unavailable");
    }
    let native = match client.submitted(Duration::from_secs(25)) {
        Ok(n) => n,
        Err(e)
            if e.downcast_ref::<crate::request_broker::InvalidGrammar>()
                .is_some() =>
        {
            return refuse(socket, 400, "request grammar cannot be enforced");
        }
        Err(_) => return refuse(socket, 503, "native submission rejected"),
    };
    let return_token_ids = body["return_token_ids"] == true;
    drop(body); // The repaired body in PreparedText owns final parsing/history data.
    let prompt = prepared.ids.len();
    let token_cache = prepared.reused_ids;
    let rid = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, random()?);
    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    if stream {
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        if chat {
            net::event(
                socket,
                &chunk(
                    &rid,
                    created,
                    &s.config.served,
                    json!({"role":"assistant"}),
                    Value::Null,
                ),
            )?;
        }
    }
    let mut last_write = Instant::now();
    let result = crate::chat_egress::consume(
        client,
        prepared,
        s.table.clone(),
        &s.preparation,
        output,
        |d| {
            if net::gone(socket) {
                return Ok(false);
            }
            if stream {
                if let Some(d) = d {
                    let value = delta(d, s.config.reasoning);
                    if !value.as_object().unwrap().is_empty() {
                        net::event(
                            socket,
                            &chunk(&rid, created, &s.config.served, value, Value::Null),
                        )?;
                        last_write = Instant::now();
                    }
                } else if last_write.elapsed() >= Duration::from_secs(1) {
                    socket.write_all(b": keepalive\n\n")?;
                    last_write = Instant::now();
                }
            }
            Ok(true)
        },
        &mut || Ok(format!("call_{}", random()?)),
    );
    let reply = match result {
        Ok(v) => v,
        Err(_) => {
            if stream {
                let _ = net::event(socket, &error(503, "native request failed or disconnected"));
                let _ = socket.write_all(b"data: [DONE]\n\n");
                return Ok(());
            }
            return refuse(socket, 503, "native request failed or disconnected");
        }
    };
    let usage = json!({"prompt_tokens":prompt,"completion_tokens":reply["completion_tokens"],"total_tokens":prompt+reply["completion_tokens"].as_u64().unwrap() as usize,"prompt_tokens_details":{"cached_tokens":reply["cached_tokens"]}});
    let mut stats = json!({"engine":"alaska-rust-experimental","incarnation":s.shared.incarnation,"request_id":external,"native_request_id":native,"mode":if background{"background"}else{"foreground"},"cached":reply["cached_tokens"],"cpu_reused_prompt_tokens":token_cache});
    if return_token_ids {
        stats["token_ids"] = reply["token_ids"].clone();
    }
    if stream {
        let tail = delta(
            &Delta {
                content: reply["final"]["content"].as_str().unwrap().into(),
                reasoning: reply["final"]["reasoning"].as_str().unwrap().into(),
            },
            s.config.reasoning,
        );
        if !tail.as_object().unwrap().is_empty() {
            net::event(
                socket,
                &chunk(&rid, created, &s.config.served, tail, Value::Null),
            )?;
        }
        for (i, call) in reply["calls"].as_array().unwrap().iter().enumerate() {
            let mut call = call.clone();
            call["index"] = json!(i);
            net::event(
                socket,
                &chunk(
                    &rid,
                    created,
                    &s.config.served,
                    json!({"tool_calls":[call]}),
                    Value::Null,
                ),
            )?;
        }
        let mut end = chunk(
            &rid,
            created,
            &s.config.served,
            json!({}),
            reply["finish"].clone(),
        );
        if include_usage {
            end["usage"] = usage;
        }
        end["tensorfold"] = stats;
        net::event(socket, &end)?;
        socket.write_all(b"data: [DONE]\n\n")?;
    } else if !chat {
        net::json(
            socket,
            200,
            &json!({"id":rid,"object":"text_completion","created":created,"model":s.config.served,"choices":[{"index":0,"text":reply["content"],"finish_reason":reply["finish"]}],"usage":usage,"tensorfold":stats}),
        )?;
    } else {
        let mut message = json!({"role":"assistant","content":if reply["content"]==""{Value::Null}else{reply["content"].clone()}});
        s.config
            .reasoning
            .insert(&mut message, reply["reasoning"].as_str().unwrap());
        if !reply["calls"].as_array().unwrap().is_empty() {
            message["tool_calls"] = reply["calls"].clone();
        }
        net::json(
            socket,
            200,
            &json!({"id":rid,"object":"chat.completion","created":created,"model":s.config.served,"choices":[{"index":0,"message":message,"finish_reason":reply["finish"]}],"usage":usage,"tensorfold":stats}),
        )?;
    }
    Ok(())
}
#[cfg(test)]
#[path = "raw_completion_tests.rs"]
mod raw_completion_tests;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn authentication_and_strict_lease_types() -> Result<()> {
        let a = Auth::new("test-secret-000000")?;
        assert!(a.matches("test-secret-000000"));
        assert!(!a.matches("test-secret-000001"));
        assert!(!a.matches(""));
        for epoch in [
            json!(true),
            json!(-1),
            json!(1.0),
            serde_json::from_str("18446744073709551616")?,
        ] {
            assert!(lease(&json!({"owner":"x","epoch":epoch,"foreground":true})).is_err());
        }
        assert!(lease(&json!({"owner":"x","epoch":u64::MAX,"foreground":false})).is_ok());
        Ok(())
    }
    #[test]
    fn refuse_unsupported_quality_controls() -> Result<()> {
        assert_eq!(request(&json!({"temperature":0}), true)?.0.requested, None);
        assert!(request(&json!({}), true)?.0.resolve()?.is_some());
        assert!(
            request(&json!({"temperature":0,"seed":3}), true)?
                .0
                .resolve()?
                .is_some_and(|policy| policy.greedy() && policy.seed == 3)
        );
        for body in [
            json!({"temperature":true}),
            json!({"temperature":0,"seed":false}),
            json!({"temperature":0,"response_format":{"type":"unknown"}}),
            json!({"temperature":0,"tools":[{"function":{"name":"f"}}],"tool_choice":"invalid"}),
            json!({"temperature":0,"max_tokens":true}),
        ] {
            assert!(request(&body, true).is_err());
        }
        assert_eq!(
            request(&json!({"temperature":0,"max_tokens":131072}), true)?
                .0
                .requested,
            Some(131072)
        );
        Ok(())
    }
    #[test]
    fn explicit_greedy_http_policy_reaches_native_queue() -> Result<()> {
        for temperature in [0.0, -0.0, -1.0] {
            let (numbers, _, _) = request(&json!({"temperature":temperature}), true)?;
            let sampling = numbers.resolve()?.context("missing HTTP greedy policy")?;
            let policy = crate::output_policy::Policy::new(false, false, &Value::Null)?
                .with_sampling(sampling)?;
            let wire = crate::output_policy::Policy::decode(&policy.encode())?;
            let mut queue = crate::request_queue::Queue::default();
            let id = queue.submit_policy(
                vec![17].into(),
                4,
                false,
                Duration::from_secs(10),
                Duration::ZERO,
                Some(&wire),
            )?;
            let retained = queue
                .request(id)?
                .sampling()
                .context("greedy policy lost at admission")?;
            assert!(retained.greedy());
            assert_eq!(retained, sampling);
            assert!(queue.request(id)?.drafts());
        }
        Ok(())
    }
    #[test]
    fn raw_flag_types_scope_and_active_grammar_reject_before_submission() -> Result<()> {
        for name in ["add_special_tokens", "ignore_eos"] {
            for value in [json!(true), json!(false)] {
                let body = json!({"prompt":[17],name:value});
                request(&body, false)?;
                request(&body, true)?;
            }
            assert_eq!(
                request(&json!({"prompt":[17],name:"x"}), true).is_err(),
                name == "ignore_eos"
            );
            for value in [
                Value::Null,
                json!(0),
                json!(1),
                json!("false"),
                json!([]),
                json!({}),
            ] {
                assert!(request(&json!({"prompt":"x",name:value}), false).is_err());
            }
        }
        for grammar in [
            json!({"guided_regex":"x"}),
            json!({"response_format":{"type":"json_object"}}),
            json!({"tools":[{"function":{"name":"f"}}],"tool_choice":"required"}),
        ] {
            let mut body = grammar;
            body["ignore_eos"] = true.into();
            assert!(request(&body, false).is_err());
            body["ignore_eos"] = false.into();
            request(&body, false)?;
        }
        request(
            &json!({"prompt":"x","ignore_eos":true,"structured_outputs":{"json_object":false}}),
            false,
        )?;
        request(&json!({"model":"alaska"}), true)?;
        assert!(request(&json!({"model":7}), true).is_err());
        request(
            &json!({"logprobs":true,"top_logprobs":2,"user":"u","metadata":{},"presence_penalty":0}),
            true,
        )?;
        assert!(request(&json!({"n":2}), true).is_err());
        Ok(())
    }
    #[test]
    fn grammar_fields_select_policy_before_native_availability() -> Result<()> {
        // Frozen parsing ignores tool_choice when no tools are supplied.
        request(&json!({"tool_choice":"invalid"}), true)?;
        for draft in [json!(true), json!(false)] {
            request(&json!({"draft": draft}), true)?;
        }
        for draft in [Value::Null, json!(0), json!("false")] {
            assert!(request(&json!({"draft": draft}), true).is_err());
        }
        for body in [
            json!({"response_format":{"type":"json_object"}}),
            json!({"guided_json":{}}),
            json!({"guided_regex":"x"}),
            json!({"guided_choice":["a","b"]}),
            json!({"guided_grammar":"root ::= \"x\""}),
            json!({"structured_outputs":{"json_object":true}}),
            json!({"tools":[{"function":{"name":"f"}}],"tool_choice":"required"}),
            json!({"tools":[{"function":{"name":"f","strict":true}}]}),
        ] {
            request(&body, true)?;
            assert!(crate::grammar_bound::Bound::from_request(&body, &[])?.is_some());
        }
        let ordinary = json!({"tools":[{"function":{"name":"f"}}]});
        request(&ordinary, true)?;
        assert!(crate::grammar_bound::Bound::from_request(&ordinary, &[])?.is_none());
        Ok(())
    }
    #[test]
    fn reasoning_modes_and_random_ids() -> Result<()> {
        for (mode, n) in [("both", 2), ("reasoning", 1), ("reasoning_content", 1)] {
            let mut v = json!({});
            Reasoning::parse(mode)?.insert(&mut v, "thought");
            assert_eq!(v.as_object().unwrap().len(), n);
        }
        let a = random()?;
        let b = random()?;
        assert_eq!(a.len(), 24);
        assert!(a.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(a, b);
        Ok(())
    }
}
