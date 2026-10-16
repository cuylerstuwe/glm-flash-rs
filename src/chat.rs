//! Bounded text preparation using the canonical model template. No Python,
//! generic message normalization, GPU work, or public serving surface.
use crate::{chat_history::Memory, chat_json as j, prompt_cache::PromptCache, tokenizer};
use anyhow::{Context, Result, ensure};
use minijinja::{
    Environment, Error, ErrorKind, Value as Jinja,
    value::{Kwargs, ValueKind, from_args},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

const TEMPLATE: &str = include_str!("../templates/glm53.jinja");
// MiniJinja requires [0] for four numeric sequence lookups and does not reset
// conditional assignments between loop iterations like Jinja. Explicitly clear
// reasoning_content to Undefined before every assistant, preventing leakage.
// Map items filters preserve Python method precedence over an "items" data key.
const COMPATIBLE: &str = include_str!("../templates/glm53-minijinja.jinja");
const TEMPLATE_SHA: &str = "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3";
const FUEL: u64 = 5_000_000;
const FORMAT_BYTES: usize = 128 << 20;
/// `ignore_eos` applies to chat and raw completions alike; chat requests ignore
/// `add_special_tokens` like any other unknown field, as the production
/// compatible service does. ByteLevel post-processing is ID-preserving for both
/// special flags; PromptCache checks that tokenizer contract during construction.
pub(crate) fn raw_flags(body: &Value, chat: bool) -> Result<bool> {
    let names: &[&str] = if chat {
        &["ignore_eos"]
    } else {
        &["add_special_tokens", "ignore_eos"]
    };
    for name in names {
        if let Some(value) = body.get(*name) {
            ensure!(value.is_boolean(), "{name} must be a boolean");
        }
    }
    let ignore = body["ignore_eos"] == true;
    ensure!(
        !ignore || crate::grammar_request::parse(body)?.is_none(),
        "ignore_eos is unavailable with an active grammar"
    );
    Ok(ignore)
}
#[derive(Clone)]
pub(crate) struct Policy {
    pub thinking: bool,
    pub effort: Option<&'static str>,
    pub clear: Option<bool>,
}
pub(crate) struct Renderer {
    env: Environment<'static>,
    pub policy: Policy,
    pub memory: Memory,
}
pub(crate) struct Rendered {
    pub text: String,
    pub thinking: bool,
    pub body: Value,
}
struct Limited {
    bytes: Vec<u8>,
    max: usize,
}
impl Write for Limited {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.max - self.bytes.len() {
            return Err(std::io::Error::other("render exceeds 32 MiB"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn template_error(error: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::InvalidOperation, error.to_string())
}
fn to_jinja(value: &Value) -> Jinja {
    match value {
        Value::Null => Jinja::from(()),
        Value::Bool(b) => Jinja::from(*b),
        Value::String(s) => Jinja::from(s.clone()),
        Value::Number(n) if n.is_f64() => Jinja::from(n.as_f64().unwrap()),
        Value::Number(n) if n.as_i64().is_some() => Jinja::from(n.as_i64().unwrap()),
        Value::Number(n) => Jinja::from(n.as_u64().unwrap()),
        Value::Array(a) => a.iter().map(to_jinja).collect(),
        Value::Object(m) => m.iter().map(|(k, v)| (k.clone(), to_jinja(v))).collect(),
    }
}
fn reasoning_effort(value: Option<&Value>) -> Result<Option<(bool, &'static str)>> {
    value
        .filter(|v| !v.is_null())
        .map(|value| {
            let asked = j::strip(
                value
                    .as_str()
                    .context("reasoning_effort must be a string")?,
            )
            .to_ascii_lowercase();
            Ok(match asked.as_str() {
                "none" => (false, "none"),
                "minimal" | "low" => (true, "low"),
                "medium" | "high" => (true, "high"),
                "xhigh" | "max" => (true, "max"),
                _ => anyhow::bail!("unsupported reasoning_effort"),
            })
        })
        .transpose()
}
impl Renderer {
    pub(crate) fn new(policy: Policy) -> Result<Self> {
        ensure!(
            format!("{:x}", Sha256::digest(TEMPLATE)) == TEMPLATE_SHA,
            "canonical template changed"
        );
        ensure!(
            matches!(policy.effort, None | Some("low" | "high")),
            "invalid default effort"
        );
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_fuel(Some(FUEL));
        env.set_debug(false);
        // MiniJinja lets None iterate as empty; Python/Jinja does not. Undefined
        // remains an empty iterable. Inputs contain only checked JSON values.
        env.add_test("iterable", |v: Jinja| {
            matches!(
                v.kind(),
                ValueKind::String | ValueKind::Seq | ValueKind::Map | ValueKind::Undefined
            )
        });
        env.add_filter(
            "tojson",
            |value: Jinja, kwargs: Kwargs| -> std::result::Result<String, Error> {
                let ascii: Option<bool> = kwargs.get("ensure_ascii")?;
                kwargs.assert_all_used()?;
                if ascii == Some(true) {
                    return Err(template_error("ASCII JSON is outside the pinned template"));
                }
                j::json(
                    &serde_json::to_value(value).map_err(template_error)?,
                    false,
                    false,
                )
                .map_err(template_error)
            },
        );
        env.set_unknown_method_callback(|state, value, method, args| {
            if value.kind() == ValueKind::Map && method == "items" {
                let _: () = from_args(args)?;
                state.apply_filter("items", std::slice::from_ref(value))
            } else if let Some(text) = value.as_str() {
                match method {
                    "strip" => {
                        let _: () = from_args(args)?;
                        Ok(Jinja::from(j::strip(text)))
                    }
                    "split" => {
                        let (separator,): (&str,) = from_args(args)?;
                        if separator.is_empty() {
                            return Err(template_error("empty split separator"));
                        }
                        Ok(text.split(separator).map(Jinja::from).collect())
                    }
                    _ => Err(Error::from(ErrorKind::UnknownMethod)),
                }
            } else {
                Err(Error::from(ErrorKind::UnknownMethod))
            }
        });
        ensure!(
            TEMPLATE.matches(".0.").count() == 4 && TEMPLATE.replace(".0.", "[0].").replace(
                "{%- set content = visible_text(m.content) %}",
                    "{%- set reasoning_content = undefined -%}\n{%- set content = visible_text(m.content) %}")
                    .replace("tool.items()", "tool|items").replace("_args.items()", "_args|items") == COMPATIBLE,
            "unexpected template compatibility edit"
        );
        env.add_template("canonical", COMPATIBLE)?;
        Ok(Self {
            env,
            policy,
            memory: Memory::new(1024, 32 << 20, 144 << 20),
        })
    }
    pub(crate) fn render(&mut self, bytes: &[u8]) -> Result<Rendered> {
        Ok(self.render_kind(bytes, false)?.0)
    }
    fn render_kind(
        &mut self,
        bytes: &[u8],
        images: bool,
    ) -> Result<(Rendered, Vec<crate::vision_markup::Reference>)> {
        let mut body = j::parse(bytes)?;
        shape(&body, images)?;
        let mut kwargs = match body.get("chat_template_kwargs") {
            None | Some(Value::Null) => serde_json::Map::new(),
            Some(Value::Object(m)) => m.clone(),
            _ => anyhow::bail!("chat_template_kwargs must be an object"),
        };
        ensure!(
            kwargs.keys().all(|k| matches!(
                k.as_str(),
                "enable_thinking" | "reasoning_effort" | "clear_thinking"
            )),
            "unsupported chat_template_kwargs"
        );
        // Validate both inputs before applying explicit template precedence, so
        // an invalid shadowed value cannot silently reach the model template.
        let top = reasoning_effort(body.get("reasoning_effort"))?;
        let template = reasoning_effort(kwargs.get("reasoning_effort"))?;
        if let Some((thinking, effort)) = template.or(top) {
            kwargs.entry("enable_thinking").or_insert(json!(thinking));
            kwargs.insert("reasoning_effort".into(), json!(effort));
        } else {
            kwargs.shift_remove("reasoning_effort");
            if kwargs
                .get("enable_thinking")
                .map_or(self.policy.thinking, j::truthy)
                && let Some(effort) = self.policy.effort
            {
                kwargs.insert("reasoning_effort".into(), json!(effort));
            }
        }
        if !kwargs.contains_key("clear_thinking") {
            let clear = body
                .get("thinking")
                .and_then(|v| v.get("clear_thinking"))
                .map(j::truthy)
                .or(self.policy.clear);
            if let Some(clear) = clear {
                kwargs.insert("clear_thinking".into(), json!(clear));
            }
        }
        let thinking = kwargs
            .get("enable_thinking")
            .map_or(self.policy.thinking, j::truthy);
        body["chat_template_kwargs"] = Value::Object(kwargs.clone());
        crate::chat_history::prepare(&mut body, &mut self.memory)?;
        // Retained reasoning and decoded argument objects also count toward the
        // preparation budget; checking only the wire body would miss them.
        j::check(&body)?;
        let mut context = serde_json::Map::new();
        let mut messages = body["messages"].clone();
        // The normalized body and retained-history signature keep the original
        // image parts. Only this private template context contains sentinels.
        let markup = images
            .then(|| crate::vision_markup::Markup::mark(&mut messages))
            .transpose()?;
        context.insert("messages".into(), messages);
        context.insert("tools".into(), body["tools"].clone());
        context.insert("add_generation_prompt".into(), json!(true));
        context.insert("enable_thinking".into(), json!(thinking));
        context.extend(kwargs);
        // Clone the compiled environment to give each render its own expansion
        // meter. It also charges intermediate macro captures, not just final IO.
        let mut env = self.env.clone();
        let bytes_formatted = AtomicUsize::new(0);
        env.set_formatter(move |out, _state, value| {
            let value = match value.kind() {
                ValueKind::None => std::borrow::Cow::Borrowed("None"),
                ValueKind::Bool => {
                    std::borrow::Cow::Borrowed(if value.is_true() { "True" } else { "False" })
                }
                _ => value
                    .as_str()
                    .map(std::borrow::Cow::Borrowed)
                    .unwrap_or_else(|| std::borrow::Cow::Owned(value.to_string())),
            };
            let used = bytes_formatted.fetch_add(value.len(), Ordering::Relaxed);
            if value.len() > FORMAT_BYTES.saturating_sub(used) {
                return Err(template_error("template expansion bound"));
            }
            write!(out, "{value}").map_err(template_error)
        });
        let mut output = Limited {
            bytes: Vec::new(),
            max: j::RENDER_BYTES,
        };
        env.get_template("canonical")?
            .render_to_write(to_jinja(&Value::Object(context)), &mut output)?;
        let mut text = String::from_utf8(output.bytes)?;
        if !thinking {
            text = text.replacen("<|system|>Reasoning Effort: Max", "", 1);
            if text.ends_with("<|assistant|><think>") {
                text.push_str("</think>");
            }
        }
        ensure!(
            text.len() <= j::RENDER_BYTES,
            "wrapped render exceeds bound"
        );
        let (text, references) = if let Some(markup) = markup {
            let image = markup.splice(text)?;
            (image.text, image.references)
        } else {
            (text, Vec::new())
        };
        Ok((
            Rendered {
                text,
                thinking,
                body,
            },
            references,
        ))
    }
}
fn content(v: &Value, images: bool) -> Result<()> {
    ensure!(
        v.is_null() || v.is_string() || v.is_array(),
        "unsupported message content"
    );
    if let Some(parts) = v.as_array() {
        for p in parts {
            if p.is_string() {
                continue;
            }
            ensure!(p.is_object(), "content parts must be objects or strings");
            identifiers(p)?;
            if let Some(output) = p.get("output") {
                // Frozen image markup visits direct content parts only.
                content(output, false)?;
                continue;
            }
            match p["type"].as_str() {
                Some("image" | "image_url" | "input_image") if images => {}
                Some(
                    "image" | "image_url" | "input_image" | "video" | "video_url" | "audio"
                    | "audio_url" | "input_audio",
                ) => anyhow::bail!(
                    "image content requires the image preparation path; refusing text-only substitution"
                ),
                Some("text") => ensure!(p["text"].is_string(), "text part needs text"),
                Some("tool_reference") => {
                    ensure!(p["name"].is_string(), "tool reference needs name")
                }
                _ => {}
            }
        }
    }
    Ok(())
}
fn identifiers(v: &Value) -> Result<()> {
    for key in ["id", "tool_call_id"] {
        ensure!(
            v.get(key).is_none_or(|id| id.is_null() || id.is_string()),
            "tool identifiers must be strings"
        );
    }
    Ok(())
}
fn shape(body: &Value, images: bool) -> Result<()> {
    ensure!(body.is_object(), "chat body must be an object");
    let messages = body["messages"]
        .as_array()
        .context("messages must be an array")?;
    ensure!(messages.len() <= 8192, "too many messages");
    for m in messages {
        identifiers(m)?;
        ensure!(
            m.is_object()
                && matches!(
                    m["role"].as_str(),
                    Some("system" | "user" | "assistant" | "tool")
                ),
            "unsupported message role/shape"
        );
        content(&m["content"], images)?;
        if let Some(calls) = m.get("tool_calls").filter(|v| !v.is_null()) {
            let calls = calls.as_array().context("tool_calls must be an array")?;
            ensure!(calls.len() <= 256, "too many tool calls");
            for call in calls {
                identifiers(call)?;
            }
        }
    }
    if let Some(tools) = body.get("tools").filter(|v| !v.is_null()) {
        let tools = tools.as_array().context("tools must be an array")?;
        ensure!(tools.len() <= 1024, "too many tools");
        for t in tools {
            let f = if t["function"].is_object() {
                &t["function"]
            } else {
                t
            };
            ensure!(
                f.is_object() && f["name"].is_string(),
                "tool must have a function name"
            );
        }
    }
    Ok(())
}

pub(crate) struct PreparedText {
    pub rendered: Rendered,
    pub ids: Arc<[i32]>,
    pub reused_ids: usize,
}
/// OpenAI's context-length wording, which agents parse to compact a
/// conversation (the compatible service sends the same).
#[derive(Debug)]
pub(crate) struct ContextExceeded {
    pub prompt: usize,
    pub requested: Option<usize>,
    pub chat: bool,
}
impl ContextExceeded {
    pub fn param(&self) -> &'static str {
        if self.chat { "messages" } else { "prompt" }
    }
}
impl std::fmt::Display for ContextExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (limit, place) = (crate::cache_pages::CONTEXT, self.param());
        write!(f, "This model's maximum context length is {limit} tokens. ")?;
        match self.requested {
            Some(asked) => write!(
                f,
                "However, you requested {} tokens ({} in the {place}, {asked} in the completion). \
                 Please reduce the length of the {place} or completion.",
                self.prompt.saturating_add(asked),
                self.prompt
            ),
            None => write!(
                f,
                "However, your {place} resulted in {} tokens. Please reduce the length of the {place}.",
                self.prompt
            ),
        }
    }
}
impl std::error::Error for ContextExceeded {}
/// Text is vocabulary-only at this stage. The native image preparation owner
/// binds decoded rows and validates the expanded total context before submission.
#[allow(dead_code)] // Intermediate text/reference carrier for image preparation.
pub(crate) struct PreparedImages {
    pub text: PreparedText,
    pub references: Vec<crate::vision_markup::Reference>,
}
pub(crate) struct Preparer {
    renderer: Renderer,
    cache: PromptCache,
    live_clear: Option<crate::service_state::History>,
    vocabulary: Vec<bool>,
}
impl Preparer {
    pub(crate) fn finish_output(
        &mut self,
        stream: crate::output::Stream,
        body: &Value,
        ids: &mut impl FnMut() -> Result<String>,
    ) -> Result<Value> {
        stream.reply(body, &mut self.renderer.memory, ids)
    }
    pub(crate) fn new(path: &Path, policy: Policy) -> Result<Self> {
        let cache = PromptCache::new(path, 32, 256 << 20)?;
        let vocabulary = cache.vocabulary();
        Ok(Self {
            renderer: Renderer::new(policy)?,
            cache,
            vocabulary,
            live_clear: None,
        })
    }
    pub(crate) fn live_history(&mut self, path: std::path::PathBuf, default: Option<bool>) {
        self.live_clear = Some(crate::service_state::History::new(path, default));
    }
    pub(crate) fn tokenize(&mut self, body: &[u8]) -> Result<Arc<[i32]>> {
        let value = j::parse(body)?;
        ensure!(value.is_object(), "tokenize requires an object");
        let prepared = if value["messages"].is_array() {
            self.encode(value, true)?
        } else {
            ensure!(
                value["prompt"].is_string(),
                "tokenize requires string prompt or messages"
            );
            // The pinned ByteLevel postprocessor preserves IDs with either
            // add_special_tokens setting; PromptCache validates that contract.
            self.encode(value, false)?
        };
        ensure!(
            prepared.ids.len() <= crate::cache_pages::CONTEXT,
            "tokenize context bound"
        );
        Ok(prepared.ids)
    }
    pub(crate) fn prepare_completion(
        &mut self,
        body: &[u8],
        output: usize,
    ) -> Result<PreparedText> {
        self.prepare_kind(body, output, false)
    }
    fn encode(&mut self, body: Value, chat: bool) -> Result<PreparedText> {
        raw_flags(&body, chat)?;
        if let Some(history) = &self.live_clear {
            self.renderer.policy.clear = history.current();
        }
        let mut rendered = if chat {
            self.renderer.render(&serde_json::to_vec(&body)?)?
        } else {
            ensure!(body.is_object(), "completion requires an object");
            let text = if let Some(text) = body["prompt"].as_str() {
                text.to_owned()
            } else {
                ensure!(
                    body["prompt"].is_array(),
                    "prompt requires string or token IDs"
                );
                String::new()
            };
            Rendered {
                text,
                thinking: false,
                body,
            }
        };
        let (ids, reused_ids) = if !chat && let Some(given) = rendered.body["prompt"].as_array() {
            ensure!(
                !given.is_empty() && given.len() <= crate::cache_pages::CONTEXT,
                "token prompt size"
            );
            let ids = given
                .iter()
                .map(|v| {
                    let id = v.as_u64().context("token ID must be unsigned integer")?;
                    ensure!(
                        id < self.vocabulary.len() as u64 && self.vocabulary[id as usize],
                        "token ID outside tokenizer vocabulary"
                    );
                    Ok(id as i32)
                })
                .collect::<Result<Arc<[i32]>>>()?;
            (ids, 0)
        } else {
            let encoded = self.cache.encode(&rendered.text)?;
            (
                encoded.ids.iter().map(|id| *id as i32).collect(),
                encoded.reused_ids,
            )
        };
        if !chat {
            rendered
                .body
                .as_object_mut()
                .unwrap()
                .shift_remove("prompt");
        }
        Ok(PreparedText {
            rendered,
            ids,
            reused_ids,
        })
    }
    fn prepare_kind(&mut self, body: &[u8], output: usize, chat: bool) -> Result<PreparedText> {
        ensure!(
            (1..=crate::cache_pages::MAX_OUTPUT).contains(&output),
            "maximum output outside policy"
        );
        let prepared = self.encode(j::parse(body)?, chat)?;
        ensure!(
            !prepared.ids.is_empty() && prepared.ids.len() <= crate::cache_pages::CONTEXT - output,
            "prompt plus output exceeds context"
        );
        Ok(prepared)
    }
    pub(crate) fn prepare(&mut self, body: &[u8], output: usize) -> Result<PreparedText> {
        self.prepare_kind(body, output, true)
    }
    /// An HTTP request: any positive requested output is accepted when the
    /// prompt plus it (one token when absent) fits the context. Returns the
    /// output budget, clamped to the output policy and the remaining context.
    pub(crate) fn prepare_request(
        &mut self,
        body: &[u8],
        requested: Option<usize>,
        chat: bool,
    ) -> Result<(PreparedText, usize)> {
        use crate::cache_pages::{CONTEXT, MAX_OUTPUT};
        let prepared = self.encode(j::parse(body)?, chat)?;
        let prompt = prepared.ids.len();
        ensure!(prompt > 0, "empty prompt");
        if prompt.saturating_add(requested.unwrap_or(1)) > CONTEXT {
            return Err(ContextExceeded {
                prompt,
                requested,
                chat,
            }
            .into());
        }
        let output = requested
            .unwrap_or(MAX_OUTPUT)
            .min(MAX_OUTPUT)
            .min(CONTEXT - prompt);
        Ok((prepared, output))
    }
    #[allow(dead_code)] // Direct preparation entry point for bounded image requests.
    pub(crate) fn prepare_images(&mut self, body: &[u8], output: usize) -> Result<PreparedImages> {
        ensure!(
            (1..=crate::cache_pages::MAX_OUTPUT).contains(&output),
            "maximum output outside policy"
        );
        self.encode_images(body, output)
    }
    pub(crate) fn tokenize_images(&mut self, body: &[u8]) -> Result<PreparedImages> {
        self.encode_images(body, 0)
    }
    fn encode_images(&mut self, body: &[u8], output: usize) -> Result<PreparedImages> {
        let body = j::parse(body)?;
        raw_flags(&body, true)?;
        if let Some(history) = &self.live_clear {
            self.renderer.policy.clear = history.current();
        }
        let (rendered, references) = self
            .renderer
            .render_kind(&serde_json::to_vec(&body)?, true)?;
        let encoded = self.cache.encode(&rendered.text)?;
        ensure!(
            !encoded.ids.is_empty() && encoded.ids.len() <= crate::cache_pages::CONTEXT - output,
            "prompt plus output exceeds context"
        );
        Ok(PreparedImages {
            text: PreparedText {
                rendered,
                ids: encoded.ids.iter().map(|id| *id as i32).collect(),
                reused_ids: encoded.reused_ids,
            },
            references,
        })
    }
}

#[path = "chat_probe.rs"]
mod qualification;
pub(crate) fn probe(path: &Path) -> Result<Value> {
    qualification::probe(path)
}
#[cfg(test)]
#[path = "chat_tests.rs"]
mod tests;
