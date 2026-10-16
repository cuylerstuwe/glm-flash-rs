//! Bounded token-to-text stream. Linear incremental reasoning/tool visibility
//! and a failure-linked stop trie; completed history is never decoded again.
use crate::{
    chat_json as j,
    output_decode::{self as decode, Table, Utf8},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    path::Path,
    sync::Arc,
};

const THINK: &str = "</think>";
const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

#[derive(Default)]
struct Node {
    next: BTreeMap<u8, usize>,
    fail: usize,
    depth: usize,
    matched: usize,
}
struct Stops {
    nodes: Vec<Node>,
    state: usize,
    pending: VecDeque<(u8, usize)>,
    seen: usize,
    emitted: usize,
    hit: Option<(usize, usize)>,
}
impl Stops {
    fn new(value: &Value) -> Result<Self> {
        let strings = crate::output_policy::parse_stops(value)?;
        let mut nodes = vec![Node::default()];
        for s in strings {
            let mut at = 0;
            for byte in s.bytes() {
                at = if let Some(&next) = nodes[at].next.get(&byte) {
                    next
                } else {
                    let next = nodes.len();
                    nodes.push(Node {
                        depth: nodes[at].depth + 1,
                        ..Node::default()
                    });
                    nodes[at].next.insert(byte, next);
                    next
                };
            }
            nodes[at].matched = s.len();
        }
        let mut queue: VecDeque<_> = nodes[0].next.values().copied().collect();
        while let Some(at) = queue.pop_front() {
            let edges: Vec<_> = nodes[at].next.iter().map(|(&b, &n)| (b, n)).collect();
            for (byte, next) in edges {
                let mut fail = nodes[at].fail;
                while fail != 0 && !nodes[fail].next.contains_key(&byte) {
                    fail = nodes[fail].fail;
                }
                nodes[next].fail = nodes[fail].next.get(&byte).copied().unwrap_or(0);
                nodes[next].matched = nodes[next].matched.max(nodes[nodes[next].fail].matched);
                queue.push_back(next);
            }
        }
        Ok(Self {
            nodes,
            state: 0,
            pending: VecDeque::new(),
            seen: 0,
            emitted: 0,
            hit: None,
        })
    }
    fn push(&mut self, ch: char, raw: usize, output: &mut String) -> Result<()> {
        if self.nodes.len() == 1 {
            output.push(ch);
            self.seen += ch.len_utf8();
            self.emitted += ch.len_utf8();
            return Ok(());
        }
        let mut encoded = [0; 4];
        for (i, byte) in ch.encode_utf8(&mut encoded).bytes().enumerate() {
            self.pending.push_back((byte, raw + i));
            self.seen += 1;
            while self.state != 0 && !self.nodes[self.state].next.contains_key(&byte) {
                self.state = self.nodes[self.state].fail;
            }
            self.state = self.nodes[self.state].next.get(&byte).copied().unwrap_or(0);
            let matched = self.nodes[self.state].matched;
            if matched != 0 {
                let cut = self.pending.len() - matched;
                let raw = self.pending[cut].1;
                let start = self.seen - matched;
                if self.hit.is_none_or(|(prior, _)| start < prior) {
                    self.hit = Some((start, raw));
                }
            }
        }
        self.emit(self.pending.len() - self.nodes[self.state].depth, output)?;
        Ok(())
    }
    fn emit(&mut self, n: usize, output: &mut String) -> Result<()> {
        // Stops are UTF-8 strings. At a character boundary, an active prefix
        // starts on a character boundary too; never emit a partial character.
        let mut remaining = n;
        while remaining != 0 {
            let first = self.pending.front().unwrap().0;
            let width = if first < 128 {
                1
            } else if first < 224 {
                2
            } else if first < 240 {
                3
            } else {
                4
            };
            ensure!(width <= remaining, "stop split a UTF-8 character");
            let mut bytes = [0; 4];
            for byte in &mut bytes[..width] {
                *byte = self.pending.pop_front().unwrap().0;
            }
            output.push_str(std::str::from_utf8(&bytes[..width])?);
            remaining -= width;
        }
        self.emitted += n;
        Ok(())
    }
    fn boundary(&mut self, prior: usize, output: &mut String) -> Result<Option<usize>> {
        if let Some((cut, raw)) = self.hit {
            // Choose the earliest visible start among all matches in this
            // decoded token, as the frozen serial callback does. An earlier
            // prefix can complete later in the same token ("abcd" vs "bc").
            ensure!(cut >= prior, "stop would retract previously emitted text");
            if cut > self.emitted {
                self.emit(cut - self.emitted, output)?;
            }
            output.truncate(cut - prior);
            self.emitted = cut;
            self.pending.clear();
            Ok(Some(raw))
        } else {
            Ok(None)
        }
    }
    fn finish(&mut self, output: &mut String) -> Result<()> {
        self.emit(self.pending.len(), output)
    }
}

#[derive(Clone, Copy)]
enum Phase {
    Reasoning,
    LeadingLf,
    Answer,
    Tool,
}
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Delta {
    pub reasoning: String,
    pub content: String,
}
pub(crate) struct Text {
    phase: Phase,
    tools: bool,
    tag: VecDeque<(char, usize)>,
    stops: Stops,
    pub reasoning: String,
    pub raw_answer: String,
    pub visible: String,
    pub stopped: bool,
    bytes: usize,
}
impl Text {
    pub(crate) fn new(thinking: bool, tools: bool, stops: &Value) -> Result<Self> {
        Ok(Self {
            phase: if thinking {
                Phase::Reasoning
            } else {
                Phase::Answer
            },
            tools,
            tag: VecDeque::with_capacity(16),
            stops: Stops::new(stops)?,
            reasoning: String::new(),
            raw_answer: String::new(),
            visible: String::new(),
            stopped: false,
            bytes: 0,
        })
    }
    fn visible(&mut self, ch: char, raw: usize, delta: &mut Delta) -> Result<()> {
        self.stops.push(ch, raw, &mut delta.content)
    }
    fn step(&mut self, ch: char, delta: &mut Delta) -> Result<()> {
        if matches!(self.phase, Phase::LeadingLf) {
            if ch == '\n' {
                return Ok(());
            }
            self.phase = Phase::Answer;
        }
        let raw = self.raw_answer.len();
        if matches!(self.phase, Phase::Answer | Phase::Tool) {
            self.raw_answer.push(ch);
        }
        if matches!(self.phase, Phase::Answer) && !self.tools {
            return self.visible(ch, raw, delta);
        }
        let tag = match self.phase {
            Phase::Reasoning => THINK,
            Phase::Answer => OPEN,
            Phase::Tool => CLOSE,
            Phase::LeadingLf => unreachable!(),
        };
        self.tag.push_back((ch, raw));
        loop {
            if self
                .tag
                .iter()
                .map(|(c, _)| *c)
                .eq(tag.chars().take(self.tag.len()))
            {
                if self.tag.len() == tag.len() {
                    self.tag.clear();
                    self.phase = match self.phase {
                        Phase::Reasoning => Phase::LeadingLf,
                        Phase::Answer => Phase::Tool,
                        Phase::Tool => Phase::Answer,
                        Phase::LeadingLf => unreachable!(),
                    };
                }
                break;
            }
            let (ch, raw) = self.tag.pop_front().unwrap();
            match self.phase {
                Phase::Reasoning => delta.reasoning.push(ch),
                Phase::Answer => self.visible(ch, raw, delta)?,
                Phase::Tool => {}
                Phase::LeadingLf => unreachable!(),
            }
        }
        Ok(())
    }
    pub(crate) fn push(&mut self, text: &str) -> Result<Delta> {
        ensure!(!self.stopped, "text after stop");
        ensure!(
            text.len() <= decode::MAX_BYTES - self.bytes,
            "text output bound"
        );
        self.bytes += text.len();
        let mut delta = Delta::default();
        for ch in text.chars() {
            self.step(ch, &mut delta)?;
        }
        if let Some(cut) = self
            .stops
            .boundary(self.visible.len(), &mut delta.content)?
        {
            self.raw_answer.truncate(cut);
            self.stopped = true;
            self.tag.clear();
        }
        self.reasoning.push_str(&delta.reasoning);
        self.visible.push_str(&delta.content);
        Ok(delta)
    }
    fn finish(&mut self) -> Result<Delta> {
        let mut delta = Delta::default();
        if !self.stopped {
            while let Some((ch, raw)) = self.tag.pop_front() {
                match self.phase {
                    Phase::Reasoning => delta.reasoning.push(ch),
                    Phase::Answer => self.visible(ch, raw, &mut delta)?,
                    _ => {}
                }
            }
            if let Some(cut) = self
                .stops
                .boundary(self.visible.len(), &mut delta.content)?
            {
                self.raw_answer.truncate(cut);
                self.stopped = true;
            } else {
                self.stops.finish(&mut delta.content)?;
            }
            self.reasoning.push_str(&delta.reasoning);
            self.visible.push_str(&delta.content);
        }
        Ok(delta)
    }
}

/// Prefix of one accepted verifier window consumed by the serial output parser.
/// A terminal token is included in keep; later accepted tokens remain untouched.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Clipped {
    pub keep: usize,
    pub ended: bool,
}

pub(crate) struct Stream {
    table: Arc<Table>,
    utf8: Utf8,
    pub text: Text,
    pub ids: Vec<u32>,
    limit: usize,
    ended: bool,
    finished: bool,
    failed: bool,
    thinking: bool,
    ignore_eos: bool,
}
impl Stream {
    pub(crate) fn new(
        table: Arc<Table>,
        thinking: bool,
        tools: bool,
        stops: &Value,
        limit: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=crate::cache_pages::MAX_OUTPUT).contains(&limit),
            "output token limit"
        );
        Ok(Self {
            table,
            utf8: Utf8::default(),
            text: Text::new(thinking, tools, stops)?,
            ids: Vec::new(),
            limit,
            ended: false,
            finished: false,
            failed: false,
            thinking,
            ignore_eos: false,
        })
    }
    /// The frozen CUDA renderer always hides model EOS markers, even when
    /// model termination at those IDs is disabled for this request.
    pub(crate) fn with_ignore_eos(mut self, ignore: bool) -> Self {
        self.ignore_eos = ignore;
        self
    }
    /// Returns true at the exact token completing EOS/visible stop/length.
    /// Callers must submit this decision to the native owner at a common boundary.
    pub(crate) fn push(&mut self, id: u32) -> Result<(Delta, bool)> {
        ensure!(!self.ended && !self.finished, "tokens after stream ended");
        self.ended = true;
        self.failed = true;
        let bytes = self.table.get(id)?;
        self.ids.push(id);
        let mut delta = Delta::default();
        if !decode::EOS.contains(&id) {
            delta = self.text.push(&self.utf8.push(bytes)?)?;
        }
        self.ended = (!self.ignore_eos && decode::EOS.contains(&id))
            || self.text.stopped
            || self.ids.len() == self.limit;
        self.failed = false;
        Ok((delta, self.ended))
    }
    /// Apply an already accepted 1..=16-token window in exact serial order.
    /// Stop at the first EOS, visible stop or output limit, including its token.
    /// This advances the existing parser once per kept token: previous output is
    /// never cloned/redecoded, and a clipped suffix is not looked up or consumed.
    /// Deltas belong to the separate egress stream; the native owner needs only
    /// the keep boundary before committing model state. Any error poisons this
    /// owner rather than resetting or replaying a partially advanced parser.
    pub(crate) fn clip_accepted(&mut self, accepted: &[u32]) -> Result<Clipped> {
        if !(1..=16).contains(&accepted.len()) || self.ended || self.finished || self.failed {
            self.ended = true;
            self.failed = true;
            anyhow::bail!("invalid accepted window or ended output owner");
        }
        for (index, &id) in accepted.iter().enumerate() {
            // push marks the stream failed before any fallible mutation and
            // clears that marker only after the complete token was consumed.
            if self.push(id)?.1 {
                return Ok(Clipped {
                    keep: index + 1,
                    ended: true,
                });
            }
        }
        Ok(Clipped {
            keep: accepted.len(),
            ended: false,
        })
    }
    fn finish(&mut self) -> Result<Delta> {
        ensure!(!self.finished, "stream finalized twice");
        ensure!(
            self.ended && !self.failed,
            "cannot finalize an active/failed stream"
        );
        self.finished = true;
        self.failed = true;
        let mut delta = Delta::default();
        if !self.text.stopped {
            delta = self.text.push(&self.utf8.finish()?)?;
            let tail = self.text.finish()?;
            delta.reasoning += &tail.reasoning;
            delta.content += &tail.content;
        }
        self.failed = false;
        Ok(delta)
    }
    pub(crate) fn reply(
        mut self,
        body: &Value,
        memory: &mut crate::chat_history::Memory,
        ids: &mut impl FnMut() -> Result<String>,
    ) -> Result<Value> {
        j::check(body)?;
        let sent = self.text.visible.len();
        let final_delta = self.finish()?;
        let tools = body["tools"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        ensure!(
            self.text.tools == !tools.is_empty(),
            "reply tools differ from streaming policy"
        );
        let (mut content, mut calls) =
            crate::output_tools::parse(&self.text.raw_answer, tools, ids)?;
        let mut reasoning = self.text.reasoning.clone();
        if !tools.is_empty() {
            if calls.is_empty()
                && self.thinking
                && !self.text.stopped
                && j::strip(&content).is_empty()
                && self.ids.last() == Some(&decode::OBSERVATION)
            {
                let (head, tail) = crate::output_tools::trailing(&reasoning)?;
                if !tail.is_empty() {
                    let (_, found) = crate::output_tools::parse(tail, tools, ids)?;
                    if !found.is_empty() {
                        calls = found;
                        reasoning = head.to_owned();
                    }
                }
            }
            if body["parallel_tool_calls"] == false {
                calls.truncate(1);
            }
            if let Some(at) = content
                .rfind(OPEN)
                .filter(|&at| !content[at..].contains(CLOSE))
            {
                content.truncate(at);
                content = content.trim_end_matches(j::whitespace).to_owned();
            }
        }
        let finish = if !calls.is_empty() {
            "tool_calls"
        } else if self.text.stopped || self.ids.last().is_some_and(|t| decode::EOS.contains(t)) {
            "stop"
        } else {
            "length"
        };
        if !calls.is_empty() && self.thinking && !reasoning.is_empty() {
            let user = body["messages"]
                .as_array()
                .and_then(|a| a.iter().rfind(|m| m["role"] == "user"))
                .map(|m| crate::chat_history::text(&m["content"]))
                .unwrap_or_default();
            memory.remember(&user, &calls, &reasoning)?;
        }
        // The frozen final parser strips whitespace and can reveal an unknown
        // call, or reclassify trailing reasoning as a call. It cannot retract
        // deltas already emitted. Preserve its prefix check for the final tail.
        let tail = if content.starts_with(&self.text.visible[..sent]) {
            &content[sent..]
        } else {
            ""
        };
        Ok(
            json!({"content":content,"reasoning":reasoning,"calls":calls,"finish":finish,
            "final":{"reasoning":final_delta.reasoning,"content":tail},
            "completion_tokens":self.ids.len(),"token_ids":self.ids}),
        )
    }
}

#[path = "output_probe.rs"]
mod probe_impl;
pub(crate) fn probe(path: &Path) -> Result<Value> {
    probe_impl::probe(path)
}
#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
