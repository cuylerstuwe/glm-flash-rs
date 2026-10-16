//! Trusted in-process output bridge. Native successful completion must arrive
//! before final parsing/history insertion. Callback failure drops the client.
use crate::{
    chat::{PreparedText, Preparer},
    output::{Delta, Stream},
    output_decode::Table,
    request_broker::{Client, Event},
    request_queue::End,
};
use anyhow::{Result, bail};
use serde_json::Value;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

pub(crate) fn consume(
    client: Client,
    prepared: PreparedText,
    table: Arc<Table>,
    preparation: &Mutex<Preparer>,
    output: usize,
    mut emit: impl FnMut(Option<&Delta>) -> Result<bool>,
    ids: &mut impl FnMut() -> Result<String>,
) -> Result<Value> {
    let PreparedText {
        rendered,
        ids: prompt,
        ..
    } = prepared;
    drop(prompt); // The submitted native request owns its immutable prompt Arc.
    let crate::chat::Rendered {
        text,
        thinking,
        body,
    } = rendered;
    drop(text); // No need to retain a second rendered prompt during generation.
    let body = &body;
    let mut stream = Stream::new(
        table,
        thinking,
        body["tools"].as_array().is_some_and(|a| !a.is_empty()),
        &body["stop"],
        output,
    )?
    .with_ignore_eos(body["ignore_eos"] == true);
    loop {
        match client.receive(Duration::from_secs(1))? {
            Some(Event::Token(token)) => {
                let (delta, _) = stream.push(u32::try_from(token)?)?;
                if !emit(Some(&delta))? {
                    bail!("output consumer disconnected");
                }
            }
            Some(Event::End(End::Complete)) => {
                let mut reply = preparation
                    .lock()
                    .map_err(|_| anyhow::anyhow!("preparation owner failed"))?
                    .finish_output(stream, body, ids)?;
                reply["cached_tokens"] = serde_json::json!(client.cached_tokens()?);
                return Ok(reply);
            }
            Some(Event::End(reason)) => bail!("native request ended without success: {reason:?}"),
            None => {
                if !emit(None)? {
                    bail!("output consumer disconnected");
                }
            }
        }
    }
}
