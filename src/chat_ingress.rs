//! CPU preparation after the qualified native receipt. The caller must reserve
//! the Client before handing over a body reader, and enforce reader IO deadlines.
use crate::{
    chat::{PreparedText, Preparer},
    chat_json::BODY_BYTES,
    request_broker::Client,
};
use anyhow::{Context, Result, ensure};
use std::{io::Read, sync::Mutex, time::Duration};

fn body(client: &Client, input: impl Read, wait: Duration) -> Result<Vec<u8>> {
    client.prepared(wait)?;
    let mut bytes = Vec::new();
    input
        .take((BODY_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= BODY_BYTES, "chat body exceeds 16 MiB");
    Ok(bytes)
}
/// Owning Client guarantees errors drop/cancel its native preparation ticket.
/// Wait before taking the tokenizer lock; never make another CPU preparer wait
/// behind a native receipt. GPU values and CUDA calls stay on the model owner.
pub(crate) fn submit(
    client: Client,
    input: impl Read,
    preparation: &Mutex<Preparer>,
    output: usize,
    wait: Duration,
) -> Result<(Client, PreparedText)> {
    let bytes = body(&client, input, wait)?;
    let prepared = preparation
        .lock()
        .map_err(|_| anyhow::anyhow!("preparation owner failed"))?
        .prepare(&bytes, output)
        .context("chat preparation failed")?;
    ensure!(
        prepared
            .rendered
            .body
            .get("draft")
            .is_none_or(serde_json::Value::is_boolean),
        "draft boolean required"
    );
    let policy = crate::output_policy::Policy::new(
        prepared.rendered.thinking,
        prepared.rendered.body["tools"]
            .as_array()
            .is_some_and(|a| !a.is_empty()),
        &prepared.rendered.body["stop"],
    )?
    .with_thinking_budget(crate::thinking_budget::parse(
        &prepared.rendered.body,
        true,
        prepared.rendered.thinking,
    )?)?
    .with_grammar(crate::grammar_bound::Bound::from_request(
        &prepared.rendered.body,
        &prepared.ids,
    )?)
    .with_drafts(prepared.rendered.body.get("draft") != Some(&serde_json::Value::Bool(false)));
    client.submit_policy(prepared.ids.clone(), output, Some(policy))?;
    Ok((client, prepared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_broker::{Action, Broker, Receipt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Count<'a>(&'a AtomicUsize);
    impl Read for Count<'_> {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        }
    }
    #[test]
    fn no_body_read_without_native_receipt() {
        let (broker, _owner) = Broker::new();
        let client = broker.begin(false, Duration::from_secs(10)).unwrap();
        let reads = AtomicUsize::new(0);
        assert!(body(&client, Count(&reads), Duration::ZERO).is_err());
        assert_eq!(reads.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn oversized_read_is_bounded_and_owned_failure_releases_ticket() {
        let (broker, owner) = Broker::new();
        let client = broker.begin(false, Duration::from_secs(10)).unwrap();
        let action = owner.next(Duration::ZERO);
        owner
            .applied(
                action,
                Some(7),
                None,
                Receipt {
                    applied: 1,
                    held: true,
                },
                None,
            )
            .unwrap();
        assert!(body(&client, std::io::repeat(b'x'), Duration::ZERO).is_err());
        drop(client);
        assert!(matches!(owner.next(Duration::ZERO), Action::Release(..)));
    }
}
