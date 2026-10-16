//! Real loopback HTTP/SSE and canonical tokenizer; scripted model tokens only.
use super::*;
use crate::{
    http_probe::{auth, call, empty, events},
    output::Stream,
    request_broker::{Action, Command, Receipt},
    request_queue::{Completion, End, Queue},
};

#[test]
#[ignore = "requires ALASKA_CANONICAL_TOKENIZER; CPU-only, no model or CUDA"]
fn raw_flags_json_sse_and_token_ids_cross_eos_with_canonical_tokenizer() -> Result<()> {
    let path = std::env::var("ALASKA_CANONICAL_TOKENIZER").context("canonical tokenizer path")?;
    let path = std::path::Path::new(&path);
    let tokenizer_bytes = std::fs::read(path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&tokenizer_bytes))
            == "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "canonical tokenizer pin"
    );
    let tokenizer = crate::tokenizer::load(path)?;
    let table = Arc::new(Table::new(&tokenizer)?);
    let one = |text| -> Result<i32> {
        let ids = tokenizer
            .encode_fast(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(ids.len() == 1, "one-token fixture");
        Ok(ids.get_ids()[0] as i32)
    };
    let [a, b, c, d] = [one("A")?, one("B")?, one("C")?, one("D")?];
    let [e0, e1, e2, e3] = crate::output_decode::EOS.map(|id| id as i32);
    let generated = [a, e0, b, e1, e2, c, e3, d];
    let (broker, owner) = Broker::new();
    let server = crate::http_probe::start(path, broker.clone(), table.clone())?;
    let address = server.address;
    let join = thread::spawn(move || -> Result<()> {
        let mut queue = Queue::default();
        let mut streams = BTreeMap::new();
        let mut ticket = 0;
        loop {
            let action = owner.next(Duration::from_millis(1));
            let (mut prepared, mut submitted) = (None, None);
            let stopping = matches!(action, Action::Stop);
            match action.command() {
                Command::Prepare { .. } => {
                    ticket += 1;
                    prepared = Some(ticket);
                }
                Command::Submit {
                    tokens,
                    output,
                    background,
                    deadline,
                    policy,
                } => {
                    let policy = policy.context("missing raw policy")?;
                    let policy = crate::output_policy::Policy::decode(&policy.encode())?;
                    ensure!(
                        policy.sampling().is_some_and(|p| p.greedy()),
                        "HTTP greedy policy lost"
                    );
                    let id = queue.submit_policy(
                        tokens,
                        output,
                        background,
                        deadline,
                        owner.now(),
                        Some(&policy),
                    )?;
                    queue.admit(id, 0, owner.now())?;
                    let stream = Stream::new(
                        table.clone(),
                        policy.thinking(),
                        policy.tools(),
                        &policy.stops(),
                        output,
                    )?
                    .with_ignore_eos(policy.ignore_eos());
                    streams.insert(id, stream);
                    submitted = Some(id);
                }
                Command::Cancel(id) => {
                    queue.cancel(id);
                    queue.finish(id, End::Cancelled, owner.now())?;
                    streams.remove(&id);
                    owner.publish(id, None, Some(End::Cancelled))?;
                }
                Command::SubmitGrammar { .. } => {
                    anyhow::bail!("conflicting grammar reached native preparation")
                }
                _ => {}
            }
            owner.applied(
                action,
                prepared,
                submitted,
                Receipt {
                    applied: 0,
                    held: false,
                },
                None,
            )?;
            if stopping {
                break;
            }
            if let Some(id) = queue.runnable() {
                let request = queue.request(id)?;
                let position = request.outputs.len();
                let stream = streams.get_mut(&id).context("scripted stream owner")?;
                let (kept, mut done) = if position == 0 {
                    let prompt = request.tokens.len();
                    stream.push(a as u32)?;
                    (1, queue.worked(id, prompt, a, Duration::ZERO)?)
                } else {
                    let offered = &generated[position..(position + 3).min(generated.len())];
                    let keep = request.sampled_prefix(offered)?;
                    let offered: Vec<_> = offered[..keep].iter().map(|id| *id as u32).collect();
                    let keep = stream.clip_accepted(&offered)?.keep;
                    (
                        keep,
                        queue.worked_many(
                            id,
                            keep,
                            &generated[position..position + keep],
                            Duration::ZERO,
                        )?,
                    )
                };
                if queue.request(id)?.stops_at(generated[position + kept - 1]) {
                    queue.terminal(id, Completion::Eos)?;
                    done = true;
                } else if stream.text.stopped {
                    queue.terminal(id, Completion::Stop)?;
                    done = true;
                }
                owner.publish_many(
                    id,
                    &generated[position..position + kept],
                    done.then_some(End::Complete),
                )?;
                if done {
                    queue.finish(id, End::Complete, owner.now())?;
                    streams.remove(&id);
                }
            }
        }
        ensure!(queue.count() == 0, "scripted requests leaked");
        Ok(())
    });
    let exercise = (|| -> Result<()> {
        let prompt = "🦀 raw <|assistant|>\n";
        let expected = tokenizer
            .encode_fast(prompt, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        for special in [false, true] {
            assert_eq!(
                tokenizer
                    .encode_fast(prompt, special)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .get_ids(),
                expected
            );
            let (status, body) = call(
                address,
                "POST",
                "/tokenize",
                &auth(),
                &json!({"prompt":prompt,"add_special_tokens":special}),
            )?;
            assert_eq!(status, 200, "{body}");
            assert_eq!(
                serde_json::from_str::<Value>(&body)?["tokens"],
                json!(expected)
            );
            for given in [json!(prompt), json!(expected)] {
                for streaming in [false, true] {
                    for (ignore, cap, stop, text, used, finish) in [
                        (false, 8, Value::Null, "A", 2, "stop"),
                        (true, 8, Value::Null, "ABCD", 8, "length"),
                        (true, 7, Value::Null, "ABC", 7, "stop"),
                        (true, 8, json!("BC"), "A", 6, "stop"),
                        (true, 8, json!("<|user|>"), "ABCD", 8, "length"),
                    ] {
                        let body = json!({"prompt":given,"temperature":0,"ignore_eos":ignore,"add_special_tokens":special,
                            "max_tokens":cap,"stop":stop,"stream":streaming,"stream_options":{"include_usage":true},"return_token_ids":true});
                        let (status, reply) =
                            call(address, "POST", "/v1/completions", &auth(), &body)?;
                        assert_eq!(status, 200, "{reply}");
                        let (reply, actual) = if streaming {
                            let values = events(&reply)?;
                            let text = values
                                .iter()
                                .filter_map(|v| v["choices"][0]["text"].as_str())
                                .collect::<String>();
                            (values.last().context("empty SSE")?.clone(), text)
                        } else {
                            let reply: Value = serde_json::from_str(&reply)?;
                            let text = reply["choices"][0]["text"]
                                .as_str()
                                .context("text completion")?
                                .to_owned();
                            (reply, text)
                        };
                        assert_eq!(actual, text);
                        assert_eq!(reply["choices"][0]["finish_reason"], finish);
                        assert_eq!(reply["usage"]["completion_tokens"], used);
                        assert_eq!(reply["usage"]["prompt_tokens"], expected.len());
                        assert_eq!(reply["tensorfold"]["token_ids"], json!(&generated[..used]));
                    }
                }
            }
        }
        for name in ["ignore_eos", "add_special_tokens"] {
            for value in [Value::Null, json!(0), json!("false"), json!([])] {
                let (status, _) = call(
                    address,
                    "POST",
                    "/v1/completions",
                    &auth(),
                    &json!({"prompt":prompt,name:value}),
                )?;
                assert_eq!(status, 400);
            }
        }
        for body in [
            json!({"prompt":prompt,"ignore_eos":true,"guided_regex":"x","stream":true}),
            json!({"prompt":prompt,"ignore_eos":true,"response_format":{"type":"json_object"}}),
        ] {
            let (status, _) = call(address, "POST", "/v1/completions", &auth(), &body)?;
            assert_eq!(status, 400);
        }
        empty(&broker)?;
        Ok(())
    })();
    drop(server);
    broker.shutdown();
    join.join()
        .map_err(|_| anyhow::anyhow!("scripted owner panic"))??;
    exercise
}
