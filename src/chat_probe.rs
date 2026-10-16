use super::*;
fn cases() -> Result<Vec<Value>> {
    Ok(serde_json::from_str(include_str!(
        "../fixtures/chat/rendering.json"
    ))?)
}
fn policy(v: &Value) -> Policy {
    Policy {
        thinking: v["thinking"].as_bool().unwrap(),
        clear: v["clear"].as_bool(),
        effort: match v["effort"].as_str() {
            Some("high") => Some("high"),
            Some("low") => Some("low"),
            _ => None,
        },
    }
}
#[cfg(test)]
pub(super) fn render_cases() -> Result<Vec<(String, String)>> {
    cases()?
        .into_iter()
        .map(|case| {
            let mut renderer = Renderer::new(policy(&case["policy"]))?;
            let rendered = renderer.render(&serde_json::to_vec(&case["body"])?)?;
            ensure!(
                rendered.text == case["expected"],
                "fixture {} differs: {:?}",
                case["name"],
                rendered.text
            );
            Ok((case["name"].as_str().unwrap().to_owned(), rendered.text))
        })
        .collect()
}
pub(super) fn probe(path: &Path) -> Result<Value> {
    let tokenizer = tokenizer::load(path)?;
    let mut prep = Preparer::new(
        path,
        Policy {
            thinking: true,
            effort: Some("high"),
            clear: Some(false),
        },
    )?;
    let mut records = Vec::new();
    for case in cases()? {
        prep.renderer.policy = policy(&case["policy"]);
        let body = serde_json::to_vec(&case["body"])?;
        let prepared = prep.prepare(&body, 128)?;
        ensure!(
            prepared.rendered.text == case["expected"],
            "fixture {} differs: {:?}",
            case["name"],
            prepared.rendered.text
        );
        let expected = tokenizer
            .encode_fast(prepared.rendered.text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            prepared
                .ids
                .iter()
                .map(|id| *id as u32)
                .eq(expected.get_ids().iter().copied()),
            "render token IDs differ"
        );
        let repeat = prep.prepare(&body, 128)?;
        ensure!(
            repeat.ids == prepared.ids && repeat.reused_ids == prepared.ids.len(),
            "repeat prompt cache miss"
        );
        records.push(json!({"name":case["name"],"prompt_sha256":format!("{:x}",Sha256::digest(&prepared.rendered.text)),
            "tokens":prepared.ids.len(),"token_sha256":format!("{:x}",Sha256::digest(prepared.ids.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())),
            "thinking":prepared.rendered.thinking,"repeat_reused_ids":repeat.reused_ids}));
    }
    // Existing all-layer controls were prepared without thinking or an effort
    // line. Verify these precise prompt bytes rather than inventing new controls.
    let texts: Vec<String> =
        serde_json::from_str(include_str!("../fixtures/chat/original-suite-texts.json"))?;
    prep.renderer.policy = Policy {
        thinking: false,
        effort: Some("high"),
        clear: Some(false),
    };
    for expected in &texts {
        let user = expected
            .strip_prefix("[gMASK]<sop><|user|>")
            .unwrap()
            .strip_suffix("<|assistant|><think></think>")
            .unwrap();
        let body = serde_json::to_vec(&json!({"messages":[{"role":"user","content":user}]}))?;
        ensure!(
            prep.prepare(&body, 64)?.rendered.text == *expected,
            "frozen model prompt changed"
        );
    }
    let calls = json!([{"id":"call_0123456789abcdef01234567","function":{"name":"f","arguments":{"z":1,"a":2}}}]);
    prep.renderer
        .memory
        .remember("ask", calls.as_array().unwrap(), "Remember 🦀.")?;
    let body = serde_json::to_vec(&json!({"messages":[{"role":"user","content":"ask"},
        {"role":"assistant","content":null,"tool_calls":calls},{"role":"tool","tool_call_id":"call_0123456789abcdef01234567","content":"ok"}]}))?;
    let kept = prep.prepare(&body, 64)?;
    ensure!(
        kept.rendered.text.contains("<think>Remember 🦀.</think>"),
        "retained reasoning missing"
    );
    ensure!(
        kept.rendered.body["messages"][1]["reasoning_content"] == "Remember 🦀.",
        "normalized body lost reasoning"
    );
    // Full policy is exercised in host preparation only, without allocating a
    // model or implying a 1M forward-pass qualification.
    let empty = serde_json::to_vec(&json!({"messages":[{"role":"user","content":""}]}))?;
    let overhead = prep.prepare(&empty, 1)?.ids.len();
    let mut boundaries = Vec::new();
    for output in [crate::cache_pages::MAX_OUTPUT, 1] {
        let target = crate::cache_pages::CONTEXT - output;
        let payload = " x".repeat(target - overhead);
        let bytes = serde_json::to_vec(&json!({"messages":[{"role":"user","content":payload}]}))?;
        let exact = prep.prepare(&bytes, output)?;
        ensure!(
            exact.ids.len() == target,
            "full-policy token fixture length differs"
        );
        let repeat = prep.prepare(&bytes, output)?;
        ensure!(
            repeat.ids == exact.ids && repeat.reused_ids == target,
            "long host cache differs"
        );
        let too_long = serde_json::to_vec(
            &json!({"messages":[{"role":"user","content":format!("{payload} x")}]}),
        )?;
        ensure!(
            prep.prepare(&too_long, output).is_err(),
            "context overflow accepted"
        );
        boundaries.push(json!({"prompt_tokens":target,"output_tokens":output,"repeat_reused_ids":repeat.reused_ids,"overflow_rejected":true}));
    }
    ensure!(
        prep.prepare(b"{}", 0).is_err()
            && prep
                .prepare(b"{}", crate::cache_pages::MAX_OUTPUT + 1)
                .is_err(),
        "output bounds failed"
    );
    let mut raw_checks = 0;
    for text in [
        "",
        " x",
        "🦀中é\n<|assistant|><think></think>",
        texts[0].as_str(),
    ] {
        let bytes = serde_json::to_vec(&json!({"prompt":text}))?;
        let ids = prep.tokenize(&bytes)?;
        for special in [false, true] {
            let flagged = serde_json::to_vec(&json!({"prompt":text,"add_special_tokens":special}))?;
            let actual = prep.tokenize(&flagged)?;
            let expected = tokenizer
                .encode_fast(text, special)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            ensure!(
                actual == ids
                    && actual
                        .iter()
                        .map(|id| *id as u32)
                        .eq(expected.get_ids().iter().copied()),
                "tokenize special setting IDs"
            );
            if !ids.is_empty() {
                ensure!(
                    prep.prepare_completion(&flagged, 8)?.ids == ids,
                    "raw special-token flag changed IDs"
                );
                let given = serde_json::to_vec(
                    &json!({"prompt":ids.as_ref(),"add_special_tokens":special}),
                )?;
                ensure!(
                    prep.prepare_completion(&given, 8)?.ids == ids,
                    "given token IDs changed by special flag"
                );
            }
            raw_checks += 1;
        }
        if !ids.is_empty() {
            let raw = prep.prepare_completion(&bytes, 8)?;
            let given =
                prep.prepare_completion(&serde_json::to_vec(&json!({"prompt":ids.as_ref()}))?, 8)?;
            ensure!(
                raw.ids == ids
                    && given.ids == ids
                    && !raw.rendered.thinking
                    && !given.rendered.thinking,
                "raw preparation equality"
            );
        }
    }
    let raw_full = vec![19; crate::cache_pages::CONTEXT - 1];
    let body = serde_json::to_vec(&json!({"prompt":raw_full}))?;
    ensure!(
        prep.prepare_completion(&body, 1)?.ids.len() == crate::cache_pages::CONTEXT - 1,
        "full raw prompt"
    );
    ensure!(
        prep.prepare_completion(&body, 2).is_err(),
        "raw total context overflow"
    );
    for prompt in [
        json!([]),
        json!([true]),
        json!([19.0]),
        json!([-1]),
        json!([154879]),
        json!([[19]]),
        json!(["19"]),
    ] {
        ensure!(
            prep.prepare_completion(&serde_json::to_vec(&json!({"prompt":prompt}))?, 8)
                .is_err(),
            "invalid raw prompt"
        );
    }
    let ingress = ingress_probe(prep)?;
    Ok(
        json!({"template_sha256":TEMPLATE_SHA,"fixtures":records,"frozen_model_prompts":texts.len(),"retained_reasoning":true,"host_context_boundaries":boundaries,"ingress":ingress,
        "raw_tokenization_checks":raw_checks,"raw_full_prompt":crate::cache_pages::CONTEXT-1,
        "bounded_body_bytes":j::BODY_BYTES,"bounded_render_bytes":j::RENDER_BYTES,"template_fuel":FUEL,
        "note":"Text preparation only. Hand-specified canonical prompt goldens; no Python oracle executed. Does not evaluate serving, image preparation or model quality/performance."}),
    )
}

fn ingress_probe(prep: Preparer) -> Result<Value> {
    use crate::request_broker::{Action, Broker, Command, Receipt};
    use std::{sync::Mutex, time::Duration};
    let preparation = Mutex::new(prep);
    let (broker, owner) = Broker::new();
    let bytes = serde_json::to_vec(
        &json!({"messages":[{"role":"user","content":"What is 2 + 2? Answer with one numeral."}]}),
    )?;
    let expected = preparation.lock().unwrap().prepare(&bytes, 64)?.ids;
    std::thread::scope(|scope| -> Result<()> {
        let client = broker.begin(false, Duration::from_secs(30))?;
        let worker = scope.spawn(|| {
            crate::chat_ingress::submit(
                client,
                bytes.as_slice(),
                &preparation,
                64,
                Duration::from_secs(10),
            )
        });
        let action = owner.next(Duration::from_secs(10));
        ensure!(
            matches!(action, Action::Prepare(..)),
            "missing native preparation action"
        );
        owner.applied(
            action,
            Some(77),
            None,
            Receipt {
                applied: 1,
                held: true,
            },
            None,
        )?;
        let action = owner.next(Duration::from_secs(10));
        match &action {
            Action::Submit(
                _,
                Command::Submit {
                    tokens,
                    output,
                    policy,
                    ..
                },
            ) => ensure!(
                tokens == &expected && *output == 64 && policy.is_some(),
                "prepared submission differs"
            ),
            _ => anyhow::bail!("missing prepared submission"),
        }
        owner.applied(
            action,
            None,
            Some(88),
            Receipt {
                applied: 2,
                held: true,
            },
            None,
        )?;
        let (client, prepared) = worker.join().unwrap()?;
        ensure!(prepared.ids == expected, "CPU client lost immutable prompt");
        drop(client);
        let action = owner.next(Duration::ZERO);
        ensure!(
            matches!(action, Action::Release(_, Command::Release(77))),
            "preparation ticket leaked"
        );
        owner.applied(
            action,
            None,
            None,
            Receipt {
                applied: 3,
                held: false,
            },
            None,
        )?;
        ensure!(
            matches!(
                owner.next(Duration::ZERO),
                Action::Cancel(_, Command::Cancel(88))
            ),
            "disconnect lost request cancellation"
        );
        Ok(())
    })?;
    // Malformed body cleanup uses a separate incarnation with no submitted GPU work.
    let (broker, owner) = Broker::new();
    let client = broker.begin(false, Duration::from_secs(30))?;
    let action = owner.next(Duration::ZERO);
    owner.applied(
        action,
        Some(99),
        None,
        Receipt {
            applied: 1,
            held: true,
        },
        None,
    )?;
    ensure!(
        crate::chat_ingress::submit(client, &b"{bad"[..], &preparation, 64, Duration::ZERO)
            .is_err(),
        "invalid JSON accepted"
    );
    ensure!(
        matches!(
            owner.next(Duration::ZERO),
            Action::Release(_, Command::Release(99))
        ),
        "failed preparation ticket leaked"
    );
    Ok(
        json!({"prepared_ids_exact":true,"receipt_before_body":true,"disconnect_cleanup":true,"invalid_json_cleanup":true,
        "scope":"CPU broker simulation; existing four-rank owner regression qualifies native receipts separately."}),
    )
}
