use super::*;
use sha2::{Digest, Sha256};
use std::time::Instant;

fn ids() -> impl FnMut() -> Result<String> {
    let mut next = 0u64;
    move || {
        next += 1;
        Ok(format!("call_{next:024x}"))
    }
}
fn check_decode(table: &Table, tok: &crate::tokenizer::NativeTokenizer, ids: &[u32]) -> Result<()> {
    let expected = tok.decode(ids, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut actual = String::new();
    let mut decoder = Utf8::default();
    for &id in ids {
        actual.push_str(&decoder.push(table.get(id)?)?);
    }
    actual.push_str(&decoder.finish()?);
    ensure!(actual == expected, "tokenizer decode differs for {ids:?}");
    Ok(())
}
pub(super) fn probe(path: &Path) -> Result<Value> {
    let tok = crate::tokenizer::load(path)?;
    let table = Arc::new(Table::new(&tok)?);
    let mut vocab: Vec<_> = tok.get_vocab(true).values().copied().collect();
    vocab.sort_unstable();
    for &id in &vocab {
        check_decode(&table, &tok, &[id])?;
    }
    let mut byte_ids = [u32::MAX; 256];
    for &id in &vocab {
        if let [b] = table.get(id)? {
            byte_ids[*b as usize] = id;
        }
    }
    ensure!(
        byte_ids.iter().all(|&id| id != u32::MAX),
        "missing byte alphabet"
    );
    for first in byte_ids {
        for second in byte_ids {
            check_decode(&table, &tok, &[first, second])?;
        }
    }
    let mut seed = 0x35749204352au64;
    for length in 1..=97 {
        for _ in 0..32 {
            let sequence: Vec<_> = (0..length)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    vocab[seed as usize % vocab.len()]
                })
                .collect();
            check_decode(&table, &tok, &sequence)?;
        }
    }
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../fixtures/chat/output-text.json"))?;
    let mut texts = Vec::new();
    for f in &fixtures {
        let input = f["input"].as_str().unwrap();
        let encoded = tok
            .encode_fast(input, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut stream = Stream::new(
            table.clone(),
            f["thinking"].as_bool().unwrap(),
            f["tools"].as_bool().unwrap(),
            &f["stop"],
            encoded.len(),
        )?;
        let mut chunks = Delta::default();
        for &id in encoded.get_ids() {
            let (d, done) = stream.push(id)?;
            chunks.reasoning += &d.reasoning;
            chunks.content += &d.content;
            if done {
                break;
            }
        }
        let d = stream.finish()?;
        chunks.reasoning += &d.reasoning;
        chunks.content += &d.content;
        ensure!(
            chunks.reasoning == f["reasoning"]
                && chunks.content == f["visible"]
                && stream.text.raw_answer == f["raw_answer"]
                && stream.text.stopped == f["stopped"],
            "tokenized output fixture: {}",
            f["name"]
        );
        texts.push(json!({"name":f["name"],"input_tokens":encoded.len(),"used_tokens":stream.ids.len(),"stopped":stream.text.stopped}));
    }
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../fixtures/chat/output-tools.json"))?;
    let mut memory = crate::chat_history::Memory::new(1024, 32 << 20, 144 << 20);
    for f in &fixtures {
        let (content, calls) = crate::output_tools::parse(
            f["input"].as_str().unwrap(),
            f["tools"].as_array().unwrap(),
            &mut ids(),
        )?;
        ensure!(
            content == f["content"] && &calls == f["calls"].as_array().unwrap(),
            "output tool fixture: {}",
            f["name"]
        );
        let encoded = tok
            .encode_fast(f["input"].as_str().unwrap(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut stream = Stream::new(
            table.clone(),
            false,
            !f["tools"].as_array().unwrap().is_empty(),
            &Value::Null,
            131072,
        )?;
        for &id in encoded.get_ids() {
            stream.push(id)?;
        }
        stream.push(decode::OBSERVATION)?;
        let reply = stream.reply(&json!({"tools":f["tools"]}), &mut memory, &mut ids())?;
        let content = if f["name"] == "incomplete-retained-by-parser" {
            "a"
        } else {
            f["content"].as_str().unwrap()
        };
        ensure!(
            reply["content"] == content
                && reply["calls"] == f["calls"]
                && reply["completion_tokens"] == encoded.len() + 1,
            "final tool reply: {}",
            f["name"]
        );
    }
    // Exercise final reply, observation-in-reasoning recovery and memory insertion.
    let body = json!({"messages":[{"role":"user","content":"ask"}],"tools":[{"type":"function","function":{"name":"Run"}}],"parallel_tool_calls":false});
    let input = "Reason.\n<tool_call>Run</tool_call>\n<tool_call>Run</tool_call>";
    let encoded = tok
        .encode_fast(input, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut stream = Stream::new(table.clone(), true, true, &Value::Null, 131072)?;
    for &id in encoded.get_ids() {
        ensure!(!stream.push(id)?.1, "early finish");
    }
    ensure!(
        stream.push(decode::OBSERVATION)?.1,
        "observation did not finish"
    );
    let reply = stream.reply(&body, &mut memory, &mut ids())?;
    ensure!(
        reply["reasoning"] == "Reason."
            && reply["content"] == ""
            && reply["calls"].as_array().unwrap().len() == 1
            && reply["finish"] == "tool_calls"
            && reply["completion_tokens"] == encoded.len() + 1,
        "thinking-call recovery differs"
    );
    let mut followup = json!({"messages":[{"role":"user","content":"ask"},
        {"role":"assistant","content":null,"tool_calls":reply["calls"]},
        {"role":"tool","tool_call_id":reply["calls"][0]["id"],"content":"ok"}]});
    crate::chat_history::prepare(&mut followup, &mut memory)?;
    ensure!(
        followup["messages"][1]["reasoning_content"] == "Reason.",
        "output history roundtrip differs"
    );
    let original: Vec<Vec<u32>> =
        serde_json::from_str(include_str!("../fixtures/chat/original-output-ids.json"))?;
    let mut originals = Vec::new();
    for sequence in &original {
        let mut s = Stream::new(table.clone(), false, false, &Value::Null, 131072)?;
        for &id in sequence {
            s.push(id)?;
        }
        let reply = s.reply(&json!({}), &mut memory, &mut ids())?;
        ensure!(
            reply["finish"] == "stop" && reply["completion_tokens"] == sequence.len(),
            "original EOS/usage differs"
        );
        let expected = tok
            .decode(
                &sequence
                    .iter()
                    .copied()
                    .filter(|id| !decode::EOS.contains(id))
                    .collect::<Vec<_>>(),
                false,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            reply["content"] == expected,
            "original reply decode differs"
        );
        originals.push(reply);
    }
    // Full output-token policy, CPU only. No claim of 128K GPU generation.
    let start = Instant::now();
    let mut s = Stream::new(table.clone(), false, false, &Value::Null, 131072)?;
    for i in 0..131072 {
        let (_, done) = s.push(19)?;
        ensure!(done == (i == 131071), "length stopped incorrectly");
    }
    s.finish()?;
    ensure!(
        s.text.visible == "4".repeat(131072) && s.ids.len() == 131072,
        "full output bound"
    );
    ensure!(
        s.push(19).is_err() && s.finish().is_err(),
        "terminal stream accepted mutation"
    );
    let maximum_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut prefix = Stream::new(
        table.clone(),
        false,
        false,
        &json!("4".repeat(4096) + "X"),
        131072,
    )?;
    for _ in 0..131072 {
        prefix.push(19)?;
    }
    let reply = prefix.reply(&json!({}), &mut memory, &mut ids())?;
    ensure!(
        reply["content"] == "4".repeat(131072)
            && reply["finish"] == "length"
            && reply["final"]["content"] == "4".repeat(4096),
        "long stop-prefix holdback"
    );
    let maximum_stop_seconds = start.elapsed().as_secs_f64();
    let mut stopped = Stream::new(table.clone(), false, false, &json!("44"), 131072)?;
    ensure!(
        !stopped.push(19)?.1 && stopped.push(19)?.1,
        "stop token boundary"
    );
    let reply = stopped.reply(&json!({}), &mut memory, &mut ids())?;
    ensure!(
        reply["completion_tokens"] == 2
            && reply["token_ids"] == json!([19, 19])
            && reply["content"] == ""
            && reply["finish"] == "stop",
        "stop usage"
    );
    for eos in decode::EOS {
        let mut end = Stream::new(table.clone(), false, false, &Value::Null, 131072)?;
        end.push(19)?;
        ensure!(end.push(eos)?.1, "EOS not terminal");
        let reply = end.reply(&json!({}), &mut memory, &mut ids())?;
        ensure!(
            reply["content"] == "4" && reply["completion_tokens"] == 2 && reply["finish"] == "stop",
            "EOS usage"
        );
    }
    let mut bad = Stream::new(table.clone(), false, false, &Value::Null, 131072)?;
    ensure!(
        bad.push(154879).is_err()
            && bad.push(19).is_err()
            && bad.reply(&json!({}), &mut memory, &mut ids()).is_err(),
        "output error did not poison stream"
    );
    let active = Stream::new(table.clone(), false, false, &Value::Null, 131072)?;
    ensure!(
        active.reply(&json!({}), &mut memory, &mut ids()).is_err(),
        "active request finalized successfully"
    );
    ensure!(
        Stream::new(table.clone(), false, false, &Value::Null, 131073).is_err()
            && table.get(154879).is_err(),
        "bounds not refused"
    );
    Ok(
        json!({"vocabulary_ids":vocab.len(),"byte_pairs":65536,"random_sequences":97*32,
        "text_goldens":texts,"tool_goldens":fixtures.len(),"final_tool_goldens":fixtures.len(),"thinking_tool_recovery":true,
        "parallel_false":true,"retained_reasoning_roundtrip":true,"original_model_outputs":originals,
        "maximum_output_tokens":131072,"maximum_output_bytes":decode::MAX_BYTES,
        "full_output_cpu_seconds":maximum_seconds,"long_stop_prefix_cpu_seconds":maximum_stop_seconds,
        "stop_usage_and_all_eos_checked":true,"failed_and_active_streams_refused":true,
        "maximum_output_sha256":format!("{:x}",Sha256::digest(&s.text.visible)),
        "terminal_and_bounds_refused":true,"python_executed":false,"model_executed":false,
        "note":"CPU component qualification. Tokenizer whole-sequence differential and hand-specified frozen output goldens. Scheduler termination, SSE, sampling and native request/cache integration require separate serving tests."}),
    )
}
