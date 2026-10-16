//! Sampled socket requests checked against independent native cold references.
use crate::{http_api::Server, http_probe::*, request_broker::Broker};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{io::Read, net::SocketAddr, thread};

fn sampled() -> Value {
    let mut b = chat();
    b.as_object_mut().unwrap().remove("temperature");
    b["seed"] = json!(450);
    b
}
fn reply(addr: SocketAddr, b: &Value, expected: &Value, reference: &str) -> Result<Value> {
    let (status, text) = call(addr, "POST", "/v1/chat/completions", &auth(), b)?;
    ensure!(status == 200, "sampled HTTP status {status}: {text}");
    let v: Value = if b["stream"] == true {
        let chunks = events(&text)?;
        let content: String = chunks
            .iter()
            .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
            .collect();
        ensure!(content == "4", "sampled SSE content");
        chunks.last().unwrap().clone()
    } else {
        serde_json::from_str(&text)?
    };
    ensure!(
        v["tensorfold"]["token_ids"] == *expected
            && v["usage"]["completion_tokens"] == expected.as_array().unwrap().len(),
        "sampled HTTP tokens/usage"
    );
    Ok(json!({"reference":reference,"response":v}))
}
pub(crate) fn exercise(
    server: Server,
    broker: &Broker,
    eos: &Value,
    wide: &Value,
) -> Result<Value> {
    let addr = server.address;
    for change in [
        json!({"temperature":"nan"}),
        json!({"seed":true}),
        json!({"top_k":20.5}),
        json!({"temperature":0,"top_p":false}),
        json!({"seed":"20.0"}),
    ] {
        let mut b = sampled();
        b["stream"] = json!(true);
        for (k, v) in change.as_object().unwrap() {
            b[k] = v.clone();
        }
        let (status, text) = call(addr, "POST", "/v1/chat/completions", &auth(), &b)?;
        ensure!(
            status == 400 && serde_json::from_str::<Value>(&text)?["error"].is_object(),
            "invalid sampled input reached SSE headers"
        );
    }
    let mut results = Vec::new();
    results.push(reply(addr, &sampled(), eos, "eos")?);
    let mut explicit = sampled();
    explicit["temperature"] = json!(1);
    explicit["top_k"] = json!(20);
    explicit["top_p"] = json!(0.95);
    results.push(reply(addr, &explicit, eos, "eos")?);
    let mut strings = sampled();
    strings["temperature"] = json!("1");
    strings["top_k"] = json!("2_0");
    strings["top_p"] = json!("0.95");
    strings["seed"] = json!("450");
    strings["max_tokens"] = json!("8");
    results.push(reply(addr, &strings, eos, "eos")?);
    let mut fallback = sampled();
    fallback["seed"] = json!(-1);
    fallback["top_k"] = json!(504);
    fallback["top_p"] = json!(0.5);
    results.push(reply(addr, &fallback, wide, "wide")?);
    let mut stream = sampled();
    stream["stream"] = json!(true);
    stream["stream_options"] = json!({"include_usage":true});
    results.push(reply(addr, &stream, eos, "eos")?);
    let mut stopped = sampled();
    stopped["stop"] = json!("4");
    let v = reply(addr, &stopped, &json!([19]), "one")?;
    ensure!(
        v["response"]["choices"][0]["message"]["content"].is_null()
            && v["response"]["choices"][0]["finish_reason"] == "stop",
        "sampled visible stop"
    );
    results.push(v);
    let mut limited = sampled();
    limited["max_tokens"] = json!(1);
    let v = reply(addr, &limited, &json!([19]), "one")?;
    ensure!(
        v["response"]["choices"][0]["finish_reason"] == "length",
        "sampled output limit"
    );
    results.push(v);
    control(addr, 50, true)?;
    let mut bg = open(
        addr,
        "POST",
        "/v1/chat/completions",
        &(auth() + "X-LocalAI-Mode: background\r\n"),
        &serde_json::to_vec(&stream)?,
    )?;
    role(&mut bg)?;
    results.push(reply(addr, &fallback, wide, "wide")?);
    control(addr, 51, false)?;
    let mut text = String::new();
    bg.take(64 << 20).read_to_string(&mut text)?;
    let chunks = events(&text)?;
    let end = chunks.last().unwrap();
    ensure!(
        end["tensorfold"]["token_ids"] == *eos
            && end["usage"]["completion_tokens"] == eos.as_array().unwrap().len(),
        "sampled HTTP background resume"
    );
    results.push(json!({"reference":"eos","response":end}));
    let expected = [eos.clone(), wide.clone()];
    let jobs = [sampled(), fallback]
        .into_iter()
        .zip(expected)
        .enumerate()
        .map(|(i, (b, e))| {
            thread::spawn(move || reply(addr, &b, &e, if i == 0 { "eos" } else { "wide" }))
        })
        .collect::<Vec<_>>();
    for j in jobs {
        results.push(
            j.join()
                .map_err(|_| anyhow::anyhow!("sampled HTTP client panic"))??,
        );
    }
    empty(broker)?;
    drop(server);
    ensure!(broker.status()["clients"] == 0, "sampled HTTP client leak");
    Ok(
        json!({"results":results,"invalid_before_sse":true,"default_policy_explicit_seed":true,
        "numeric_string_compatibility":"ASCII decimal","seed_modulo_64":true,"json_sse_stops_limits":true,
        "native_background_resume":true,"concurrent_clients":true,"production_ready":false}),
    )
}
