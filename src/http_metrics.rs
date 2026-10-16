//! Bounded, read-only views of native accounting. No GPU access or HTTP-body data.
//! Phase time includes synchronized model, sampling and attached snapshot work.
use serde_json::{Value, json};

fn number(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or(0.0)
}
fn count(v: &Value, key: &str) -> u64 {
    v[key].as_u64().unwrap_or(0)
}
fn requests(broker: &Value) -> &[Value] {
    broker["native"]["queue"]["requests"]
        .as_array()
        .map_or(&[], Vec::as_slice)
}
pub(crate) fn health(broker: Value, incarnation: &str) -> Value {
    let totals = &broker["native"]["queue"]["totals"];
    let active = requests(&broker);
    let outputs = active
        .iter()
        .fold(count(totals, "completion_tokens"), |n, r| {
            n.saturating_add(count(r, "output_tokens"))
        });
    let uptime = number(&broker, "uptime_ns") / 1e9;
    let last = totals["last_finished_ns"].as_f64();
    let fatal = broker["failed"] == true;
    json!({"ok":broker["available"]==true,"mode":"native","experimental":true,
        "production_ready":false,"incarnation":incarnation,"backend":"alaska-rust",
        "uptime_s":uptime,"busy":!active.is_empty(),"inflight":active.len(),"requests_running":active.len(),
        "oldest_s":active.iter().map(|r|number(r,"age_ns")/1e9).fold(0.0,f64::max),
        "idle_s":if active.is_empty(){last.map(|n|(uptime-n/1e9).max(0.0))}else{None},
        "requests":count(totals,"requests"),"errors":count(totals,"errors"),
        "requests_total":count(totals,"requests"),"prompt_tokens_total":count(totals,"prompt_tokens"),
        "completion_tokens_total":outputs,"cached_tokens_total":count(totals,"cached_tokens"),
        "rounds_total":count(totals,"decode_rounds"),"drafted_total":0,"accepted_total":0,
        "prefill_seconds_total":number(totals,"prefill_ns")/1e9,
        "decode_seconds_total":number(totals,"decode_ns")/1e9,
        "context_length":crate::cache_pages::CONTEXT,
        "streams":{"max":crate::cache_pages::SLOTS,
            "decoding":active.iter().filter(|r|r["phase"]=="decoding").count(),
            "prefilling":active.iter().filter(|r|r["phase"]=="prefill"||r["phase"]=="restoring").count(),
            "paused":active.iter().filter(|r|r["phase"]=="paused").count()},
        "fatal":if fatal{Some("native owner failed")}else{None},
        "timing_scope":"native active phase time including sampling and snapshot copies",
        "broker":broker})
}
pub(crate) fn metrics(broker: &Value, served: &str) -> String {
    let totals = &broker["native"]["queue"]["totals"];
    let mut out = String::new();
    let label = served
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    let mut metric = |name: &str, kind: &str, help: &str, value: f64| {
        use std::fmt::Write;
        writeln!(
            out,
            "# HELP {name} {help}\n# TYPE {name} {kind}\n{name}{{model=\"{label}\"}} {value:.6}"
        )
        .unwrap();
    };
    for (name, key, help) in [
        (
            "tensorfold_requests_total",
            "requests",
            "Native requests retired, including cancelled and failed requests",
        ),
        (
            "tensorfold_request_errors_total",
            "errors",
            "Native output errors and timeouts",
        ),
        (
            "tensorfold_requests_rejected_total",
            "rejected",
            "Requests rejected by the native owner after submission",
        ),
        (
            "tensorfold_prompt_tokens_total",
            "prompt_tokens",
            "Input tokens of retired native requests",
        ),
        (
            "tensorfold_cached_tokens_total",
            "cached_tokens",
            "Input tokens restored by retired native requests",
        ),
        (
            "tensorfold_completion_tokens_total",
            "completion_tokens",
            "Generated tokens of retired native requests",
        ),
        (
            "tensorfold_decode_rounds_total",
            "decode_rounds",
            "Serial decode calls, excluding the first prefill token",
        ),
        (
            "alaska_rust_requests_cancelled_total",
            "cancelled",
            "Cancelled native requests",
        ),
        (
            "alaska_rust_requests_timed_out_total",
            "timed_out",
            "Native request deadlines reached",
        ),
    ] {
        metric(name, "counter", help, number(totals, key));
    }
    metric(
        "tensorfold_decode_seconds_total",
        "counter",
        "Active decode phase seconds including sampling",
        number(totals, "decode_ns") / 1e9,
    );
    metric(
        "tensorfold_prefill_seconds_total",
        "counter",
        "Active prefill phase seconds including snapshot work",
        number(totals, "prefill_ns") / 1e9,
    );
    metric(
        "tensorfold_requests_inflight",
        "gauge",
        "Native requests queued, restoring, active or paused",
        requests(broker).len() as f64,
    );
    metric(
        "tensorfold_engine_fatal",
        "gauge",
        "One after the native owner fails",
        u8::from(broker["failed"] == true) as f64,
    );
    metric(
        "tensorfold_uptime_seconds",
        "gauge",
        "Seconds since native broker creation",
        number(broker, "uptime_ns") / 1e9,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_tokens_and_phase_timing_are_explicit_and_bounded() {
        let b = json!({"available":true,"failed":false,"uptime_ns":10_000_000_000u64,
            "native":{"queue":{"totals":{"requests":20,"completion_tokens":100,"prefill_ns":3_000_000_000u64,"last_finished_ns":8_000_000_000u64},
                "requests":[{"phase":"paused","output_tokens":7,"age_ns":4_000_000_000u64},{"phase":"prefill","output_tokens":0}]}}});
        let h = health(b.clone(), "test");
        assert_eq!(h["requests_total"], 20);
        assert_eq!(h["completion_tokens_total"], 107);
        assert_eq!(h["prefill_seconds_total"], 3.0);
        assert_eq!(h["streams"]["paused"], 1);
        assert!(h["idle_s"].is_null());
        let m = metrics(&b, "test\"\\\nmodel");
        assert!(m.contains("model=\"test\\\"\\\\\\nmodel\""));
        assert!(m.contains(" 100.000000\n"));
        assert!(!m.contains("NaN"));
        let absent = health(json!({"available":false,"failed":true}), "test");
        assert_eq!(absent["ok"], false);
        assert_eq!(absent["requests_total"], 0);
        assert!(!absent["fatal"].is_null());
    }
}
