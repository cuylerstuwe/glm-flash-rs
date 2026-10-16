mod accepted_features;
mod batch_copy;
mod cache_pages;
mod cache_watch;
mod candidates;
mod candidates_probe;
mod chat;
mod chat_egress;
mod chat_history;
mod chat_ingress;
mod chat_json;
mod comm;
mod context_memory;
mod copy_capture_probe;
mod copy_gate;
mod cpu_pin;
mod cuda;
mod depth_policy;
mod draft_blas;
mod draft_chain;
mod draft_q4;
mod draft_q4_probe;
mod draft_readback;
mod drafter_weights;
mod dsa;
mod experts;
mod fast_experts;
mod front;
mod grammar_bound;
mod grammar_native;
mod grammar_prepare;
mod grammar_request;
mod grammar_runtime;
mod grammar_serial;
mod grammar_vocab;
mod graph_transition_probe;
mod grouping;
mod http_api;
mod http_metrics;
mod http_numbers;
mod http_probe;
mod http_sampling_probe;
mod http_transport;
mod image_gate;
mod index_projection;
mod indexer;
mod kda_commit;
mod kda_commit_plan;
mod kda_window;
mod kda_window_probe;
mod l2_prefetch;
mod l2_prefetch_plan;
mod masked_candidates;
mod masked_candidates_probe;
mod memory_budget;
mod model;
mod moe;
mod output;
mod output_decode;
mod output_policy;
mod output_tools;
mod packed_candidates;
mod packed_candidates_layout;
mod packed_candidates_probe;
mod phase_profile;
mod prefill;
mod prefill_partial;
mod prefill_partial_probe;
mod prepared;
mod priority;
mod production_qualification;
mod prompt_cache;
mod prompt_lookup;
mod request_broker;
mod request_metrics;
mod request_queue;
mod roce;
mod roce_config;
mod round_timing;
mod sampling;
mod sampling_probe;
mod select;
mod service_config;
mod service_health;
mod service_state;
mod session_index;
mod shared_expert;
mod sparse_attention;
mod sparse_select;
mod ssd_format;
mod ssd_lifecycle;
mod ssd_reader;
mod ssd_store;
mod ssd_worker;
mod target_taps;
mod thinking_budget;
mod tokenizer;
mod verifier_dsa_window;
mod verifier_graph_cache;
// Vision modules also serve their standalone qualification harnesses;
// the service uses a subset of each.
#[allow(dead_code)]
mod vision_codecs;
#[allow(dead_code)]
mod vision_decode;
#[allow(dead_code)]
mod vision_fetch;
mod vision_fetch_policy;
mod vision_gif;
#[allow(dead_code)]
mod vision_input;
mod vision_markup;
#[allow(dead_code)]
mod vision_native;
#[allow(dead_code)]
mod vision_preprocess;
#[allow(dead_code)]
mod vision_request;
#[allow(dead_code)]
mod vision_resize;
mod vision_service;
mod vision_source;
mod vision_tower;
#[allow(dead_code)]
mod vision_weights;
mod window_experts_probe;
mod window_gemm;
mod window_gemm_probe;
mod window_ops;
mod window_ops_probe;
use anyhow::{Context, Result, ensure};
use cuda::{Device, Kernel, arg};
use prepared::Prepared;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

fn cached(
    device: &std::rc::Rc<Device>,
    root: &Path,
    key: &str,
    name: &str,
    sha: &str,
) -> Result<Kernel> {
    let path = root.join(key).join(format!("{name}.cubin"));
    ensure!(
        format!("{:x}", Sha256::digest(std::fs::read(&path)?)) == sha,
        "kernel hash mismatch: {name}"
    );
    let metadata: serde_json::Value =
        serde_json::from_reader(File::open(path.with_extension("json"))?)?;
    ensure!(
        metadata["target"]["arch"].as_i64() == Some(device.arch as i64),
        "kernel/GPU architecture mismatch"
    );
    ensure!(
        metadata["num_ctas"] == 1
            && metadata["global_scratch_size"] == 0
            && metadata["profile_scratch_size"] == 0,
        "unsupported kernel scratch/cluster ABI"
    );
    ensure!(metadata["name"] == name, "kernel entry name mismatch");
    Kernel::load(
        device,
        &path,
        name,
        metadata["num_warps"].as_u64().context("warps")? as u32 * 32,
        metadata["shared"].as_u64().context("shared memory")? as u32,
    )
}
fn main() -> Result<()> {
    let a: Vec<_> = env::args().collect();
    if a.get(1).map(String::as_str) == Some("copy-capture-probe") {
        println!(
            "{}",
            serde_json::to_string_pretty(&copy_capture_probe::run()?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("graph-transition-probe") {
        ensure!(
            (6..=7).contains(&a.len()),
            "graph-transition-probe KERNEL_CACHE SHORT LONG REPLAYS [INIT]"
        );
        println!(
            "{}",
            graph_transition_probe::run(
                Path::new(&a[2]),
                a[3].parse()?,
                a[4].parse()?,
                a[5].parse()?,
                a.get(6).map_or("", String::as_str),
            )?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("l2-prefetch-probe") {
        ensure!(a.len() == 4, "l2-prefetch-probe NATIVE_LIBRARY SHA256");
        println!(
            "{}",
            serde_json::to_string_pretty(&l2_prefetch::probe(Path::new(&a[2]), &a[3])?)?
        );
        return Ok(());
    }
    if matches!(
        a.get(1).map(String::as_str),
        Some(
            "model-retained-speculation-roce"
                | "model-retained-speculation-nccl"
                | "model-exact-prompt-roce"
                | "model-exact-prompt-nccl"
        )
    ) {
        ensure!(
            a.len() == 10,
            "model-retained-speculation-{{roce|nccl}} TARGET_MANIFEST DRAFT_MANIFEST RANK KERNEL_CACHE NATIVE NCCL MASTER TOKEN_SETS"
        );
        let rank: usize = a[4].parse()?;
        let p = prepared::Prepared::open(Path::new(&a[2]), rank as u64)?;
        let sets: Vec<Vec<i32>> = serde_json::from_slice(&std::fs::read(&a[9])?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&model::run_retained_speculation(
                &p,
                Path::new(&a[3]),
                rank,
                Path::new(&a[5]),
                Path::new(&a[6]),
                Path::new(&a[7]),
                a[8].parse()?,
                &sets,
                a[1].ends_with("-roce"),
                a[1].starts_with("model-exact-prompt-"),
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("speculative-workloads-roce") {
        ensure!(
            a.len() == 11,
            "speculative-workloads-roce TARGET_MANIFEST DRAFT_MANIFEST RANK KERNEL_CACHE NATIVE NCCL MASTER TOKENIZER WORKLOADS"
        );
        let rank: usize = a[4].parse()?;
        let p = prepared::Prepared::open(Path::new(&a[2]), rank as u64)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&model::run_speculative_workloads(
                &p,
                Path::new(&a[3]),
                rank,
                Path::new(&a[5]),
                Path::new(&a[6]),
                Path::new(&a[7]),
                a[8].parse()?,
                Path::new(&a[9]),
                Path::new(&a[10]),
            )?)?
        );
        return Ok(());
    }
    if matches!(
        a.get(1).map(String::as_str),
        Some("model-speculative-roce" | "model-speculative-eager-roce")
    ) {
        ensure!(
            a.len() == 10,
            "model-speculative-roce TARGET_MANIFEST DRAFT_MANIFEST RANK KERNEL_CACHE NATIVE NCCL MASTER TOKEN_SETS"
        );
        let rank: usize = a[4].parse()?;
        let p = prepared::Prepared::open(Path::new(&a[2]), rank as u64)?;
        let sets: Vec<Vec<i32>> = serde_json::from_slice(&std::fs::read(&a[9])?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&model::run_speculative(
                &p,
                Path::new(&a[3]),
                rank,
                Path::new(&a[5]),
                Path::new(&a[6]),
                Path::new(&a[7]),
                a[8].parse()?,
                &sets,
                a[1] != "model-speculative-eager-roce",
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("drafter-forward-roce") {
        ensure!(
            a.len() == 9,
            "drafter-forward-roce TARGET_MANIFEST DRAFT_MANIFEST RANK KERNEL_CACHE NATIVE NCCL MASTER"
        );
        let rank: usize = a[4].parse()?;
        let p = prepared::Prepared::open(Path::new(&a[2]), rank as u64)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&model::probe_drafter(
                &p,
                Path::new(&a[3]),
                rank,
                Path::new(&a[5]),
                Path::new(&a[6]),
                Path::new(&a[7]),
                a[8].parse()?
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("drafter-q4") {
        ensure!(a.len() == 5, "drafter-q4 MANIFEST KERNEL_CACHE RANK");
        println!(
            "{}",
            serde_json::to_string_pretty(&draft_q4_probe::probe(
                Path::new(&a[2]),
                Path::new(&a[3]),
                a[4].parse()?
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("drafter-weights") {
        ensure!(a.len() == 4, "drafter-weights MANIFEST RANK");
        println!(
            "{}",
            serde_json::to_string_pretty(&drafter_weights::probe(
                Path::new(&a[2]),
                a[3].parse()?
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("window-experts") {
        ensure!(
            a.len() == 6,
            "window-experts MANIFEST KERNEL_CACHE NATIVE_DIR RANK"
        );
        let p = Prepared::open(Path::new(&a[2]), a[5].parse()?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&window_experts_probe::probe(
                &p,
                Path::new(&a[3]),
                Path::new(&a[4])
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("window-ops") {
        ensure!(a.len() == 5, "window-ops MANIFEST KERNEL_CACHE RANK");
        let p = Prepared::open(Path::new(&a[2]), a[4].parse()?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&window_ops_probe::probe(&p, Path::new(&a[3]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("window-gemm") {
        ensure!(a.len() == 5, "window-gemm MANIFEST KERNEL_CACHE RANK");
        let p = Prepared::open(Path::new(&a[2]), a[4].parse()?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&window_gemm_probe::probe(&p, Path::new(&a[3]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("kda-window") {
        ensure!(
            a.len() == 6,
            "kda-window MANIFEST KERNEL_CACHE NATIVE_DIR RANK"
        );
        let p = Prepared::open(Path::new(&a[2]), a[5].parse()?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&kda_window_probe::probe(
                &p,
                Path::new(&a[3]),
                Path::new(&a[4])
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("memory-view") {
        ensure!(a.len() == 2, "memory-view");
        println!(
            "{}",
            serde_json::to_string_pretty(&Device::new()?.memory_view()?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("serve") {
        ensure!(a.len() == 3, "serve CONFIG_JSON");
        return model::serve(service_config::Config::load(Path::new(&a[2]))?);
    }
    if a.get(1).map(String::as_str) == Some("http-probe") {
        ensure!(a.len() == 3, "http-probe TOKENIZER_JSON");
        println!(
            "{}",
            serde_json::to_string_pretty(&http_probe::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("chat-output") {
        ensure!(a.len() == 3, "chat-output TOKENIZER_JSON");
        println!(
            "{}",
            serde_json::to_string_pretty(&output::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("prepared-identity") {
        ensure!(a.len() == 4, "prepared-identity MANIFEST RANK");
        let p = Prepared::open(Path::new(&a[2]), a[3].parse()?)?;
        println!(
            "{}",
            json!({"rank":p.manifest.key["rank"],"identity":p.model_identity()?})
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("chat-prepare") {
        ensure!(a.len() == 3, "chat-prepare TOKENIZER_JSON");
        println!(
            "{}",
            serde_json::to_string_pretty(&chat::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("ssd-catalog") {
        ensure!(a.len() == 4, "ssd-catalog NEW_ISOLATED_DIRECTORY RANK");
        println!(
            "{}",
            serde_json::to_string_pretty(&ssd_lifecycle::probe(Path::new(&a[2]), a[3].parse()?)?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("ssd-reader") {
        ensure!(a.len() == 4, "ssd-reader NEW_ISOLATED_DIRECTORY RANK");
        println!(
            "{}",
            serde_json::to_string_pretty(&ssd_reader::probe(Path::new(&a[2]), a[3].parse()?)?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("ssd-worker") {
        ensure!(a.len() == 4, "ssd-worker NEW_ISOLATED_DIRECTORY RANK");
        println!(
            "{}",
            serde_json::to_string_pretty(&ssd_worker::probe(Path::new(&a[2]), a[3].parse()?)?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("ssd-store") {
        ensure!(a.len() == 4, "ssd-store NEW_ISOLATED_DIRECTORY RANK");
        println!(
            "{}",
            serde_json::to_string_pretty(&ssd_store::probe(Path::new(&a[2]), a[3].parse()?)?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("session-index") {
        ensure!(a.len() == 2, "session-index");
        println!(
            "{}",
            serde_json::to_string_pretty(&session_index::probe()?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("context-memory") {
        ensure!(a.len() == 2, "context-memory");
        println!(
            "{}",
            serde_json::to_string_pretty(&context_memory::probe()?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("sparse-attention") {
        ensure!(a.len() == 3, "sparse-attention KERNEL_CACHE");
        println!(
            "{}",
            serde_json::to_string_pretty(&sparse_attention::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("sparse-select-split") {
        ensure!(a.len() == 3, "sparse-select-split KERNEL_CACHE");
        println!(
            "{}",
            serde_json::to_string_pretty(&sparse_select::probe_split(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("sparse-select") {
        ensure!(a.len() == 3, "sparse-select KERNEL_CACHE");
        println!(
            "{}",
            serde_json::to_string_pretty(&sparse_select::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("vision-serve") {
        ensure!(a.len() == 4, "vision-serve SOCKET CONFIG_JSON");
        return vision_service::serve(Path::new(&a[2]), Path::new(&a[3]));
    }
    if a.get(1).map(String::as_str) == Some("draft-readback") {
        ensure!(a.len() == 2, "draft-readback");
        println!(
            "{}",
            serde_json::to_string_pretty(&draft_readback::probe()?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("grouping") {
        ensure!(a.len() == 3, "grouping NATIVE_DIR");
        println!(
            "{}",
            serde_json::to_string_pretty(&grouping::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("prompt-cache-probe") {
        ensure!(a.len() == 4, "prompt-cache-probe TOKENIZER_JSON TEXTS_JSON");
        println!(
            "{}",
            serde_json::to_string_pretty(&prompt_cache::probe(
                Path::new(&a[2]),
                Path::new(&a[3])
            )?)?
        );
        return Ok(());
    }
    if matches!(a.get(1).map(String::as_str), Some("roce" | "roce-failure")) {
        ensure!(
            a.len() == 6,
            "roce RANK NATIVE_DIR NCCL_LIBRARY MASTER_IP:PORT"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&(if a[1] == "roce" {
                roce::probe
            } else {
                roce::failure_probe
            })(
                a[2].parse()?,
                Path::new(&a[3]),
                Path::new(&a[4]),
                a[5].parse()?
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("packed-candidates-probe") {
        ensure!(
            a.len() == 4,
            "packed-candidates-probe NATIVE_DIR ORACLE_BIN"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&packed_candidates_probe::run(
                Path::new(&a[2]),
                Path::new(&a[3])
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("candidates-probe") {
        ensure!(a.len() == 4, "candidates-probe NATIVE_DIR ORACLE_BIN");
        println!(
            "{}",
            serde_json::to_string_pretty(&candidates_probe::probe(
                Path::new(&a[2]),
                Path::new(&a[3])
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("masked-candidates-probe") {
        ensure!(a.len() == 3, "masked-candidates-probe NATIVE_DIR");
        println!(
            "{}",
            serde_json::to_string_pretty(&masked_candidates_probe::run(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if matches!(
        a.get(1).map(String::as_str),
        Some("candidates-collective" | "candidates-collective-roce")
    ) {
        ensure!(
            a.len() == 7,
            "candidates-collective[-roce] RANK NATIVE_DIR NCCL_LIBRARY MASTER_IP:PORT ORACLE_BIN"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&candidates_probe::collective(
                a[2].parse()?,
                Path::new(&a[3]),
                Path::new(&a[4]),
                a[5].parse()?,
                Path::new(&a[6]),
                a[1].ends_with("-roce")
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("sampling-probe") {
        ensure!(a.len() == 3, "sampling-probe NATIVE_DIR");
        println!(
            "{}",
            serde_json::to_string_pretty(&sampling_probe::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("select") {
        ensure!(a.len() == 3, "select NATIVE_DIR");
        println!(
            "{}",
            serde_json::to_string_pretty(&select::probe(Path::new(&a[2]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("tokenizer-bench") {
        ensure!(
            a.len() == 5,
            "tokenizer-bench TOKENIZER_JSON TEXTS_JSON ROUNDS"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&tokenizer::bench(
                Path::new(&a[2]),
                Path::new(&a[3]),
                a[4].parse()?
            )?)?
        );
        return Ok(());
    }
    if matches!(
        a.get(1).map(String::as_str),
        Some("prefill-blas" | "prefill-blas-exact")
    ) {
        ensure!(a.len() == 4, "prefill-blas[-exact] KERNEL_CACHE NATIVE_DIR");
        let probe = if a[1] == "prefill-blas" {
            prefill::blas_probe
        } else {
            prefill::blas_exact_probe
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&probe(Path::new(&a[2]), Path::new(&a[3]))?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("tokenize") {
        ensure!(a.len() == 4, "tokenize TOKENIZER_JSON TEXT_FILE");
        let t = tokenizer::load(Path::new(&a[2]))?;
        let enc = t
            .encode_fast(std::fs::read_to_string(&a[3])?, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("{}", serde_json::to_string(enc.get_ids())?);
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("tokenize-suite") {
        ensure!(a.len() == 4, "tokenize-suite TOKENIZER_JSON TEXTS_JSON");
        let mut cache = prompt_cache::PromptCache::new(Path::new(&a[2]), 32, 256 * 1024 * 1024)?;
        let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&a[3])?)?;
        let ids = texts
            .into_iter()
            .map(|s| cache.encode(&s).map(|e| e.ids.to_vec()))
            .collect::<Result<Vec<_>>>()?;
        println!("{}", serde_json::to_string(&ids)?);
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("decode") {
        ensure!(a.len() == 4, "decode TOKENIZER_JSON TOKEN_IDS_JSON");
        let t = tokenizer::load(Path::new(&a[2]))?;
        let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(&a[3])?)?;
        println!(
            "{}",
            t.decode(&ids, false).map_err(|e| anyhow::anyhow!("{e}"))?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("prefill-split-comm") {
        ensure!(
            a.len() == 5,
            "prefill-split-comm RANK NCCL_LIBRARY MASTER_IP:PORT"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&model::probe_prefill_split(
                a[2].parse()?,
                Path::new(&a[3]),
                a[4].parse()?
            )?)?
        );
        return Ok(());
    }
    if a.get(1).map(String::as_str) == Some("comm") {
        ensure!(a.len() == 5, "comm RANK NCCL_LIBRARY MASTER_IP:PORT");
        println!(
            "{}",
            serde_json::to_string_pretty(&comm::probe(
                a[2].parse()?,
                Path::new(&a[3]),
                a[4].parse()?
            )?)?
        );
        return Ok(());
    }
    ensure!(
        a.len() >= 4,
        "usage: glm-flash-rs inspect|probe MANIFEST RANK [KERNEL_CACHE]"
    );
    let rank: u64 = a[3].parse()?;
    let p = Prepared::open(Path::new(&a[2]), rank)?;
    if matches!(
        a[1].as_str(),
        "model-batch"
            | "model-batch-roce"
            | "model-long"
            | "model-long-roce"
            | "model-profile"
            | "model-profile-roce"
            | "model-profile-batches-roce"
            | "model-profile-large-batches-roce"
            | "model-verifier-roce"
            | "model-verifier-nccl"
            | "model-verifier-sampled-roce"
            | "model-verifier-sampled-nccl"
            | "model-profile-hc-slabs-roce"
            | "model-profile-overlap-roce"
            | "model-profile-hc-pipeline-roce"
            | "model-prefill-head"
            | "model-prefill-head-roce"
            | "model-prefill-partial"
            | "model-prefill-partial-roce"
            | "model-prefill-row"
            | "model-prefill-row-roce"
            | "model-sampling"
            | "model-sampling-roce"
            | "model-restored-requests"
            | "model-restored-requests-roce"
            | "model-disk-requests"
            | "model-disk-requests-roce"
            | "model-cached-requests"
            | "model-cached-requests-roce"
            | "model-sampled-requests"
            | "model-sampled-requests-roce"
            | "model-reuse"
            | "model-reuse-roce"
            | "model-store"
            | "model-store-roce"
            | "model-ssd"
            | "model-ssd-roce"
            | "model-ssd-cooperative"
            | "model-ssd-cooperative-roce"
            | "model-owner"
            | "model-owner-roce"
            | "model-termination"
            | "model-termination-roce"
            | "model-requests"
            | "model-requests-roce"
            | "model-priority"
            | "model-priority-roce"
            | "model-ssd-controller"
            | "model-ssd-controller-roce"
            | "model-ssd-restore"
            | "model-ssd-restore-roce"
            | "model-ssd-produce"
            | "model-ssd-produce-roce"
            | "model-ssd-recover"
            | "model-ssd-recover-roce"
    ) {
        let recovery =
            a[1].starts_with("model-ssd-produce") || a[1].starts_with("model-ssd-recover");
        ensure!(
            a.len() == if recovery { 11 } else { 10 },
            "model-batch MANIFEST RANK KERNEL_CACHE NATIVE_DIR NCCL_LIBRARY MASTER_IP:PORT TOKEN_IDS_JSON MAX_NEW [RECOVERY_BUNDLE]"
        );
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[8])?)?;
        let tokens: Vec<Vec<i32>> = if raw.get(0).is_some_and(|v| v.is_array()) {
            serde_json::from_value(raw)?
        } else {
            vec![serde_json::from_value(raw)?]
        };
        let result = if recovery {
            model::run_ssd_recovery(
                &p,
                rank as usize,
                Path::new(&a[4]),
                Path::new(&a[5]),
                Path::new(&a[6]),
                a[7].parse()?,
                &tokens,
                a[9].parse()?,
                a[1].ends_with("-roce"),
                Path::new(&a[10]),
                a[1].starts_with("model-ssd-produce"),
            )?
        } else {
            (if a[1].starts_with("model-restored-requests") {
                model::run_restored_requests
            } else if a[1].starts_with("model-disk-requests") {
                model::run_disk_requests
            } else if a[1].starts_with("model-cached-requests") {
                model::run_cached_requests
            } else if a[1].starts_with("model-sampled-requests") {
                model::run_sampled_requests
            } else if a[1].starts_with("model-sampling") {
                model::run_sampling
            } else if a[1].starts_with("model-termination") {
                model::run_termination
            } else if a[1].starts_with("model-owner") {
                model::run_owner
            } else if a[1].starts_with("model-requests") {
                model::run_requests
            } else if a[1].starts_with("model-priority") {
                model::run_priority
            } else if a[1].starts_with("model-ssd-controller") {
                model::run_ssd_controller
            } else if a[1].starts_with("model-ssd-restore") {
                model::run_ssd_restore
            } else if a[1].starts_with("model-ssd-cooperative") {
                model::run_ssd_cooperative
            } else if a[1].starts_with("model-ssd") {
                model::run_ssd
            } else if a[1].starts_with("model-store") {
                model::run_store
            } else if a[1].starts_with("model-reuse") {
                model::run_reuse
            } else if a[1].starts_with("model-prefill-head") {
                model::run_prefill_head
            } else if a[1].starts_with("model-prefill-partial") {
                model::run_prefill_partial
            } else if a[1].starts_with("model-prefill-row") {
                model::run_prefill_row
            } else if a[1] == "model-profile-hc-pipeline-roce" {
                model::run_profile_hc_pipeline
            } else if a[1] == "model-profile-overlap-roce" {
                model::run_profile_overlap
            } else if a[1] == "model-profile-hc-slabs-roce" {
                model::run_profile_hc_slabs
            } else if matches!(
                a[1].as_str(),
                "model-verifier-sampled-roce" | "model-verifier-sampled-nccl"
            ) {
                model::run_verifier_sampled
            } else if matches!(a[1].as_str(), "model-verifier-roce" | "model-verifier-nccl") {
                model::run_verifier
            } else if a[1] == "model-profile-large-batches-roce" {
                model::run_profile_large_batches
            } else if a[1] == "model-profile-batches-roce" {
                model::run_profile_batches
            } else if a[1].starts_with("model-profile") {
                model::run_profile
            } else if a[1].starts_with("model-long") {
                model::run_long
            } else {
                model::run_batch
            })(
                &p,
                rank as usize,
                Path::new(&a[4]),
                Path::new(&a[5]),
                Path::new(&a[6]),
                a[7].parse()?,
                &tokens,
                a[9].parse()?,
                a[1].ends_with("-roce"),
            )?
        };
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    if matches!(
        a[1].as_str(),
        "model" | "model-fast" | "model-roce" | "model-roce-audit"
    ) {
        ensure!(
            a.len() == 10,
            "model MANIFEST RANK KERNEL_CACHE NATIVE_DIR NCCL_LIBRARY MASTER_IP:PORT TOKEN_IDS_JSON MAX_NEW"
        );
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[8])?)?;
        let tokens: Vec<Vec<i32>> = if raw.get(0).is_some_and(|v| v.is_array()) {
            serde_json::from_value(raw)?
        } else {
            vec![serde_json::from_value(raw)?]
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&model::run(
                &p,
                rank as usize,
                Path::new(&a[4]),
                Path::new(&a[5]),
                Path::new(&a[6]),
                a[7].parse()?,
                &tokens,
                a[9].parse()?,
                a[1] == "model" || a[1] == "model-roce-audit",
                a[1].starts_with("model-roce")
            )?)?
        );
        return Ok(());
    }
    if a[1] == "index-projection" {
        ensure!(a.len() == 5, "index-projection MANIFEST RANK KERNEL_CACHE");
        println!(
            "{}",
            serde_json::to_string_pretty(&index_projection::probe(&p, Path::new(&a[4]))?)?
        );
        return Ok(());
    }
    if a[1] == "indexer" {
        ensure!(a.len() == 5, "indexer MANIFEST RANK KERNEL_CACHE");
        println!(
            "{}",
            serde_json::to_string_pretty(&indexer::probe(&p, Path::new(&a[4]))?)?
        );
        return Ok(());
    }
    if a[1] == "fast-experts" {
        ensure!(a.len() == 5, "fast-experts MANIFEST RANK NATIVE_DIR");
        println!(
            "{}",
            serde_json::to_string_pretty(&fast_experts::probe(&p, Path::new(&a[4]))?)?
        );
        return Ok(());
    }
    if a[1] == "prefill-partial" {
        ensure!(
            a.len() == 6,
            "prefill-partial MANIFEST RANK KERNEL_CACHE NATIVE_DIR"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&prefill_partial_probe::run(
                &p,
                Path::new(&a[4]),
                Path::new(&a[5]),
            )?)?
        );
        return Ok(());
    }
    if a[1] == "prefill" {
        ensure!(
            a.len() == 6,
            "prefill MANIFEST RANK KERNEL_CACHE NATIVE_DIR"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&prefill::probe(&p, Path::new(&a[4]), Path::new(&a[5]))?)?
        );
        return Ok(());
    }
    if a[1] == "dsa" {
        ensure!(a.len() == 5, "dsa MANIFEST RANK KERNEL_CACHE");
        println!(
            "{}",
            serde_json::to_string_pretty(&dsa::probe(&p, Path::new(&a[4]))?)?
        );
        return Ok(());
    }
    if a[1] == "experts" {
        ensure!(a.len() == 5, "experts MANIFEST RANK EXL3_CUBIN");
        println!(
            "{}",
            serde_json::to_string_pretty(&experts::probe(&p, Path::new(&a[4]))?)?
        );
        return Ok(());
    }
    if a[1] == "front" || a[1] == "layer" {
        ensure!(
            (a[1] == "front" && a.len() == 6) || (a[1] == "layer" && a.len() == 8),
            "front MANIFEST RANK KERNEL_CACHE KDA_CUBIN; layer additionally NCCL_LIBRARY MASTER_IP:PORT"
        );
        let tp = if a[1] == "layer" {
            Some((rank as usize, Path::new(&a[6]), a[7].parse()?))
        } else {
            None
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&front::run(&p, Path::new(&a[4]), Path::new(&a[5]), tp)?)?
        );
        return Ok(());
    }
    if a[1] == "inspect" {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"rank":rank,"format":p.manifest.format,"storages":p.manifest.storages.len(),"tensor_views":p.tensors.len(),"file_bytes":p.manifest.file_size,"dtypes":p.tensors.values().map(|t|t.dtype.clone()).collect::<std::collections::BTreeSet<_>>()})
            )?
        );
        return Ok(());
    }
    ensure!(
        a[1] == "probe" && a.len() == 5,
        "expected probe MANIFEST RANK KERNEL_CACHE"
    );
    let d = Device::new()?;
    p.expect("weights.embed", "bfloat16", &[154880, 4096])?;
    ensure!(d.arch == 121, "this experiment targets GB10 sm121");
    let m = p.load(&d)?;
    let kernel = cached(
        &d,
        Path::new(&a[4]),
        "WABKC7362HBZHW5CASQJVYZZZSF6SDDOX33P7Q4H26HJAGVXLGSQ",
        "_embed_b16",
        "c5177ba3d9e89ef218348674466c693bad91d1ae081e3d97844fecb748d8e4c5",
    )?;
    let ids: [i32; 8] = [0, 1, 42, 8192, 154819, 154820, 154829, 154879];
    let id_data: Vec<_> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let input = d.alloc(id_data.len())?;
    input.write(0, &id_data)?;
    let output = d.alloc(ids.len() * 4 * 4096 * 2)?;
    output.zero()?;
    let stream = d.stream()?;
    let launch = || -> Result<()> {
        let mut i = input.ptr;
        let mut w = m.pointer(&p, "weights.embed")?;
        let mut o = output.ptr;
        let mut scratch = 0u64;
        let mut profile = 0u64;
        unsafe {
            kernel.launch(
                &stream,
                [ids.len() as u32, 64, 1],
                &mut [
                    arg(&mut i),
                    arg(&mut w),
                    arg(&mut o),
                    arg(&mut scratch),
                    arg(&mut profile),
                ],
            )
        }
    };
    launch()?;
    stream.sync()?;
    let actual = output.read(0, output.bytes)?;
    let t = p.tensor("weights.embed")?;
    ensure!(
        t.dtype == "bfloat16" && t.shape == [154880, 4096] && t.stride == [4096, 1],
        "embedding layout changed"
    );
    let mut file = File::open(&p.data)?;
    let mut row = vec![0u8; 8192];
    for (r, &id) in ids.iter().enumerate() {
        file.seek(SeekFrom::Start(
            p.manifest.storages[t.storage].offset + t.off * 2 + id as u64 * 8192,
        ))?;
        file.read_exact(&mut row)?;
        for s in 0..4 {
            ensure!(
                actual[(r * 4 + s) * 8192..(r * 4 + s + 1) * 8192] == row,
                "embedding mismatch token {id}, stream {s}"
            );
        }
    }
    // Captured and eager launches must produce the same bytes. This only times embedding dispatch,
    // and is not an inference-throughput measurement.
    let graph = unsafe { stream.capture(launch)? };
    for _ in 0..10 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    let start = Instant::now();
    for _ in 0..1000 {
        launch()?;
    }
    stream.sync()?;
    let eager_us = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    for _ in 0..1000 {
        unsafe {
            graph.launch(&stream)?;
        }
    }
    stream.sync()?;
    let graph_us = start.elapsed().as_secs_f64() * 1000.;
    ensure!(
        output.read(0, output.bytes)? == actual,
        "graph/eager mismatch"
    );
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    ensure!(
        !maps.contains("libpython") && !maps.contains("libtorch"),
        "unexpected Python/Torch runtime dependency"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"milestone":"native_weight_load_and_embedding","full_inference":false,"rank":rank,"device":d.name,"arch":d.arch,"verified_weight_bytes":p.manifest.file_size,"gpu_weight_bytes":m.bytes,"load_seconds":m.seconds,"tensor_views":p.tensors.len(),"embedding_tokens":ids,"embedding_byte_equal":true,"graph_byte_equal":true,"eager_embedding_us":eager_us,"graph_embedding_us":graph_us,"python_or_torch_loaded":false,"output_sha256":format!("{:x}",Sha256::digest(&actual))})
        )?
    );
    Ok(())
}
