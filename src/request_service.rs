//! Persistent isolated native server. This entrypoint deliberately carries no
//! deployment/router registration and keeps diagnostic hashing disabled.
use super::*;
use crate::{
    chat::{Policy, Preparer},
    http_api,
    request_broker::Broker,
    service_config::{Config, Transport, secret},
};
use std::sync::atomic::{AtomicBool, Ordering};
static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn signal(_: libc::c_int) {
    STOP.store(true, Ordering::Release);
}
struct SignalWatch {
    done: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}
impl SignalWatch {
    fn start(broker: Broker) -> Result<Self> {
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = signal as *const () as usize;
        unsafe { libc::sigemptyset(&mut action.sa_mask) };
        ensure!(
            unsafe { libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) } == 0
                && unsafe { libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) } == 0,
            "signal setup"
        );
        let done = Arc::new(AtomicBool::new(false));
        let finished = done.clone();
        let join = std::thread::spawn(move || {
            while !finished.load(Ordering::Acquire) {
                if STOP.load(Ordering::Acquire) {
                    broker.shutdown();
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        Ok(Self {
            done,
            join: Some(join),
        })
    }
}
impl Drop for SignalWatch {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}
pub fn serve(config: Config) -> Result<()> {
    crate::service_state::lease(&config.state)?;
    crate::cpu_pin::init()?;
    let rank = config.rank;
    // Resolve CPU/config errors before allocating the model or joining peers.
    let tokenizer = crate::tokenizer::load(&config.tokenizer)?;
    let table = Arc::new(Table::new(&tokenizer)?);
    let share_roles = ["<|user|>", "<|assistant|>", "<|observation|>"]
        .iter()
        .map(|role| {
            tokenizer
                .token_to_id(role)
                .map(|id| id as i32)
                .ok_or_else(|| anyhow::anyhow!("tokenizer lacks the {role} role token"))
        })
        .collect::<Result<Vec<_>>>()?;
    // Native compiler and mask workers are created before CUDA ownership/CPU
    // affinity is established. Compilation never runs on the model owner.
    let grammar = if let Some(grammar) = &config.grammar {
        let bytes = std::fs::read(&config.tokenizer)?;
        let views = Arc::new(crate::grammar_vocab::Views::new(
            &tokenizer,
            &serde_json::from_slice(&bytes)?,
        )?);
        let native = crate::grammar_native::Native::open(&grammar.library, &grammar.sha256)?;
        Some(super::grammar::State::new(native, views, &grammar.sha256)?)
    } else {
        None
    };
    drop(tokenizer);
    let mut prep = Preparer::new(
        &config.tokenizer,
        Policy {
            thinking: config.thinking,
            effort: Some(if config.reasoning_effort == "high" {
                "high"
            } else {
                "low"
            }),
            clear: config.clear_thinking_default,
        },
    )?;
    prep.live_history(
        config.clear_thinking_file.clone(),
        config.clear_thinking_default,
    );
    let http = if rank == 0 {
        let api = config.api_key_file.as_deref().map(secret).transpose()?;
        let mut c = http_api::Config::new(api.as_deref(), &secret(&config.scheduler_key_file)?)?;
        c.public = config.public_listener;
        c.qualified = !config.experimental;
        c.served = config.served_model.clone();
        c.listen = config.listen;
        c.request_timeout = Duration::from_millis(config.request_timeout_ms);
        c.vision = config.vision.as_ref().map(|v| v.socket.clone());
        c.image_fetch = config.vision.as_ref().is_some_and(|v| v.fetch);
        Some(c)
    } else {
        None
    };
    if std::env::var("ALASKA_TRANSITION_PROBE").as_deref() == Ok("1") {
        let device = crate::cuda::Device::new()?;
        crate::graph_transition_probe::service_stage(&device, &config.kernels, "before engine");
    }
    let p = Prepared::open(&config.manifest, rank as u64)?;
    let mut engine = Engine::new(
        &p,
        rank,
        &config.kernels,
        &config.native,
        &config.nccl,
        config.master,
        config.audit_logits,
        matches!(config.transport, Transport::Roce),
    )?;
    engine.compile_graphs(154822)?;
    if config.prefill_tokens == 8192 {
        engine.batch = Some(crate::model::batch::Batch::new_with_rows(
            &engine.device,
            &engine,
            &config.kernels,
            &config.native,
            8192,
        )?);
        let batch = engine.batch.as_mut().unwrap();
        batch.hc_slabs = true;
        batch.hc_pipeline = true;
        batch.overlap = true;
        batch.prepare_pipe(&engine.device)?;
    } else {
        engine.enable_prefill(&config.kernels, &config.native)?;
    }
    let mut long = Long::new(engine, &config.kernels)?;
    let bootstrap = Control::new(&long.engine.device)?;
    let graph_policy = crate::verifier_graph_cache::Policy::environment();
    let commit_policy = crate::kda_commit_plan::Policy::environment();
    let dsa_window = crate::verifier_dsa_window::Policy::environment();
    let draft_readback = crate::draft_readback::enabled();
    let shared_overlap = crate::shared_expert::enabled();
    let multi_greedy = match std::env::var("ALASKA_MULTI_GREEDY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => Ok(0u8),
        Ok("1") => Ok(1u8),
        Ok("2") => Ok(2u8),
        Ok("3") => Ok(3u8),
        Ok("4") => Ok(4u8),
        Ok("5") => Ok(5u8),
        Ok("6") => Ok(6u8),
        _ => Err(anyhow::anyhow!("ALASKA_MULTI_GREEDY must be 0/1/2/3/4/5/6")),
    };
    let mut policy = Sha256::new();
    policy.update(b"native-service-configuration-v19");
    policy.update([*multi_greedy.as_ref().unwrap_or(&0)]);
    policy.update([u8::from(*draft_readback.as_ref().unwrap_or(&false))]);
    policy.update([u8::from(*shared_overlap.as_ref().unwrap_or(&false))]);
    policy.update(graph_policy.as_ref().map_or([0; 40], |p| p.identity()));
    policy.update(commit_policy.as_ref().map_or([0; 16], |p| p.identity()));
    policy.update(dsa_window.as_ref().map_or([0; 16], |p| p.identity()));
    policy.update(long.engine.prefetch_identity);
    policy.update(long.engine.direct_prefill.identity());
    policy.update(long.engine.row_prefill.identity());
    policy.update(long.split_identity);
    policy.update([
        u8::from(config.audit_logits),
        u8::from(config.speculation.is_some()),
        u8::from(config.grammar.is_some()),
        u8::from(config.final_prefill_head_only),
        u8::from(config.exact_prompt_batch),
        u8::from(config.greedy_speculation),
    ]);
    policy.update((config.prefill_tokens as u64).to_le_bytes());
    policy.update(config.request_timeout_ms.to_le_bytes());
    if let Some(speculation) = &config.speculation {
        policy.update(speculation.digest()?);
    }
    if let Some(grammar) = &config.grammar {
        policy.update(grammar.sha256.as_bytes());
    }
    let policy: [u8; 32] = policy.finalize().into();
    // Configuration errors and mixed rank settings are rejected together before
    // any rank enters optional drafter bootstrap or graph-policy-dependent work.
    ensure!(
        bootstrap.agree(
            &long,
            848,
            policy,
            graph_policy.is_ok()
                && commit_policy.is_ok()
                && dsa_window.is_ok()
                && draft_readback.is_ok()
                && shared_overlap.is_ok()
                && multi_greedy.is_ok()
        )?,
        "service verifier policy configuration disagreed"
    );
    let graph_policy = graph_policy?;
    let commit_policy = commit_policy?;
    let dsa_window = dsa_window?;
    let _ = draft_readback?;
    let _ = shared_overlap?;
    let multi_greedy = multi_greedy?;
    let enabled = if let Some(speculation) = &config.speculation {
        long.enable_speculation_with_policies(
            &speculation.manifest,
            &config.kernels,
            &config.native,
            graph_policy,
            commit_policy,
            dsa_window,
        )
    } else {
        Ok(())
    };
    long.agree_speculation_work(&bootstrap, 849, policy, enabled)?;
    crate::graph_transition_probe::service_stage(
        &long.engine.device,
        &config.kernels,
        "engine ready",
    );
    let warmed = long.prewarm_speculation_graphs();
    long.agree_speculation_work(&bootstrap, 846, policy, warmed)?;
    crate::graph_transition_probe::service_stage(&long.engine.device, &config.kernels, "prewarmed");
    let mut scheduler = Scheduler::new(&long, false)?;
    scheduler.final_prefill_head_only = config.final_prefill_head_only;
    scheduler.exact_prompt_batch = config.exact_prompt_batch;
    scheduler.multi_greedy = multi_greedy;
    scheduler.share_roles = share_roles;
    scheduler.install_sampler(&long, &config.native)?;
    scheduler.install_output_table(&long, table.clone())?;
    if let Some(grammar) = grammar {
        scheduler.install_grammar(&long, grammar, &config.native)?;
    }
    if let Some(speculation) = &config.speculation {
        scheduler.install_speculation(
            &long,
            speculation.costs.clone(),
            speculation.lookup,
            config.greedy_speculation,
        )?;
    }
    scheduler.install_cache(&long, config.ram_bytes)?;
    scheduler.install_disk(&mut long, &config.state.join("cache"), config.ssd_bytes)?;
    let (broker, owner) = Broker::new();
    broker.ready(false);
    if rank == 0 {
        scheduler.broker = Some(broker.clone());
    }
    // Catalog initialization is cooperative and bounded, using the same clock
    // as subsequent HTTP requests. Do not advertise readiness beforehand.
    while scheduler.disk.as_ref().unwrap().controller.busy() {
        scheduler.pump(
            &mut long,
            (rank == 0).then(|| (Command::Tick, owner.now(), false)),
        )?;
        if rank == 0 {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    ensure!(
        scheduler.disk.as_ref().unwrap().controller.enabled(),
        "service SSD cache unavailable at startup"
    );
    let watch = if rank == 0 {
        SignalWatch::start(broker.clone()).map(Some)
    } else {
        Ok(None)
    };
    let server = if let Some(http) = http {
        http_api::Server::start(http, broker.clone(), prep, table).map(Some)
    } else {
        Ok(None)
    };
    // All ranks acknowledge readiness before a listener can execute a request.
    ensure!(
        scheduler
            .wire
            .control
            .agree(&long, 850, policy, server.is_ok() && watch.is_ok())?,
        "service readiness disagreement"
    );
    let watch = watch?;
    let server = server?;
    broker.ready(true);
    eprintln!(
        "native service ready rank={rank} experimental={} deployment_qualified={}",
        config.experimental, !config.experimental
    );
    crate::cpu_pin::serve_thread();
    let result = scheduler.serve(&mut long, (rank == 0).then_some(&owner));
    crate::round_timing::report();
    drop(server);
    drop(watch);
    result?;
    eprintln!("native service stopped rank={rank} drained=true");
    Ok(())
}
