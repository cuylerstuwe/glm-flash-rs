use super::super::super::store::finish_reference;
use super::*;
use crate::request_broker::{Broker, Client, Event};
fn collect(c: &Client) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    loop {
        match c.receive(Duration::from_secs(20))? {
            Some(Event::Token(t)) => out.push(t),
            Some(Event::End(End::Complete)) => return Ok(out),
            other => anyhow::bail!("unexpected client event {other:?}"),
        }
    }
}
fn foreground(b: &Broker, tokens: Arc<[i32]>) -> Result<Vec<i32>> {
    let c = b.begin(false, Duration::from_secs(30))?;
    ensure!(
        c.prepared(Duration::from_secs(5))?.held,
        "foreground preparation not held"
    );
    c.submit(tokens, 4)?;
    collect(&c)
}
#[allow(clippy::too_many_arguments)]
pub fn run(
    p: &Prepared,
    rank: usize,
    root: &Path,
    native: &Path,
    nccl: &Path,
    master: SocketAddr,
    sets: &[Vec<i32>],
    _new_tokens: usize,
    use_roce: bool,
) -> Result<serde_json::Value> {
    let tokens = sets.last().context("owner fixture")?;
    ensure!(tokens.len() == 16384, "owner fixture must be 16K");
    let mut engine = Engine::new(p, rank, root, native, nccl, master, true, use_roce)?;
    engine.compile_graphs(tokens[0])?;
    engine.enable_prefill(root, native)?;
    let mut long = Long::new(engine, root)?;
    // Qualify command transport at each chunk edge and the complete policy
    // payload without claiming a full-context model forward pass.
    let mut wire = Wire::new(&long)?;
    let sizes = [65535usize, 65536, 65537, crate::cache_pages::CONTEXT - 1];
    for n in sizes {
        let input: Arc<[i32]> = (0..n)
            .map(|i| ((i * 17 + 3) % 154880) as i32)
            .collect::<Vec<_>>()
            .into();
        let head = (rank == 0).then(|| {
            (
                Command::Submit {
                    tokens: input.clone(),
                    output: 1,
                    background: true,
                    deadline: Duration::from_secs(60),
                    policy: None,
                },
                Duration::ZERO,
                true,
            )
        });
        let (command, _, control) = wire.receive(&long, head)?;
        let Command::Submit {
            tokens: received, ..
        } = command
        else {
            anyhow::bail!("wire fixture kind")
        };
        ensure!(
            control && received == input,
            "multi-chunk wire changed tokens"
        );
    }
    long.start(tokens.len(), 4)?;
    let cold = finish_reference(&mut long, tokens)?;
    let fg: Arc<[i32]> = vec![45; 2177].into();
    long.start(fg.len(), 4)?;
    let reference_fg = finish_reference(&mut long, &fg)?;
    let (broker, owner) = Broker::new();
    let join = if rank == 0 {
        let b = broker.clone();
        let prompt: Arc<[i32]> = tokens.clone().into();
        let fg = fg.clone();
        Some(std::thread::spawn(move || -> Result<serde_json::Value> {
            let result = (|| -> Result<serde_json::Value> {
                // Real expiry; after these submissions the owner receives no input
                // until the lease timer must resume background work.
                let held = b
                    .lease("fixture", 1, true, 1.0)?
                    .recv_timeout(Duration::from_secs(5))?
                    .map_err(anyhow::Error::msg)?;
                ensure!(held.held, "lease acknowledged without hold");
                let applied = held.applied;
                let start = Instant::now();
                let bg = b.begin(true, Duration::from_secs(30))?;
                bg.prepared(Duration::from_secs(5))?;
                bg.submit(prompt.clone(), 4)?;
                let outputs_bg = collect(&bg)?;
                let outputs_fg = foreground(&b, fg.clone())?;
                ensure!(
                    start.elapsed() >= Duration::from_millis(900),
                    "lease expiry did not delay background"
                );
                drop(bg);
                // Last foreground preparation disconnects without a request;
                // its pending release must be applied even when queue is empty.
                let abandoned = b.begin(false, Duration::from_secs(30))?;
                abandoned.prepared(Duration::from_secs(5))?;
                drop(abandoned);
                let rx = b.lease("fixture", 2, false, 1.0)?;
                let released = rx
                    .recv_timeout(Duration::from_secs(5))?
                    .map_err(anyhow::Error::msg)?;
                ensure!(
                    !released.held && released.applied > applied,
                    "idle preparation release not applied before receipt"
                );
                // Preparation validation error, preparation timeout, and a
                // disconnect after submission all retire without ticket leaks.
                let invalid = b.begin(false, Duration::from_secs(30))?;
                invalid.prepared(Duration::from_secs(5))?;
                ensure!(
                    invalid.submit(vec![154880].into(), 4).is_err(),
                    "invalid tokens accepted"
                );
                drop(invalid);
                let expiry = b.begin(false, Duration::from_millis(50))?;
                expiry.prepared(Duration::from_secs(5))?;
                ensure!(
                    expiry.receive(Duration::from_secs(5))? == Some(Event::End(End::TimedOut)),
                    "preparation timeout did not retire"
                );
                drop(expiry);
                let cancelled = b.begin(true, Duration::from_secs(30))?;
                cancelled.prepared(Duration::from_secs(5))?;
                cancelled.submit(prompt, 4)?;
                let cancelled_id = cancelled.submitted(Duration::from_secs(5))?;
                drop(cancelled);
                // Genuine concurrent client threads; each receives its own output.
                let mut joins = Vec::new();
                for _ in 0..3 {
                    let b = b.clone();
                    let fg = fg.clone();
                    joins.push(std::thread::spawn(move || foreground(&b, fg)));
                }
                let mut peers = Vec::new();
                for j in joins {
                    peers.push(j.join().map_err(|_| anyhow::anyhow!("client panic"))??);
                }
                Ok(
                    serde_json::json!({"background_outputs":outputs_bg,"foreground_outputs":outputs_fg,"concurrent_outputs":peers,"lease_elapsed_ms":start.elapsed().as_millis(),"disconnected_request_id":cancelled_id}),
                )
            })();
            b.shutdown();
            result
        }))
    } else {
        None
    };
    let mut s = Scheduler::new(&long, true)?;
    let ticks = s.serve(&mut long, (rank == 0).then_some(&owner))?;
    let client = join
        .map(|j| {
            j.join()
                .map_err(|_| anyhow::anyhow!("client fixture panic"))
        })
        .transpose()?
        .transpose()?;
    ensure!(
        s.queue.count() == 0 && s.queue.gate.local_count() == 0 && !long.priority_hold,
        "owner shutdown leaked request/ticket/hold"
    );
    let mut background = 0;
    let mut foreground = 0;
    for c in &s.queue.completed {
        if c.reason == End::Complete {
            let reference = if c.status.background {
                background += 1;
                ensure!(
                    c.status.queued_ns >= 900_000_000,
                    "idle lease timer did not hold queued work"
                );
                &cold
            } else {
                foreground += 1;
                &reference_fg
            };
            let actual = s
                .traces
                .get(&c.status.id)
                .context("missing owner trace")?
                .value(&c.outputs);
            ensure!(
                actual == *reference,
                "concurrent owner changed model output/state"
            );
        }
    }
    ensure!(background == 1 && foreground == 4, "owner completion count");
    if let Some(client) = &client {
        let id = client["disconnected_request_id"]
            .as_u64()
            .context("missing disconnected ID")?;
        ensure!(
            s.queue
                .completed
                .iter()
                .any(|c| c.status.id == id && c.reason == End::Cancelled),
            "submitted disconnect did not retire its owner"
        );
    }
    long.memory.book().check()?;
    // Same aggregate hold and revision, but a different lease identity on one
    // rank: must poison before publishing any native receipt or doing work.
    let now = s.stamp + Duration::from_secs(1);
    s.queue.gate.update(
        if rank == 3 { "different" } else { "consistent" },
        1,
        false,
        1.0,
        now,
    )?;
    let mismatch = s.pump(&mut long, (rank == 0).then_some((Command::Tick, now, true)));
    ensure!(
        mismatch.is_err() && !long.healthy,
        "lease ownership mismatch did not poison before receipt"
    );
    Ok(
        serde_json::json!({"milestone":"bounded_concurrent_ingress_owner_loop","rank":rank,"transport":if use_roce{"roce"}else{"nccl"},"prompt_tokens":tokens.len(),"reference":cold,"foreground_reference":reference_fg,"client":client,"completed":s.queue.completed,"ticks":ticks,"wire_payload_token_counts":[65535,65536,65537,1048575],"concurrent_clients_exact":true,"native_receipt_before_work":true,"lease_expiry_without_input":true,"idle_prepare_disconnect_release":true,"preparation_timeout_and_validation_cleanup":true,"shutdown_no_ticket_leak":true,"gate_identity_disagreement_before_receipt":true,"submitted_disconnect_cancelled":true,"production_ready":false,"http":false,"automatic_cache_policy":false}),
    )
}
