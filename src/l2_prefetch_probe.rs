//! Real CUDA lifecycle/canary probe. Only explicit CLI invocation runs this.
use super::*;
use serde_json::json;
pub(crate) fn run(library: &Path, pin: &str) -> Result<serde_json::Value> {
    let device = Device::new()?;
    ensure!(device.arch == 121, "L2 probe requires GB10 sm121");
    let main = device.stream()?;
    let native = Native::open(library, pin)?;
    let mut sms = 0;
    let mut l2 = 0;
    check(unsafe { (native.properties)(&mut sms, &mut l2) })?;
    ensure!(sms > 0 && l2 >= 0, "L2 properties");
    let bytes = 288 * 65536;
    let pattern = (0..bytes)
        .map(|i| ((i * 37 + i / 128) & 255) as u8)
        .collect::<Vec<_>>();
    let gate = device.alloc(bytes)?;
    let up = device.alloc(bytes)?;
    gate.write(0, &pattern)?;
    up.write(0, &pattern)?;
    let reference = format!("{:x}", Sha256::digest(&pattern));
    let canary = device.alloc(4096)?;
    let marker = vec![0xa7; 4096];
    canary.write(0, &marker)?;
    let ids = device.alloc(289 * 4)?;
    let count = device.alloc(4)?;
    let mut owner = Owner {
        tables: BTreeMap::new(),
        experts: BTreeMap::new(),
        fork: device.fence()?,
        joined: device.fence()?,
        side: device.stream()?,
        native,
        sms: sms as u32,
        pending: false,
        captured: false,
        stats: Stats::default(),
    };
    let tensor = |buffer: &Buffer| Tensor {
        address: buffer.ptr,
        bytes: buffer.bytes as u64,
        contiguous: true,
    };
    let expert = plan::experts(tensor(&gate), tensor(&up), 288)?.context("expert plan")?;
    let bases = expert
        .bases
        .into_iter()
        .flat_map(u64::to_ne_bytes)
        .collect::<Vec<_>>();
    let base_buffer = device.alloc(bases.len())?;
    base_buffer.write(0, &bases)?;
    owner.experts.insert(
        0,
        ExpertTable {
            buffer: base_buffer,
            plan: expert,
        },
    );
    let spans = plan::take(
        &[
            Item::Tensor(Tensor {
                address: gate.ptr + 3,
                bytes: 61,
                contiguous: true,
            }),
            Item::Scaled {
                scale: Tensor {
                    address: up.ptr,
                    bytes: 32,
                    contiguous: true,
                },
                weight: tensor(&gate),
            },
        ],
        plan::budget(l2 as u64),
    )?;
    for site in [
        Site::AttentionGather,
        Site::FeedForwardGather,
        Site::AttentionCore,
    ] {
        owner
            .tables
            .insert((0, site), table(&device, &spans, sms as u32)?);
    }
    let mut cases = Vec::new();
    for rows in [1usize, 2, 4, 8, 16, 64, 65, 512, 8192] {
        for active in [0i32, -1, 9, 400] {
            let values = (0..289)
                .map(|i| match i % 7 {
                    0 => -1,
                    1 => 288,
                    2 => 999,
                    3 => 287,
                    _ => i % 288,
                })
                .flat_map(i32::to_ne_bytes)
                .collect::<Vec<_>>();
            ids.write(0, &values)?;
            count.write(0, &active.to_ne_bytes())?;
            let before = owner.stats.clone();
            let mut work = || -> Result<()> {
                for site in [
                    Site::AttentionGather,
                    Site::AttentionCore,
                    Site::FeedForwardGather,
                ] {
                    owner.mark(&main, 0, site, rows)?;
                }
                owner.experts(&main, 0, rows, &ids, &count)?;
                owner.join(&main)
            };
            work()?;
            main.sync()?;
            let graph = unsafe { main.capture(&mut work)? };
            for _ in 0..3 {
                unsafe {
                    graph.launch(&main)?;
                }
            }
            main.sync()?;
            drop(graph);
            work()?; // Capture→eager with the exact same reusable fences.
            main.sync()?;
            let on = plan::rows_enabled(rows);
            ensure!(
                owner.stats.launches - before.launches == if on { 9 } else { 0 },
                "L2 static hook count"
            );
            ensure!(
                owner.stats.expert_launches - before.expert_launches == if on { 3 } else { 0 },
                "L2 expert hook count"
            );
            ensure!(!owner.pending, "L2 forward left an open fork");
            cases.push(
                json!({"rows":rows,"count":active,"enabled":on,"eager_capture_replay_eager":true}),
            );
        }
    }
    main.sync()?;
    ensure!(
        format!("{:x}", Sha256::digest(gate.read(0, bytes)?)) == reference,
        "L2 changed gate weights"
    );
    ensure!(
        format!("{:x}", Sha256::digest(up.read(0, bytes)?)) == reference,
        "L2 changed up weights"
    );
    ensure!(canary.read(0, 4096)? == marker, "L2 changed output canary");
    Ok(
        json!({"all":true,"native_sha256":pin,"weight_sha256":reference,"sms":sms,"l2_bytes":l2,"cases":cases,"stats":owner.stats,"scope":"retained read-hint CUDA ABI, bounded plans and eager/capture lifecycle; not model or performance qualification"}),
    )
}
