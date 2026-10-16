use crate::kda_window::MAX_ROWS;
use crate::{
    cuda::{Buffer, Device, Stream},
    dsa::words,
    front::load_kernel,
    prepared::Prepared,
    window_ops::{HcWeights, Ops},
};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
fn fill(b: &Buffer, fp32: bool, seed: u32) -> Result<()> {
    let mut r = seed;
    let v = (0..b.bytes / if fp32 { 4 } else { 2 })
        .flat_map(|_| {
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            let v = (r % 8193) as f32 / 4096. - 1.;
            if fp32 {
                v.to_le_bytes().to_vec()
            } else {
                half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec()
            }
        })
        .collect::<Vec<_>>();
    b.write(0, &v)
}
fn bytes(bs: &[(&Buffer, usize)], n: usize) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    for (b, stride) in bs {
        v.extend(b.read(0, n * stride)?);
    }
    Ok(v)
}
fn check_case(
    s: &Stream,
    name: &str,
    run: impl Fn(usize) -> Result<()>,
    outputs: &[(&Buffer, usize)],
    expected: &[Vec<u8>],
) -> Result<serde_json::Value> {
    let expected_at = |n: usize| {
        let mut v = Vec::new();
        for (i, (_, stride)) in outputs.iter().enumerate() {
            v.extend(&expected[i][..n * stride]);
        }
        v
    };
    for n in 1..=MAX_ROWS {
        run(n)?;
        s.sync()?;
        ensure!(
            bytes(outputs, n)? == expected_at(n),
            "{name} differs at rows={n}"
        );
    }
    let graph = unsafe { s.capture(|| run(MAX_ROWS))? };
    for _ in 0..2 {
        unsafe { graph.launch(s)? };
        s.sync()?;
        ensure!(
            bytes(outputs, MAX_ROWS)? == expected_at(MAX_ROWS),
            "{name} graph differs"
        );
    }
    ensure!(
        run(0).is_err() && run(MAX_ROWS + 1).is_err(),
        "{name} invalid row count accepted"
    );
    s.sync()?;
    ensure!(
        bytes(outputs, MAX_ROWS)? == expected_at(MAX_ROWS),
        "{name} invalid rows changed output"
    );
    Ok(
        serde_json::json!({"component":name,"rows":(1..=MAX_ROWS).collect::<Vec<_>>(),"exact":true,"graph_replays":2,"invalid_rows_rejected":true,"serial_sha256":format!("{:x}",Sha256::digest(expected_at(MAX_ROWS)))}),
    )
}
pub fn probe(p: &Prepared, root: &Path) -> Result<serde_json::Value> {
    let d = Device::new()?;
    let s = d.stream()?;
    let ops = Ops::new(&d, root)?;
    let m = p.load_selected(
        &d,
        &[
            "weights.head",
            "weights.norm",
            "weights.layers[3].moe.router",
            "weights.layers[3].moe.bias",
            "weights.layers[3].dsa.kv_k",
            "weights.layers[3].dsa.kv_v",
            "weights.layers[3].attn_hc",
            "weights.layers[3].in_norm",
        ],
    )?;
    let w = |name: &str| m.pointer(p, name);
    let mut k = BTreeMap::new();
    for name in [
        "head",
        "stream_mean",
        "final_norm",
        "router_part",
        "router_sum",
        "topk",
        "absorb",
        "expand",
        "hc_partial",
        "hc_finish",
        "hc_post",
    ] {
        k.insert(name, load_kernel(&d, root, name)?);
    }
    let x = d.alloc(MAX_ROWS * 4 * 4096 * 2)?;
    fill(&x, false, 82)?;
    let q = d.alloc(MAX_ROWS * 4096 * 2)?;
    fill(&q, false, 93)?;
    let u = d.alloc(MAX_ROWS * 16 * 512 * 4)?;
    fill(&u, true, 94)?;
    let out = d.alloc(38720 * 2)?;
    let part = d.alloc(16 * 32 * 4 + 8 * 288 * 4)?;
    let norm = d.alloc(4096 * 2)?;
    let xs = d.alloc(64 * 4)?;
    let hidden = d.alloc(4096 * 2)?;
    let rl = d.alloc(288 * 4)?;
    let picks = d.alloc(9 * 4)?;
    let rw = d.alloc(9 * 4)?;
    let qa = d.alloc(16 * 512 * 2)?;
    let expanded = d.alloc(4096 * 2)?;
    let post = d.alloc(4 * 4)?;
    let comb = d.alloc(16 * 4)?;
    let hc_out = d.alloc(4 * 4096 * 2)?;
    let hcw = HcWeights {
        function: w("weights.layers[3].attn_hc.fn")?,
        base: w("weights.layers[3].attn_hc.base")?,
        scale: w("weights.layers[3].attn_hc.scale")?,
        norm: w("weights.layers[3].in_norm")?,
    };
    let router = w("weights.layers[3].moe.router")?;
    let bias = w("weights.layers[3].moe.bias")?;
    let wk = w("weights.layers[3].dsa.kv_k.weight")?;
    let wv = w("weights.layers[3].dsa.kv_v.weight")?;
    let wn = w("weights.norm")?;
    let wh = w("weights.head.weight")?;
    // Distinct rank contributions, including cancellation-sensitive magnitudes.
    let gathered = d.alloc(4 * MAX_ROWS * 4096 * 4)?;
    fill(&gathered, true, 96)?;
    let single_g = d.alloc(4 * 4096 * 4)?;
    let all_g = gathered.read(0, gathered.bytes)?;
    let mut head_ref = vec![Vec::new()];
    let mut route_ref = vec![Vec::new(), Vec::new(), Vec::new()];
    let mut absorb_ref = vec![Vec::new()];
    let mut expand_ref = vec![Vec::new()];
    let mut hc_ref = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut hc_post_ref = [Vec::new()];
    for r in 0..MAX_ROWS {
        let xr = x.ptr + (r * 4 * 4096 * 2) as u64;
        let qr = q.ptr + (r * 4096 * 2) as u64;
        words(&k["stream_mean"], &s, [1, 4, 1], &[xr, hidden.ptr])?;
        words(
            &k["final_norm"],
            &s,
            [1, 1, 1],
            &[
                hidden.ptr,
                4096,
                wn,
                norm.ptr,
                4096,
                xs.ptr,
                1e-5f32.to_bits() as u64,
            ],
        )?;
        words(
            &k["head"],
            &s,
            [1, 605, 1],
            &[norm.ptr, wh, out.ptr, out.ptr, 4096],
        )?;
        words(
            &k["router_part"],
            &s,
            [1, 9, 8],
            &[qr, router, part.ptr, 4096],
        )?;
        words(&k["router_sum"], &s, [1, 1, 1], &[part.ptr, rl.ptr, 288])?;
        words(
            &k["topk"],
            &s,
            [1, 1, 1],
            &[rl.ptr, bias, picks.ptr, rw.ptr, 2.5f32.to_bits() as u64],
        )?;
        words(&k["absorb"], &s, [128, 1, 1], &[qr, wk, wk, wk, qa.ptr])?;
        words(
            &k["expand"],
            &s,
            [64, 1, 1],
            &[u.ptr + (r * 16 * 512 * 4) as u64, wv, wv, wv, expanded.ptr],
        )?;
        words(
            &k["hc_partial"],
            &s,
            [1, 16, 1],
            &[xr, hcw.function, part.ptr],
        )?;
        words(
            &k["hc_finish"],
            &s,
            [1, 1, 1],
            &[
                xr,
                part.ptr,
                hcw.base,
                hcw.scale,
                hcw.norm,
                norm.ptr,
                xs.ptr,
                post.ptr,
                comb.ptr,
                1e-5f32.to_bits() as u64,
                1e-6f32.to_bits() as u64,
            ],
        )?;
        s.sync()?;
        head_ref[0].extend(out.read(0, out.bytes)?);
        for (dst, b) in route_ref.iter_mut().zip([&rl, &picks, &rw]) {
            dst.extend(b.read(0, b.bytes)?);
        }
        absorb_ref[0].extend(qa.read(0, qa.bytes)?);
        expand_ref[0].extend(expanded.read(0, expanded.bytes)?);
        for (dst, b) in hc_ref.iter_mut().zip([&norm, &xs, &post, &comb]) {
            dst.extend(b.read(0, b.bytes)?);
        }
        for rank in 0..4 {
            let a = (rank * MAX_ROWS + r) * 4096 * 4;
            single_g.write(rank * 4096 * 4, &all_g[a..a + 4096 * 4])?;
        }
        words(
            &k["hc_post"],
            &s,
            [1, 4, 1],
            &[xr, hc_out.ptr, single_g.ptr, post.ptr, comb.ptr, 4096],
        )?;
        s.sync()?;
        hc_post_ref[0].extend(hc_out.read(0, hc_out.bytes)?);
    }
    let mut cases = vec![
        check_case(
            &s,
            "head",
            |n| unsafe { ops.head(&s, &x, wn, wh, n) },
            &[(&ops.logits, 38720 * 2)],
            &head_ref,
        )?,
        check_case(
            &s,
            "router",
            |n| unsafe { ops.route(&s, &q, router, bias, n) },
            &[
                (&ops.router_logits, 288 * 4),
                (&ops.picks, 9 * 4),
                (&ops.weights, 9 * 4),
            ],
            &route_ref,
        )?,
        check_case(
            &s,
            "absorb",
            |n| unsafe { ops.absorb(&s, &q, wk, n) },
            &[(&ops.qa, 16 * 512 * 2)],
            &absorb_ref,
        )?,
        check_case(
            &s,
            "expand",
            |n| unsafe { ops.expand(&s, &u, wv, n) },
            &[(&ops.expanded, 4096 * 2)],
            &expand_ref,
        )?,
        check_case(
            &s,
            "hc_pre",
            |n| unsafe { ops.hc_pre(&s, &x, hcw, n) },
            &[
                (&ops.norm, 4096 * 2),
                (&ops.sums, 64 * 4),
                (&ops.post, 4 * 4),
                (&ops.comb, 16 * 4),
            ],
            &hc_ref,
        )?,
    ];
    // Each n uses a contiguous [4,n,4096] gather. Build outside graph capture.
    let gout = d.alloc(MAX_ROWS * 4 * 4096 * 2)?;
    for n in 1..=MAX_ROWS {
        let g = d.alloc(4 * n * 4096 * 4)?;
        for rank in 0..4 {
            let a = rank * MAX_ROWS * 4096 * 4;
            g.write(rank * n * 4096 * 4, &all_g[a..a + n * 4096 * 4])?;
        }
        unsafe {
            ops.hc_pre(&s, &x, hcw, n)?;
            ops.hc_post(&s, &x, &g, &gout, n)?;
        }
        s.sync()?;
        ensure!(
            gout.read(0, n * 4 * 4096 * 2)? == hc_post_ref[0][..n * 4 * 4096 * 2],
            "hc_post rows={n}"
        );
    }
    unsafe {
        ops.hc_pre(&s, &x, hcw, MAX_ROWS)?;
    }
    let graph = unsafe { s.capture(|| ops.hc_post(&s, &x, &gathered, &gout, MAX_ROWS))? };
    for _ in 0..2 {
        unsafe {
            graph.launch(&s)?;
        }
        s.sync()?;
        ensure!(gout.read(0, gout.bytes)? == hc_post_ref[0], "hc_post graph");
    }
    ensure!(
        unsafe { ops.hc_post(&s, &x, &gathered, &gout, 0) }.is_err()
            && unsafe { ops.hc_post(&s, &x, &gathered, &gout, MAX_ROWS + 1) }.is_err(),
        "hc_post invalid rows"
    );
    cases.push(serde_json::json!({"component":"hc_post","rows":(1..=MAX_ROWS).collect::<Vec<_>>(),"exact":true,"graph_replays":2,"invalid_rows_rejected":true,"serial_sha256":format!("{:x}",Sha256::digest(&hc_post_ref[0]))}));
    Ok(
        serde_json::json!({"rank":p.manifest.key["rank"],"device":d.name,"cases":cases,"scope":"exact verifier row operations; no speculative serving or throughput claim"}),
    )
}
