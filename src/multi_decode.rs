//! Bounded multi-request target verification for sampled and greedy cohorts.
//! Each attention state is independent; projections and experts share a
//! row-invariant GPU batch.
use super::*;

fn multi_geometry(position: usize, window: usize) -> Result<(usize, usize)> {
    ensure!(
        position > 0
            && matches!(window, 2 | 3 | 4 | 8)
            && position
                .checked_add(window)
                .is_some_and(|end| end <= crate::cache_pages::CONTEXT)
            && (position >= 2051 || position + window <= 2051),
        "multi attention geometry crosses an incompatible dense/sparse ABI"
    );
    let end = position + window;
    Ok((
        if position < 2051 {
            end.div_ceil(512)
        } else {
            0
        },
        if position >= 2051 {
            (end / 4)
                .next_power_of_two()
                .clamp(1024, crate::cache_pages::CONTEXT / 4)
        } else {
            0
        },
    ))
}

fn accepted_counts(all: &[[u8; 64]], record: &[u8; 64]) -> Result<Vec<usize>> {
    let width = record[32] as usize;
    let window = record[38];
    ensure!(
        matches!(width, 2..=4)
            && all.len() == 4
            && record[33] == 1
            && matches!(window, 2 | 3 | 4 | 8)
            && all.iter().all(|r| r[..34] == record[..34]
                && r[38..] == record[38..]
                && r[34..34 + width].iter().all(|&keep| keep <= window)),
        "multi acceptance disagreement/invalid clipping"
    );
    (0..width)
        .map(|row| {
            if all.iter().any(|r| r[34 + row] == 0) {
                return Ok(0);
            }
            let keep = record[34 + row];
            ensure!(
                all.iter().all(|r| r[34 + row] == keep),
                "multi prefix counts disagreed"
            );
            Ok(keep as usize)
        })
        .collect()
}
impl Verifier {
    fn multi_select(&mut self, long: &Long<'_>, rows: usize) -> Result<()> {
        self.greedy_rows.local(
            &long.engine.stream,
            &self.ops.logits,
            rows,
            long.engine.rank,
        )?;
        unsafe {
            long.engine.gather_prefix(
                &self.greedy_rows.local,
                &self.greedy_rows.gathered,
                rows * 16,
            )?;
        }
        self.greedy_rows.merge(&long.engine.stream, rows)
    }

    pub(in crate::model) fn prewarm_multi(
        &mut self,
        long: &Long<'_>,
        window: usize,
    ) -> Result<usize> {
        self.prepare_multi_spec(long)?;
        ensure!(matches!(window, 2 | 3 | 4 | 8), "multi prewarm window");
        let mut layouts = if window <= 4 {
            vec![vec![0, 1, 2, 3]]
        } else {
            Vec::new()
        };
        if window <= 4 {
            for omitted in 0..4 {
                layouts.push((0..4).filter(|&slot| slot != omitted).collect());
            }
        }
        for a in 0..4 {
            for b in a + 1..4 {
                layouts.push(vec![a, b]);
            }
        }
        for slots in layouts {
            for mask in 0..1usize << slots.len() {
                let physical: Vec<_> = (0..slots.len()).map(|row| (mask >> row) & 1).collect();
                let positions: Vec<_> = slots
                    .iter()
                    .zip(&physical)
                    .map(|(&slot, &parity)| {
                        2 + (parity ^ usize::from(long.sessions[slot].kda_swapped))
                    })
                    .collect();
                let geometry = positions
                    .iter()
                    .map(|&p| multi_geometry(p, window))
                    .collect::<Result<Vec<_>>>()?;
                let key = (window, slots.clone(), physical, geometry, true);
                let graph = unsafe {
                    long.engine.stream.capture(|| {
                        self.multi_spec_forward_inner(long, &slots, &positions, window)?;
                        self.multi_select(long, slots.len() * window)
                    })?
                };
                ensure!(
                    self.multi_spec_graphs
                        .insert(key, graph.node_count(), graph)
                        .is_ok(),
                    "multi prewarm graph node bound"
                );
            }
        }
        let expected = if window <= 4 { 72 } else { 24 };
        ensure!(
            self.multi_spec_graphs.stats().retained_graphs == expected,
            "multi prewarm bound"
        );
        Ok(expected)
    }
    pub(in crate::model) fn enable_multi_audit(&mut self) {
        self.audit = true;
    }
    pub(in crate::model) fn multi_logits_hashes(&self, rows: usize) -> Result<Vec<String>> {
        ensure!(self.audit && rows <= MAX_ROWS, "multi logit audit bounds");
        let row_bytes = 38720 * 2;
        let bytes = self.all_logits.read(0, rows * row_bytes * 4)?;
        Ok((0..rows)
            .map(|row| {
                let mut hash = Sha256::new();
                for rank in 0..4 {
                    let at = (rank * rows + row) * row_bytes;
                    hash.update(&bytes[at..at + row_bytes]);
                }
                format!("{:x}", hash.finalize())
            })
            .collect())
    }
    pub(in crate::model) fn multi_greedy(
        &mut self,
        long: &mut Long<'_>,
        positions: &[usize; 4],
    ) -> Result<Vec<i32>> {
        ensure!(
            positions.iter().all(|&p| p > 0 && p < 512),
            "multi dense geometry"
        );
        let tokens: Vec<i32> = long
            .sessions
            .iter()
            .map(|s| s.pending.context("multi pending"))
            .collect::<Result<_>>()?;
        self.ids.write(
            0,
            &tokens
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.taps
            .as_ref()
            .context("multi taps")?
            .set_position(&long.engine.stream, 0)?;
        let key = std::array::from_fn(|i| long.kda_phys(i, positions[i]));
        if !self.multi_graphs.contains_key(&key) {
            ensure!(self.multi_graphs.len() < 16, "multi graph bound");
            let graph = unsafe {
                long.engine
                    .stream
                    .capture(|| self.multi_forward_inner(long, positions))?
            };
            self.multi_graphs.insert(key, graph);
        }
        unsafe {
            self.multi_graphs[&key].launch(&long.engine.stream)?;
        }
        self.greedy_rows
            .local(&long.engine.stream, &self.ops.logits, 4, long.engine.rank)?;
        unsafe {
            long.engine.gather_prefix(
                &self.greedy_rows.local,
                &self.greedy_rows.gathered,
                4 * 16,
            )?;
        }
        self.greedy_rows.merge(&long.engine.stream, 4)?;
        long.engine.sync()?;
        self.greedy_rows.read(4)
    }
    fn multi_forward_inner(
        &mut self,
        long: &Long<'_>,
        positions_by_slot: &[usize; 4],
    ) -> Result<()> {
        let e = &long.engine;
        let s = &e.stream;
        let n = 4;
        let mark = |_label: &str, _layer: usize| -> Result<()> { Ok(()) };
        let Self {
            shared_branch,
            audit,
            dsa_proj,
            dsa_qr,
            dsa_latent,
            dsa_q,
            dsa_attended,
            taps,
            k,
            projections,
            ops,
            experts,
            group,
            kda,
            ids,
            x,
            partial,
            out,
            gathered,
            gu,
            activation,
            sums,
            shared,
            all_logits,
            ..
        } = self;
        let mat_on = |stream: &crate::cuda::Stream,
                      alias: &str,
                      stem: &str,
                      input: &Buffer,
                      offset,
                      stride,
                      output: &Buffer|
         -> Result<()> {
            unsafe {
                projections[alias].run_offset(
                    stream,
                    input,
                    offset,
                    stride,
                    e.m.pointer(e.p, &format!("{stem}.weight"))?,
                    e.m.pointer(e.p, &format!("{stem}.scale"))?,
                    output,
                    partial,
                    n,
                )
            }
        };
        let mat = |alias: &str, stem: &str, input: &Buffer, offset, stride, output: &Buffer| {
            mat_on(s, alias, stem, input, offset, stride, output)
        };
        let finish = || -> Result<()> {
            unsafe {
                e.gather_prefix(out, gathered, n * 4096 * 4)?;
                ops.hc_post(s, x, gathered, x, n)
            }
        };
        words(
            &k["embed"],
            s,
            [n as u32, 64, 1],
            &[ids.ptr, e.m.pointer(e.p, "weights.embed")?, x.ptr],
        )?;
        let mut saves = kda.iter_mut();
        let mut recurrent_layer = 0;
        for (layer, attn) in e.attention.iter().enumerate() {
            let stem = |name: &str| format!("weights.layers[{layer}].{name}");
            let w = |name: &str| e.m.pointer(e.p, &stem(name));
            let pre = |branch: &str, norm: &str| -> Result<()> {
                unsafe {
                    ops.hc_pre(
                        s,
                        x,
                        HcWeights {
                            function: w(&format!("{branch}.fn"))?,
                            base: w(&format!("{branch}.base"))?,
                            scale: w(&format!("{branch}.scale"))?,
                            norm: w(norm)?,
                        },
                        n,
                    )
                }
            };
            mark("hc_attention", layer)?;
            pre("attn_hc", "in_norm")?;
            mark(
                if matches!(attn, Attention::Kda(_)) {
                    "kda"
                } else {
                    "dsa_projection"
                },
                layer,
            )?;
            match attn {
                Attention::Kda(a) => {
                    let save = saves.next().context("KDA verifier geometry")?;
                    mat(
                        "kda_proj",
                        &stem("kda.proj"),
                        &ops.norm,
                        0,
                        4096,
                        &save.proj,
                    )?;
                    mat(
                        "kda_gate",
                        &stem("kda.fb"),
                        &save.proj,
                        6144 * 2,
                        6416,
                        &save.ga,
                    )?;
                    mat(
                        "kda_gate",
                        &stem("kda.gb"),
                        &save.proj,
                        6272 * 2,
                        6416,
                        &save.gg,
                    )?;
                    e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, n)?;
                    for (slot, &pos) in positions_by_slot.iter().enumerate() {
                        let retained = &long.sessions[slot].recurrent[recurrent_layer];
                        unsafe {
                            a.proj
                                .copy_from(s, 0, &save.proj, slot * 6416 * 2, 6416 * 2)?;
                            e.ga.copy_from(s, 0, &save.ga, slot * 2048 * 2, 2048 * 2)?;
                            e.gg.copy_from(s, 0, &save.gg, slot * 2048 * 2, 2048 * 2)?;
                            let window = save.scratch.evaluate_into(
                                s,
                                kda_window::Inputs {
                                    state: &retained.states[pos % 2],
                                    conv: &retained.conv,
                                    projection: &a.proj,
                                    gate_a: &e.ga,
                                    gate_g: &e.gg,
                                },
                                kda_window::Weights {
                                    conv: w("kda.conv")?,
                                    a_log: w("kda.a_log")?,
                                    dt_bias: w("kda.dt_bias")?,
                                    norm: w("kda.norm")?,
                                },
                                1,
                                &retained.states[(pos + 1) % 2],
                            )?;
                            activation.copy_from(
                                s,
                                slot * 2048 * 2,
                                window.output(),
                                0,
                                2048 * 2,
                            )?;
                            drop(window);
                            save.scratch.commit_conv(s, &retained.conv, &a.proj, 1, 1)?;
                            retained.conv.copy_from(
                                s,
                                0,
                                &save.scratch.committed_conv,
                                0,
                                kda_window::CONV_BYTES,
                            )?;
                        }
                    }
                    recurrent_layer += 1;
                    mat("kda_out", &stem("kda.o"), activation, 0, 2048, out)?;
                }

                Attention::Dsa(a) => {
                    mat("dsa_proj", &stem("dsa.proj"), &ops.norm, 0, 4096, dsa_proj)?;
                    for (alias, off, name, dst, width) in [
                        ("rms1536", 0, "dsa.q_norm", dsa_qr.ptr, 1536),
                        ("rms512", 3072, "dsa.kv_norm", dsa_latent.ptr, 512),
                    ] {
                        words(
                            &k[alias],
                            s,
                            [n as u32, 1, 1],
                            &[
                                dsa_proj.ptr + off,
                                2048,
                                w(name)?,
                                dst,
                                width,
                                sums.ptr,
                                1e-5f32.to_bits() as u64,
                            ],
                        )?;
                    }
                    mat("dsa_qb", &stem("dsa.q_b"), dsa_qr, 0, 1536, dsa_q)?;
                    unsafe {
                        ops.absorb(s, dsa_q, w("dsa.kv_k.weight")?, n)?;
                    }
                    e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, n)?;
                    for (slot, &pos) in positions_by_slot.iter().enumerate() {
                        let position = &long.sessions[slot].position;
                        let pages = long.memory.table(slot)?;
                        words(
                            &k["latent_write"],
                            s,
                            [1, 1, 1],
                            &[
                                dsa_latent.ptr + (slot * 512 * 2) as u64,
                                a.cache.ptr,
                                a.cache.ptr,
                                position.ptr,
                                pages.ptr,
                            ],
                        )?;
                        let norm = ops.norm.ptr + (slot * 4096 * 2) as u64;
                        let qr = dsa_qr.ptr + (slot * 1536 * 2) as u64;
                        long.projection
                            .run(s, e.p, &e.m, layer, norm, qr, 1, pos, false)?;
                        long.attend_projected_slot(
                            crate::model::batch::Inputs {
                                layer,
                                position: pos,
                                norm,
                                qr,
                                qa: ops.qa.ptr + (slot * 16 * 512 * 2) as u64,
                                attended: dsa_attended.ptr + (slot * 16 * 512 * 4) as u64,
                                rows: 1,
                                dsa: a,
                            },
                            pos,
                            false,
                            long.projection.raw.ptr,
                            long.projection.gate.ptr,
                            long.projection.query.ptr,
                            slot,
                        )?;
                    }
                    unsafe {
                        mark("dsa_expand_output", layer)?;
                        ops.expand(s, dsa_attended, w("dsa.kv_v.weight")?, n)?;
                    }
                    mat("dsa_out", &stem("dsa.o"), &ops.expanded, 0, 4096, out)?;
                }
            }

            mark("attention_exchange", layer)?;
            e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionGather, n)?;
            finish()?;
            mark("hc_ffn", layer)?;
            pre("ffn_hc", "post_norm")?;
            mark(if layer < 3 { "dense_ffn" } else { "experts" }, layer)?;
            if layer < 3 {
                mat("mlp_gu", &stem("mlp.gu"), &ops.norm, 0, 4096, gu)?;
                words(
                    &k["swiglu"],
                    s,
                    [n as u32, 6, 1],
                    &[gu.ptr, activation.ptr, sums.ptr, 10f32.to_bits() as u64],
                )?;
                mat("mlp_down", &stem("mlp.down"), activation, 0, 3072, out)?;
            } else {
                unsafe {
                    ops.route(s, &ops.norm, w("moe.router")?, w("moe.bias")?, n)?;
                }
                group.run(s, ops.picks.ptr, n)?;
                let routed = || {
                    experts.run_rows(
                        s,
                        e.p,
                        &e.m,
                        layer,
                        ops.norm.ptr,
                        ops.picks.ptr,
                        group.ids.ptr,
                        group.count.ptr,
                        group.members.ptr,
                        n,
                        MAX_ROWS,
                    )
                };
                e.prefetch_experts(layer, n, &group.ids, &group.count)?;
                let shared_work = |stream: &crate::cuda::Stream| -> Result<()> {
                    mat_on(
                        stream,
                        "shared_gu",
                        &stem("moe.shared.gu"),
                        &ops.norm,
                        0,
                        4096,
                        gu,
                    )?;
                    words(
                        &k["shared_swiglu"],
                        stream,
                        [n as u32, 1, 1],
                        &[gu.ptr, activation.ptr, sums.ptr, 10f32.to_bits() as u64],
                    )?;
                    mat_on(
                        stream,
                        "shared_down",
                        &stem("moe.shared.down"),
                        activation,
                        0,
                        512,
                        shared,
                    )
                };
                if let Some(branch) = shared_branch.as_mut() {
                    // Routed ExpertGpu owns separate xg/xu/xd/z/y scratch.
                    // Both branches only read ops.norm; combine stays on the
                    // original stream after their join, with unchanged sums.
                    branch.run(s, shared_work, routed)?;
                } else {
                    if e.prefetch.is_none() {
                        routed()?;
                    }
                    shared_work(s)?;
                    if e.prefetch.is_some() {
                        routed()?;
                    }
                }
                words(
                    &k["combine"],
                    s,
                    [n as u32, 4, 1],
                    &[experts.y.ptr, shared.ptr, ops.weights.ptr, out.ptr],
                )?;
            }
            mark("ffn_exchange", layer)?;
            e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::FeedForwardGather, n)?;
            finish()?;
            if let Some(taps) = taps.as_ref() {
                taps.capture(s, x, n, layer)?;
            }
        }
        mark("head", 45)?;
        unsafe {
            ops.head(
                s,
                x,
                e.m.pointer(e.p, "weights.norm")?,
                e.m.pointer(e.p, "weights.head.weight")?,
                n,
            )?;
            if *audit {
                e.gather_prefix(&ops.logits, all_logits, n * 38720 * 2)?;
            }
        }
        mark("sampling_commit", 45)?;
        e.prefetch_join()
    }
}

pub(super) struct Saved {
    // The replay tables point into the saves below; retire their kernel owner first.
    // Separate pinned counters per request: uploads may still be in flight when
    // the next independent request stages its own acceptance count.
    replay: Vec<crate::kda_commit::Batch>,
    kda: Vec<Vec<KdaSave>>,
    undo: Vec<Vec<Undo>>,
}
impl Verifier {
    pub(in crate::model) fn prepare_multi_spec(&mut self, long: &Long<'_>) -> Result<()> {
        if self.multi_save.is_some() {
            return Ok(());
        }
        let d = &long.engine.device;
        let kda = (0..self.kda.len())
            .map(|_| {
                (0..4)
                    .map(|_| {
                        Ok(KdaSave {
                            scratch: kda_window::Scratch::new(
                                d,
                                &self.multi_native.join("kda.cubin"),
                            )?,
                            proj: d.alloc(8 * 6416 * 2)?,
                            ga: d.alloc(8 * 2048 * 2)?,
                            gg: d.alloc(8 * 2048 * 2)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let pointers = (0..4)
            .flat_map(|slot| {
                let kda = &kda;
                (0..2).map(move |physical| {
                    long.sessions[slot]
                        .recurrent
                        .iter()
                        .zip(kda)
                        .map(|(state, saves)| {
                            let save = &saves[slot];
                            let logical = physical ^ usize::from(long.sessions[slot].kda_swapped);
                            save.scratch.replay_pointers_into(
                                &state.states[logical],
                                &save.scratch.committed_state,
                            )
                        })
                        .collect()
                })
            })
            .collect::<Vec<_>>();
        let replay = (0..4)
            .map(|_| crate::kda_commit::Batch::new(d, &self.multi_native, &pointers))
            .collect::<Result<Vec<_>>>()?;
        let undo = (0..4)
            .map(|_| {
                (0..self.undo.len())
                    .map(|_| {
                        Ok(Undo {
                            keys: d.alloc(MAX_ROWS * crate::indexer::IndexCache::ROW_BYTES)?,
                            gates: d.alloc(MAX_ROWS * crate::indexer::IndexCache::ROW_BYTES)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let packed = crate::packed_candidates::Packed::new(d, &self.multi_native)?;
        self.multi_save = Some(Saved { replay, kda, undo });
        self.multi_packed = Some(packed);
        Ok(())
    }
    pub(in crate::model) fn multi_speculative(
        &mut self,
        long: &mut Long<'_>,
        slots: &[usize],
        inputs: &[Vec<i32>],
        stop_eos: &[bool],
        policies: &[Sampling],
        mut clip: impl FnMut(usize, &[i32]) -> Result<usize>,
    ) -> Result<Vec<Vec<i32>>> {
        ensure!(
            matches!(slots.len(), 2..=4)
                && slots.len() == inputs.len()
                && stop_eos.len() == slots.len()
                && policies.len() == slots.len()
                && policies.iter().all(|p| p.greedy() == policies[0].greedy())
                && (policies[0].greedy()
                    || policies
                        .iter()
                        .all(|p| p.local_count() == policies[0].local_count())),
            "multi speculative width"
        );
        let window = inputs[0].len();
        let positions: Vec<usize> = slots
            .iter()
            .map(|&slot| long.sessions[slot].tokens.len())
            .collect();
        let greedy = policies[0].greedy();
        ensure!(
            matches!(window, 2 | 3 | 4 | 8)
                && slots.len() * window <= MAX_ROWS
                && positions.iter().all(|&p| multi_geometry(p, window).is_ok())
                && inputs.iter().all(|t| t.len() == window),
            "multi dense bounds"
        );
        self.ids.write(
            0,
            &inputs
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.taps
            .as_ref()
            .context("multi taps")?
            .set_position(&long.engine.stream, 0)?;
        for (&slot, &pos) in slots.iter().zip(&positions) {
            let (table, layers, ring_rows) = undo_table(
                &long.sessions[slot].index,
                &self.multi_save.as_ref().context("multi undo")?.undo[slot],
            )?;
            unsafe {
                self.copies.index_undo(
                    &long.engine.stream,
                    &table,
                    layers,
                    pos,
                    0,
                    window,
                    ring_rows,
                    crate::indexer::IndexCache::ROW_BYTES,
                    false,
                )?;
            }
        }
        let key = (
            window,
            slots.to_vec(),
            slots
                .iter()
                .zip(&positions)
                .map(|(&slot, &pos)| long.kda_phys(slot, pos))
                .collect(),
            positions
                .iter()
                .map(|&p| multi_geometry(p, window))
                .collect::<Result<Vec<_>>>()?,
            greedy,
        );
        let rows = slots.len() * window;
        crate::round_timing::enter(crate::round_timing::Phase::VerifyLaunch);
        match self.multi_spec_graphs.action(&key) {
            GraphAction::Replay => unsafe {
                self.multi_spec_graphs
                    .get(&key)
                    .launch(&long.engine.stream)?;
            },
            GraphAction::Eager => {
                self.multi_spec_forward_inner(long, slots, &positions, window)?;
                if greedy {
                    self.multi_select(long, rows)?;
                }
            }
            GraphAction::Capture => {
                let graph = unsafe {
                    long.engine.stream.capture(|| {
                        self.multi_spec_forward_inner(long, slots, &positions, window)?;
                        if greedy {
                            self.multi_select(long, rows)?;
                        }
                        Ok(())
                    })?
                };
                match self
                    .multi_spec_graphs
                    .insert(key.clone(), graph.node_count(), graph)
                {
                    Ok(()) => unsafe {
                        self.multi_spec_graphs
                            .get(&key)
                            .launch(&long.engine.stream)?;
                    },
                    Err(graph) => unsafe {
                        graph.launch(&long.engine.stream)?;
                    },
                }
            }
        }
        let row_policies: Vec<_> = policies
            .iter()
            .zip(&positions)
            .flat_map(|(&policy, &pos)| {
                (0..window).map(move |offset| (policy, (pos + offset + 1) as u64))
            })
            .collect();
        let layout = if greedy {
            None
        } else {
            let layout =
                crate::packed_candidates_layout::Layout::new(rows, policies[0].local_count())?;
            let packed = self.multi_packed.as_mut().context("multi sampled owner")?;
            packed.local(
                &long.engine.stream,
                &self.ops.logits,
                long.engine.rank,
                layout,
            )?;
            unsafe {
                long.engine
                    .gather_prefix(&packed.local, &packed.gathered, layout.rank_bytes)?;
                if policies.iter().all(|p| p.gpu_fits(4 * layout.k)) {
                    packed.gpu_policies(&long.engine.stream, layout, &row_policies)?;
                }
            }
            Some(layout)
        };
        crate::round_timing::enter(crate::round_timing::Phase::VerifyWait);
        long.engine.sync()?;
        crate::round_timing::enter(crate::round_timing::Phase::VerifyPost);
        let selection = if let Some(layout) = layout {
            self.multi_packed
                .as_mut()
                .unwrap()
                .read_policies(layout, &row_policies)
                .map(|chosen| chosen[..rows].iter().map(|r| r.0 as i32).collect())
        } else {
            self.greedy_rows.read(rows)
        };
        let valid_selection = selection.is_ok();
        let samples = selection.unwrap_or_else(|_| vec![0; rows]);
        let decisions: Vec<Result<usize>> = inputs
            .iter()
            .zip(stop_eos)
            .enumerate()
            .map(|(row, (tokens, &eos))| {
                clipped_keep(
                    tokens,
                    &samples[row * window..row * window + window],
                    eos,
                    |tokens| clip(row, tokens),
                )
            })
            .collect();
        let mut hash = Sha256::new();
        hash.update(b"multi-spec-accept-v1");
        for &slot in slots {
            hash.update((slot as u64).to_le_bytes());
        }
        for p in policies {
            hash.update(p.encode());
        }
        for token in &samples {
            hash.update(token.to_le_bytes());
        }
        let mut record = [0u8; 64];
        record[..32].copy_from_slice(&hash.finalize());
        record[32] = slots.len() as u8;
        record[33] = u8::from(valid_selection);
        record[38] = window as u8;
        for (row, decision) in decisions.iter().enumerate() {
            match decision {
                Ok(keep) if (1..=window).contains(keep) => record[34 + row] = *keep as u8,
                Err(e) if e.is::<accepted_path::ParserFailure>() => (),
                _ => record[33] = 0,
            }
        }
        let all = self.control.exchange(long, &record)?;
        let keeps = accepted_counts(&all, &record)?;
        let mut result = Vec::new();
        for (row, (&slot, &pos)) in slots.iter().zip(&positions).enumerate() {
            let keep = keeps[row];
            if keep > 0 {
                let saved = self.multi_save.as_mut().unwrap();
                if keep < window {
                    unsafe {
                        saved.replay[slot].replay(
                            &long.engine.stream,
                            slot,
                            long.kda_phys(slot, pos),
                            window,
                            keep,
                        )?;
                    }
                }
                let mut table = crate::batch_copy::KdaCommit::default();
                for (layer, saves) in saved.kda.iter().enumerate() {
                    let retained = &long.sessions[slot].recurrent[layer];
                    let save = &saves[slot];
                    table.conv[layer] = retained.conv.ptr;
                    table.proj[layer] = save.proj.ptr;
                    table.tail[layer] = save.scratch.committed_conv.ptr;
                    table.state_src[layer] = if keep == window {
                        save.scratch.proposal_state.ptr
                    } else {
                        save.scratch.committed_state.ptr
                    };
                    table.state_dst[layer] = retained.states[(pos + keep) % 2].ptr;
                }
                unsafe {
                    self.copies.kda_commit(
                        &long.engine.stream,
                        &table,
                        keep,
                        kda_window::CONV_CHANNELS * 2,
                        kda_window::PROJ_STRIDE * 2,
                        Some(kda_window::STATE_BYTES),
                    )?;
                }
            }
            let (table, layers, ring_rows) = undo_table(
                &long.sessions[slot].index,
                &self.multi_save.as_ref().unwrap().undo[slot],
            )?;
            unsafe {
                self.copies.index_undo(
                    &long.engine.stream,
                    &table,
                    layers,
                    pos,
                    keep,
                    window,
                    ring_rows,
                    crate::indexer::IndexCache::ROW_BYTES,
                    true,
                )?;
                long.sessions[slot]
                    .position
                    .set_i32(&long.engine.stream, (pos + keep) as i32)?;
            }
            result.push(samples[row * window..row * window + keep].to_vec());
        }
        long.engine.sync()?;
        Ok(result)
    }
    fn multi_spec_forward_inner(
        &mut self,
        long: &Long<'_>,
        slots: &[usize],
        positions_by_row: &[usize],
        window: usize,
    ) -> Result<()> {
        let e = &long.engine;
        let s = &e.stream;
        let n = slots.len() * window;
        let mark = |_label: &str, _layer: usize| -> Result<()> { Ok(()) };
        let Self {
            shared_branch,
            audit,
            dsa_proj,
            dsa_qr,
            dsa_latent,
            dsa_q,
            dsa_attended,
            taps,
            k,
            projections,
            ops,
            experts,
            group,
            kda,
            multi_save,
            ids,
            x,
            partial,
            out,
            gathered,
            gu,
            activation,
            sums,
            shared,
            all_logits,
            ..
        } = self;
        let mat_on = |stream: &crate::cuda::Stream,
                      alias: &str,
                      stem: &str,
                      input: &Buffer,
                      offset,
                      stride,
                      output: &Buffer|
         -> Result<()> {
            unsafe {
                projections[alias].run_offset(
                    stream,
                    input,
                    offset,
                    stride,
                    e.m.pointer(e.p, &format!("{stem}.weight"))?,
                    e.m.pointer(e.p, &format!("{stem}.scale"))?,
                    output,
                    partial,
                    n,
                )
            }
        };
        let mat = |alias: &str, stem: &str, input: &Buffer, offset, stride, output: &Buffer| {
            mat_on(s, alias, stem, input, offset, stride, output)
        };
        let finish = || -> Result<()> {
            unsafe {
                e.gather_prefix(out, gathered, n * 4096 * 4)?;
                ops.hc_post(s, x, gathered, x, n)
            }
        };
        words(
            &k["embed"],
            s,
            [n as u32, 64, 1],
            &[ids.ptr, e.m.pointer(e.p, "weights.embed")?, x.ptr],
        )?;
        let mut saves = kda.iter_mut();
        let saved = multi_save.as_mut().context("multi scratch")?;
        let mut recurrent_layer = 0;
        for (layer, attn) in e.attention.iter().enumerate() {
            let stem = |name: &str| format!("weights.layers[{layer}].{name}");
            let w = |name: &str| e.m.pointer(e.p, &stem(name));
            let pre = |branch: &str, norm: &str| -> Result<()> {
                unsafe {
                    ops.hc_pre(
                        s,
                        x,
                        HcWeights {
                            function: w(&format!("{branch}.fn"))?,
                            base: w(&format!("{branch}.base"))?,
                            scale: w(&format!("{branch}.scale"))?,
                            norm: w(norm)?,
                        },
                        n,
                    )
                }
            };
            mark("hc_attention", layer)?;
            pre("attn_hc", "in_norm")?;
            mark(
                if matches!(attn, Attention::Kda(_)) {
                    "kda"
                } else {
                    "dsa_projection"
                },
                layer,
            )?;
            match attn {
                Attention::Kda(_) => {
                    let save = saves.next().context("KDA verifier geometry")?;
                    mat(
                        "kda_proj",
                        &stem("kda.proj"),
                        &ops.norm,
                        0,
                        4096,
                        &save.proj,
                    )?;
                    mat(
                        "kda_gate",
                        &stem("kda.fb"),
                        &save.proj,
                        6144 * 2,
                        6416,
                        &save.ga,
                    )?;
                    mat(
                        "kda_gate",
                        &stem("kda.gb"),
                        &save.proj,
                        6272 * 2,
                        6416,
                        &save.gg,
                    )?;
                    e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, n)?;
                    for (row, (&slot, &pos)) in slots.iter().zip(positions_by_row).enumerate() {
                        let retained = &long.sessions[slot].recurrent[recurrent_layer];
                        let saved_row = &mut saved.kda[recurrent_layer][slot];
                        unsafe {
                            saved_row.proj.copy_from(
                                s,
                                0,
                                &save.proj,
                                row * window * 6416 * 2,
                                window * 6416 * 2,
                            )?;
                            saved_row.ga.copy_from(
                                s,
                                0,
                                &save.ga,
                                row * window * 2048 * 2,
                                window * 2048 * 2,
                            )?;
                            saved_row.gg.copy_from(
                                s,
                                0,
                                &save.gg,
                                row * window * 2048 * 2,
                                window * 2048 * 2,
                            )?;
                            let proposed = saved_row.scratch.evaluate(
                                s,
                                kda_window::Inputs {
                                    state: &retained.states[pos % 2],
                                    conv: &retained.conv,
                                    projection: &saved_row.proj,
                                    gate_a: &saved_row.ga,
                                    gate_g: &saved_row.gg,
                                },
                                kda_window::Weights {
                                    conv: w("kda.conv")?,
                                    a_log: w("kda.a_log")?,
                                    dt_bias: w("kda.dt_bias")?,
                                    norm: w("kda.norm")?,
                                },
                                window,
                            )?;
                            activation.copy_from(
                                s,
                                row * window * 2048 * 2,
                                proposed.output(),
                                0,
                                window * 2048 * 2,
                            )?;
                        }
                    }
                    recurrent_layer += 1;
                    mat("kda_out", &stem("kda.o"), activation, 0, 2048, out)?;
                }

                Attention::Dsa(a) => {
                    mat("dsa_proj", &stem("dsa.proj"), &ops.norm, 0, 4096, dsa_proj)?;
                    for (alias, off, name, dst, width) in [
                        ("rms1536", 0, "dsa.q_norm", dsa_qr.ptr, 1536),
                        ("rms512", 3072, "dsa.kv_norm", dsa_latent.ptr, 512),
                    ] {
                        words(
                            &k[alias],
                            s,
                            [n as u32, 1, 1],
                            &[
                                dsa_proj.ptr + off,
                                2048,
                                w(name)?,
                                dst,
                                width,
                                sums.ptr,
                                1e-5f32.to_bits() as u64,
                            ],
                        )?;
                    }
                    mat("dsa_qb", &stem("dsa.q_b"), dsa_qr, 0, 1536, dsa_q)?;
                    unsafe {
                        ops.absorb(s, dsa_q, w("dsa.kv_k.weight")?, n)?;
                    }
                    e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionCore, n)?;
                    for (row, (&slot, &pos)) in slots.iter().zip(positions_by_row).enumerate() {
                        let position = &long.sessions[slot].position;
                        let pages = long.memory.table(slot)?;
                        words(
                            &k["latent_write"],
                            s,
                            [window as u32, 1, 1],
                            &[
                                dsa_latent.ptr + (row * window * 512 * 2) as u64,
                                a.cache.ptr,
                                a.cache.ptr,
                                position.ptr,
                                pages.ptr,
                            ],
                        )?;
                        let norm = ops.norm.ptr + (row * window * 4096 * 2) as u64;
                        let qr = dsa_qr.ptr + (row * window * 1536 * 2) as u64;
                        long.projection
                            .run(s, e.p, &e.m, layer, norm, qr, window, pos, false)?;
                        long.attend_projected_slot(
                            crate::model::batch::Inputs {
                                layer,
                                position: pos,
                                norm,
                                qr,
                                qa: ops.qa.ptr + (row * window * 16 * 512 * 2) as u64,
                                attended: dsa_attended.ptr + (row * window * 16 * 512 * 4) as u64,
                                rows: window,
                                dsa: a,
                            },
                            pos,
                            false,
                            long.projection.raw.ptr,
                            long.projection.gate.ptr,
                            long.projection.query.ptr,
                            slot,
                        )?;
                    }
                    unsafe {
                        mark("dsa_expand_output", layer)?;
                        ops.expand(s, dsa_attended, w("dsa.kv_v.weight")?, n)?;
                    }
                    mat("dsa_out", &stem("dsa.o"), &ops.expanded, 0, 4096, out)?;
                }
            }

            mark("attention_exchange", layer)?;
            e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::AttentionGather, n)?;
            finish()?;
            mark("hc_ffn", layer)?;
            pre("ffn_hc", "post_norm")?;
            mark(if layer < 3 { "dense_ffn" } else { "experts" }, layer)?;
            if layer < 3 {
                mat("mlp_gu", &stem("mlp.gu"), &ops.norm, 0, 4096, gu)?;
                words(
                    &k["swiglu"],
                    s,
                    [n as u32, 6, 1],
                    &[gu.ptr, activation.ptr, sums.ptr, 10f32.to_bits() as u64],
                )?;
                mat("mlp_down", &stem("mlp.down"), activation, 0, 3072, out)?;
            } else {
                unsafe {
                    ops.route(s, &ops.norm, w("moe.router")?, w("moe.bias")?, n)?;
                }
                group.run(s, ops.picks.ptr, n)?;
                let routed = || {
                    experts.run_rows(
                        s,
                        e.p,
                        &e.m,
                        layer,
                        ops.norm.ptr,
                        ops.picks.ptr,
                        group.ids.ptr,
                        group.count.ptr,
                        group.members.ptr,
                        n,
                        MAX_ROWS,
                    )
                };
                e.prefetch_experts(layer, n, &group.ids, &group.count)?;
                let shared_work = |stream: &crate::cuda::Stream| -> Result<()> {
                    mat_on(
                        stream,
                        "shared_gu",
                        &stem("moe.shared.gu"),
                        &ops.norm,
                        0,
                        4096,
                        gu,
                    )?;
                    words(
                        &k["shared_swiglu"],
                        stream,
                        [n as u32, 1, 1],
                        &[gu.ptr, activation.ptr, sums.ptr, 10f32.to_bits() as u64],
                    )?;
                    mat_on(
                        stream,
                        "shared_down",
                        &stem("moe.shared.down"),
                        activation,
                        0,
                        512,
                        shared,
                    )
                };
                if let Some(branch) = shared_branch.as_mut() {
                    // Routed ExpertGpu owns separate xg/xu/xd/z/y scratch.
                    // Both branches only read ops.norm; combine stays on the
                    // original stream after their join, with unchanged sums.
                    branch.run(s, shared_work, routed)?;
                } else {
                    if e.prefetch.is_none() {
                        routed()?;
                    }
                    shared_work(s)?;
                    if e.prefetch.is_some() {
                        routed()?;
                    }
                }
                words(
                    &k["combine"],
                    s,
                    [n as u32, 4, 1],
                    &[experts.y.ptr, shared.ptr, ops.weights.ptr, out.ptr],
                )?;
            }
            mark("ffn_exchange", layer)?;
            e.prefetch_mark(layer, crate::l2_prefetch_plan::Site::FeedForwardGather, n)?;
            finish()?;
            if let Some(taps) = taps.as_ref() {
                taps.capture(s, x, n, layer)?;
            }
        }
        mark("head", 45)?;
        unsafe {
            ops.head(
                s,
                x,
                e.m.pointer(e.p, "weights.norm")?,
                e.m.pointer(e.p, "weights.head.weight")?,
                n,
            )?;
            if *audit {
                e.gather_prefix(&ops.logits, all_logits, n * 38720 * 2)?;
            }
        }
        mark("sampling_commit", 45)?;
        e.prefetch_join()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn long_graph_geometry_tracks_dense_chunks_sparse_buckets_and_bounds() -> Result<()> {
        assert_eq!(multi_geometry(508, 4)?, (1, 0));
        assert_eq!(multi_geometry(509, 4)?, (2, 0));
        assert_eq!(multi_geometry(2047, 4)?, (5, 0));
        assert!(multi_geometry(2048, 4).is_err());
        assert!(multi_geometry(2050, 4).is_err());
        assert_eq!(multi_geometry(2051, 4)?, (0, 1024));
        assert_eq!(multi_geometry(4093, 4)?, (0, 1024));
        assert_eq!(multi_geometry(4097, 4)?, (0, 2048));
        assert_eq!(multi_geometry(33000, 4)?, multi_geometry(33008, 4)?);
        assert_eq!(
            multi_geometry(crate::cache_pages::CONTEXT - 4, 4)?,
            (0, crate::cache_pages::CONTEXT / 4)
        );
        assert!(multi_geometry(crate::cache_pages::CONTEXT - 3, 4).is_err());
        assert!(multi_geometry(usize::MAX, 4).is_err());
        Ok(())
    }

    #[test]
    fn common_parser_discard_preserves_other_independent_prefixes() -> Result<()> {
        let mut record = [0u8; 64];
        record[32] = 4;
        record[33] = 1;
        record[38] = 4;
        record[34..38].copy_from_slice(&[1, 2, 3, 4]);
        let mut ranks = [record; 4];
        assert_eq!(accepted_counts(&ranks, &record)?, [1, 2, 3, 4]);
        ranks[2][34] = 0;
        assert_eq!(accepted_counts(&ranks, &record)?, [0, 2, 3, 4]);
        ranks[1][37] = 0;
        assert_eq!(accepted_counts(&ranks, &record)?, [0, 2, 3, 0]);
        ranks[0][35] = 1;
        assert!(accepted_counts(&ranks, &record).is_err());
        Ok(())
    }

    #[test]
    fn mismatched_work_or_unclassified_errors_cannot_grant_discard() {
        let mut record = [0u8; 64];
        record[32] = 2;
        record[33] = 1;
        record[38] = 4;
        record[34..36].copy_from_slice(&[2, 4]);
        for (at, value) in [(0, 1), (32, 4), (33, 0), (35, 5)] {
            let mut ranks = [record; 4];
            ranks[1][at] = value;
            assert!(accepted_counts(&ranks, &record).is_err());
        }
        assert!(accepted_counts(&[record; 3], &record).is_err());
    }

    #[test]
    fn triplet_clipping_and_one_rank_parser_discard_are_independent() -> Result<()> {
        let mut record = [0u8; 64];
        record[32] = 3;
        record[33] = 1;
        record[38] = 4;
        record[34..37].copy_from_slice(&[4, 2, 3]);
        let mut ranks = [record; 4];
        assert_eq!(accepted_counts(&ranks, &record)?, [4, 2, 3]);
        ranks[2][35] = 0;
        assert_eq!(accepted_counts(&ranks, &record)?, [4, 0, 3]);
        ranks[1][36] = 2;
        assert!(accepted_counts(&ranks, &record).is_err());
        Ok(())
    }
}
