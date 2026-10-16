//! Single-row routing and all nine expert contributions, with reusable Rust-owned storage.
use crate::{
    cuda::{Buffer, Device, Kernel, Stream},
    dsa::words,
    experts::ExpertGpu,
    front::{fmm, launch_ptrs, load_kernel, reduce},
    prepared::{Loaded, Prepared},
};
use anyhow::Result;
use std::{collections::BTreeMap, path::Path, rc::Rc};
pub struct Moe {
    k: BTreeMap<&'static str, Kernel>,
    expert: ExpertGpu,
    part: Buffer,
    logits: Buffer,
    picks: Buffer,
    weights: Buffer,
    ids: Buffer,
    count: Buffer,
    members: Buffer,
    gu: Buffer,
    activation: Buffer,
    sums: Buffer,
    shared: Buffer,
    pub output: Buffer,
}
impl Moe {
    pub fn new(d: &Rc<Device>, root: &Path, native: &Path) -> Result<Self> {
        let mut k = BTreeMap::new();
        for n in [
            "router_part",
            "router_sum",
            "topk",
            "group",
            "combine",
            "shared_gu",
            "shared_down",
            "shared_swiglu",
            "reduce8_bf16",
        ] {
            k.insert(n, load_kernel(d, root, n)?);
        }
        let alloc = |n| -> Result<Buffer> {
            let b = d.alloc(n)?;
            b.zero()?;
            Ok(b)
        };
        Ok(Self {
            k,
            expert: ExpertGpu::new(d, native)?,
            part: alloc(8 * 4096 * 4)?,
            logits: alloc(288 * 4)?,
            picks: alloc(9 * 4)?,
            weights: alloc(9 * 4)?,
            ids: alloc(9 * 4)?,
            count: alloc(4)?,
            members: alloc(9 * 4)?,
            gu: alloc(1024 * 2)?,
            activation: alloc(512 * 2)?,
            sums: alloc(64 * 4)?,
            shared: alloc(4096 * 4)?,
            output: alloc(4096 * 4)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_prefetched(
        &self,
        s: &Stream,
        p: &Prepared,
        m: &Loaded,
        layer: usize,
        input: u64,
        prefetch: Option<&std::cell::RefCell<crate::l2_prefetch::Owner>>,
    ) -> Result<()> {
        let w = |n: &str| m.pointer(p, &format!("weights.layers[{layer}].moe.{n}"));
        let k = &self.k;
        words(
            &k["router_part"],
            s,
            [1, 9, 8],
            &[input, w("router")?, self.part.ptr, 4096],
        )?;
        words(
            &k["router_sum"],
            s,
            [1, 1, 1],
            &[self.part.ptr, self.logits.ptr, 288],
        )?;
        words(
            &k["topk"],
            s,
            [1, 1, 1],
            &[
                self.logits.ptr,
                w("bias")?,
                self.picks.ptr,
                self.weights.ptr,
                2.5f32.to_bits() as u64,
            ],
        )?;
        launch_ptrs(
            &k["group"],
            s,
            [1, 1, 1],
            &[
                self.picks.ptr,
                self.ids.ptr,
                self.count.ptr,
                self.members.ptr,
            ],
        )?;
        let routed = || {
            self.expert.run(
                s,
                p,
                m,
                layer,
                input,
                self.picks.ptr,
                self.ids.ptr,
                self.count.ptr,
                self.members.ptr,
            )
        };
        if let Some(prefetch) = prefetch {
            prefetch
                .borrow_mut()
                .experts(s, layer, 1, &self.ids, &self.count)?;
        } else {
            routed()?;
        }
        fmm(
            &k["shared_gu"],
            s,
            input,
            w("shared.gu.weight")?,
            w("shared.gu.scale")?,
            self.gu.ptr,
            self.part.ptr,
            4096,
            1024,
            8,
        )?;
        reduce(&k["reduce8_bf16"], s, self.part.ptr, self.gu.ptr, 1024)?;
        words(
            &k["shared_swiglu"],
            s,
            [1, 1, 1],
            &[
                self.gu.ptr,
                self.activation.ptr,
                self.sums.ptr,
                10f32.to_bits() as u64,
            ],
        )?;
        fmm(
            &k["shared_down"],
            s,
            self.activation.ptr,
            w("shared.down.weight")?,
            w("shared.down.scale")?,
            self.shared.ptr,
            self.part.ptr,
            512,
            4096,
            1,
        )?;
        if prefetch.is_some() {
            routed()?;
        }
        launch_ptrs(
            &k["combine"],
            s,
            [1, 4, 1],
            &[
                self.expert.y.ptr,
                self.shared.ptr,
                self.weights.ptr,
                self.output.ptr,
            ],
        )
    }
}
