//! L2 read-hint planning. No model arithmetic or device work.
//! Only the retained bulk/a,f,o,e/8 MiB policy is exposed; no tuning sweep.
use anyhow::{Context, Result, ensure};

pub(crate) const ALIGN: u64 = 16;
pub(crate) const BUDGET: u64 = 8 << 20;
pub(crate) const L2_FALLBACK: u64 = 24 << 20;
pub(crate) const EXPERT_BYTES: u64 = 64 << 10;
pub(crate) const MAX_ROWS: usize = 64;
pub(crate) const CHUNK: u64 = 32 << 10;
pub(crate) const THREADS: u64 = 128;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Policy {
    #[default]
    Off,
    Retained,
}
impl Policy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        Ok(match value.trim() {
            "" | "0" => Self::Off,
            "1" => Self::Retained,
            _ => anyhow::bail!("ALASKA_L2_PREFETCH accepts only 0 or retained policy 1"),
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Site {
    AttentionGather,
    FeedForwardGather,
    AttentionCore,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tensor {
    pub(crate) address: u64,
    pub(crate) bytes: u64,
    pub(crate) contiguous: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) address: u64,
    pub(crate) bytes: u64,
}
impl Tensor {
    fn span(self, left: u64) -> Result<Option<Span>> {
        if !self.contiguous || self.bytes == 0 || left < ALIGN {
            return Ok(None);
        }
        ensure!(self.address != 0, "null L2 prefetch address");
        let end = self
            .address
            .checked_add(self.bytes)
            .context("L2 tensor extent overflow")?;
        let address = self
            .address
            .checked_add(ALIGN - 1)
            .context("L2 alignment overflow")?
            / ALIGN
            * ALIGN;
        let bytes = left.min(end.saturating_sub(address)) / ALIGN * ALIGN;
        Ok((bytes > 0).then_some(Span { address, bytes }))
    }
}
/// `split_k` is the actual retained matrix schedule, not a new arithmetic choice.
pub(crate) enum Item {
    Tensor(Tensor),
    Scaled {
        scale: Tensor,
        weight: Tensor,
    },
    #[cfg(test)]
    Q4 {
        scales: Tensor,
        biases: Tensor,
        weight: Tensor,
        n: u64,
        split_k: u64,
    },
}
pub(crate) fn budget(l2_bytes: u64) -> u64 {
    BUDGET.min(if l2_bytes == 0 { L2_FALLBACK } else { l2_bytes } / 2)
}
pub(crate) fn rows_enabled(rows: usize) -> bool {
    rows > 0 && rows <= MAX_ROWS
}
pub(crate) fn grid(bytes: u64, sms: u32) -> Result<u32> {
    ensure!(
        bytes > 0 && bytes <= BUDGET && sms > 0,
        "L2 segment grid bounds"
    );
    Ok((bytes.div_ceil(CHUNK).div_ceil(THREADS) as u32).clamp(1, sms))
}
pub(crate) fn expert_grid(
    rows: usize,
    experts: u64,
    capacity: usize,
    sms: u32,
) -> Result<(i32, i32)> {
    ensure!(
        rows_enabled(rows) && experts > 0 && sms > 0,
        "L2 expert grid policy"
    );
    // forward.Buffers.group(R): min(R * top_k, experts) + shared slot.
    // Physical Rust scratch is larger; do not accidentally launch that capacity.
    let maxu = ((rows as u64 * 8).min(experts) + 1) as usize;
    ensure!(
        maxu <= capacity && maxu <= i32::MAX as usize,
        "L2 expert group capacity"
    );
    Ok((maxu as i32, sms.min(maxu as u32) as i32))
}
pub(crate) fn take(items: &[Item], budget: u64) -> Result<Vec<Span>> {
    ensure!(budget > 0 && budget <= BUDGET, "L2 site budget bound");
    let mut spans = Vec::new();
    let mut left = budget;
    fn add(spans: &mut Vec<Span>, left: &mut u64, tensor: Tensor) -> Result<()> {
        if let Some(span) = tensor.span(*left)? {
            *left -= span.bytes;
            spans.push(span);
        }
        Ok(())
    }
    for item in items {
        if left < ALIGN {
            break;
        }
        match *item {
            Item::Tensor(tensor) => add(&mut spans, &mut left, tensor)?,
            Item::Scaled { scale, weight } => {
                add(&mut spans, &mut left, scale)?;
                add(&mut spans, &mut left, weight)?;
            }
            #[cfg(test)]
            Item::Q4 {
                scales,
                biases,
                weight,
                n,
                split_k,
            } => {
                add(&mut spans, &mut left, scales)?;
                add(&mut spans, &mut left, biases)?;
                let chunks = n
                    .div_ceil(64)
                    .checked_mul(split_k)
                    .context("L2 Q4 schedule overflow")?;
                let size = weight.bytes.checked_div(chunks).unwrap_or(0);
                let per = size.min(left.checked_div(chunks).unwrap_or(0)) / ALIGN * ALIGN;
                if (1..=192).contains(&chunks)
                    && weight.bytes % chunks == 0
                    && per >= 512
                    && weight.contiguous
                    && weight.address % ALIGN == 0
                    && size % ALIGN == 0
                {
                    ensure!(weight.address != 0, "null L2 Q4 address");
                    weight
                        .address
                        .checked_add(weight.bytes)
                        .context("L2 Q4 extent overflow")?;
                    for chunk in 0..chunks {
                        spans.push(Span {
                            address: weight.address + chunk * size,
                            bytes: per,
                        });
                    }
                    left -= chunks * per;
                } else {
                    add(&mut spans, &mut left, weight)?;
                }
            }
        }
    }
    Ok(spans)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertPlan {
    pub(crate) bases: [u64; 2],
    pub(crate) stride: u64,
    pub(crate) bytes: u64,
    pub(crate) experts: u64,
}
pub(crate) fn experts(gate: Tensor, up: Tensor, count: u64) -> Result<Option<ExpertPlan>> {
    if count == 0
        || gate.bytes != up.bytes
        || !gate.contiguous
        || !up.contiguous
        || !gate.bytes.is_multiple_of(count)
    {
        return Ok(None);
    }
    let stride = gate.bytes / count;
    let bytes = EXPERT_BYTES.min(stride) / ALIGN * ALIGN;
    if bytes == 0
        || !stride.is_multiple_of(ALIGN)
        || !gate.address.is_multiple_of(ALIGN)
        || !up.address.is_multiple_of(ALIGN)
    {
        return Ok(None);
    }
    ensure!(
        gate.address != 0 && up.address != 0,
        "null L2 expert address"
    );
    gate.address
        .checked_add(gate.bytes)
        .context("L2 gate extent overflow")?;
    up.address
        .checked_add(up.bytes)
        .context("L2 up extent overflow")?;
    ensure!(
        count <= i64::MAX as u64 && stride <= i64::MAX as u64,
        "L2 expert launch bounds"
    );
    Ok(Some(ExpertPlan {
        bases: [gate.address, up.address],
        stride,
        bytes,
        experts: count,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(address: u64, bytes: u64) -> Tensor {
        Tensor {
            address,
            bytes,
            contiguous: true,
        }
    }
    #[test]
    fn retained_policy_is_bounded_and_skips_prefill() {
        assert_eq!(Policy::default(), Policy::Off);
        assert_ne!(Policy::Retained, Policy::Off);
        assert_eq!(Policy::parse(" 1 ").unwrap(), Policy::Retained);
        assert_eq!(Policy::parse("").unwrap(), Policy::Off);
        assert!(Policy::parse("12MiB").is_err());
        assert_eq!(budget(24 << 20), BUDGET);
        assert_eq!(budget(0), BUDGET);
        assert_eq!(budget(8 << 20), 4 << 20);
        for rows in [1, 2, 8, 16, 64] {
            assert!(rows_enabled(rows));
        }
        for rows in [0, 65, 512, 8192] {
            assert!(!rows_enabled(rows));
        }
        assert_eq!(grid(BUDGET, 48).unwrap(), 2);
        assert_eq!(grid(16, 48).unwrap(), 1);
        assert!(grid(BUDGET + 1, 48).is_err());
        assert!(grid(16, 0).is_err());
        assert_ne!(Site::AttentionGather, Site::FeedForwardGather);
        assert_ne!(Site::AttentionCore, Site::AttentionGather);
    }
    #[test]
    fn aligned_read_hints_never_exceed_tensor_or_site() {
        let out = take(
            &[
                Item::Tensor(tensor(0x1003, 61)),
                Item::Tensor(tensor(0x2000, 64)),
            ],
            80,
        )
        .unwrap();
        assert_eq!(
            out,
            vec![
                Span {
                    address: 0x1010,
                    bytes: 48
                },
                Span {
                    address: 0x2000,
                    bytes: 32
                }
            ]
        );
        assert!(take(&[Item::Tensor(tensor(u64::MAX - 31, 64))], 64).is_err());
        assert!(take(&[Item::Tensor(tensor(0, 64))], 64).is_err());
        assert!(
            take(
                &[Item::Tensor(Tensor {
                    contiguous: false,
                    ..tensor(0x1000, 64)
                })],
                64
            )
            .unwrap()
            .is_empty()
        );
    }
    #[test]
    fn canonical_fp8_reads_scales_before_words() {
        let out = take(
            &[Item::Scaled {
                scale: tensor(0x1000, 4096),
                weight: tensor(0x10000, 16 << 20),
            }],
            BUDGET,
        )
        .unwrap();
        assert_eq!(
            out,
            vec![
                Span {
                    address: 0x1000,
                    bytes: 4096
                },
                Span {
                    address: 0x10000,
                    bytes: BUDGET - 4096
                }
            ]
        );
    }
    #[test]
    fn q4_single_wave_heads_and_larger_launch_fallback_match_reference() {
        let item = |n| Item::Q4 {
            scales: tensor(0x1000, 16),
            biases: tensor(0x2000, 16),
            weight: tensor(0x10000, 8 << 20),
            n,
            split_k: 2,
        };
        let out = take(&[item(4096)], 1 << 20).unwrap();
        assert_eq!(out.len(), 130);
        assert_eq!(
            out[2],
            Span {
                address: 0x10000,
                bytes: 8176
            }
        );
        assert_eq!(
            out[129],
            Span {
                address: 0x10000 + 127 * 65536,
                bytes: 8176
            }
        );
        let out = take(&[item(8192)], 1 << 20).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[2].bytes, (1 << 20) - 32);
        let out = take(&[item(4096)], 1024).unwrap();
        assert_eq!(out.len(), 3);
    }
    #[test]
    fn routed_gate_up_heads_keep_shared_expert_outside_the_weight_extent() {
        let plan = experts(
            tensor(0x10000, 288 * 131072),
            tensor(0x5000000, 288 * 131072),
            288,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            plan,
            ExpertPlan {
                bases: [0x10000, 0x5000000],
                stride: 131072,
                bytes: 65536,
                experts: 288
            }
        );
        assert!(
            experts(
                tensor(0x10001, 288 * 131072),
                tensor(0x5000000, 288 * 131072),
                288
            )
            .unwrap()
            .is_none()
        );
        assert!(
            experts(tensor(0x10000, 32), tensor(0x5000000, 16), 1)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            experts(tensor(0x10000, 32), tensor(0x5000000, 32), 1)
                .unwrap()
                .unwrap()
                .bytes,
            32
        );
        assert_eq!(expert_grid(1, 288, 289, 48).unwrap(), (9, 9));
        assert_eq!(expert_grid(4, 288, 289, 48).unwrap(), (33, 33));
        assert_eq!(expert_grid(16, 288, 289, 48).unwrap(), (129, 48));
        assert_eq!(expert_grid(64, 288, 289, 48).unwrap(), (289, 48));
        assert!(expert_grid(4, 288, 9, 48).is_err());
    }
}
