//! Rust-owned vision tower. Every operation delegates to a pinned native
//! BF16/FP32 ATen primitive; no Python, model orchestration in C++, or target FP8.
//! Intermediate and final tensors support native differential qualification.
use crate::{
    vision_native::{Kind, Native, OpCode as O, Tensor},
    vision_preprocess::{PATCH_WIDTH, Prepared, Settings},
    vision_weights::Weights,
};
use anyhow::{Context, Result, ensure};
use std::{cell::Cell, collections::BTreeMap, path::Path};
const WIDTH: i64 = 1024;
const HEADS: i64 = 16;
const HD: i64 = 64;
type Observe<'a> = &'a mut dyn FnMut(&str, &Tensor) -> Result<()>;
pub(crate) struct Tower {
    native: Native,
    weights: BTreeMap<String, Tensor>,
    healthy: Cell<bool>,
}
impl Tower {
    pub(crate) fn load(native: Native, path: &Path) -> Result<Self> {
        let mut source = Weights::open(path)?;
        let mut weights = BTreeMap::new();
        source.load(1024 * 1024, |spec, offset, bytes| {
            if offset == 0 {
                let shape: Vec<i64> = spec
                    .shape
                    .iter()
                    .map(|&d| i64::try_from(d))
                    .collect::<std::result::Result<_, _>>()?;
                ensure!(
                    !weights.contains_key(&spec.name),
                    "duplicate loaded vision weight"
                );
                weights.insert(spec.name.clone(), native.empty(Kind::Bf16, 0, &shape)?);
            }
            weights
                .get_mut(&spec.name)
                .context("missing provisional vision weight")?
                .upload(offset, bytes)
        })?;
        native.sync()?;
        ensure!(weights.len() == 347, "vision weight set incomplete");
        Ok(Self {
            native,
            weights,
            healthy: Cell::new(true),
        })
    }
    fn w(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(&format!("model.visual.{name}"))
            .context("canonical vision weight absent")
    }
    fn op(&self, code: O, x: &[&Tensor], dims: &[i64], scalars: &[f64]) -> Result<Tensor> {
        self.native.op(code, x, dims, scalars)
    }
    fn shape(&self, x: &Tensor, dims: &[i64]) -> Result<Tensor> {
        self.op(O::Reshape, &[x], dims, &[])
    }
    fn cast(&self, x: &Tensor, kind: Kind) -> Result<Tensor> {
        self.op(O::Cast, &[x], &[kind as i64], &[])
    }
    fn linear(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        if let Some(b) = bias {
            self.op(O::Linear, &[x, weight, b], &[], &[])
        } else {
            self.op(O::Linear, &[x, weight], &[], &[])
        }
    }
    fn rms(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let f = self.cast(x, Kind::F32)?;
        let square = self.op(O::PowScalar, &[&f], &[], &[2.])?;
        let mean = self.op(O::Mean, &[&square], &[-1], &[])?;
        drop(square);
        let offset = self.op(O::AddScalar, &[&mean], &[], &[1e-5])?;
        drop(mean);
        let inv = self.op(O::Rsqrt, &[&offset], &[], &[])?;
        drop(offset);
        let h = self.op(O::Mul, &[&f, &inv], &[], &[])?;
        drop((f, inv));
        let rounded = self.cast(&h, Kind::Bf16)?;
        drop(h);
        self.op(O::Mul, &[w, &rounded], &[], &[])
    }
    fn swiglu(&self, x: &Tensor, prefix: &str, bias: bool) -> Result<Tensor> {
        let mut gate = self.linear(
            x,
            self.w(&format!("{prefix}gate_proj.weight"))?,
            if bias {
                Some(self.w(&format!("{prefix}gate_proj.bias"))?)
            } else {
                None
            },
        )?;
        gate = self.op(O::ClampMax, &[&gate], &[], &[10.])?;
        let mut up = self.linear(
            x,
            self.w(&format!("{prefix}up_proj.weight"))?,
            if bias {
                Some(self.w(&format!("{prefix}up_proj.bias"))?)
            } else {
                None
            },
        )?;
        up = self.op(O::Clamp, &[&up], &[], &[-10., 10.])?;
        gate = self.op(O::Silu, &[&gate], &[], &[])?;
        gate = self.op(O::Mul, &[&gate, &up], &[], &[])?;
        drop(up);
        self.linear(
            &gate,
            self.w(&format!("{prefix}down_proj.weight"))?,
            if bias {
                Some(self.w(&format!("{prefix}down_proj.bias"))?)
            } else {
                None
            },
        )
    }
    fn mlp(&self, x: &Tensor, prefix: &str, rows: i64) -> Result<Tensor> {
        if rows <= 8192 {
            return self.swiglu(x, prefix, true);
        }
        let mut pieces = Vec::new();
        for start in (0..rows).step_by(8192) {
            let piece = self.op(O::Slice, &[x], &[0, start, (start + 8192).min(rows)], &[])?;
            pieces.push(self.swiglu(&piece, prefix, true)?);
        }
        self.op(O::Cat, &pieces.iter().collect::<Vec<_>>(), &[0], &[])
    }
    fn rope(&self, gh: i64, gw: i64) -> Result<(Tensor, Tensor)> {
        // Frozen inverse frequencies are Float32 CPU pow/div/reciprocal before
        // copying to the dedicated CUDA stream. Do not compute in Rust f64.
        let range = self.op(O::Arange, &[], &[0, HD / 2, 2, Kind::F32 as i64, -1], &[])?;
        let exponent = self.op(O::DivScalar, &[&range], &[], &[(HD / 2) as f64])?;
        let power = self.op(O::ScalarPow, &[&exponent], &[], &[10000.])?;
        let inverse = self.op(O::Reciprocal, &[&power], &[], &[])?;
        let inverse = self.op(O::ToDevice, &[&inverse], &[0], &[])?;
        let axis = |size: i64, vertical: bool| -> Result<Tensor> {
            let values = self.op(O::Arange, &[], &[0, size, 1, Kind::I64 as i64, 0], &[])?;
            let dims = if vertical { [gh, 1] } else { [1, gw] };
            let values = self.shape(&values, &dims)?;
            let grid = self.op(O::Expand, &[&values], &[gh, gw], &[])?;
            let tiles = self.shape(&grid, &[gh / 2, 2, gw / 2, 2])?;
            let tiles = self.op(O::Transpose, &[&tiles], &[1, 2], &[])?;
            self.shape(&tiles, &[gh * gw, 1])
        };
        let h = axis(gh, true)?;
        let w = axis(gw, false)?;
        let positions = self.op(O::Cat, &[&h, &w], &[1], &[])?;
        let positions = self.op(O::Unsqueeze, &[&positions], &[-1], &[])?;
        let positions = self.cast(&positions, Kind::F32)?;
        let freq = self.op(O::Mul, &[&positions, &inverse], &[], &[])?;
        let recompose = |value: Tensor| -> Result<Tensor> {
            let h = self.op(O::Select, &[&value], &[1, 0], &[])?;
            let w = self.op(O::Select, &[&value], &[1, 1], &[])?;
            let hw = self.op(O::Cat, &[&h, &w], &[-1], &[])?;
            self.op(O::Cat, &[&hw, &hw], &[-1], &[])
        };
        Ok((
            recompose(self.op(O::Cos, &[&freq], &[], &[])?)?,
            recompose(self.op(O::Sin, &[&freq], &[], &[])?)?,
        ))
    }
    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = self.cast(x, Kind::F32)?;
        let a = self.op(O::Slice, &[&x], &[-1, HD / 2, HD], &[])?;
        let a = self.op(O::Neg, &[&a], &[], &[])?;
        let b = self.op(O::Slice, &[&x], &[-1, 0, HD / 2], &[])?;
        let rotated = self.op(O::Cat, &[&a, &b], &[-1], &[])?;
        drop((a, b));
        let c = self.op(O::Unsqueeze, &[cos], &[-2], &[])?;
        let s = self.op(O::Unsqueeze, &[sin], &[-2], &[])?;
        let left = self.op(O::Mul, &[&x, &c], &[], &[])?;
        let right = self.op(O::Mul, &[&rotated, &s], &[], &[])?;
        drop((x, rotated));
        self.cast(&self.op(O::Add, &[&left, &right], &[], &[])?, Kind::Bf16)
    }
    fn block(
        &self,
        x: &Tensor,
        rows: i64,
        layer: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let p = format!("blocks.{layer}.");
        let h = self.rms(x, self.w(&format!("{p}norm1.weight"))?)?;
        let mut qkv = self.linear(
            &h,
            self.w(&format!("{p}attn.qkv.weight"))?,
            Some(self.w(&format!("{p}attn.qkv.bias"))?),
        )?;
        drop(h);
        qkv = self.shape(&qkv, &[rows, 3, HEADS, HD])?;
        qkv = self.op(O::Permute, &[&qkv], &[1, 0, 2, 3], &[])?;
        let mut q = self.op(O::Select, &[&qkv], &[0, 0], &[])?;
        let mut k = self.op(O::Select, &[&qkv], &[0, 1], &[])?;
        let mut v = self.op(O::Select, &[&qkv], &[0, 2], &[])?;
        drop(qkv);
        q = self.rms(&q, self.w(&format!("{p}attn.q_norm.weight"))?)?;
        k = self.rms(&k, self.w(&format!("{p}attn.k_norm.weight"))?)?;
        q = self.rotate(&q, cos, sin)?;
        k = self.rotate(&k, cos, sin)?;
        let attention_shape = |t: &Tensor| -> Result<Tensor> {
            let t = self.op(O::Transpose, &[t], &[0, 1], &[])?;
            self.op(O::Unsqueeze, &[&t], &[0], &[])
        };
        q = attention_shape(&q)?;
        k = attention_shape(&k)?;
        v = attention_shape(&v)?;
        let mut a = self.op(O::Sdpa, &[&q, &k, &v], &[], &[0.125])?;
        drop((q, k, v));
        a = self.op(O::Transpose, &[&a], &[1, 2], &[])?;
        a = self.shape(&a, &[rows, WIDTH])?;
        let projected = self.linear(
            &a,
            self.w(&format!("{p}attn.proj.weight"))?,
            Some(self.w(&format!("{p}attn.proj.bias"))?),
        )?;
        drop(a);
        let x = self.op(O::Add, &[x, &projected], &[], &[])?;
        drop(projected);
        let h = self.rms(&x, self.w(&format!("{p}norm2.weight"))?)?;
        let m = self.mlp(&h, &format!("{p}mlp."), rows)?;
        self.op(O::Add, &[&x, &m], &[], &[])
    }
    /// Bounded Float32 host staging:1024 patch rows. Full matrix is device-only
    /// and is released after its BF16 cast. Observer readbacks are diagnostics;
    /// normal use supplies a no-op closure. The completed tensor crosses the
    /// ownership edge only after this dedicated stream successfully drains.
    pub(crate) fn encode(
        &self,
        image: &Prepared,
        settings: &Settings,
        observe: Observe<'_>,
    ) -> Result<Tensor> {
        ensure!(
            self.healthy.replace(false),
            "vision owner failed or already encoding"
        );
        let result = self.encode_inner(image, settings, observe);
        if result.is_ok() {
            self.healthy.set(true);
        }
        result
    }
    fn encode_inner(
        &self,
        image: &Prepared,
        settings: &Settings,
        observe: Observe<'_>,
    ) -> Result<Tensor> {
        let [frames, gh, gw] = image.plan().grid();
        let n = gh * gw;
        ensure!(
            frames == 1 && gh % 2 == 0 && gw % 2 == 0 && n > 0 && n <= 32000,
            "vision patch grid"
        );
        let mut pixels = self
            .native
            .empty(Kind::F32, 0, &[n as i64, PATCH_WIDTH as i64])?;
        let mut values = vec![0.0f32; 1024 * PATCH_WIDTH];
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for start in (0..n).step_by(1024) {
            let rows = 1024.min(n - start);
            let values = &mut values[..rows * PATCH_WIDTH];
            image.write_patch_rows(settings, start, rows, values)?;
            bytes.clear();
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            pixels.upload((start * PATCH_WIDTH * 4) as u64, &bytes)?;
        }
        observe("pixels-f32", &pixels)?;
        let mut x = self.cast(&pixels, Kind::Bf16)?;
        drop(pixels);
        drop((values, bytes));
        observe("pixels-bf16", &x)?;
        let patch = self.shape(self.w("patch_embed.proj.weight")?, &[WIDTH, -1])?;
        x = self.linear(&x, &patch, Some(self.w("patch_embed.proj.bias")?))?;
        observe("patch-projection", &x)?;
        let (cos, sin) = self.rope(gh as i64, gw as i64)?;
        observe("rope-cos", &cos)?;
        observe("rope-sin", &sin)?;
        for layer in 0..24 {
            x = self.block(&x, n as i64, layer, &cos, &sin)?;
            observe(&format!("block-{layer}"), &x)?;
        }
        x = self.rms(&x, self.w("post_layernorm.weight")?)?;
        observe("post-norm", &x)?;
        x = self.shape(&x, &[-1, 2, 2, WIDTH])?;
        x = self.op(O::Permute, &[&x], &[0, 3, 1, 2], &[])?;
        x = self.shape(&x, &[-1, 4096])?;
        observe("merged-order", &x)?;
        let down = self.shape(self.w("downsample.weight")?, &[4096, -1])?;
        x = self.linear(&x, &down, Some(self.w("downsample.bias")?))?;
        observe("downsample", &x)?;
        x = self.linear(&x, self.w("merger.proj.weight")?, None)?;
        observe("merger-projection", &x)?;
        x = self.op(
            O::LayerNorm,
            &[
                &x,
                self.w("merger.post_projection_norm.weight")?,
                self.w("merger.post_projection_norm.bias")?,
            ],
            &[4096],
            &[1e-5],
        )?;
        observe("merger-layernorm", &x)?;
        x = self.op(O::Gelu, &[&x], &[], &[])?;
        observe("merger-gelu", &x)?;
        x = self.swiglu(&x, "merger.", false)?;
        observe("output", &x)?;
        let (kind, shape, bytes) = x.info()?;
        ensure!(
            kind == Kind::Bf16 && shape == [(n / 4) as i64, 4096] && bytes == n / 4 * 4096 * 2,
            "vision output shape/dtype"
        );
        self.native.sync()?;
        Ok(x)
    }
}
