//! Image geometry, canonical canvas identity and streamed patch layout.
//! Image decoding is a separate stage. The resize implementation follows
//! ATen integer arithmetic; see LICENSES/PyTorch.txt and THIRD_PARTY_NOTICES.md.
use crate::{chat_json, vision_input::ImageKey};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const PATCH: usize = 14;
const MERGE: usize = 2;
const TEMPORAL: usize = 2;
const FACTOR: usize = PATCH * MERGE;
const PIXEL_CAP: usize = 64 * 1024 * 1024;
pub(crate) const PATCH_WIDTH: usize = 3 * TEMPORAL * PATCH * PATCH;

pub(crate) struct Settings {
    key: Vec<u8>,
    mean255: [f32; 3],
    std255: [f32; 3],
}
impl Settings {
    pub(crate) fn canonical(config: &Value, processor: &Value) -> Result<Self> {
        let v = &config["vision_config"];
        let p = &processor["image_processor"];
        for (name, expected) in [
            ("depth", 24),
            ("hidden_size", 1024),
            ("num_heads", 16),
            ("out_hidden_size", 4096),
            ("patch_size", 14),
            ("spatial_merge_size", 2),
            ("temporal_patch_size", 2),
            ("in_channels", 3),
            ("intermediate_size", 4096),
            ("projection_intermediate_size", 10240),
        ] {
            ensure!(
                v[name].as_u64() == Some(expected),
                "unsupported vision {name}"
            );
        }
        ensure!(
            v["hidden_act"] == "silu"
                && v["rms_norm_eps"].as_f64() == Some(1e-5)
                && v["swiglu_limit"].as_f64() == Some(10.0),
            "unsupported vision math"
        );
        ensure!(
            v["attention_dropout"].as_f64() == Some(0.0) && v["attention_bias"] == true,
            "unsupported vision attention"
        );
        if let Some(rope) = v.get("rope_parameters") {
            ensure!(
                rope["rope_type"].as_str().unwrap_or("axial") == "axial"
                    && rope["rope_theta"].as_f64().unwrap_or(10000.0) == 10000.0,
                "unsupported vision RoPE"
            );
        }
        for (name, expected) in [
            ("patch_size", 14),
            ("merge_size", 2),
            ("temporal_patch_size", 2),
            ("patch_expand_factor", 1),
            ("min_image_tokens", 16),
            ("max_image_tokens", 8000),
        ] {
            ensure!(
                p[name].as_u64() == Some(expected),
                "unsupported processor {name}"
            );
        }
        ensure!(p["do_rescale"] == true, "unsupported processor rescale");
        for (name, expected) in [
            ("image_token_id", 154854),
            ("image_start_token_id", 154830),
            ("image_end_token_id", 154831),
        ] {
            ensure!(
                config[name].as_u64() == Some(expected),
                "unsupported image token"
            );
        }
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];
        for (c, (&m, &s)) in mean.iter().zip(&std).enumerate() {
            ensure!(
                p["image_mean"][c].as_f64() == Some(m) && p["image_std"][c].as_f64() == Some(s),
                "unsupported image normalization"
            );
        }
        ensure!(
            p["image_mean"].as_array().is_some_and(|a| a.len() == 3)
                && p["image_std"].as_array().is_some_and(|a| a.len() == 3),
            "image channels"
        );
        let key=chat_json::json(&json!({"mean":mean,"std":std,"patch":PATCH,"merge":MERGE,"temporal":TEMPORAL,"rescale":1.0f64/255.0,"vision":v}),true,false)?.into_bytes();
        Ok(Self {
            key,
            mean255: mean.map(|v| v as f32 * 255.0f32),
            std255: std.map(|v| v as f32 * 255.0f32),
        })
    }
    pub(crate) fn key(&self) -> &[u8] {
        &self.key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Plan {
    source_h: usize,
    source_w: usize,
    height: usize,
    width: usize,
    content_h: usize,
    content_w: usize,
}
impl Plan {
    pub(crate) fn new(height: usize, width: usize, max_tokens: usize) -> Result<Self> {
        ensure!(
            height > 0 && width > 0 && height.checked_mul(width).is_some_and(|p| p <= PIXEL_CAP),
            "decoded image pixel bound"
        );
        ensure!(
            (16..=8000).contains(&max_tokens),
            "image token budget bound"
        );
        let align = |v: usize| v.div_ceil(FACTOR) * FACTOR;
        let min_pixels = 16 * FACTOR * FACTOR;
        let max_pixels = max_tokens * FACTOR * FACTOR;
        let mut th = align(height);
        let mut tw = align(width);
        if th * tw < min_pixels {
            let scale = (min_pixels as f64 / (height * width) as f64).sqrt();
            th = align(((height as f64 * scale).ceil() as usize).max(1));
            tw = align(((width as f64 * scale).ceil() as usize).max(1));
        }
        if th * tw > max_pixels {
            let (mut low, mut high) = (1, height);
            (th, tw) = (FACTOR, FACTOR);
            while low <= high {
                let h = (low + high) / 2;
                let w = ((width as f64 * h as f64 / height as f64).floor() as usize).max(1);
                let (ah, aw) = (align(h), align(w));
                if ah * aw <= max_pixels {
                    (th, tw) = (ah, aw);
                    low = h + 1;
                } else {
                    high = h - 1;
                }
            }
        }
        let mut scale = (th as f64 / height as f64).min(tw as f64 / width as f64);
        if height * width >= min_pixels {
            scale = scale.min(1.0);
        }
        let ch = ((height as f64 * scale).floor() as usize).clamp(1, th);
        let cw = ((width as f64 * scale).floor() as usize).clamp(1, tw);
        Ok(Self {
            source_h: height,
            source_w: width,
            height: th,
            width: tw,
            content_h: ch,
            content_w: cw,
        })
    }
    pub(crate) fn canvas_size(&self) -> (usize, usize) {
        (self.height, self.width)
    }
    pub(crate) fn content_size(&self) -> (usize, usize) {
        (self.content_h, self.content_w)
    }
    pub(crate) fn grid(&self) -> [usize; 3] {
        [1, self.height / PATCH, self.width / PATCH]
    }
    pub(crate) fn tokens(&self) -> usize {
        self.height * self.width / (FACTOR * FACTOR)
    }
    pub(crate) fn needs_resize(&self) -> bool {
        (self.source_h, self.source_w) != self.content_size()
    }
}

/// Retain only uint8 CHW canvas; generate bounded float32 patch slabs on demand.
/// This avoids retaining the8x larger temporally duplicated float patch matrix.
pub(crate) struct Prepared {
    plan: Plan,
    canvas: Vec<u8>,
    key: ImageKey,
}
impl Prepared {
    /// Decoded RGB follows the specified uint8 resize, padding and normalization.
    pub(crate) fn from_rgb(plan: Plan, rgb: Vec<u8>, settings: &Settings) -> Result<Self> {
        Self::from_rgb_checked(plan, rgb, settings, || Ok(()))
    }
    pub(crate) fn from_rgb_checked(
        plan: Plan,
        rgb: Vec<u8>,
        settings: &Settings,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        let resized = crate::vision_resize::resize_rgb_checked(
            rgb,
            plan.source_h,
            plan.source_w,
            plan.content_h,
            plan.content_w,
            &mut check,
        )?;
        Self::from_resized_rgb_checked(plan, &resized, settings, check)
    }
    /// Input is decoded RGB, already resized by the exact canonical resizer to
    /// content_size. Padding is black at right/bottom, before normalization.
    pub(crate) fn from_resized_rgb(plan: Plan, rgb: &[u8], settings: &Settings) -> Result<Self> {
        Self::from_resized_rgb_checked(plan, rgb, settings, || Ok(()))
    }
    fn from_resized_rgb_checked(
        plan: Plan,
        rgb: &[u8],
        settings: &Settings,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        check()?;
        ensure!(
            rgb.len() == plan.content_h * plan.content_w * 3,
            "resized RGB dimensions"
        );
        let area = plan.height * plan.width;
        let mut canvas = vec![0u8; area * 3];
        for y in 0..plan.content_h {
            check()?;
            for x in 0..plan.content_w {
                for c in 0..3 {
                    canvas[c * area + y * plan.width + x] = rgb[(y * plan.content_w + x) * 3 + c];
                }
            }
        }
        let mut h = Sha256::new();
        h.update(b"tensorfold-glm5-vision/1");
        h.update(settings.key());
        h.update(
            chat_json::json(
                &json!([
                    plan.height,
                    plan.width,
                    plan.height / PATCH,
                    plan.width / PATCH
                ]),
                false,
                false,
            )?
            .as_bytes(),
        );
        for bytes in canvas.chunks(1024 * 1024) {
            check()?;
            h.update(bytes);
        }
        check()?;
        Ok(Self {
            plan,
            canvas,
            key: ImageKey(h.finalize().into()),
        })
    }
    pub(crate) fn key(&self) -> ImageKey {
        self.key
    }
    pub(crate) fn plan(&self) -> Plan {
        self.plan
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        self.canvas.len()
    }
    pub(crate) fn write_patch_rows(
        &self,
        settings: &Settings,
        start: usize,
        rows: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let gh = self.plan.height / PATCH;
        let gw = self.plan.width / PATCH;
        ensure!(
            start <= gh * gw && rows <= gh * gw - start && rows <= 8192,
            "patch slab bound"
        );
        ensure!(out.len() == rows * PATCH_WIDTH, "patch output size");
        let area = self.plan.height * self.plan.width;
        for (r, dst) in out.as_chunks_mut::<PATCH_WIDTH>().0.iter_mut().enumerate() {
            let row = start + r;
            let group = row / (MERGE * MERGE);
            let inside = row % (MERGE * MERGE);
            let y = (group / (gw / MERGE)) * MERGE + inside / MERGE;
            let x = (group % (gw / MERGE)) * MERGE + inside % MERGE;
            for c in 0..3 {
                for t in 0..TEMPORAL {
                    for py in 0..PATCH {
                        for px in 0..PATCH {
                            let v = self.canvas
                                [c * area + (y * PATCH + py) * self.plan.width + x * PATCH + px]
                                as f32;
                            dst[((c * TEMPORAL + t) * PATCH + py) * PATCH + px] =
                                (v - settings.mean255[c]) / settings.std255[c];
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "vision_fixtures.rs"]
mod fixtures;
#[cfg(test)]
mod tests {
    use super::*;
    fn settings() -> Settings {
        Settings::canonical(
            &serde_json::from_str(fixtures::CONFIG).unwrap(),
            &serde_json::from_str(fixtures::PROCESSOR).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn canonical_geometry_boundaries_and_aspect() {
        assert_eq!(Plan::new(1, 1, 8000).unwrap().canvas_size(), (112, 112));
        let p = Plan::new(512, 512, 8000).unwrap();
        assert_eq!(p.canvas_size(), (532, 532));
        assert_eq!(p.content_size(), (512, 512));
        assert_eq!(p.tokens(), 361);
        assert!(!p.needs_resize());
        let p = Plan::new(1024, 1024, 8000).unwrap();
        assert_eq!(p.canvas_size(), (1036, 1036));
        assert_eq!(p.tokens(), 1369);
        let p = Plan::new(4096, 4096, 8000).unwrap();
        assert_eq!(p.canvas_size(), (2492, 2492));
        assert_eq!(p.tokens(), 7921);
        assert!(p.needs_resize());
        let p = Plan::new(300, 900, 256).unwrap();
        assert_eq!(p.canvas_size(), (252, 756));
        assert_eq!(p.content_size(), (252, 756));
        assert_eq!(p.tokens(), 243);
        for (h, w) in [(1, 64 * 1024 * 1024), (8000, 8000), (28, 448), (1, 1)] {
            let p = Plan::new(h, w, 16).unwrap();
            assert!(p.tokens() <= 16);
            assert!(p.content_h <= p.height && p.content_w <= p.width);
        }
        assert!(Plan::new(0, 12, 16).is_err());
        assert!(Plan::new(usize::MAX, 2, 16).is_err());
        assert!(Plan::new(1, 1, 15).is_err());
    }
    #[test]
    fn channel_temporal_patch_order_padding_and_chunk_equivalence() {
        let s = settings();
        let p = Plan::new(113, 113, 8000).unwrap();
        let (h, w) = p.content_size();
        assert_eq!((h, w), (113, 113));
        let rgb: Vec<u8> = (0..h * w)
            .flat_map(|i| [(i / w) as u8, (i % w) as u8, 200])
            .collect();
        let a = Prepared::from_resized_rgb(p, &rgb, &s).unwrap();
        assert_eq!(a.retained_bytes(), 140 * 140 * 3);
        assert_eq!(a.plan().grid(), [1, 10, 10]);
        let mut whole = vec![0.0; 100 * PATCH_WIDTH];
        a.write_patch_rows(&s, 0, 100, &mut whole).unwrap();
        let mut pieces = vec![0.0; whole.len()];
        for start in (0..100).step_by(7) {
            let n = 7.min(100 - start);
            a.write_patch_rows(
                &s,
                start,
                n,
                &mut pieces[start * PATCH_WIDTH..(start + n) * PATCH_WIDTH],
            )
            .unwrap();
        }
        assert_eq!(whole, pieces);
        for (row, (y, x)) in [(0, 0), (0, 14), (14, 0), (14, 14), (0, 28)]
            .into_iter()
            .enumerate()
        {
            let r = &whole[row * PATCH_WIDTH..(row + 1) * PATCH_WIDTH];
            assert_eq!(r[0], (y as f32 - s.mean255[0]) / s.std255[0]);
            assert_eq!(r[392], (x as f32 - s.mean255[1]) / s.std255[1]);
            assert_eq!(&r[..196], &r[196..392]);
        }
        let last = &whole[99 * PATCH_WIDTH..];
        assert_eq!(last[0], -s.mean255[0] / s.std255[0]);
        let before = whole.clone();
        assert!(a.write_patch_rows(&s, 99, 2, &mut whole).is_err());
        assert_eq!(whole, before);
        let mut changed = rgb;
        changed[0] ^= 1;
        let b = Prepared::from_resized_rgb(p, &changed, &s).unwrap();
        assert_ne!(a.key(), b.key());
    }
    #[test]
    fn decoded_rgb_constant_resize_matches_canonical_patch_path() {
        let s = settings();
        let p = Plan::new(1, 1, 8000).unwrap();
        let direct = Prepared::from_rgb(p, vec![19, 102, 185], &s).unwrap();
        let reference =
            Prepared::from_resized_rgb(p, &[19, 102, 185].repeat(112 * 112), &s).unwrap();
        assert_eq!(direct.key(), reference.key());
        let mut a = vec![0.0; 64 * PATCH_WIDTH];
        let mut b = vec![0.0; a.len()];
        direct.write_patch_rows(&s, 0, 64, &mut a).unwrap();
        reference.write_patch_rows(&s, 0, 64, &mut b).unwrap();
        assert_eq!(a, b);
        assert!(Prepared::from_rgb(p, vec![0; 2], &s).is_err());
    }
    #[test]
    fn pinned_metadata_and_digest_json() {
        let s = settings();
        let key = std::str::from_utf8(s.key()).unwrap();
        assert!(key.contains("\"rms_norm_eps\": 1e-05"));
        assert!(key.contains("\"rescale\": 0.00392156862745098"));
        let mut config: Value = serde_json::from_str(fixtures::CONFIG).unwrap();
        let proc: Value = serde_json::from_str(fixtures::PROCESSOR).unwrap();
        config["vision_config"]["hidden_size"] = json!(512);
        assert!(Settings::canonical(&config, &proc).is_err());
    }
}
