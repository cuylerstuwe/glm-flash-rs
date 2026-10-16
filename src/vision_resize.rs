//! Frozen ATen CPU uint8 antialiased bicubic: horizontal then vertical,
//! float64 Keys coefficients quantized to int16, rounded/clipped after each pass.
//! Adapted ATen resize arithmetic; see LICENSES/PyTorch.txt and THIRD_PARTY_NOTICES.md.
use anyhow::{Result, ensure};

const PIXELS: usize = 64 * 1024 * 1024;
const OUTPUT_PIXELS: usize = 8000 * 28 * 28;
const OUTPUT_SIDE: usize = 8000 * 28;
const CACHED_WEIGHTS: usize = 1024 * 1024;

fn cubic(mut x: f64) -> f64 {
    x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}
#[derive(Clone, Copy)]
struct Span {
    first: usize,
    count: usize,
    center: f64,
    total: f64,
}
struct Axis {
    spans: Vec<Span>,
    invscale: f64,
    precision: u32,
}
impl Axis {
    fn new(input: usize, output: usize) -> Result<Self> {
        ensure!(
            input > 0 && input <= PIXELS && output > 0 && output <= OUTPUT_SIDE,
            "resize axis bound"
        );
        let scale = input as f64 / output as f64;
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
        let max_interp = support.ceil() as usize * 2 + 1;
        let mut spans = Vec::with_capacity(output);
        let mut maximum = 0.0f64;
        for i in 0..output {
            let center = scale * (i as f64 + 0.5);
            // ATen converts to signed int64 by truncation before clamping.
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let last = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let count = last.saturating_sub(first).min(max_interp);
            ensure!(count > 0 && first + count <= input, "empty resize support");
            let mut total = 0.0;
            for j in 0..count {
                total += cubic((j as f64 + first as f64 - center + 0.5) * invscale);
            }
            ensure!(total != 0.0 && total.is_finite(), "invalid resize weights");
            for j in 0..count {
                maximum =
                    maximum.max(cubic((j as f64 + first as f64 - center + 0.5) * invscale) / total);
            }
            spans.push(Span {
                first,
                count,
                center,
                total,
            });
        }
        let mut precision = 0;
        while precision < 22 {
            let next = (0.5 + maximum * f64::from(1u32 << (precision + 1))) as i32;
            if next >= 1 << 15 {
                break;
            }
            precision += 1;
        }
        ensure!(precision > 0, "resize coefficient precision");
        Ok(Self {
            spans,
            invscale,
            precision,
        })
    }
    fn weight(&self, span: Span, j: usize) -> i16 {
        let w =
            cubic((j as f64 + span.first as f64 - span.center + 0.5) * self.invscale) / span.total;
        let v = w * f64::from(1u32 << self.precision);
        (if v < 0.0 { -0.5 + v } else { 0.5 + v }) as i16
    }
    fn coefficients(&self, span: Span, cached: &mut Vec<i16>) {
        cached.clear();
        if span.count <= CACHED_WEIGHTS {
            cached.extend((0..span.count).map(|j| self.weight(span, j)));
        }
    }
    fn pixel(&self, span: Span, cached: &[i16], source: impl Fn(usize) -> u8) -> u8 {
        // Same integer arithmetic as ATen within its i32 range; the wider
        // accumulator additionally avoids overflow on hostile geometry.
        let mut value = 1i64 << (self.precision - 1);
        for j in 0..span.count {
            let weight = if cached.is_empty() {
                self.weight(span, j)
            } else {
                cached[j]
            };
            value += i64::from(source(span.first + j)) * i64::from(weight);
        }
        (value >> self.precision).clamp(0, 255) as u8
    }
}

/// Consumes RGB bytes, returning the allocation unchanged for identity resize.
/// RGB channel order is independent of the frozen contiguous CHW scalar path.
pub(crate) fn resize_rgb(
    rgb: Vec<u8>,
    height: usize,
    width: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Vec<u8>> {
    resize_rgb_checked(rgb, height, width, target_h, target_w, || Ok(()))
}

/// The caller owns native hold/cancellation policy. Checks occur at completed
/// column/row boundaries and do not alter coefficient or pixel arithmetic.
pub(crate) fn resize_rgb_checked(
    rgb: Vec<u8>,
    height: usize,
    width: usize,
    target_h: usize,
    target_w: usize,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    check()?;
    let bounded =
        |h: usize, w: usize, cap| h > 0 && w > 0 && h.checked_mul(w).is_some_and(|p| p <= cap);
    ensure!(
        bounded(height, width, PIXELS) && rgb.len() == height * width * 3,
        "decoded RGB bound/length"
    );
    ensure!(
        bounded(target_h, target_w, OUTPUT_PIXELS)
            && target_h <= OUTPUT_SIDE
            && target_w <= OUTPUT_SIDE,
        "resized RGB bound"
    );
    ensure!(
        bounded(height, target_w, PIXELS),
        "resize intermediate bound"
    );
    if (height, width) == (target_h, target_w) {
        return Ok(rgb);
    }
    let intermediate = if width == target_w {
        rgb
    } else {
        let axis = Axis::new(width, target_w)?;
        let mut output = vec![0; height * target_w * 3];
        let mut cached = Vec::new();
        for (x, span) in axis.spans.iter().copied().enumerate() {
            check()?;
            axis.coefficients(span, &mut cached);
            for y in 0..height {
                for c in 0..3 {
                    output[(y * target_w + x) * 3 + c] =
                        axis.pixel(span, &cached, |i| rgb[(y * width + i) * 3 + c]);
                }
            }
        }
        output
    };
    if height == target_h {
        return Ok(intermediate);
    }
    let axis = Axis::new(height, target_h)?;
    let mut output = vec![0; target_h * target_w * 3];
    let mut cached = Vec::new();
    for (y, span) in axis.spans.iter().copied().enumerate() {
        check()?;
        axis.coefficients(span, &mut cached);
        for x in 0..target_w {
            for c in 0..3 {
                output[(y * target_w + x) * 3 + c] =
                    axis.pixel(span, &cached, |i| intermediate[(i * target_w + x) * 3 + c]);
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_constant_channels_and_degenerate_axes() -> Result<()> {
        let input = vec![7, 99, 243];
        let pointer = input.as_ptr();
        let output = resize_rgb(input, 1, 1, 1, 1)?;
        assert_eq!(output.as_ptr(), pointer);
        for (h, w, oh, ow) in [
            (1, 1, 28, 28),
            (7, 1, 28, 3),
            (1, 13, 5, 3),
            (9, 11, 3, 7),
            (3, 7, 9, 11),
        ] {
            let rgb = [7, 99, 243].repeat(h * w);
            assert_eq!(resize_rgb(rgb, h, w, oh, ow)?, [7, 99, 243].repeat(oh * ow));
        }
        Ok(())
    }
    #[test]
    fn axis_canonical_integer_coefficients_and_edge_normalization() -> Result<()> {
        let axis = Axis::new(2, 4)?;
        assert_eq!(axis.precision, 14);
        let got = axis
            .spans
            .iter()
            .map(|s| {
                let mut cached = Vec::new();
                axis.coefficients(*s, &mut cached);
                (s.first, s.count, cached)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            got,
            vec![
                (0, 2, vec![17830, -1446]),
                (0, 2, vec![12990, 3394]),
                (0, 2, vec![3394, 12990]),
                (0, 2, vec![-1446, 17830])
            ]
        );
        let resized = resize_rgb(vec![0, 10, 200, 255, 110, 0], 1, 2, 1, 4)?;
        assert_eq!(
            resized,
            vec![0, 1, 218, 53, 31, 159, 202, 89, 41, 255, 119, 0]
        );
        Ok(())
    }
    #[test]
    fn validates_all_buffer_geometry_before_resize_work() {
        for (bytes, h, w, oh, ow) in [
            (vec![], 0, 1, 1, 1),
            (vec![0; 3], 2, 2, 2, 2),
            (vec![0; 3], 1, 1, 0, 1),
            (vec![0; 3], 1, 1, 1, OUTPUT_SIDE + 1),
        ] {
            assert!(resize_rgb(bytes, h, w, oh, ow).is_err());
        }
    }
}
