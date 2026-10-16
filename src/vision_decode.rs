//! Bounded first-frame PNG decoding with Pillow-compatible RGB conversion.
use anyhow::{Context, Result, ensure};
use png::{BitDepth, ColorType, InterlaceInfo};
use std::io::Cursor;

pub(crate) const MAX_ENCODED_BYTES: usize = 20 * 1024 * 1024;
pub(crate) const MAX_PIXELS: usize = 64 * 1024 * 1024;
const DECODER_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const DEFAULT_WORKSPACE_BYTES: usize = 384 * 1024 * 1024;

pub(crate) struct Decoded {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) rgb: Vec<u8>,
    /// Conservative admission charge; encoded bytes remain caller-owned.
    pub(crate) workspace_charge: usize,
}

/// `check` is the request owner's cancellation/deadline check, called before
/// allocations and between decoded rows. Never publishes a partial image.
pub(crate) fn png(
    encoded: &[u8],
    workspace_bytes: usize,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Decoded> {
    check()?;
    ensure!(
        !encoded.is_empty() && encoded.len() <= MAX_ENCODED_BYTES,
        "encoded image byte bound"
    );
    ensure!(
        encoded.starts_with(b"\x89PNG\r\n\x1a\n"),
        "expected PNG image"
    );
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(encoded),
        png::Limits {
            bytes: DECODER_BYTES,
        },
    );
    // Frozen Pillow path performs no color management and uses no text metadata.
    // Skipping these also avoids allocating decompressed ancillary payloads.
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let header = decoder.read_header_info().context("PNG header")?;
    let (width, height) = (header.width as usize, header.height as usize);
    let pixels = width
        .checked_mul(height)
        .context("PNG dimensions overflow")?;
    ensure!(
        width > 0 && height > 0 && pixels <= MAX_PIXELS,
        "decoded image pixel bound"
    );
    let color = header.color_type;
    let depth = header.bit_depth;
    let bits = depth as usize * color.samples();
    let raw_stride = width
        .checked_mul(bits)
        .context("PNG row overflow")?
        .div_ceil(8)
        + 1;
    let decoder_stride = raw_stride.max(width * 4 + 1);
    let rgb_stride = width
        .checked_add(7)
        .and_then(|n| n.checked_mul(3))
        .context("PNG RGB row overflow")?;
    // png's documented Limits is best-effort, not a total allocator budget.
    // Its pinned row decoder additionally owns a sliding unfilter buffer and
    // scratch rows. Charge 16 raw rows plus fixed slack rather than assuming
    // that the 64-MiB metadata limit accounts for those. Very wide images may
    // require a larger admitted workspace; never silently exceed the caller's.
    let workspace_charge = pixels
        .checked_mul(3)
        .and_then(|n| {
            decoder_stride
                .checked_mul(16)
                .and_then(|r| n.checked_add(r))
        })
        .and_then(|n| n.checked_add(rgb_stride))
        .and_then(|n| n.checked_add(DECODER_BYTES + 1024 * 1024))
        .context("PNG workspace overflow")?;
    ensure!(
        workspace_charge <= workspace_bytes,
        "PNG workspace admission bound"
    );
    check()?;
    // Only palette expansion is desired. Global EXPAND would turn RGB/L tRNS
    // into alpha, unlike the frozen to_rgb path for these modes. This also
    // lets the vetted decoder discard palette row padding before conversion.
    if color == ColorType::Indexed {
        decoder.set_transformations(png::Transformations::EXPAND);
    }
    let mut reader = decoder.read_info().context("PNG metadata")?;
    let info = reader.info();
    let palette = info.palette.as_deref().unwrap_or(&[]).to_vec();
    let transparency = info.trns.as_deref().unwrap_or(&[]).to_vec();
    if color == ColorType::Indexed {
        ensure!(
            !palette.is_empty() && palette.len() % 3 == 0 && palette.len() <= 768,
            "PNG palette bound"
        );
    }
    let (frame_w, frame_h, offset_x, offset_y) =
        info.frame_control.map_or((width, height, 0, 0), |f| {
            (
                f.width as usize,
                f.height as usize,
                f.x_offset as usize,
                f.y_offset as usize,
            )
        });
    ensure!(
        frame_w > 0 && frame_h > 0 && offset_x + frame_w <= width && offset_y + frame_h <= height,
        "PNG first-frame geometry"
    );
    let mut rgb = zeroed(pixels * 3)?;
    let source_mode = Mode {
        color,
        depth,
        palette: &palette,
        transparency: &transparency,
    };
    // An APNG initial sub-frame leaves the remaining native image zero-filled.
    // Convert that zero pixel through the same palette/alpha semantics.
    let background = source_mode.pixel(&[0; 8], 0)?;
    if background != [0; 3] {
        for pixel in rgb.as_chunks_mut::<3>().0 {
            *pixel = background;
        }
    }
    let (out_color, out_depth) = reader.output_color_type();
    let direct_rgb = out_color == ColorType::Rgb && out_depth == BitDepth::Eight;
    let mut row_rgb = zeroed(if direct_rgb { 0 } else { rgb_stride })?;
    let mode = Mode {
        color: out_color,
        depth: out_depth,
        palette: &[],
        transparency: &[],
    };
    let out_bits = out_depth as usize * out_color.samples();
    let mut rows = 0;
    while let Some(row) = reader.next_interlaced_row().context("PNG image data")? {
        check()?;
        let count = row.data().len() * 8 / out_bits;
        ensure!(count <= width + 7, "PNG decoded row bound");
        let converted = if direct_rgb {
            row.data()
        } else {
            for (i, pixel) in row_rgb[..count * 3]
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .enumerate()
            {
                *pixel = mode.pixel(row.data(), i)?;
            }
            &row_rgb[..count * 3]
        };
        match row.interlace() {
            InterlaceInfo::Null(_) => {
                ensure!(rows < frame_h && count >= frame_w, "PNG row geometry");
                let start = ((offset_y + rows) * width + offset_x) * 3;
                rgb[start..start + frame_w * 3].copy_from_slice(&converted[..frame_w * 3]);
                rows += 1;
            }
            InterlaceInfo::Adam7(pass) => {
                let start = (offset_y * width + offset_x) * 3;
                png::expand_interlaced_row(&mut rgb[start..], width * 3, converted, pass, 24);
            }
        }
    }
    // Select the initial/default frame only. The
    // row decoder has already checked this frame's compressed data and CRC.
    // Do not decode later animation frames or let them change the chosen image.
    check()?;
    Ok(Decoded {
        width,
        height,
        rgb,
        workspace_charge,
    })
}

pub(crate) fn zeroed(len: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).context("PNG allocation")?;
    bytes.resize(len, 0);
    Ok(bytes)
}

struct Mode<'a> {
    color: ColorType,
    depth: BitDepth,
    palette: &'a [u8],
    transparency: &'a [u8],
}
impl Mode<'_> {
    fn pixel(&self, row: &[u8], index: usize) -> Result<[u8; 3]> {
        let depth = self.depth as usize;
        if depth < 8 {
            let bit = index * depth;
            let v = row[bit / 8] >> (8 - depth - bit % 8) & ((1 << depth) - 1) as u8;
            return if self.color == ColorType::Indexed {
                self.palette_pixel(v)
            } else {
                ensure!(self.color == ColorType::Grayscale, "PNG sub-byte color");
                Ok([v * (255 / ((1 << depth) - 1)) as u8; 3])
            };
        }
        let bytes = depth / 8;
        let start = index * bytes * self.color.samples();
        let sample = |channel| row[start + channel * bytes];
        Ok(match self.color {
            ColorType::Grayscale => {
                // Pillow I;16 -> RGB clips the numeric sample, rather than
                // selecting its high byte as its RGB/LA/RGBA PNG unpackers do.
                let value = if depth == 16 {
                    u16::from_be_bytes([row[start], row[start + 1]]).min(255) as u8
                } else {
                    sample(0)
                };
                [value; 3]
            }
            ColorType::Rgb => [sample(0), sample(1), sample(2)],
            ColorType::Indexed => return self.palette_pixel(sample(0)),
            ColorType::GrayscaleAlpha => [white(sample(0), sample(1)); 3],
            ColorType::Rgba => [
                white(sample(0), sample(3)),
                white(sample(1), sample(3)),
                white(sample(2), sample(3)),
            ],
        })
    }
    fn palette_pixel(&self, index: u8) -> Result<[u8; 3]> {
        let start = index as usize * 3;
        let rgb = self
            .palette
            .get(start..start + 3)
            .context("PNG palette index")?;
        let alpha = self
            .transparency
            .get(index as usize)
            .copied()
            .unwrap_or(255);
        Ok([
            white(rgb[0], alpha),
            white(rgb[1], alpha),
            white(rgb[2], alpha),
        ])
    }
}

/// Algebraic reduction of Pillow AlphaComposite.c for an opaque white target.
fn white(value: u8, alpha: u8) -> u8 {
    ((u32::from(value) * u32::from(alpha) + 255 * (255 - u32::from(alpha)) + 127) / 255) as u8
}

#[cfg(test)]
#[path = "vision_decode_tests.rs"]
mod tests;
