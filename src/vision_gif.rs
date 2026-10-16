//! First-frame GIF decoding; the LZW/interlace codec is the pinned Rust gif crate.
use crate::vision_decode::{Decoded, MAX_ENCODED_BYTES, MAX_PIXELS, zeroed};
use anyhow::{Context, Result, ensure};
use std::{io::Cursor, num::NonZeroU64};

pub(crate) fn decode(
    encoded: &[u8],
    workspace_bytes: usize,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Decoded> {
    check()?;
    ensure!(
        encoded.len() <= MAX_ENCODED_BYTES
            && (encoded.starts_with(b"GIF87a") || encoded.starts_with(b"GIF89a")),
        "GIF input bound/signature"
    );
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::Indexed);
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(MAX_PIXELS as u64).unwrap(),
    ));
    let mut reader = options
        .read_info(Cursor::new(encoded))
        .context("GIF header")?;
    let (screen_w, screen_h) = (usize::from(reader.width()), usize::from(reader.height()));
    ensure!(
        screen_w > 0 && screen_h > 0 && screen_w * screen_h <= MAX_PIXELS,
        "GIF screen pixel bound"
    );
    let global = reader.global_palette().map(<[u8]>::to_vec);
    let frame = reader
        .next_frame_info()
        .context("GIF frame metadata")?
        .context("GIF missing image frame")?;
    let (fw, fh, left, top) = (
        usize::from(frame.width),
        usize::from(frame.height),
        usize::from(frame.left),
        usize::from(frame.top),
    );
    let (width, height) = (screen_w.max(left + fw), screen_h.max(top + fh));
    let pixels = width * height;
    ensure!(
        fw > 0 && fh > 0 && pixels <= MAX_PIXELS,
        "GIF decoded pixel bound"
    );
    // Pillow extends the canvas to the first frame's bounds. Its default
    // strategy keeps a palette unless every entry is its identical gray index.
    let palette = frame
        .palette
        .clone()
        .or(global)
        .context("GIF missing palette")?;
    let colored = palette
        .as_chunks::<3>()
        .0
        .iter()
        .enumerate()
        .any(|(i, p)| *p != [i as u8; 3]);
    let transparent = frame.transparent;
    let interlaced = frame.interlaced;
    let workspace_charge = pixels * 3 + fw * fh + MAX_ENCODED_BYTES + 2 * 1024 * 1024;
    ensure!(
        workspace_charge <= workspace_bytes,
        "GIF workspace admission bound"
    );
    check()?;
    let mut indices = zeroed(fw * fh)?;
    if interlaced {
        // The vetted decoder performs deinterlacing. Cancellation brackets its
        // one bounded frame; noninterlaced images check between rows below.
        reader
            .read_into_buffer(&mut indices)
            .context("GIF interlaced image")?;
    } else {
        for row in indices.chunks_exact_mut(fw) {
            check()?;
            ensure!(
                reader.fill_buffer(row).context("GIF image row")?,
                "GIF truncated image row"
            );
        }
    }
    check()?;
    let convert = |v: u8| -> Result<[u8; 3]> {
        if !colored {
            return Ok([v; 3]);
        }
        if transparent == Some(v) {
            return Ok([255; 3]);
        }
        let at = usize::from(v) * 3;
        let color = palette.get(at..at + 3).context("GIF palette index")?;
        Ok([color[0], color[1], color[2]])
    };
    // Pillow fills the first canvas with the transparent index if present,
    // otherwise index zero; it does not use the logical background-color index.
    let background = convert(transparent.unwrap_or(0))?;
    let mut rgb = zeroed(pixels * 3)?;
    if background != [0; 3] {
        for pixel in rgb.as_chunks_mut::<3>().0 {
            *pixel = background;
        }
    }
    for (y, row) in indices.chunks_exact(fw).enumerate() {
        check()?;
        let offset = ((top + y) * width + left) * 3;
        for (pixel, &index) in rgb[offset..offset + fw * 3]
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(row)
        {
            *pixel = convert(index)?;
        }
    }
    Ok(Decoded {
        width,
        height,
        rgb,
        workspace_charge,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    fn fixture(palette: &[u8], transparent: Option<u8>, interlaced: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut writer = gif::Encoder::new(&mut bytes, 3, 3, palette).unwrap();
            let frame = gif::Frame {
                width: 2,
                height: 2,
                left: 1,
                top: 1,
                transparent,
                interlaced,
                buffer: Cow::Borrowed(&[0, 1, 1, 0]),
                ..Default::default()
            };
            writer.write_frame(&frame).unwrap();
            let second = gif::Frame {
                width: 3,
                height: 3,
                buffer: Cow::Borrowed(&[1; 9]),
                ..Default::default()
            };
            writer.write_frame(&second).unwrap();
        }
        bytes
    }
    #[test]
    fn colored_palette_first_subrectangle_and_transparent_background() {
        for interlaced in [false, true] {
            let bytes = fixture(&[10, 20, 30, 40, 50, 60], Some(1), interlaced);
            let result = decode(&bytes, 1024 * 1024 * 32, || Ok(())).unwrap();
            let mut expected = vec![255; 27];
            expected[12..15].copy_from_slice(&[10, 20, 30]);
            expected[24..27].copy_from_slice(&[10, 20, 30]);
            assert_eq!(result.rgb, expected);
        }
    }
    #[test]
    fn identity_gray_palette_uses_l_mode_which_ignores_transparency() {
        let bytes = fixture(&[0, 0, 0, 1, 1, 1], Some(1), false);
        let result = decode(&bytes, 1024 * 1024 * 32, || Ok(())).unwrap();
        let mut expected = vec![1; 27];
        expected[12..15].fill(0);
        expected[24..27].fill(0);
        assert_eq!(result.rgb, expected);
    }
    #[test]
    fn initial_palette_background_and_bounded_cancelled_frames() {
        let bytes = fixture(&[10, 20, 30, 40, 50, 60], None, false);
        let result = decode(&bytes, 1024 * 1024 * 32, || Ok(())).unwrap();
        assert_eq!(&result.rgb[..3], &[10, 20, 30]);
        assert!(decode(&bytes, 1, || Ok(())).is_err());
        assert!(decode(&bytes[..20], 1024 * 1024 * 32, || Ok(())).is_err());
        let mut count = 0;
        assert!(
            decode(&bytes, 1024 * 1024 * 32, || {
                count += 1;
                ensure!(count < 4, "cancelled");
                Ok(())
            })
            .is_err()
        );
    }

    #[test]
    fn nine_row_interlace_preserves_every_pixel_and_frame_extent() {
        let palette = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mut raw = Vec::new();
        for y in [0, 8, 4, 2, 6, 1, 3, 5, 7] {
            for x in 0..3 {
                raw.push(((y * 3 + x) % 4) as u8);
            }
        }
        let mut encoded = Vec::new();
        {
            let mut writer = gif::Encoder::new(&mut encoded, 3, 9, &palette).unwrap();
            writer
                .write_frame(&gif::Frame {
                    width: 3,
                    height: 9,
                    interlaced: true,
                    buffer: Cow::Borrowed(&raw),
                    ..Default::default()
                })
                .unwrap();
        }
        let result = decode(&encoded, 32 * 1024 * 1024, || Ok(())).unwrap();
        let expected: Vec<u8> = (0..27)
            .flat_map(|p| palette[(p % 4) * 3..(p % 4) * 3 + 3].iter().copied())
            .collect();
        assert_eq!(result.rgb, expected);
    }
}
