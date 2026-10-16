use super::*;

fn encoded(
    color: ColorType,
    depth: BitDepth,
    width: u32,
    height: u32,
    raw: &[u8],
    palette: Option<&[u8]>,
    trns: Option<&[u8]>,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(color);
        encoder.set_depth(depth);
        if let Some(value) = palette {
            encoder.set_palette(value);
        }
        if let Some(value) = trns {
            encoder.set_trns(value);
        }
        encoder
            .write_header()
            .unwrap()
            .write_image_data(raw)
            .unwrap();
    }
    bytes
}
fn decode(bytes: &[u8]) -> Decoded {
    png(bytes, DEFAULT_WORKSPACE_BYTES, || Ok(())).unwrap()
}

#[test]
fn rgb_l_transparency_metadata_ignored_as_frozen_to_rgb() {
    let rgb = [1, 2, 3, 0, 0, 0];
    let bytes = encoded(
        ColorType::Rgb,
        BitDepth::Eight,
        2,
        1,
        &rgb,
        None,
        Some(&[0, 1, 0, 2, 0, 3]),
    );
    assert_eq!(decode(&bytes).rgb, rgb);
    let bytes = encoded(
        ColorType::Grayscale,
        BitDepth::Eight,
        3,
        1,
        &[0, 51, 255],
        None,
        Some(&[0, 51]),
    );
    assert_eq!(decode(&bytes).rgb, [0, 0, 0, 51, 51, 51, 255, 255, 255]);
}

#[test]
fn grayscale16_clips_but_color16_and_la16_select_high_bytes() {
    let bytes = encoded(
        ColorType::Grayscale,
        BitDepth::Sixteen,
        4,
        1,
        &[0, 0, 0, 128, 1, 0, 255, 255],
        None,
        None,
    );
    assert_eq!(
        decode(&bytes).rgb,
        [0, 0, 0, 128, 128, 128, 255, 255, 255, 255, 255, 255]
    );
    let bytes = encoded(
        ColorType::Rgb,
        BitDepth::Sixteen,
        1,
        1,
        &[7, 240, 9, 250, 11, 0],
        None,
        None,
    );
    assert_eq!(decode(&bytes).rgb, [7, 9, 11]);
    let bytes = encoded(
        ColorType::GrayscaleAlpha,
        BitDepth::Sixteen,
        2,
        1,
        &[0, 255, 128, 0, 42, 255, 255, 0],
        None,
        None,
    );
    assert_eq!(decode(&bytes).rgb, [127, 127, 127, 42, 42, 42]);
}

#[test]
fn rgba_and_transparent_palette_composite_white_including_packed_padding() {
    let bytes = encoded(
        ColorType::Rgba,
        BitDepth::Eight,
        3,
        1,
        &[0, 0, 0, 128, 9, 8, 7, 0, 42, 43, 44, 255],
        None,
        None,
    );
    assert_eq!(
        decode(&bytes).rgb,
        [127, 127, 127, 255, 255, 255, 42, 43, 44]
    );
    let bytes = encoded(
        ColorType::Indexed,
        BitDepth::Two,
        3,
        1,
        &[0b00011011],
        Some(&[0, 0, 0, 60, 90, 120, 42, 43, 44]),
        Some(&[0, 128]),
    );
    assert_eq!(
        decode(&bytes).rgb,
        [255, 255, 255, 157, 172, 187, 42, 43, 44]
    );
}

#[test]
fn packed_grayscale_expands_without_transparency_or_padding_pixels() {
    for (depth, pixels, raw, expected) in [
        (BitDepth::One, 3, 0b01011111, vec![0, 255, 0]),
        (BitDepth::Two, 3, 0b00011011, vec![0, 85, 170]),
        (BitDepth::Four, 2, 0b00011111, vec![17, 255]),
    ] {
        let bytes = encoded(
            ColorType::Grayscale,
            depth,
            pixels,
            1,
            &[raw],
            None,
            Some(&[0, 1]),
        );
        let rgb: Vec<u8> = expected.into_iter().flat_map(|v| [v; 3]).collect();
        assert_eq!(decode(&bytes).rgb, rgb);
    }
}

#[test]
fn white_composite_exhaustively_matches_frozen_pillow_integer_arithmetic() {
    for value in 0..=255u32 {
        for alpha in 0..=255u32 {
            let reference = if alpha == 0 {
                255
            } else {
                let blend = 255 * (255 - alpha);
                let out_a = alpha * 255 + blend;
                let coef1 = alpha * 255 * 255 * 128 / out_a;
                let coef2 = 255 * 128 - coef1;
                let temp = value * coef1 + 255 * coef2 + (128 << 7);
                (((temp >> 8) + temp) >> 8) >> 7
            };
            assert_eq!(white(value as u8, alpha as u8), reference as u8);
        }
    }
}

#[test]
fn first_animation_frame_and_separate_default_are_selected() {
    for separate in [false, true] {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(ColorType::Rgba);
            encoder.set_depth(BitDepth::Eight);
            encoder
                .set_animated(if separate { 1 } else { 2 }, 0)
                .unwrap();
            encoder.set_sep_def_img(separate).unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[1, 2, 3, 255]).unwrap();
            writer.write_image_data(&[9, 8, 7, 255]).unwrap();
            writer.finish().unwrap();
        }
        assert_eq!(decode(&bytes).rgb, [1, 2, 3]);
    }
}

#[test]
fn admission_corruption_truncation_and_cancellation_never_return_partial_image() {
    let bytes = encoded(ColorType::Rgb, BitDepth::Eight, 2, 2, &[0; 12], None, None);
    assert!(png(&bytes, 1, || Ok(())).is_err());
    let mut calls = 0;
    assert!(
        png(&bytes, DEFAULT_WORKSPACE_BYTES, || {
            calls += 1;
            ensure!(calls < 4, "cancelled");
            Ok(())
        })
        .is_err()
    );
    assert_eq!(calls, 4);
    assert!(png(&bytes[..40], DEFAULT_WORKSPACE_BYTES, || Ok(())).is_err());
    let mut corrupt = bytes.clone();
    corrupt[16] ^= 1;
    assert!(png(&corrupt, DEFAULT_WORKSPACE_BYTES, || Ok(())).is_err());
    let mut bytes = Vec::new();
    png::Encoder::new(&mut bytes, 65536, 65536)
        .write_header()
        .unwrap();
    assert!(
        png(&bytes, DEFAULT_WORKSPACE_BYTES, || Ok(()))
            .unwrap_err_message()
            .contains("pixel bound")
    );
}

trait ErrorMessage {
    fn unwrap_err_message(self) -> String;
}
impl ErrorMessage for Result<Decoded> {
    fn unwrap_err_message(self) -> String {
        match self {
            Err(error) => format!("{error:#}"),
            Ok(_) => panic!("unexpected decode success"),
        }
    }
}

#[test]
fn odd_sized_adam7_matches_plain_and_semantic_goldens_all_png_modes() {
    use sha2::{Digest, Sha256};
    // The decode harness crate and the service crate both compile these tests.
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = [
        crate_root.join("fixtures"),
        crate_root.join("native/vision-decode/fixtures"),
    ]
    .into_iter()
    .find(|p| p.join("manifest.json").exists())
    .unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 30);
    for case in cases {
        let encoded = std::fs::read(root.join(case["file"].as_str().unwrap())).unwrap();
        let expected = std::fs::read(root.join(case["rgb"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&encoded)),
            case["png_sha256"].as_str().unwrap()
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&expected)),
            case["rgb_sha256"].as_str().unwrap()
        );
        let result = decode(&encoded);
        assert_eq!((result.width, result.height), (29, 19));
        assert_eq!(result.rgb, expected, "{}", case["file"]);
    }
}
