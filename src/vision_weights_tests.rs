use super::*;
use std::{
    fs,
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
};

fn header(specs: &BTreeMap<String, Spec>) -> Vec<u8> {
    let mut offset = 0;
    let value = specs.iter().map(|(name, spec)| {
        let start = offset; offset += spec.bytes();
        (name.clone(), serde_json::json!({"dtype":"BF16", "shape":spec.shape, "data_offsets":[start,offset]}))
    }).collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&value).unwrap()
}
fn fixture_specs() -> BTreeMap<String, Spec> {
    [("a", 3, Role::PatchBias), ("b", 5, Role::PostNorm)]
        .into_iter()
        .map(|(name, rows, role)| {
            (
                name.into(),
                Spec {
                    name: name.into(),
                    shape: vec![rows],
                    role,
                },
            )
        })
        .collect()
}
fn hashes(specs: &BTreeMap<String, Spec>) -> BTreeMap<String, String> {
    specs.keys().map(|k| (k.clone(), "a".repeat(64))).collect()
}

#[test]
fn complete_tower_map_and_manifest_preserve_canonical_bf16_shapes() -> Result<()> {
    let specs = canonical_specs();
    let pins = manifest()?;
    assert_eq!(specs.len(), 347);
    assert_eq!(specs.values().map(Spec::bytes).sum::<u64>(), PAYLOAD);
    assert_eq!(
        specs
            .values()
            .filter(|s| matches!(s.role, Role::Block { .. }))
            .count(),
        24 * 14
    );
    assert_eq!(
        specs["model.visual.patch_embed.proj.weight"].bytes(),
        1024 * 1176 * 2
    );
    assert_eq!(
        specs["model.visual.downsample.weight"].bytes(),
        4096 * 4096 * 2
    );
    let bytes = header(&specs);
    let mapped = layout(
        &bytes,
        8 + bytes.len() as u64 + PAYLOAD,
        &specs,
        &pins.tensors,
    )?;
    assert_eq!(mapped.len(), 347);
    assert_eq!(
        mapped.last().unwrap().offset + mapped.last().unwrap().spec.bytes(),
        8 + bytes.len() as u64 + PAYLOAD
    );
    Ok(())
}
#[test]
fn header_rejects_duplicate_shape_dtype_overflow_overlap_gap_and_trailing_bytes() -> Result<()> {
    let specs = fixture_specs();
    let pins = hashes(&specs);
    let valid = header(&specs);
    let file_bytes = 8 + valid.len() as u64 + 16;
    layout(&valid, file_bytes, &specs, &pins)?;
    assert!(layout(&valid, file_bytes - 1, &specs, &pins).is_err());
    assert!(layout(&valid, file_bytes + 1, &specs, &pins).is_err());
    for (key, field, value) in [
        ("a", "dtype", serde_json::json!("F16")),
        ("a", "shape", serde_json::json!([6])),
        ("b", "data_offsets", serde_json::json!([0, 10])),
        ("b", "data_offsets", serde_json::json!([8, 18])),
        (
            "b",
            "data_offsets",
            serde_json::json!([u64::MAX - 9, u64::MAX]),
        ),
    ] {
        let mut raw: serde_json::Value = serde_json::from_slice(&valid)?;
        raw[key][field] = value;
        let bytes = serde_json::to_vec(&raw)?;
        assert!(layout(&bytes, 8 + bytes.len() as u64 + 16, &specs, &pins).is_err());
    }
    let duplicate = br#"{"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]},"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#;
    assert!(layout(duplicate, 1000, &specs, &pins).is_err());
    let duplicate_field =
        br#"{"a":{"dtype":"BF16","dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#;
    assert!(layout(duplicate_field, 1000, &specs, &pins).is_err());
    Ok(())
}
struct Temp(std::path::PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
#[test]
fn public_open_checks_real_pinned_header_before_accepting_the_tower() -> Result<()> {
    let path = std::env::temp_dir().join(format!("alaska-vision-header-{}", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)?;
    let temp = Temp(path.clone());
    assert!(Weights::open(&path).is_err(), "short file rejected");
    let pins = manifest()?;
    let header = include_bytes!("../fixtures/vision/standalone-header.json");
    assert_eq!(hash(header), pins.header_sha256);
    // A sparse test file exercises the actual 1.1 GB layout without allocating
    // or reading tensor bodies. Only the 38,920-byte public header is copied.
    file.set_len(pins.file_bytes)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(header)?;
    file.sync_all()?;
    let tower = Weights::open(&path)?;
    assert_eq!(tower.mapped.len(), 347);
    assert_eq!(
        tower.mapped.iter().map(|t| t.spec.bytes()).sum::<u64>(),
        PAYLOAD
    );
    file.seek(SeekFrom::Start(8))?;
    file.write_all(b"[")?;
    file.sync_all()?;
    assert!(Weights::open(&path).is_err(), "changed header rejected");
    drop(tower);
    drop(file);
    drop(temp);
    Ok(())
}
fn file_fixture(payload: &[u8], expected: &[u8]) -> Result<(Temp, Weights)> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "alaska-vision-weight-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)?;
    file.write_all(payload)?;
    file.sync_all()?;
    drop(file);
    let temp = Temp(path.clone());
    let file = File::open(path)?;
    let mapped = fixture_specs()
        .into_values()
        .enumerate()
        .map(|(i, spec)| {
            let offset = if i == 0 { 0 } else { 6 };
            let sha256 = hash(&expected[offset..offset + spec.bytes() as usize]);
            Mapped {
                spec,
                offset: offset as u64,
                sha256,
            }
        })
        .collect();
    let initial = FileState::read(&file.metadata()?)?;
    Ok((
        temp,
        Weights {
            file,
            initial,
            mapped,
            identity: "a".repeat(64),
        },
    ))
}
#[test]
fn bounded_stream_preserves_bytes_offsets_roles_and_returns_identity_only_when_verified()
-> Result<()> {
    let bytes = (0..16).collect::<Vec<u8>>();
    let (_temp, mut weights) = file_fixture(&bytes, &bytes)?;
    let mut output = Vec::new();
    let mut offsets = Vec::new();
    let verified = weights.load(4, |spec, offset, chunk| {
        assert!(chunk.len() <= 4);
        offsets.push((spec.name.clone(), offset));
        output.extend_from_slice(chunk);
        Ok(())
    })?;
    assert_eq!(output, bytes);
    assert_eq!(
        offsets,
        [
            ("a".into(), 0),
            ("a".into(), 4),
            ("b".into(), 0),
            ("b".into(), 4),
            ("b".into(), 8)
        ]
    );
    assert_eq!(verified.identity, "a".repeat(64));
    assert_eq!(verified.payload_bytes, 16);
    assert_eq!(verified.tensors, 2);
    assert_eq!(verified.staging_bytes, 4);
    for size in [0, 1, CHUNK_LIMIT + 2] {
        assert!(weights.load(size, |_, _, _| unreachable!()).is_err());
    }
    Ok(())
}
#[test]
fn corrupted_payload_and_sink_errors_cannot_return_verified_identity() -> Result<()> {
    let expected = (0..16).collect::<Vec<u8>>();
    let mut corrupt = expected.clone();
    corrupt[14] ^= 1;
    let (_temp, mut weights) = file_fixture(&corrupt, &expected)?;
    assert!(
        weights
            .load(4, |_, _, _| Ok(()))
            .unwrap_err()
            .to_string()
            .contains("payload mismatch")
    );
    let (_temp2, mut weights) = file_fixture(&expected, &expected)?;
    let mut calls = 0;
    assert!(
        weights
            .load(4, |_, _, _| {
                calls += 1;
                anyhow::bail!("sink failed")
            })
            .is_err()
    );
    assert_eq!(calls, 1);
    Ok(())
}
