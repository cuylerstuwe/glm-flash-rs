//! Frozen BF16 tower mapping and bounded byte-exact loading. No CUDA or Python.
//! A sink is provisional until every tensor hash and final file identity pass.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, Metadata},
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::SystemTime,
};

const PREFIX: &str = "model.visual.";
const PAYLOAD: u64 = 1_127_254_016;
const HEADER_LIMIT: u64 = 1024 * 1024;
const CHUNK_LIMIT: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    PatchWeight,
    PatchBias,
    Block { layer: usize, field: &'static str },
    PostNorm,
    DownWeight,
    DownBias,
    MergeProjection,
    MergeNormWeight,
    MergeNormBias,
    MergeGate,
    MergeUp,
    MergeDown,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Spec {
    pub name: String,
    pub shape: Vec<u64>,
    pub role: Role,
}
impl Spec {
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<u64>() * 2
    }
}
pub(crate) fn canonical_specs() -> BTreeMap<String, Spec> {
    let mut specs = BTreeMap::new();
    let mut add = |name: String, shape: &[u64], role| {
        let name = format!("{PREFIX}{name}");
        specs.insert(
            name.clone(),
            Spec {
                name,
                shape: shape.to_vec(),
                role,
            },
        );
    };
    add(
        "patch_embed.proj.weight".into(),
        &[1024, 3, 2, 14, 14],
        Role::PatchWeight,
    );
    add("patch_embed.proj.bias".into(), &[1024], Role::PatchBias);
    for layer in 0..24 {
        for (field, shape) in [
            ("norm1.weight", &[1024][..]),
            ("norm2.weight", &[1024][..]),
            ("attn.qkv.weight", &[3072, 1024][..]),
            ("attn.qkv.bias", &[3072][..]),
            ("attn.proj.weight", &[1024, 1024][..]),
            ("attn.proj.bias", &[1024][..]),
            ("attn.q_norm.weight", &[64][..]),
            ("attn.k_norm.weight", &[64][..]),
            ("mlp.gate_proj.weight", &[4096, 1024][..]),
            ("mlp.gate_proj.bias", &[4096][..]),
            ("mlp.up_proj.weight", &[4096, 1024][..]),
            ("mlp.up_proj.bias", &[4096][..]),
            ("mlp.down_proj.weight", &[1024, 4096][..]),
            ("mlp.down_proj.bias", &[1024][..]),
        ] {
            add(
                format!("blocks.{layer}.{field}"),
                shape,
                Role::Block { layer, field },
            );
        }
    }
    for (name, shape, role) in [
        ("post_layernorm.weight", &[1024][..], Role::PostNorm),
        (
            "downsample.weight",
            &[4096, 1024, 2, 2][..],
            Role::DownWeight,
        ),
        ("downsample.bias", &[4096][..], Role::DownBias),
        (
            "merger.proj.weight",
            &[4096, 4096][..],
            Role::MergeProjection,
        ),
        (
            "merger.post_projection_norm.weight",
            &[4096][..],
            Role::MergeNormWeight,
        ),
        (
            "merger.post_projection_norm.bias",
            &[4096][..],
            Role::MergeNormBias,
        ),
        (
            "merger.gate_proj.weight",
            &[10240, 4096][..],
            Role::MergeGate,
        ),
        ("merger.up_proj.weight", &[10240, 4096][..], Role::MergeUp),
        (
            "merger.down_proj.weight",
            &[4096, 10240][..],
            Role::MergeDown,
        ),
    ] {
        add(name.into(), shape, role);
    }
    specs
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}
#[derive(Default)]
struct Header(BTreeMap<String, RawTensor>);
impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Header;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("unique safetensor entries")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Header, M::Error> {
                let mut result = Header::default();
                let mut metadata = false;
                while let Some(name) = map.next_key::<String>()? {
                    if name == "__metadata__" {
                        if metadata {
                            return Err(serde::de::Error::custom("duplicate metadata"));
                        }
                        let _: BTreeMap<String, String> = map.next_value()?;
                        metadata = true;
                    } else {
                        if result.0.contains_key(&name) {
                            return Err(serde::de::Error::custom("duplicate tensor"));
                        }
                        result.0.insert(name, map.next_value()?);
                    }
                }
                Ok(result)
            }
        }
        d.deserialize_map(Visitor)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: u32,
    model: String,
    tensor_dtype: String,
    header_sha256: String,
    file_bytes: u64,
    payload_bytes: u64,
    source_body_evidence_sha256: String,
    tensors: BTreeMap<String, String>,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_hash(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn manifest() -> Result<Manifest> {
    let m: Manifest = serde_json::from_str(include_str!("../native/vision-manifest.json"))?;
    ensure!(
        m.format == 1
            && m.model == "Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdbf168"
            && m.tensor_dtype == "BF16"
            && m.payload_bytes == PAYLOAD,
        "vision manifest identity"
    );
    ensure!(
        valid_hash(&m.header_sha256)
            && valid_hash(&m.source_body_evidence_sha256)
            && m.tensors.values().all(|h| valid_hash(h)),
        "vision manifest hash"
    );
    ensure!(
        m.tensors.keys().eq(canonical_specs().keys()),
        "vision manifest tensor set"
    );
    Ok(m)
}
struct Mapped {
    spec: Spec,
    offset: u64,
    sha256: String,
}
fn layout(
    header: &[u8],
    file_bytes: u64,
    specs: &BTreeMap<String, Spec>,
    hashes: &BTreeMap<String, String>,
) -> Result<Vec<Mapped>> {
    ensure!(header.len() as u64 <= HEADER_LIMIT, "vision header bound");
    let parsed: Header = serde_json::from_slice(header)?;
    ensure!(
        parsed.0.keys().eq(specs.keys()) && hashes.keys().eq(specs.keys()),
        "vision tensor set"
    );
    let start = 8 + header.len() as u64;
    let mut mapped = Vec::with_capacity(specs.len());
    for (name, raw) in parsed.0 {
        let spec = &specs[&name];
        ensure!(
            raw.dtype == "BF16" && raw.shape == spec.shape,
            "vision tensor dtype/shape: {name}"
        );
        ensure!(
            raw.data_offsets[0].is_multiple_of(2)
                && raw.data_offsets[1].checked_sub(raw.data_offsets[0]) == Some(spec.bytes()),
            "vision tensor range: {name}"
        );
        let end = start
            .checked_add(raw.data_offsets[1])
            .context("vision range overflow")?;
        ensure!(
            end <= file_bytes && valid_hash(&hashes[&name]),
            "vision tensor extent/hash: {name}"
        );
        mapped.push(Mapped {
            spec: spec.clone(),
            offset: start + raw.data_offsets[0],
            sha256: hashes[&name].clone(),
        });
    }
    mapped.sort_by_key(|t| t.offset);
    let mut end = start;
    for tensor in &mapped {
        ensure!(tensor.offset == end, "vision overlapping/gapped payload");
        end += tensor.spec.bytes();
    }
    ensure!(end == file_bytes, "vision trailing payload");
    Ok(mapped)
}
#[derive(PartialEq, Eq)]
struct FileState {
    bytes: u64,
    modified: SystemTime,
}
impl FileState {
    fn read(m: &Metadata) -> Result<Self> {
        Ok(Self {
            bytes: m.len(),
            modified: m.modified()?,
        })
    }
}
pub(crate) struct Weights {
    file: File,
    initial: FileState,
    mapped: Vec<Mapped>,
    identity: String,
}
#[derive(Debug)]
pub(crate) struct Verified {
    pub identity: String,
    pub payload_bytes: u64,
    pub tensors: usize,
    pub staging_bytes: usize,
}
impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let m = manifest()?;
        let mut file = File::open(path)?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file() && metadata.len() == m.file_bytes,
            "vision file identity/size"
        );
        let initial = FileState::read(&metadata)?;
        let mut size = [0; 8];
        file.read_exact(&mut size)?;
        let size = u64::from_le_bytes(size);
        ensure!(size > 0 && size <= HEADER_LIMIT, "vision header size");
        let mut header = vec![0; size as usize];
        file.read_exact(&mut header)?;
        ensure!(hash(&header) == m.header_sha256, "vision header identity");
        let mapped = layout(&header, initial.bytes, &canonical_specs(), &m.tensors)?;
        Ok(Self {
            file,
            initial,
            mapped,
            identity: hash(include_bytes!("../native/vision-manifest.json")),
        })
    }
    /// Bounded raw BF16 chunks, in file order. A sink may enqueue native copies,
    /// but must discard all provisional allocations if any error is returned.
    /// It must copy the borrowed bytes before returning; no conversion is done.
    pub fn load(
        &mut self,
        chunk_bytes: usize,
        mut sink: impl FnMut(&Spec, u64, &[u8]) -> Result<()>,
    ) -> Result<Verified> {
        ensure!(
            chunk_bytes > 0 && chunk_bytes <= CHUNK_LIMIT && chunk_bytes.is_multiple_of(2),
            "vision staging bound/alignment"
        );
        ensure!(
            FileState::read(&self.file.metadata()?)? == self.initial,
            "vision file changed before loading"
        );
        let mut staging = vec![0; chunk_bytes];
        let mut total = 0;
        for tensor in &self.mapped {
            self.file.seek(SeekFrom::Start(tensor.offset))?;
            let mut digest = Sha256::new();
            let mut offset = 0;
            while offset < tensor.spec.bytes() {
                let n = (tensor.spec.bytes() - offset).min(chunk_bytes as u64) as usize;
                let bytes = &mut staging[..n];
                self.file.read_exact(bytes)?;
                digest.update(&*bytes);
                sink(&tensor.spec, offset, bytes)?;
                offset += n as u64;
            }
            ensure!(
                format!("{:x}", digest.finalize()) == tensor.sha256,
                "vision tensor payload mismatch: {}",
                tensor.spec.name
            );
            total += offset;
        }
        ensure!(
            FileState::read(&self.file.metadata()?)? == self.initial,
            "vision file changed while loading"
        );
        Ok(Verified {
            identity: self.identity.clone(),
            payload_bytes: total,
            tensors: self.mapped.len(),
            staging_bytes: staging.len(),
        })
    }
}

#[cfg(test)]
#[path = "vision_weights_tests.rs"]
mod tests;
