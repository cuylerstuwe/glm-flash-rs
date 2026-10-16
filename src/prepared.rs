use crate::cuda::{Buffer, Device, Pinned};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

#[derive(Deserialize, Serialize, Debug)]
pub struct Storage {
    pub offset: u64,
    pub nbytes: u64,
    pub dev: String,
}
#[derive(Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub file_size: u64,
    pub chunk: usize,
    pub chunk_sha256: Vec<String>,
    pub storages: Vec<Storage>,
    pub tree: Value,
    pub key: Value,
}
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Tensor {
    #[serde(rename = "T")]
    pub storage: usize,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    pub off: u64,
    pub dev: String,
}
impl Tensor {
    pub fn element_bytes(&self) -> Result<u64> {
        Ok(match self.dtype.as_str() {
            "bool" | "uint8" | "int8" | "float8_e4m3fn" | "float8_e5m2" => 1,
            "bfloat16" | "float16" | "int16" | "uint16" => 2,
            "float32" | "int32" | "uint32" => 4,
            "float64" | "int64" | "uint64" => 8,
            _ => bail!("unsupported dtype {}", self.dtype),
        })
    }
    pub fn extent(&self) -> Result<u64> {
        ensure!(
            self.shape.len() == self.stride.len(),
            "shape/stride rank differs"
        );
        if self.shape.contains(&0) {
            return self
                .off
                .checked_mul(self.element_bytes()?)
                .context("empty tensor overflow");
        }
        let mut last = self.off;
        for (&n, &s) in self.shape.iter().zip(&self.stride) {
            last = last
                .checked_add((n - 1).checked_mul(s).context("tensor stride overflow")?)
                .context("tensor extent overflow")?;
        }
        last.checked_add(1)
            .and_then(|v| v.checked_mul(self.element_bytes().ok()?))
            .context("tensor bytes overflow")
    }
}
pub struct Prepared {
    pub manifest: Manifest,
    pub tensors: BTreeMap<String, Tensor>,
    pub data: PathBuf,
}
/// Chat needs insertion order and arbitrary-precision parsing for explicit range
/// checks. Manifest identity uses sorted keys and finite-f64 JSON values.
pub(crate) fn canonical_metadata(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.sort_keys();
            for value in map.values_mut() {
                canonical_metadata(value);
            }
        }
        Value::Array(array) => {
            for value in array {
                canonical_metadata(value);
            }
        }
        Value::Number(number) if number.is_f64() => {
            if let Some(normalized) = number.as_f64().and_then(serde_json::Number::from_f64) {
                *number = normalized;
            }
        }
        _ => {}
    }
}
fn walk(node: &Value, path: &str, tensors: &mut BTreeMap<String, Tensor>) -> Result<()> {
    let o = node.as_object().context("prepared node is not an object")?;
    if o.contains_key("T") {
        tensors.insert(path.to_owned(), serde_json::from_value(node.clone())?);
    } else if let Some(f) = o.get("f") {
        ensure!(
            o.get("O").and_then(Value::as_str).is_some(),
            "untyped object"
        );
        for (k, v) in f.as_object().context("object fields")? {
            walk(v, &format!("{path}.{k}"), tensors)?;
        }
    } else if let Some(a) = o.get("L").or_else(|| o.get("U")) {
        for (i, v) in a.as_array().context("sequence")?.iter().enumerate() {
            walk(v, &format!("{path}[{i}]"), tensors)?;
        }
    } else if let Some(a) = o.get("D") {
        for pair in a.as_array().context("dict")? {
            let pair = pair.as_array().context("dict pair")?;
            ensure!(pair.len() == 2, "dict pair length");
            walk(&pair[1], &format!("{path}[{}]", pair[0]), tensors)?;
        }
    } else if o.contains_key("N") {
        bail!("NumPy storage is unsupported: {path}");
    } else {
        ensure!(
            o.len() == 1 && (o.contains_key("P") || o.contains_key("dev")),
            "unknown prepared node {path}"
        );
    }
    Ok(())
}
impl Prepared {
    pub fn model_identity(&self) -> Result<String> {
        let manifest = &self.manifest;
        let mut tree = manifest.tree.clone();
        let mut key = manifest.key.clone();
        canonical_metadata(&mut tree);
        canonical_metadata(&mut key);
        let identity = serde_json::to_vec(&(
            manifest.format,
            manifest.file_size,
            manifest.chunk,
            &manifest.chunk_sha256,
            &manifest.storages,
            &tree,
            &key,
        ))?;
        Ok(format!("{:x}", Sha256::digest(identity)))
    }
    pub fn open(path: &Path, rank: u64) -> Result<Self> {
        let manifest: Manifest = serde_json::from_reader(File::open(path)?)?;
        ensure!(manifest.format == 1, "unsupported prepared format");
        ensure!(
            manifest.key["rank"].as_u64() == Some(rank)
                && manifest.key["world"].as_u64() == Some(4),
            "expected TP4 rank {rank}"
        );
        ensure!(
            manifest.key["model"] == "Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdbf168",
            "unexpected model/revision"
        );
        ensure!(
            manifest.tree["O"] == "tensorfold.families.glm5_next.cuda.weights:Weights",
            "unexpected root object"
        );
        for (k, v) in [
            ("hidden", 4096),
            ("layers", 45),
            ("vocab", 154880),
            ("streams", 4),
            ("lin_heads", 64),
            ("lin_dim", 128),
        ] {
            ensure!(
                manifest.tree["f"]["cfg"]["f"][k]["P"].as_u64() == Some(v),
                "unsupported geometry: {k}"
            );
        }
        let mut tensors = BTreeMap::new();
        walk(&manifest.tree, "weights", &mut tensors)?;
        Self::open_parts(path, manifest, tensors)
    }
    /// Storage validation shared by strictly typed target/drafter tree parsers.
    /// Each caller must validate its own model, rank and geometry before entry.
    pub(crate) fn open_parts(
        path: &Path,
        manifest: Manifest,
        tensors: BTreeMap<String, Tensor>,
    ) -> Result<Self> {
        ensure!(manifest.format == 1, "unsupported prepared format");
        ensure!(
            manifest.chunk > 0 && manifest.chunk <= 128 * 1024 * 1024,
            "invalid chunk size"
        );
        ensure!(
            manifest.chunk_sha256.len() as u64
                == manifest.file_size.div_ceil(manifest.chunk as u64),
            "chunk hash count"
        );
        let data = path.with_file_name("data.bin");
        ensure!(
            data.metadata()?.len() == manifest.file_size,
            "data file size mismatch"
        );
        let mut prev = 0;
        for s in &manifest.storages {
            ensure!(
                s.nbytes > 0 && s.offset >= prev,
                "empty/overlapping/out-of-order storage"
            );
            prev = s.offset.checked_add(s.nbytes).context("storage overflow")?;
            ensure!(prev <= manifest.file_size, "storage outside data file");
            ensure!(s.dev == "cuda" || s.dev == "cpu", "unknown storage device");
        }
        for (name, t) in &tensors {
            let s = manifest
                .storages
                .get(t.storage)
                .context("missing storage")?;
            ensure!(t.extent()? <= s.nbytes, "tensor {name} escapes storage");
            ensure!(t.dev == s.dev, "tensor {name} changes device");
        }
        ensure!(!tensors.is_empty(), "no tensors");
        Ok(Self {
            manifest,
            tensors,
            data,
        })
    }
    pub fn tensor(&self, path: &str) -> Result<&Tensor> {
        self.tensors
            .get(path)
            .with_context(|| format!("missing tensor {path}"))
    }
    pub fn expect(&self, path: &str, dtype: &str, shape: &[u64]) -> Result<()> {
        let t = self.tensor(path)?;
        ensure!(
            t.dtype == dtype && t.shape == shape && t.dev == "cuda",
            "unsupported tensor {path}"
        );
        let mut stride = 1;
        for (&n, &s) in t.shape.iter().zip(&t.stride).rev() {
            ensure!(s == stride, "noncontiguous tensor {path}");
            stride *= n;
        }
        Ok(())
    }
    pub fn read_tensor(&self, path: &str) -> Result<Vec<u8>> {
        let t = self.tensor(path)?;
        self.read_tensor_segment(path, 0, t.extent()? - t.off * t.element_bytes()?)
    }
    pub fn read_tensor_segment(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        let t = self.tensor(path)?;
        let s = &self.manifest.storages[t.storage];
        let start = t.off * t.element_bytes()?;
        ensure!(
            offset <= t.extent()? - start && len <= t.extent()? - start - offset,
            "tensor read outside extent"
        );
        let mut f = File::open(&self.data)?;
        f.seek(SeekFrom::Start(s.offset + start + offset))?;
        let mut data = vec![0u8; len as usize];
        f.read_exact(&mut data)?;
        Ok(data)
    }
    pub fn load(&self, device: &Rc<Device>) -> Result<Loaded> {
        self.load_selected(device, &[])
    }
    pub fn load_selected(&self, device: &Rc<Device>, prefixes: &[&str]) -> Result<Loaded> {
        let mut selected = vec![prefixes.is_empty(); self.manifest.storages.len()];
        for (name, t) in &self.tensors {
            if prefixes.iter().any(|p| name.starts_with(p)) {
                selected[t.storage] = true;
            }
        }
        let bytes: u64 = self
            .manifest
            .storages
            .iter()
            .zip(&selected)
            .filter(|(s, b)| **b && s.dev == "cuda")
            .map(|(s, _)| s.nbytes)
            .sum();
        let (free, _) = device.allocatable_memory()?;
        ensure!(
            bytes + 8 * 1024 * 1024 * 1024 < free as u64,
            "not enough memory with 8 GiB reserve"
        );
        let started = Instant::now();
        let mut storages = Vec::with_capacity(self.manifest.storages.len());
        for (s, &selected) in self.manifest.storages.iter().zip(&selected) {
            storages.push(if !selected {
                LoadedStorage::Absent
            } else if s.dev == "cuda" {
                LoadedStorage::Gpu(device.alloc(s.nbytes as usize)?)
            } else {
                LoadedStorage::Cpu(vec![0u8; s.nbytes as usize])
            });
        }
        let mut staging = Pinned::new(device, self.manifest.chunk)?;
        let mut file = File::open(&self.data)?;
        let mut first = 0;
        let mut offset = 0u64;
        for (i, hash) in self.manifest.chunk_sha256.iter().enumerate() {
            let len = (self.manifest.file_size - offset).min(self.manifest.chunk as u64) as usize;
            let needed =
                self.manifest.storages.iter().zip(&selected).any(|(s, b)| {
                    *b && s.offset < offset + len as u64 && s.offset + s.nbytes > offset
                });
            if !needed {
                offset += len as u64;
                file.seek(SeekFrom::Start(offset))?;
                continue;
            }
            let data = &mut staging.bytes_mut()[..len];
            file.read_exact(data)?;
            ensure!(
                format!("{:x}", Sha256::digest(&*data)) == *hash,
                "weight hash mismatch in chunk {i}"
            );
            while first < storages.len()
                && self.manifest.storages[first].offset + self.manifest.storages[first].nbytes
                    <= offset
            {
                first += 1;
            }
            for (j, s) in self.manifest.storages.iter().enumerate().skip(first) {
                if s.offset >= offset + len as u64 {
                    break;
                }
                let lo = s.offset.max(offset);
                let hi = (s.offset + s.nbytes).min(offset + len as u64);
                let src = &data[(lo - offset) as usize..(hi - offset) as usize];
                let dst = (lo - s.offset) as usize;
                match &mut storages[j] {
                    LoadedStorage::Gpu(b) => b.write(dst, src)?,
                    LoadedStorage::Cpu(b) => b[dst..dst + src.len()].copy_from_slice(src),
                    LoadedStorage::Absent => {}
                }
            }
            offset += len as u64;
            if i % 64 == 63 {
                eprintln!(
                    "verified and loaded {:.1} GiB",
                    offset as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }
        }
        device.sync()?;
        Ok(Loaded {
            storages,
            bytes,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}
pub enum LoadedStorage {
    Gpu(Buffer),
    Cpu(Vec<u8>),
    Absent,
}
pub struct Loaded {
    pub storages: Vec<LoadedStorage>,
    pub bytes: u64,
    pub seconds: f64,
}
impl Loaded {
    pub fn pointer(&self, prepared: &Prepared, path: &str) -> Result<u64> {
        let t = prepared.tensor(path)?;
        match &self.storages[t.storage] {
            LoadedStorage::Gpu(b) => Ok(b.ptr + t.off * t.element_bytes()?),
            LoadedStorage::Cpu(_) => bail!("CPU tensor has no GPU pointer: {path}"),
            LoadedStorage::Absent => bail!("tensor was not loaded: {path}"),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t() -> Tensor {
        Tensor {
            storage: 0,
            dtype: "bfloat16".into(),
            shape: vec![2, 3],
            stride: vec![8, 1],
            off: 4,
            dev: "cuda".into(),
        }
    }
    #[test]
    fn strided_extent_includes_offset_and_holes() {
        assert_eq!(t().extent().unwrap(), 30);
    }
    #[test]
    fn reject_integer_overflow() {
        let mut t = t();
        t.stride[0] = u64::MAX;
        assert!(t.extent().is_err());
    }
    #[test]
    fn reject_rank_mismatch() {
        let mut t = t();
        t.stride.pop();
        assert!(t.extent().is_err());
    }
    #[test]
    fn empty_view_does_not_underflow() {
        let mut t = t();
        t.shape[0] = 0;
        assert_eq!(t.extent().unwrap(), 8);
    }
}
