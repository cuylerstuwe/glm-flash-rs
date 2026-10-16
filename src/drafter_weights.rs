//! Strict loader for prequantized DFlash2 prepared shards.
//! NumPy-tagged CPU arrays are raw typed storage; no interpreter is involved.
use crate::prepared::{LoadedStorage, Manifest, Prepared, Tensor};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

const MODEL: &str = "incoai--GLM-5.3-Flash-DFlash2-dc77ff1c99ee";
const Q4: &str = "tensorfold.families.glm5_next.cuda.qmm:Q4";
const LAYER: &str = "tensorfold.families.glm5_next.cuda.dflash2:DraftLayer";

fn tensor(node: &Value, dtype: &str, shape: &[u64], cpu: bool) -> Result<Tensor> {
    let t = if cpu {
        let o = node.as_object().context("drafter CPU tensor")?;
        ensure!(o.len() == 3 && o.contains_key("N"), "drafter CPU array tag");
        let storage = usize::try_from(node["N"].as_u64().context("CPU storage index")?)?;
        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        Tensor {
            storage,
            dtype: node["dtype"].as_str().context("CPU dtype")?.into(),
            shape: serde_json::from_value(node["shape"].clone())?,
            stride,
            off: 0,
            dev: "cpu".into(),
        }
    } else {
        serde_json::from_value(node.clone()).context("drafter CUDA tensor")?
    };
    ensure!(
        t.dtype == dtype
            && t.shape == shape
            && t.off == 0
            && t.dev == if cpu { "cpu" } else { "cuda" },
        "drafter tensor geometry/type"
    );
    let mut stride = 1;
    ensure!(t.stride.len() == shape.len(), "drafter tensor stride rank");
    for (&n, &s) in shape.iter().zip(&t.stride).rev() {
        ensure!(s == stride, "drafter tensor must be contiguous");
        stride *= n;
    }
    Ok(t)
}
fn put(
    out: &mut BTreeMap<String, Tensor>,
    node: &Value,
    path: &str,
    dtype: &str,
    shape: &[u64],
    cpu: bool,
) -> Result<()> {
    ensure!(
        out.insert(path.into(), tensor(node, dtype, shape, cpu)?)
            .is_none(),
        "duplicate drafter tensor"
    );
    Ok(())
}
fn q4(out: &mut BTreeMap<String, Tensor>, node: &Value, path: &str, n: u64, k: u64) -> Result<()> {
    ensure!(
        node["O"] == Q4 && node["f"].as_object().is_some_and(|o| o.len() == 5),
        "drafter Q4 type/fields"
    );
    let f = &node["f"];
    ensure!(
        f["n"]["P"].as_u64() == Some(n) && f["k"]["P"].as_u64() == Some(k),
        "drafter Q4 shape"
    );
    put(
        out,
        &f["weight"],
        &format!("{path}.weight"),
        "int32",
        &[n / 64, k / 64, 64, 8],
        false,
    )?;
    for name in ["scales", "biases"] {
        put(
            out,
            &f[name],
            &format!("{path}.{name}"),
            "bfloat16",
            &[k / 64, n],
            false,
        )?;
    }
    Ok(())
}
fn tree(value: &Value) -> Result<BTreeMap<String, Tensor>> {
    ensure!(
        value.as_object().is_some_and(|o| o.len() == 1),
        "drafter root fields"
    );
    let mut fields = BTreeMap::new();
    for pair in value["D"].as_array().context("drafter dictionary")? {
        let pair = pair.as_array().context("drafter dictionary pair")?;
        ensure!(pair.len() == 2, "drafter dictionary pair length");
        let name = pair[0].as_str().context("drafter dictionary key")?;
        ensure!(
            fields.insert(name, &pair[1]).is_none(),
            "duplicate drafter field"
        );
    }
    ensure!(
        fields.keys().copied().eq([
            "fc",
            "hidden_norm",
            "hproj",
            "layers",
            "norm",
            "pred",
            "succ"
        ]),
        "drafter dictionary fields"
    );
    let mut out = BTreeMap::new();
    q4(&mut out, fields["fc"], "drafter.fc", 4096, 20480)?;
    for name in ["hidden_norm", "norm"] {
        put(
            &mut out,
            fields[name],
            &format!("drafter.{name}"),
            "bfloat16",
            &[4096],
            false,
        )?;
    }
    put(
        &mut out,
        fields["hproj"],
        "drafter.hproj",
        "bfloat16",
        &[256, 4096],
        false,
    )?;
    for name in ["pred", "succ"] {
        put(
            &mut out,
            fields[name],
            &format!("drafter.{name}"),
            "float32",
            &[154880, 256],
            true,
        )?;
    }
    let layers = fields["layers"]["L"].as_array().context("drafter layers")?;
    ensure!(layers.len() == 5, "drafter layer count");
    for (i, layer) in layers.iter().enumerate() {
        ensure!(
            layer["O"] == LAYER && layer["f"].as_object().is_some_and(|o| o.len() == 13),
            "drafter layer type/fields"
        );
        let f = &layer["f"];
        let path = format!("drafter.layers[{i}]");
        for name in ["in_norm", "post_norm"] {
            put(
                &mut out,
                &f[name],
                &format!("{path}.{name}"),
                "bfloat16",
                &[4096],
                false,
            )?;
        }
        for name in ["a_base", "m_base"] {
            put(
                &mut out,
                &f[name],
                &format!("{path}.{name}"),
                "bfloat16",
                &[2, 2, 4096],
                false,
            )?;
        }
        for name in ["q_norm", "k_norm"] {
            put(
                &mut out,
                &f[name],
                &format!("{path}.{name}"),
                "bfloat16",
                &[128],
                false,
            )?;
        }
        for (name, n, k) in [
            ("a_kp", 1024, 4096),
            ("m_kp", 1024, 4096),
            ("qkv", 1536, 4096),
            ("kv", 512, 4096),
            ("o", 4096, 1024),
            ("gu", 6144, 4096),
            ("down", 4096, 3072),
        ] {
            q4(&mut out, &f[name], &format!("{path}.{name}"), n, k)?;
        }
    }
    Ok(out)
}
pub fn open(path: &Path, rank: u64) -> Result<Prepared> {
    let file = File::open(path)?;
    ensure!(file.metadata()?.len() <= 1 << 20, "drafter manifest bound");
    let manifest: Manifest = serde_json::from_reader(file.take((1 << 20) + 1))?;
    ensure!(
        rank < 4
            && manifest.key["rank"].as_u64() == Some(rank)
            && manifest.key["world"].as_u64() == Some(4),
        "drafter TP4 rank"
    );
    ensure!(
        manifest.key["what"] == "drafter"
            && manifest.key["model"] == MODEL
            && manifest.key["code"]
                == "7450d186e7778e8be8ace70308ffb4667ea4d29360324eead31602408c1cc12c",
        "unexpected prepared drafter identity"
    );
    ensure!(
        manifest.file_size <= 768 << 20 && manifest.storages.len() == 143,
        "drafter storage bounds"
    );
    let tensors = tree(&manifest.tree)?;
    ensure!(tensors.len() == 143, "drafter tensor count");
    let p = Prepared::open_parts(path, manifest, tensors)?;
    let mut used = std::collections::BTreeSet::new();
    for t in p.tensors.values() {
        ensure!(
            used.insert(t.storage) && t.extent()? == p.manifest.storages[t.storage].nbytes,
            "drafter storage alias or size mismatch"
        );
    }
    Ok(p)
}
pub fn probe(path: &Path, rank: u64) -> Result<Value> {
    let p = open(path, rank)?;
    let device = crate::cuda::Device::new()?;
    let loaded = p.load(&device)?;
    let mut records = Vec::new();
    let mut cpu_bytes = 0;
    for (name, t) in &p.tensors {
        let n = t.extent()? as usize;
        cpu_bytes += if t.dev == "cpu" { n } else { 0 };
        let mut hash = Sha256::new();
        for start in (0..n).step_by(8 << 20) {
            let len = (n - start).min(8 << 20);
            let source = p.read_tensor_segment(name, start as u64, len as u64)?;
            let gpu;
            let bytes = match &loaded.storages[t.storage] {
                LoadedStorage::Gpu(buffer) => {
                    gpu = buffer.read(start, len)?;
                    &gpu[..]
                }
                LoadedStorage::Cpu(bytes) => &bytes[start..start + len],
                LoadedStorage::Absent => anyhow::bail!("drafter tensor absent"),
            };
            ensure!(source == bytes, "drafter loaded bytes differ: {name}");
            if t.dtype == "float32" {
                ensure!(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|b| f32::from_le_bytes(*b).is_finite()),
                    "nonfinite drafter selector"
                );
            }
            hash.update(bytes);
        }
        records.push(
            json!({"name":name,"dtype":t.dtype,"shape":t.shape,"device":t.dev,
            "bytes":n,"sha256":format!("{:x}",hash.finalize())}),
        );
    }
    Ok(
        json!({"rank":rank,"model_identity":p.model_identity()?,"gpu_bytes":loaded.bytes,
        "cpu_bytes":cpu_bytes,"load_seconds":loaded.seconds,"exact_storage_readback":true,
        "python_executed":false,"records":records,"scope":"Frozen prepared drafter loading only; no drafting or speculative serving"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpu_arrays_are_exact_typed_contiguous_storage() {
        let good = json!({"N":6,"dtype":"float32","shape":[154880,256]});
        let t = tensor(&good, "float32", &[154880, 256], true).unwrap();
        assert_eq!(t.extent().unwrap(), 158597120);
        assert_eq!(t.stride, [256, 1]);
        for bad in [
            json!({"N":6,"dtype":"float64","shape":[154880,256]}),
            json!({"N":6,"dtype":"float32","shape":[256,154880]}),
            json!({"N":6,"dtype":"float32","shape":[154880,256],"stride":[1,154880]}),
            json!({"N":-1,"dtype":"float32","shape":[154880,256]}),
        ] {
            assert!(tensor(&bad, "float32", &[154880, 256], true).is_err());
        }
    }
    #[test]
    fn duplicate_unknown_and_incomplete_drafter_trees_reject() {
        for bad in [
            json!({"D":[["fc",{}],["fc",{}]]}),
            json!({"D":[["unknown",{}]]}),
            json!({"O":LAYER,"f":{}}),
        ] {
            assert!(tree(&bad).is_err());
        }
    }
}
