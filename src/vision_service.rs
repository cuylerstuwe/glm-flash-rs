//! Rank-zero vision encoder. The tower and JPEG/WebP codecs require a pinned
//! native runtime. This binary runs in that environment as a sidecar
//! (`vision-serve`). The model owner sends each encoded image over a Unix socket
//! and keeps the prompt, virtual IDs and embedding rows.
use crate::{
    vision_codecs::Codecs,
    vision_decode::{self, Decoded, MAX_ENCODED_BYTES},
    vision_gif,
    vision_input::{ImageKey, MAX_ROWS},
    vision_native::{Kind, Native},
    vision_preprocess::{Plan, Prepared, Settings},
    vision_tower::Tower,
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::{Path, PathBuf},
    time::Duration,
};

const SCRATCH: usize = 384 << 20;
pub(crate) const ROW_BYTES: usize = 4096 * 2;

pub(crate) struct Encoded {
    pub key: ImageKey,
    pub tokens: usize,
    pub rows: Vec<u8>,
}
/// A request's image rows on rank 0: sorted unique virtual IDs and their BF16
/// rows in the same order.
pub(crate) struct Table {
    pub keys: Vec<i32>,
    pub rows: Vec<u8>,
    _lease: Option<crate::vision_request::Lease>,
}
impl Table {
    pub(crate) fn new(mut pairs: Vec<(i32, &[u8])>) -> Result<Self> {
        pairs.sort_unstable_by_key(|p| p.0);
        pairs.dedup_by_key(|p| p.0);
        ensure!(
            pairs.iter().all(|p| p.1.len() == ROW_BYTES),
            "vision table row width"
        );
        let mut rows = Vec::with_capacity(pairs.len() * ROW_BYTES);
        for (_, row) in &pairs {
            rows.extend_from_slice(row);
        }
        Ok(Self {
            keys: pairs.iter().map(|p| p.0).collect(),
            rows,
            _lease: None,
        })
    }
    pub(crate) fn retain(mut self, lease: crate::vision_request::Lease) -> Self {
        self._lease = Some(lease);
        self
    }
    /// Rows for `keys` (sorted, unique), each of which this table must hold.
    pub(crate) fn select(&self, keys: &[i32]) -> Result<Vec<u8>> {
        let mut rows = Vec::with_capacity(keys.len() * ROW_BYTES);
        for key in keys {
            let i = self
                .keys
                .binary_search(key)
                .map_err(|_| anyhow::anyhow!("image row missing from vision table"))?;
            rows.extend_from_slice(&self.rows[i * ROW_BYTES..(i + 1) * ROW_BYTES]);
        }
        Ok(rows)
    }
}
/// The image itself was unusable (HTTP 400), as opposed to an encoder failure.
#[derive(Debug)]
pub(crate) struct ImageRejected(pub String);
impl std::fmt::Display for ImageRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ImageRejected {}
/// Connection/setup failed before a request byte could reach the encoder.
/// Unlike a missing reply, this proves no encoder GPU work was admitted.
#[derive(Debug)]
pub(crate) struct EncoderNotStarted;
impl std::fmt::Display for EncoderNotStarted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("vision encoder unavailable before admission")
    }
}
impl std::error::Error for EncoderNotStarted {}
#[derive(Debug)]
pub(crate) struct ImageCapacity;
impl std::fmt::Display for ImageCapacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("image preparation capacity unavailable")
    }
}
impl std::error::Error for ImageCapacity {}

fn send(stream: &mut UnixStream, status: u8, header: &Value, body: &[u8]) -> Result<()> {
    let header = serde_json::to_vec(header)?;
    stream.write_all(&[status])?;
    stream.write_all(&u32::try_from(header.len())?.to_le_bytes())?;
    stream.write_all(&header)?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}
fn receive(stream: &mut UnixStream, limit: usize) -> Result<(u8, Value, Vec<u8>)> {
    let mut status = [0u8; 1];
    stream.read_exact(&mut status)?;
    let mut length = [0u8; 4];
    stream.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    ensure!(length <= 4096, "vision frame header bound");
    let mut header = vec![0u8; length];
    stream.read_exact(&mut header)?;
    let header: Value = serde_json::from_slice(&header)?;
    let bytes = header["bytes"].as_u64().unwrap_or(0) as usize;
    ensure!(bytes <= limit, "vision frame body bound");
    let mut body = vec![0u8; bytes];
    stream.read_exact(&mut body)?;
    Ok((status[0], header, body))
}
fn json_file(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(
        &std::fs::read(path).with_context(|| path.display().to_string())?,
    )?)
}
fn decode(bytes: &[u8], codecs: &Codecs) -> Result<Decoded> {
    if bytes.starts_with(b"\x89PNG") {
        vision_decode::png(bytes, SCRATCH, || Ok(()))
    } else if bytes.starts_with(b"GIF") {
        vision_gif::decode(bytes, SCRATCH, || Ok(()))
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        codecs.jpeg(bytes, SCRATCH, || Ok(()))
    } else {
        codecs.webp(bytes, SCRATCH, || Ok(()))
    }
}

/// CONFIG: {"model_dir", "library", "library_sha256", "codecs"} (the codec
/// manifest). Encodes one image per connection; a tower failure ends the process.
pub(crate) fn serve(socket: &Path, config: &Path) -> Result<()> {
    let c = json_file(config)?;
    let text = |key: &str| {
        c[key]
            .as_str()
            .with_context(|| format!("vision config {key}"))
    };
    let model = PathBuf::from(text("model_dir")?);
    let settings = Settings::canonical(
        &json_file(&model.join("config.json"))?,
        &json_file(&model.join("processor_config.json"))?,
    )?;
    let codecs = Codecs::load(&json_file(Path::new(text("codecs")?))?)?;
    let native = Native::open(Path::new(text("library")?), text("library_sha256")?)?;
    let tower = Tower::load(native, &model.join("vision.safetensors"))?;
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)?;
    eprintln!("vision encoder ready");
    for stream in listener.incoming() {
        let mut stream = stream?;
        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
        let prepared = (|| -> Result<(Option<Prepared>, usize)> {
            let (_, header, bytes) = receive(&mut stream, MAX_ENCODED_BYTES)?;
            let max_tokens = header["max_tokens"].as_u64().context("max_tokens")? as usize;
            let decoded = decode(&bytes, &codecs)?;
            let plan = Plan::new(decoded.height, decoded.width, max_tokens)?;
            let pixels = if header["operation"] == "count" {
                None
            } else {
                Some(Prepared::from_rgb(plan, decoded.rgb, &settings)?)
            };
            Ok((pixels, plan.tokens()))
        })();
        let (prepared, tokens) = match prepared {
            Ok(v) => v,
            Err(error) => {
                let _ = send(&mut stream, 1, &json!({"error": error.to_string()}), &[]);
                continue;
            }
        };
        let Some(prepared) = prepared else {
            let _ = send(&mut stream, 0, &json!({"tokens":tokens,"bytes":0}), &[]);
            continue;
        };
        let rows = (|| -> Result<Vec<u8>> {
            let rows = tower.encode(&prepared, &settings, &mut |_, _| Ok(()))?;
            let (kind, shape, _) = rows.info()?;
            ensure!(
                matches!(kind, Kind::Bf16) && shape == [tokens as i64, 4096],
                "vision rows shape"
            );
            rows.bytes()
        })();
        match rows {
            Ok(rows) => {
                let key: String = prepared
                    .key()
                    .0
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                let header = json!({"key": key, "tokens": tokens, "bytes": rows.len()});
                if let Err(error) = send(&mut stream, 0, &header, &rows) {
                    eprintln!("vision reply failed: {error}");
                }
            }
            Err(error) => {
                let _ = send(&mut stream, 2, &json!({"error": error.to_string()}), &[]);
                return Err(error.context("vision tower failed"));
            }
        }
    }
    Ok(())
}

pub(crate) struct Client {
    socket: PathBuf,
}
impl Client {
    pub(crate) fn new(socket: &Path) -> Self {
        Self {
            socket: socket.to_owned(),
        }
    }
    fn connect(&self) -> Result<UnixStream> {
        let stream = UnixStream::connect(&self.socket).map_err(|_| EncoderNotStarted)?;
        stream
            .set_read_timeout(Some(Duration::from_secs(120)))
            .map_err(|_| EncoderNotStarted)?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(|_| EncoderNotStarted)?;
        Ok(stream)
    }
    pub(crate) fn count(&self, bytes: &[u8], max_tokens: usize) -> Result<usize> {
        let mut stream = self.connect()?;
        send(
            &mut stream,
            0,
            &json!({"operation":"count","max_tokens":max_tokens,"bytes":bytes.len()}),
            bytes,
        )?;
        let (status, header, _) = receive(&mut stream, 0)?;
        if status == 1 {
            return Err(
                ImageRejected(header["error"].as_str().unwrap_or("invalid image").into()).into(),
            );
        }
        ensure!(status == 0, "vision encoder count failed");
        let count = header["tokens"].as_u64().context("vision count")? as usize;
        ensure!((1..=MAX_ROWS).contains(&count), "vision count bound");
        Ok(count)
    }
    pub(crate) fn encode(&self, bytes: &[u8], max_tokens: usize) -> Result<Encoded> {
        let mut stream = self.connect()?;
        send(
            &mut stream,
            0,
            &json!({"max_tokens": max_tokens, "bytes": bytes.len()}),
            bytes,
        )?;
        let (status, header, rows) = receive(&mut stream, MAX_ROWS * ROW_BYTES)?;
        let error = || {
            header["error"]
                .as_str()
                .unwrap_or("vision encoder error")
                .to_owned()
        };
        match status {
            0 => {}
            1 => return Err(ImageRejected(error()).into()),
            _ => bail!("vision encoder failed: {}", error()),
        }
        let tokens = header["tokens"].as_u64().context("vision tokens")? as usize;
        let hex = header["key"].as_str().context("vision key")?;
        ensure!(
            (1..=MAX_ROWS).contains(&tokens) && rows.len() == tokens * ROW_BYTES && hex.len() == 64,
            "vision encoder reply"
        );
        let mut key = [0u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
        }
        Ok(Encoded {
            key: ImageKey(key),
            tokens,
            rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_table_retains_admission_through_the_last_reader() {
        use std::sync::Arc;
        let budget = crate::vision_request::Budget::new(1 << 20).unwrap();
        let lease = budget.reserve(1, 65536).unwrap();
        let row = vec![0u8; ROW_BYTES];
        let table = Arc::new(
            Table::new(vec![(1, &row), (1, &row)])
                .unwrap()
                .retain(lease),
        );
        assert_eq!(table.rows.len(), ROW_BYTES);
        let other = table.clone();
        drop(table);
        assert_eq!(budget.used().unwrap(), 65536);
        drop(other);
        assert_eq!(budget.used().unwrap(), 0);
    }
    #[test]
    fn count_protocol_and_proven_not_started_errors() {
        let path = std::env::temp_dir().join(format!(
            "alaska-count-{}-{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let client = Client::new(&path);
        assert!(
            client
                .count(b"a", 16)
                .unwrap_err()
                .is::<EncoderNotStarted>()
        );
        let listener = UnixListener::bind(&path).unwrap();
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let (_, header, bytes) = receive(&mut stream, 16).unwrap();
            assert_eq!(header["operation"], "count");
            assert_eq!(header["max_tokens"], 16);
            assert_eq!(bytes, b"a");
            send(&mut stream, 0, &json!({"tokens":16,"bytes":0}), &[]).unwrap();
            // A peer disappearing after admission is not proof that it did no work.
            let (mut stream, _) = listener.accept().unwrap();
            receive(&mut stream, 16).unwrap();
        });
        assert_eq!(client.count(b"a", 16).unwrap(), 16);
        assert!(
            !client
                .count(b"a", 16)
                .unwrap_err()
                .is::<EncoderNotStarted>()
        );
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
