//! Versioned, bounded disk objects for pure prefix snapshots. No GPU or Python.
//! Format changes are required when adding any snapshot field (e.g. MTP).
use crate::cache_pages::{CONTEXT, PAGE};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};

pub type Hash = [u8; 32];
pub const CHUNK: usize = 8 << 20;
pub const PAGE_BYTES: u64 = (11 * (PAGE * 528 + PAGE / 4 * 256)) as u64;
const MAGIC: &[u8; 8] = b"ALSSD\0\x02\0";
const FIXED: usize = 112;
pub const ARITHMETIC: &str = "alaska-rust-prefix-v1/fast-b12x4-bf16/kda64/bm32";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub namespace: Hash,
    pub rank: u32,
    pub ring_rows: usize,
    pub speculation: Option<Hash>,
}
impl Identity {
    /// Inputs must come from the verified shard manifests and kernel policy.
    /// Capacity and paths deliberately do not participate in byte compatibility.
    pub fn new(
        shards: [Hash; 4],
        kernel_policy: Hash,
        rank: u32,
        ring_rows: usize,
    ) -> Result<Self> {
        ensure!(
            rank < 4 && ring_rows.is_power_of_two() && (256..=16384).contains(&ring_rows),
            "SSD identity dimensions"
        );
        let mut h = Sha256::new();
        h.update(MAGIC);
        h.update(ARITHMETIC);
        h.update(b"pure;no-mtp;no-sampler;le;fp8-latent528;bf16-pool128;page256;layers45;dsa-layer-mod4=3;kda-conv3x6144bf16;kda-active16x128x128fp32;index-key-gate-bf16;compact-tail");
        h.update((ring_rows as u64).to_le_bytes());
        for shard in shards {
            h.update(shard);
        }
        h.update(kernel_policy);
        Ok(Self {
            namespace: h.finalize().into(),
            rank,
            ring_rows,
            speculation: None,
        })
    }
    /// All-rank companion identity, derived from verified drafter manifests.
    /// Target-only files never masquerade as complete speculative snapshots.
    pub fn with_speculation(mut self, identity: Hash) -> Self {
        let mut h = Sha256::new();
        h.update(b"alaska-dflash-pure-v1/window2047/5layers/2heads128bf16");
        h.update(self.namespace);
        h.update(identity);
        self.namespace = h.finalize().into();
        self.speculation = Some(identity);
        self
    }
    pub fn segments(&self, n: usize) -> Result<Vec<Segment>> {
        segments_with_speculation(n, self.ring_rows, self.speculation.is_some())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Tokens,
    Conv(usize),
    Recurrent(usize),
    IndexKeys(usize),
    IndexGates(usize),
    TailLatent(usize),
    TailPool(usize),
    DraftMetadata,
    DraftKv(usize),
}
#[derive(Clone, Debug)]
pub struct Segment {
    pub role: Role,
    pub offset: u64,
    pub bytes: u64,
}

pub fn segments(n: usize, ring_rows: usize) -> Result<Vec<Segment>> {
    segments_with_speculation(n, ring_rows, false)
}
pub fn segments_with_speculation(
    n: usize,
    ring_rows: usize,
    speculation: bool,
) -> Result<Vec<Segment>> {
    ensure!(
        n > 0 && n < CONTEXT && n.is_multiple_of(64),
        "SSD pure prefix length/grid"
    );
    ensure!(
        ring_rows.is_power_of_two() && (256..=16384).contains(&ring_rows),
        "SSD ring shape"
    );
    let mut v = Vec::with_capacity(113);
    let mut offset = 0;
    let mut add = |role, bytes| {
        v.push(Segment {
            role,
            offset,
            bytes,
        });
        offset += bytes;
    };
    add(Role::Tokens, n as u64 * 4);
    for layer in 0..45 {
        if layer % 4 == 3 {
            add(Role::IndexKeys(layer), ring_rows as u64 * 128 * 2);
            add(Role::IndexGates(layer), ring_rows as u64 * 128 * 2);
        } else {
            add(Role::Conv(layer), 3 * 6144 * 2);
            add(Role::Recurrent(layer), 16 * 128 * 128 * 4);
        }
    }
    // Include a full final page privately when the next token is not known.
    let tail = n - (n - 1) / PAGE * PAGE;
    for family in 0..11 {
        add(Role::TailLatent(family), tail as u64 * 528);
        add(Role::TailPool(family), (tail / 4) as u64 * 256);
    }
    if speculation {
        add(Role::DraftMetadata, 64);
        for tensor in 0..10 {
            add(Role::DraftKv(tensor), n.min(2047) as u64 * 512);
        }
    }
    Ok(v)
}
pub fn body_bytes(n: usize, id: &Identity) -> Result<u64> {
    let v = id.segments(n)?;
    let last = v.last().unwrap();
    Ok(last.offset + last.bytes)
}
fn token_hash(namespace: &Hash) -> Sha256 {
    let mut h = Sha256::new();
    h.update(b"alaska-rust-disk-tokens-v1");
    h.update(namespace);
    h
}
fn finish_key(mut h: Sha256, n: usize) -> Hash {
    h.update((n as u64).to_le_bytes());
    h.finalize().into()
}
pub fn keys(namespace: &Hash, ids: &[i32]) -> Result<(Hash, Vec<Hash>)> {
    ensure!(
        !ids.is_empty()
            && ids.len() < CONTEXT
            && ids.len().is_multiple_of(64)
            && ids.iter().all(|&t| crate::vision_input::prompt_token(t)),
        "SSD pure prefix tokens"
    );
    let mut h = token_hash(namespace);
    for id in ids {
        h.update(id.to_le_bytes());
    }
    let entry = finish_key(h, ids.len());
    let mut h = Sha256::new();
    h.update(b"alaska-rust-disk-pages-v1");
    h.update(namespace);
    let mut chain: Hash = h.finalize().into();
    let mut pages = Vec::with_capacity((ids.len() - 1) / PAGE);
    for page in 0..(ids.len() - 1) / PAGE {
        let mut h = Sha256::new();
        h.update(chain);
        for id in &ids[page * PAGE..(page + 1) * PAGE] {
            h.update(id.to_le_bytes());
        }
        chain = h.finalize().into();
        let mut h = Sha256::new();
        h.update(b"page-next-token");
        h.update(chain);
        h.update(ids[(page + 1) * PAGE].to_le_bytes());
        pages.push(h.finalize().into());
    }
    Ok((entry, pages))
}
/// Hash all request prefixes in one pass; retained metadata never stores token arrays.
pub fn candidates(namespace: &Hash, ids: &[i32]) -> Result<Vec<(usize, Hash)>> {
    ensure!(
        ids.len() <= CONTEXT && ids.iter().all(|&t| crate::vision_input::prompt_token(t)),
        "SSD request tokens"
    );
    let mut h = token_hash(namespace);
    let mut out = Vec::with_capacity(ids.len() / 64);
    for (i, t) in ids.iter().enumerate() {
        h.update(t.to_le_bytes());
        let n = i + 1;
        if n.is_multiple_of(64) && n < ids.len() {
            out.push((n, finish_key(h.clone(), n)));
        }
    }
    Ok(out)
}
#[derive(Clone, Debug)]
pub struct PageRef {
    pub key: Hash,
    pub checksum: Hash,
}
#[derive(Clone, Debug)]
pub struct Header {
    pub key: Hash,
    pub n: usize, // zero identifies a page; otherwise a pure prefix.
    pub stamp: u64,
    pub body_bytes: u64,
    pub pages: Vec<PageRef>,
    pub sums: Vec<Hash>,
}
impl Header {
    pub fn new(
        id: &Identity,
        key: Hash,
        n: usize,
        stamp: u64,
        pages: Vec<PageRef>,
    ) -> Result<Self> {
        let bytes = if n == 0 {
            PAGE_BYTES
        } else {
            body_bytes(n, id)?
        };
        ensure!(
            pages.len() == if n == 0 { 0 } else { (n - 1) / PAGE },
            "SSD page count"
        );
        Ok(Self {
            key,
            n,
            stamp,
            body_bytes: bytes,
            pages,
            sums: vec![[0; 32]; bytes.div_ceil(CHUNK as u64) as usize],
        })
    }
    pub fn header_bytes(&self) -> usize {
        FIXED + self.pages.len() * 64 + self.sums.len() * 32 + 32
    }
    pub fn file_bytes(&self) -> u64 {
        self.header_bytes() as u64 + self.body_bytes
    }
    pub fn checksum(&self) -> Hash {
        self.sums[0]
    } // pages have exactly one chunk
    pub fn encode(&self, id: &Identity) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.header_bytes());
        b.extend(MAGIC);
        b.extend((if self.n == 0 { 1u32 } else { 2 }).to_le_bytes());
        b.extend(id.rank.to_le_bytes());
        b.extend(id.namespace);
        b.extend(self.key);
        b.extend((self.n as u32).to_le_bytes());
        b.extend(0u32.to_le_bytes()); // flags: pure, no unsupported fields
        b.extend(self.body_bytes.to_le_bytes());
        b.extend(self.stamp.to_le_bytes());
        b.extend((self.pages.len() as u32).to_le_bytes());
        b.extend((self.sums.len() as u32).to_le_bytes());
        for p in &self.pages {
            b.extend(p.key);
            b.extend(p.checksum);
        }
        for sum in &self.sums {
            b.extend(sum);
        }
        let checksum = Sha256::digest(&b);
        b.extend(checksum);
        b
    }
    /// Fixed dimensions are checked before allocating any header-controlled buffer.
    pub fn read<R: Read + Seek>(r: &mut R, id: &Identity) -> Result<Self> {
        r.seek(SeekFrom::Start(0))?;
        let mut fixed = [0; FIXED];
        r.read_exact(&mut fixed)?;
        let u32at = |a| u32::from_le_bytes(fixed[a..a + 4].try_into().unwrap());
        let u64at = |a| u64::from_le_bytes(fixed[a..a + 8].try_into().unwrap());
        ensure!(
            &fixed[..8] == MAGIC
                && u32at(12) == id.rank
                && fixed[16..48] == id.namespace
                && u32at(84) == 0,
            "SSD format/rank/compatibility/flags"
        );
        let n = u32at(80) as usize;
        ensure!(u32at(8) == if n == 0 { 1 } else { 2 }, "SSD kind");
        let bytes = if n == 0 {
            PAGE_BYTES
        } else {
            body_bytes(n, id)?
        };
        let np = if n == 0 { 0 } else { (n - 1) / PAGE };
        let nc = bytes.div_ceil(CHUNK as u64) as usize;
        ensure!(
            u64at(88) == bytes && u32at(104) as usize == np && u32at(108) as usize == nc,
            "SSD body/layout bounds"
        );
        let header_len = FIXED + np * 64 + nc * 32 + 32;
        ensure!(
            r.seek(SeekFrom::End(0))? == header_len as u64 + bytes,
            "SSD truncated/overlong object"
        );
        r.seek(SeekFrom::Start(FIXED as u64))?;
        let mut rest = vec![0; header_len - FIXED];
        r.read_exact(&mut rest)?;
        let mut h = Sha256::new();
        h.update(fixed);
        h.update(&rest[..rest.len() - 32]);
        ensure!(
            h.finalize()[..] == rest[rest.len() - 32..],
            "SSD header checksum"
        );
        let pages = rest[..np * 64]
            .as_chunks::<64>()
            .0
            .iter()
            .map(|p| PageRef {
                key: p[..32].try_into().unwrap(),
                checksum: p[32..].try_into().unwrap(),
            })
            .collect();
        let sums = rest[np * 64..rest.len() - 32].as_chunks::<32>().0.to_vec();
        Ok(Self {
            key: fixed[48..80].try_into().unwrap(),
            n,
            stamp: u64at(96),
            body_bytes: bytes,
            pages,
            sums,
        })
    }
}
/// One reusable staging buffer, independent of total context and snapshot size.
pub fn write_body<W: Write + Seek>(
    w: &mut W,
    header: &mut Header,
    stage: &mut [u8],
    mut source: impl FnMut(u64, &mut [u8]) -> Result<()>,
) -> Result<()> {
    ensure!(stage.len() == CHUNK, "SSD stage size");
    w.seek(SeekFrom::Start(header.header_bytes() as u64))?;
    let mut offset = 0;
    for sum in &mut header.sums {
        let len = (header.body_bytes - offset).min(CHUNK as u64) as usize;
        let buf = &mut stage[..len];
        source(offset, buf)?;
        *sum = Sha256::digest(&*buf).into();
        w.write_all(buf)?;
        offset += len as u64;
    }
    Ok(())
}
pub fn read_body<R: Read + Seek>(
    r: &mut R,
    header: &Header,
    stage: &mut [u8],
    mut sink: impl FnMut(u64, &[u8]) -> Result<()>,
) -> Result<()> {
    ensure!(stage.len() == CHUNK, "SSD stage size");
    r.seek(SeekFrom::Start(header.header_bytes() as u64))?;
    let mut offset = 0;
    for sum in &header.sums {
        let len = (header.body_bytes - offset).min(CHUNK as u64) as usize;
        let buf = &mut stage[..len];
        r.read_exact(buf)?;
        ensure!(Sha256::digest(&*buf)[..] == sum[..], "SSD data checksum");
        sink(offset, buf)?;
        offset += len as u64;
    }
    Ok(())
}
pub fn hex(h: &Hash) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn unhex(s: &str) -> Option<Hash> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let mut h = [0; 32];
    for (i, v) in h.iter_mut().enumerate() {
        *v = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn id() -> Identity {
        Identity::new([[1; 32]; 4], [2; 32], 0, 1024).unwrap()
    }
    #[test]
    fn complete_typed_layout_and_bounds() -> Result<()> {
        for n in [64, 256, 320, 8192, CONTEXT - 64] {
            let s = segments(n, 1024)?;
            assert_eq!(s.len(), 113);
            assert_eq!(
                s.iter()
                    .filter(|s| matches!(s.role, Role::Recurrent(_)))
                    .count(),
                34
            );
            assert_eq!(
                s.iter()
                    .filter(|s| matches!(s.role, Role::IndexKeys(_)))
                    .count(),
                11
            );
            assert_eq!(
                s.last().unwrap().offset + s.last().unwrap().bytes,
                body_bytes(n, &id())?
            );
            assert!(
                s.windows(2)
                    .all(|p| p[0].offset + p[0].bytes == p[1].offset)
            );
        }
        for n in [0, 1, 63, 65, CONTEXT, usize::MAX] {
            assert!(segments(n, 1024).is_err());
        }
        assert!(segments(64, usize::MAX).is_err());
        Ok(())
    }
    #[test]
    fn drafter_companion_layout_accounting_and_namespace_are_exact() -> Result<()> {
        let plain = id();
        let draft = plain.clone().with_speculation([3; 32]);
        assert_ne!(plain.namespace, draft.namespace);
        assert_ne!(
            draft.namespace,
            plain.clone().with_speculation([4; 32]).namespace
        );
        for n in [64, 256, 2048, 8192, CONTEXT - 64] {
            let segments = draft.segments(n)?;
            assert_eq!(segments.len(), 124);
            let companion = &segments[113..];
            assert_eq!(companion[0].role, Role::DraftMetadata);
            assert_eq!(companion[0].bytes, 64);
            for (i, tensor) in companion[1..].iter().enumerate() {
                assert_eq!(tensor.role, Role::DraftKv(i));
                assert_eq!(tensor.bytes, n.min(2047) as u64 * 512);
            }
            assert!(
                segments
                    .windows(2)
                    .all(|pair| pair[0].offset + pair[0].bytes == pair[1].offset)
            );
            assert_eq!(
                body_bytes(n, &draft)? - body_bytes(n, &plain)?,
                64 + n.min(2047) as u64 * 5120
            );
            // Metadata is a single complete codec slice even at chunk boundaries.
            assert!(companion[0].offset % CHUNK as u64 + 64 <= CHUNK as u64);
        }
        let header = Header::new(&draft, [9; 32], 64, 1, vec![])?;
        let bytes = header.encode(&draft);
        assert!(Header::read(&mut Cursor::new(bytes), &plain).is_err());
        Ok(())
    }
    #[test]
    fn streamed_lookup_keys_and_next_token_dependency() -> Result<()> {
        let id = id();
        let ids = vec![1; 1025];
        let candidates = candidates(&id.namespace, &ids)?;
        for (n, key) in candidates {
            assert_eq!(key, keys(&id.namespace, &ids[..n])?.0);
        }
        let (_, a) = keys(&id.namespace, &ids[..256])?;
        assert!(a.is_empty());
        let (_, b) = keys(&id.namespace, &ids[..320])?;
        assert_eq!(b.len(), 1);
        let mut changed = ids[..320].to_vec();
        changed[256] = 2;
        assert_ne!(b, keys(&id.namespace, &changed)?.1);
        assert!(keys(&id.namespace, &vec![-1; 64]).is_err());
        Ok(())
    }
    #[test]
    fn bounded_header_and_chunk_corruption() -> Result<()> {
        let id = id();
        let mut h = Header::new(&id, [7; 32], 0, 0, Vec::new())?;
        let mut f = Cursor::new(Vec::new());
        let mut stage = vec![0; CHUNK];
        write_body(&mut f, &mut h, &mut stage, |_, b| {
            b.fill(17);
            Ok(())
        })?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&h.encode(&id))?;
        let good = f.into_inner();
        for (offset, value) in [
            (8, 3),
            (12, 1),
            (16, 99),
            (80, 255),
            (84, 1),
            (88, 255),
            (104, 255),
            (108, 255),
            (112, 99),
        ] {
            let mut bad = good.clone();
            bad[offset] = value;
            assert!(
                Header::read(&mut Cursor::new(bad), &id).is_err(),
                "offset {offset}"
            );
        }
        for size in [0, 7, 111, 176, good.len() - 1] {
            assert!(Header::read(&mut Cursor::new(&good[..size]), &id).is_err());
        }
        let mut f = Cursor::new(good.clone());
        let h = Header::read(&mut f, &id)?;
        read_body(&mut f, &h, &mut stage, |_, b| {
            assert!(b.iter().all(|b| *b == 17));
            Ok(())
        })?;
        let mut bad = good;
        *bad.last_mut().unwrap() ^= 1;
        let mut f = Cursor::new(bad);
        let h = Header::read(&mut f, &id)?;
        let mut called = false;
        assert!(
            read_body(&mut f, &h, &mut stage, |_, _| {
                called = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!called);
        Ok(())
    }
    #[test]
    fn durable_identity_binds_layout_kernels_and_shard_order() -> Result<()> {
        let a = Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [5; 32], 0, 1024)?;
        let b = Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [5; 32], 3, 1024)?;
        assert_eq!(a.namespace, b.namespace);
        assert_ne!(
            a.namespace,
            Identity::new([[2; 32], [1; 32], [3; 32], [4; 32]], [5; 32], 0, 1024)?.namespace
        );
        assert_ne!(
            a.namespace,
            Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [6; 32], 0, 1024)?.namespace
        );
        assert_ne!(
            a.namespace,
            Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [5; 32], 0, 2048)?.namespace
        );
        assert_eq!(unhex(&hex(&a.namespace)), Some(a.namespace));
        assert!(unhex(&"F".repeat(64)).is_none());
        Ok(())
    }
}
