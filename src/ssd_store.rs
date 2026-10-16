//! Single-owner synchronous SSD component. GPU staging and collective policy are
//! intentionally outside this module. Errors never authorize a partial restore.
use crate::ssd_format::{self as fmt, CHUNK, Hash, Header, Identity, PageRef, Role, hex, unhex};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

const MAX_BUDGET: u64 = 64 << 30;
const META_BUDGET: usize = 64 << 20;
const MAX_ENTRIES: usize = 4096;
const ROOT_BYTES: u64 = 32 << 10; // marker/lock/directories + atomic touch overlap
fn charge(n: u64) -> u64 {
    n.div_ceil(4096) * 4096
}
fn entry_charge(h: &Header) -> u64 {
    charge(h.file_bytes()) + 12288
} // durable LRU sidecar
fn page_charge() -> u64 {
    charge(176 + fmt::PAGE_BYTES) + 4096
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Layer(Role),
    Page {
        logical: usize,
        family: usize,
        pooled: bool,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Saved,
    Duplicate,
    Skipped,
}
#[derive(Clone)]
struct Entry {
    n: usize,
    pages: Vec<Hash>,
    bytes: u64,
    stamp: u64,
}
#[derive(Clone)]
struct Page {
    refs: usize,
    checksum: Hash,
}
pub struct Store {
    root: PathBuf,
    id: Identity,
    budget: u64,
    _lock: File,
    entries: BTreeMap<Hash, Entry>,
    pages: BTreeMap<Hash, Page>,
    stage: Vec<u8>,
    clock: u64,
    healthy: bool,
    /// Cleared by every mutable method; scheduler agreements read the digest
    /// several times a round.
    memo: std::cell::Cell<Option<Hash>>,
    // A foreground reader may retain a bounded catalog snapshot while the
    // exclusive mutable owner is parked. It never publishes, unlinks or touches
    // files. Actual tokens and every payload checksum are still verified.
    read_only: bool,
}
fn regular(path: &Path) -> Result<File> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "SSD object is not a regular file"
    );
    Ok(File::open(path)?)
}
impl Store {
    pub fn open(root: &Path, id: Identity, budget: u64) -> Result<Self> {
        ensure!(
            (ROOT_BYTES..=MAX_BUDGET).contains(&budget),
            "SSD budget exceeds 64 GiB or root floor"
        );
        if !root.exists() {
            fs::create_dir(root)?;
            fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        }
        ensure!(
            fs::symlink_metadata(root)?.file_type().is_dir(),
            "SSD root is not a directory"
        );
        let lock_path = root.join(".lock");
        if lock_path.exists() {
            ensure!(
                fs::symlink_metadata(&lock_path)?.file_type().is_file(),
                "SSD lock type"
            );
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;
        lock.try_lock().context("SSD directory already owned")?;
        let marker = format!(
            "alaska-rust-ssd-v2\n{}\n{}\n{}\n",
            hex(&id.namespace),
            id.rank,
            id.ring_rows
        );
        let marker_path = root.join("identity");
        if marker_path.exists() {
            let mut f = regular(&marker_path)?;
            ensure!(
                f.metadata()?.len() == marker.len() as u64,
                "SSD root identity length"
            );
            let mut old = String::new();
            f.read_to_string(&mut old)?;
            ensure!(
                old == marker,
                "SSD root identity mismatch; use a fresh isolated directory"
            );
        } else {
            ensure!(
                fs::read_dir(root)?.all(|p| p.is_ok_and(|p| p.file_name() == ".lock")),
                "refusing unmarked nonempty SSD directory"
            );
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&marker_path)?;
            f.write_all(marker.as_bytes())?;
            f.sync_all()?;
            File::open(root)?.sync_all()?;
        }
        for name in ["entries", "pages"] {
            let p = root.join(name);
            if !p.exists() {
                fs::create_dir(&p)?;
            }
            ensure!(
                fs::symlink_metadata(p)?.file_type().is_dir(),
                "SSD child directory type"
            );
        }
        let mut s = Self {
            root: root.to_owned(),
            id,
            budget,
            _lock: lock,
            entries: BTreeMap::new(),
            pages: BTreeMap::new(),
            stage: vec![0; CHUNK],
            clock: 0,
            healthy: true,
            memo: std::cell::Cell::new(None),
            read_only: false,
        };
        for p in fs::read_dir(root)? {
            let name = p?.file_name();
            ensure!(
                [".lock", "identity", "entries", "pages"]
                    .iter()
                    .any(|allowed| name == *allowed),
                "SSD root contains unowned data"
            );
        }
        s.scan()?;
        Ok(s)
    }
    fn path(&self, key: &Hash, ext: &str) -> PathBuf {
        self.root
            .join(if ext == "page" { "pages" } else { "entries" })
            .join(format!("{}.{}", hex(key), ext))
    }
    pub fn reader_view(&self) -> Result<Self> {
        self.ready()?;
        ensure!(!self.read_only, "cannot fork an SSD read view");
        ensure!(
            self.metadata_bytes() <= META_BUDGET,
            "SSD view metadata bound"
        );
        Ok(Self {
            root: self.root.clone(),
            id: self.id.clone(),
            budget: self.budget,
            _lock: self._lock.try_clone()?,
            entries: self.entries.clone(),
            pages: self.pages.clone(),
            stage: vec![0; CHUNK],
            clock: self.clock,
            healthy: true,
            memo: std::cell::Cell::new(None),
            read_only: true,
        })
    }
    fn sync_dir(&self, name: &str) -> Result<()> {
        File::open(self.root.join(name))?.sync_all()?;
        Ok(())
    }
    fn ready(&self) -> Result<()> {
        ensure!(
            self.healthy,
            "SSD I/O mutation failed; close and reopen before reuse"
        );
        Ok(())
    }
    pub fn bytes(&self) -> u64 {
        ROOT_BYTES
            + self.entries.values().map(|e| e.bytes).sum::<u64>()
            + self.pages.len() as u64 * page_charge()
    }
    pub fn metadata_bytes(&self) -> usize {
        self.pages.len() * 256
            + self
                .entries
                .values()
                .map(|e| 512 + e.pages.len() * 32)
                .sum::<usize>()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Rank-independent metadata digest; rank-local payload checksums are excluded.
    pub fn digest(&self) -> Hash {
        if let Some(hash) = self.memo.get() {
            return hash;
        }
        let mut h = Sha256::new();
        h.update(self.id.namespace);
        for value in [
            self.budget,
            self.clock,
            self.bytes(),
            self.entries.len() as u64,
            self.pages.len() as u64,
        ] {
            h.update(value.to_le_bytes());
        }
        for (key, e) in &self.entries {
            h.update(key);
            for value in [e.n as u64, e.bytes, e.stamp, e.pages.len() as u64] {
                h.update(value.to_le_bytes());
            }
            for page in &e.pages {
                h.update(page);
            }
        }
        let hash: Hash = h.finalize().into();
        self.memo.set(Some(hash));
        hash
    }
    pub fn prefix_len(&mut self, key: &Hash, request: &[i32]) -> Result<usize> {
        self.memo.set(None);
        self.ready()?;
        ensure!(self.entries.contains_key(key), "SSD entry absent");
        Ok(self.token_header(key, Some(request))?.n)
    }
    fn header(&self, key: &Hash, page: bool) -> Result<(File, Header)> {
        let mut f = regular(&self.path(key, if page { "page" } else { "entry" }))?;
        let h = Header::read(&mut f, &self.id)?;
        ensure!(
            h.key == *key && (h.n == 0) == page,
            "SSD filename/key/kind mismatch"
        );
        Ok((f, h))
    }
    fn token_header(&mut self, key: &Hash, request: Option<&[i32]>) -> Result<Header> {
        self.memo.set(None);
        let (mut f, h) = self.header(key, false)?;
        // The entry key hashes the exact packed IDs and namespace. Read only
        // those bytes during catalog recovery, not an 8 MiB payload chunk per
        // entry. Full body checksums are still verified during restoration.
        f.seek(SeekFrom::Start(h.header_bytes() as u64))?;
        f.read_exact(&mut self.stage[..h.n * 4])?;
        let ids: Vec<i32> = self.stage[..h.n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        let (computed, pages) = fmt::keys(&self.id.namespace, &ids)?;
        ensure!(
            computed == h.key && pages.iter().eq(h.pages.iter().map(|p| &p.key)),
            "SSD exact token/page identity"
        );
        if let Some(request) = request {
            ensure!(
                request.len() > ids.len() && request.starts_with(&ids),
                "SSD request does not strictly extend saved tokens"
            );
        }
        Ok(h)
    }
    fn insert(&mut self, h: &Header, stamp: u64) -> Result<()> {
        self.memo.set(None);
        for p in &h.pages {
            ensure!(
                !self.pages.contains_key(&p.key) || self.pages[&p.key].checksum == p.checksum,
                "SSD shared page checksum disagreement"
            );
        }
        for p in &h.pages {
            self.pages
                .entry(p.key)
                .or_insert(Page {
                    refs: 0,
                    checksum: p.checksum,
                })
                .refs += 1;
        }
        self.entries.insert(
            h.key,
            Entry {
                n: h.n,
                pages: h.pages.iter().map(|p| p.key).collect(),
                bytes: entry_charge(h),
                stamp,
            },
        );
        self.clock = self.clock.max(stamp);
        Ok(())
    }
    fn unlink(&self, path: &Path) -> Result<()> {
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
    fn evict(&mut self, key: &Hash, pinned: &BTreeSet<Hash>) -> Result<()> {
        self.memo.set(None);
        if let Some(e) = self.entries.get(key) {
            self.unlink(&self.path(key, "entry"))?;
            self.unlink(&self.path(key, "touch"))?;
            self.sync_dir("entries")?; // remove commit record before freeing dependencies
            let pages = e.pages.clone();
            self.entries.remove(key);
            for key in pages {
                self.pages.get_mut(&key).unwrap().refs -= 1;
            }
            self.gc(pinned)?;
        }
        Ok(())
    }
    fn gc(&mut self, pinned: &BTreeSet<Hash>) -> Result<()> {
        self.memo.set(None);
        let unused: Vec<_> = self
            .pages
            .iter()
            .filter(|(key, p)| p.refs == 0 && !pinned.contains(*key))
            .map(|(key, _)| *key)
            .collect();
        for key in unused {
            self.unlink(&self.path(&key, "page"))?;
            self.pages.remove(&key);
        }
        self.sync_dir("pages")?;
        Ok(())
    }
    fn oldest(&self) -> Option<Hash> {
        self.entries
            .iter()
            .min_by_key(|(key, e)| (e.stamp, **key))
            .map(|(key, _)| *key)
    }
    fn stamp(&mut self) -> Result<u64> {
        self.memo.set(None);
        self.clock = self
            .clock
            .checked_add(1)
            .context("SSD LRU counter exhausted")?;
        Ok(self.clock)
    }
    fn write_touch(&self, key: &Hash, stamp: u64) -> Result<()> {
        let path = self.path(key, "touch");
        let tmp = path.with_extension("touch.tmp");
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        let mut bytes = Vec::from(stamp.to_le_bytes());
        bytes.extend(Sha256::digest(&bytes));
        f.write_all(&bytes)?;
        f.sync_all()?;
        fs::rename(tmp, path)?;
        self.sync_dir("entries")
    }
    fn read_touch(&self, key: &Hash, fallback: u64) -> u64 {
        let read = || -> Result<u64> {
            let mut f = regular(&self.path(key, "touch"))?;
            ensure!(f.metadata()?.len() == 40, "touch length");
            let mut b = [0; 40];
            f.read_exact(&mut b)?;
            ensure!(Sha256::digest(&b[..8])[..] == b[8..], "touch checksum");
            Ok(u64::from_le_bytes(b[..8].try_into().unwrap()))
        };
        read().unwrap_or(fallback)
    }
    pub fn touch(&mut self, key: &Hash) -> Result<()> {
        self.memo.set(None);
        self.ready()?;
        ensure!(self.entries.contains_key(key), "SSD entry absent");
        if self.read_only {
            return Ok(());
        }
        self.healthy = false;
        let stamp = self.stamp()?;
        self.write_touch(key, stamp)?;
        self.entries.get_mut(key).unwrap().stamp = stamp;
        self.healthy = true;
        Ok(())
    }
    pub fn forget(&mut self, key: &Hash) -> Result<()> {
        self.memo.set(None);
        self.ready()?;
        if self.read_only {
            // Quarantine only this view. The mutable owner may still be reading
            // a shared page; its ordinary checked lookup handles persistent
            // quarantine after it regains catalog ownership.
            self.entries.remove(key);
            return Ok(());
        }
        self.healthy = false;
        self.evict(key, &BTreeSet::new())?;
        self.healthy = true;
        Ok(())
    }
    /// Rank coordinator supplies a common intersection in rank-zero LRU order.
    /// Local scanning alone does NOT establish all-rank presence or agreement.
    pub fn retain_order(&mut self, ordered: &[Hash]) -> Result<()> {
        ensure!(!self.read_only, "cannot reconcile an SSD read view");
        self.memo.set(None);
        self.ready()?;
        let keep: BTreeSet<_> = ordered.iter().copied().collect();
        ensure!(
            keep.len() == ordered.len() && keep.iter().all(|k| self.entries.contains_key(k)),
            "SSD reconciliation input"
        );
        self.healthy = false;
        let remove: Vec<_> = self
            .entries
            .keys()
            .filter(|k| !keep.contains(*k))
            .copied()
            .collect();
        for k in remove {
            self.evict(&k, &BTreeSet::new())?;
        }
        for (i, k) in ordered.iter().enumerate() {
            self.write_touch(k, i as u64 + 1)?;
            self.entries.get_mut(k).unwrap().stamp = i as u64 + 1;
        }
        self.clock = ordered.len() as u64;
        self.healthy = true;
        Ok(())
    }
    pub fn order(&self) -> Vec<Hash> {
        let mut v: Vec<_> = self.entries.iter().map(|(k, e)| (e.stamp, *k)).collect();
        v.sort();
        v.into_iter().map(|(_, k)| k).collect()
    }
    fn scan(&mut self) -> Result<()> {
        ensure!(!self.read_only, "cannot scan an SSD read view");
        self.memo.set(None);
        // Remove only this format's owned names. Unknown files stop recovery,
        // rather than being silently deleted or escaping disk accounting.
        let mut candidates = Vec::new();
        for dir in ["entries", "pages"] {
            let mut count = 0;
            for p in fs::read_dir(self.root.join(dir))? {
                count += 1;
                ensure!(count <= 65536, "SSD directory inventory bound");
                let p = p?;
                let name = p.file_name();
                let name = name.to_str().context("SSD filename encoding")?;
                ensure!(p.file_type()?.is_file(), "SSD directory contains non-file");
                let parts: Vec<_> = name.split('.').collect();
                ensure!(
                    (parts.len() == 2 || parts.len() == 3 && parts[2] == "tmp")
                        && unhex(parts[0]).is_some()
                        && if dir == "pages" {
                            parts[1] == "page"
                        } else {
                            matches!(parts[1], "entry" | "touch")
                        },
                    "SSD directory contains unowned file"
                );
                if parts.len() == 3 {
                    fs::remove_file(p.path())?;
                } else if parts[1] == "entry" {
                    ensure!(candidates.len() < MAX_ENTRIES, "SSD entry inventory bound");
                    candidates.push(unhex(parts[0]).unwrap());
                }
            }
        }
        candidates.sort();
        // Headers, exact tokens and page presence are checked; other payload
        // checksums are deferred to load. Only one token array is resident here.
        let mut valid = Vec::new();
        for key in candidates {
            let checked = (|| -> Result<Header> {
                let h = self.token_header(&key, None)?;
                for p in &h.pages {
                    let (_, ph) = self.header(&p.key, true)?;
                    ensure!(
                        ph.checksum() == p.checksum,
                        "SSD page header checksum mismatch"
                    );
                }
                Ok(h)
            })();
            match checked {
                Ok(h) => {
                    let stamp = self.read_touch(&key, h.stamp);
                    valid.push((stamp, key));
                }
                Err(_) => {
                    self.unlink(&self.path(&key, "entry"))?;
                    self.unlink(&self.path(&key, "touch"))?;
                }
            }
        }
        valid.sort();
        for (stamp, key) in valid.into_iter().rev() {
            let h = self.token_header(&key, None)?;
            let missing = h
                .pages
                .iter()
                .filter(|p| !self.pages.contains_key(&p.key))
                .count();
            if self.bytes() + entry_charge(&h) + missing as u64 * page_charge() > self.budget
                || self.metadata_bytes() + 512 + h.pages.len() * 32 + missing * 256 > META_BUDGET
                || self.insert(&h, stamp).is_err()
            {
                self.unlink(&self.path(&key, "entry"))?;
                self.unlink(&self.path(&key, "touch"))?;
            }
        }
        for dir in ["entries", "pages"] {
            for p in fs::read_dir(self.root.join(dir))? {
                let p = p?;
                let path = p.path();
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .context("SSD stem")?;
                let key = unhex(name).context("SSD orphan filename")?;
                let kept = if dir == "pages" {
                    self.pages.contains_key(&key)
                } else {
                    self.entries.contains_key(&key)
                };
                if !kept {
                    fs::remove_file(path)?;
                }
            }
            self.sync_dir(dir)?;
        }
        Ok(())
    }
    fn write_object(
        &mut self,
        h: &mut Header,
        mut read: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        self.memo.set(None);
        let path = self.path(&h.key, if h.n == 0 { "page" } else { "entry" });
        ensure!(!path.exists(), "SSD immutable object already present");
        let tmp = path.with_extension(if h.n == 0 { "page.tmp" } else { "entry.tmp" });
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        fmt::write_body(&mut f, h, &mut self.stage, &mut read)?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&h.encode(&self.id))?;
        f.sync_all()?;
        fs::rename(&tmp, &path)?;
        self.sync_dir(if h.n == 0 { "pages" } else { "entries" })
    }
    pub fn save(
        &mut self,
        ids: &[i32],
        mixed: bool,
        mut source: impl FnMut(Source, u64, &mut [u8]) -> Result<()>,
    ) -> Result<Outcome> {
        ensure!(!self.read_only, "cannot save through an SSD read view");
        self.memo.set(None);
        self.ready()?;
        ensure!(
            !mixed,
            "mixed snapshots are not supported by the SSD prefix format"
        );
        let (key, page_keys) = fmt::keys(&self.id.namespace, ids)?;
        if self.entries.contains_key(&key) {
            self.touch(&key)?;
            return Ok(Outcome::Duplicate);
        }
        let mut h = Header::new(
            &self.id,
            key,
            ids.len(),
            0,
            page_keys
                .iter()
                .map(|&key| PageRef {
                    key,
                    checksum: [0; 32],
                })
                .collect(),
        )?;
        let floor = ROOT_BYTES + entry_charge(&h) + page_keys.len() as u64 * page_charge();
        if floor > self.budget {
            return Ok(Outcome::Skipped);
        }
        let pinned: BTreeSet<_> = page_keys.iter().copied().collect();
        self.healthy = false;
        loop {
            let missing = page_keys
                .iter()
                .filter(|k| !self.pages.contains_key(*k))
                .count();
            let bytes = self.bytes() + entry_charge(&h) + missing as u64 * page_charge();
            let meta = self.metadata_bytes() + 512 + page_keys.len() * 32 + missing * 256;
            if bytes <= self.budget && meta <= META_BUDGET && self.entries.len() < MAX_ENTRIES {
                break;
            }
            self.evict(&self.oldest().context("SSD admission accounting")?, &pinned)?;
        }
        h.stamp = self.stamp()?;
        for (logical, p) in h.pages.iter_mut().enumerate() {
            if let Some(existing) = self.pages.get(&p.key) {
                p.checksum = existing.checksum;
                continue;
            }
            let mut ph = Header::new(&self.id, p.key, 0, 0, Vec::new())?;
            self.write_object(&mut ph, |offset, buf| {
                fill_page(logical, offset, buf, &mut source)
            })?;
            p.checksum = ph.checksum();
            self.pages.insert(
                p.key,
                Page {
                    refs: 0,
                    checksum: p.checksum,
                },
            );
        }
        let segments = self.id.segments(ids.len())?;
        self.write_object(&mut h, |offset, buf| {
            for s in &segments {
                intersection(
                    offset,
                    buf.len(),
                    s.offset,
                    s.bytes,
                    |within, start, len| {
                        let out = &mut buf[start..start + len];
                        if s.role == Role::Tokens {
                            for (i, b) in out.iter_mut().enumerate() {
                                let at = within as usize + i;
                                *b = ids[at / 4].to_le_bytes()[at % 4];
                            }
                            Ok(())
                        } else {
                            source(Source::Layer(s.role), within, out)
                        }
                    },
                )?;
            }
            Ok(())
        })?;
        self.insert(&h, h.stamp)?;
        self.gc(&BTreeSet::new())?;
        ensure!(
            self.bytes() <= self.budget && self.metadata_bytes() <= META_BUDGET,
            "SSD committed accounting"
        );
        self.healthy = true;
        Ok(Outcome::Saved)
    }
    /// Metadata-only candidate hint. The worker must still verify actual tokens
    /// and complete payload checksums before any restored model can execute.
    pub fn hint(&self, request: &[i32]) -> Result<Option<usize>> {
        self.ready()?;
        Ok(fmt::candidates(&self.id.namespace, request)?
            .into_iter()
            .rev()
            .find_map(|(n, key)| self.entries.get(&key).filter(|e| e.n == n).map(|_| n)))
    }
    pub fn lookup(&mut self, request: &[i32]) -> Result<Option<Hash>> {
        self.memo.set(None);
        self.ready()?;
        for (n, key) in fmt::candidates(&self.id.namespace, request)?
            .into_iter()
            .rev()
        {
            if self.entries.get(&key).is_some_and(|e| e.n == n) {
                // Never trust only a digest for a hit. Corrupt candidates are
                // discarded, allowing shorter valid prefixes to be considered.
                if self.token_header(&key, Some(request)).is_ok() {
                    return Ok(Some(key));
                }
                self.forget(&key)?;
            }
        }
        Ok(None)
    }
    /// Each chunk is checked before sink receives it. A later I/O error means
    /// ALL consumed bytes must be discarded. GPU integration must use collective
    /// fallback and must not expose partially restored state to inference.
    pub fn load(
        &mut self,
        key: &Hash,
        request: &[i32],
        mut sink: impl FnMut(Source, u64, &[u8]) -> Result<()>,
    ) -> Result<()> {
        self.memo.set(None);
        self.ready()?;
        ensure!(self.entries.contains_key(key), "SSD entry absent");
        ensure!(
            request.len() <= crate::cache_pages::CONTEXT
                && request
                    .iter()
                    .all(|&t| crate::vision_input::prompt_token(t)),
            "SSD request token bounds"
        );
        let h = self.token_header(key, Some(request))?;
        for (logical, p) in h.pages.iter().enumerate() {
            let (mut f, ph) = self.header(&p.key, true)?;
            ensure!(
                ph.checksum() == p.checksum,
                "SSD shared page digest changed"
            );
            fmt::read_body(&mut f, &ph, &mut self.stage, |offset, buf| {
                visit_page(logical, offset, buf, &mut sink)
            })?;
        }
        let segments = self.id.segments(h.n)?;
        let mut f = regular(&self.path(key, "entry"))?;
        fmt::read_body(&mut f, &h, &mut self.stage, |offset, buf| {
            for s in &segments {
                if s.role == Role::Tokens {
                    continue;
                }
                intersection(
                    offset,
                    buf.len(),
                    s.offset,
                    s.bytes,
                    |within, start, len| {
                        sink(Source::Layer(s.role), within, &buf[start..start + len])
                    },
                )?;
            }
            Ok(())
        })?;
        self.touch(key)?;
        Ok(())
    }
}
fn intersection(
    offset: u64,
    len: usize,
    start: u64,
    bytes: u64,
    mut f: impl FnMut(u64, usize, usize) -> Result<()>,
) -> Result<()> {
    let lo = offset.max(start);
    let hi = (offset + len as u64).min(start + bytes);
    if lo < hi {
        f(lo - start, (lo - offset) as usize, (hi - lo) as usize)?;
    }
    Ok(())
}
fn page_parts(mut visit: impl FnMut(usize, bool, u64, u64) -> Result<()>) -> Result<()> {
    let mut at = 0;
    for family in 0..11 {
        for (pooled, bytes) in [(false, 256 * 528), (true, 64 * 256)] {
            visit(family, pooled, at, bytes)?;
            at += bytes;
        }
    }
    Ok(())
}
pub(crate) fn fill_page(
    logical: usize,
    offset: u64,
    buf: &mut [u8],
    source: &mut impl FnMut(Source, u64, &mut [u8]) -> Result<()>,
) -> Result<()> {
    page_parts(|family, pooled, start, bytes| {
        intersection(offset, buf.len(), start, bytes, |within, i, n| {
            source(
                Source::Page {
                    logical,
                    family,
                    pooled,
                },
                within,
                &mut buf[i..i + n],
            )
        })
    })
}
fn visit_page(
    logical: usize,
    offset: u64,
    buf: &[u8],
    sink: &mut impl FnMut(Source, u64, &[u8]) -> Result<()>,
) -> Result<()> {
    page_parts(|family, pooled, start, bytes| {
        intersection(offset, buf.len(), start, bytes, |within, i, n| {
            sink(
                Source::Page {
                    logical,
                    family,
                    pooled,
                },
                within,
                &buf[i..i + n],
            )
        })
    })
}

fn fixture_identity(rank: u32) -> Result<Identity> {
    Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [5; 32], rank, 1024)
}
fn pattern(source: Source, offset: u64, bytes: &mut [u8], entry: &Hash, pages: &[Hash], rank: u32) {
    let (code, key) = match source {
        Source::Page {
            logical,
            family,
            pooled,
        } => (family * 2 + usize::from(pooled), pages[logical][0]),
        Source::Layer(role) => {
            let code = match role {
                Role::Conv(l) => 32 + l * 2,
                Role::Recurrent(l) => 33 + l * 2,
                Role::IndexKeys(l) => 128 + l * 2,
                Role::IndexGates(l) => 129 + l * 2,
                Role::TailLatent(f) => 224 + f * 2,
                Role::TailPool(f) => 225 + f * 2,
                Role::DraftMetadata => 247,
                Role::DraftKv(i) => 248 + i,
                Role::Tokens => unreachable!(),
            };
            (code, entry[0])
        }
    };
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = ((offset + i as u64) % 251) as u8 ^ code as u8 ^ key ^ rank as u8;
    }
}
fn save_fixture(s: &mut Store, ids: &[i32]) -> Result<Outcome> {
    let (entry, pages) = fmt::keys(&s.id.namespace, ids)?;
    let rank = s.id.rank;
    s.save(ids, false, |source, offset, buf| {
        pattern(source, offset, buf, &entry, &pages, rank);
        Ok(())
    })
}
fn check_fixture(s: &mut Store, ids: &[i32]) -> Result<u64> {
    let (entry, pages) = fmt::keys(&s.id.namespace, ids)?;
    let mut request = ids.to_vec();
    request.extend([8, 9, 10]);
    let rank = s.id.rank;
    ensure!(s.lookup(&request)? == Some(entry), "fixture lookup");
    let mut bytes = 0;
    let mut expected = vec![0; CHUNK];
    s.load(&entry, &request, |source, offset, buf| {
        pattern(
            source,
            offset,
            &mut expected[..buf.len()],
            &entry,
            &pages,
            rank,
        );
        ensure!(
            buf == &expected[..buf.len()],
            "fixture restored field bytes"
        );
        bytes += buf.len() as u64;
        Ok(())
    })?;
    let h = Header::new(
        &s.id,
        entry,
        ids.len(),
        0,
        pages
            .iter()
            .map(|&key| PageRef {
                key,
                checksum: [0; 32],
            })
            .collect(),
    )?;
    ensure!(
        bytes == h.body_bytes - ids.len() as u64 * 4 + pages.len() as u64 * fmt::PAGE_BYTES,
        "fixture complete schema coverage"
    );
    Ok(bytes)
}
/// CPU/disk component qualification; synthetic bytes, never a model-quality claim.
pub fn probe(root: &Path, rank: u32) -> Result<serde_json::Value> {
    ensure!(!root.exists(), "probe requires a new directory");
    let id = fixture_identity(rank)?;
    let budget = 100 << 20;
    let mut s = Store::open(root, id.clone(), budget)?;
    let a = vec![20; 256];
    let mut b = a.clone();
    b.extend(vec![21; 64]);
    let mut c = b.clone();
    c.extend(vec![22; 64]);
    ensure!(save_fixture(&mut s, &a)? == Outcome::Saved, "first save");
    let full = check_fixture(&mut s, &a)?;
    ensure!(
        save_fixture(&mut s, &a)? == Outcome::Duplicate,
        "duplicate save"
    );
    ensure!(save_fixture(&mut s, &b)? == Outcome::Saved, "branch save");
    check_fixture(&mut s, &b)?;
    ensure!(
        save_fixture(&mut s, &c)? == Outcome::Saved,
        "shared branch save"
    );
    let partial = check_fixture(&mut s, &c)?;
    ensure!(
        s.len() == 2 && s.pages.len() == 1,
        "bounded eviction/shared page"
    );
    let bkey = fmt::keys(&id.namespace, &b)?.0;
    let ckey = fmt::keys(&id.namespace, &c)?.0;
    let page = fmt::keys(&id.namespace, &c)?.1[0];
    let before = s.bytes();
    let order = s.order();
    ensure!(order == [bkey, ckey], "durable LRU order");
    ensure!(
        Store::open(root, id.clone(), budget).is_err(),
        "exclusive writer"
    );
    drop(s);
    let mut s = Store::open(root, id.clone(), budget)?;
    ensure!(s.order() == order && s.bytes() == before, "restart catalog");
    check_fixture(&mut s, &b)?;
    check_fixture(&mut s, &c)?;
    drop(s);
    let mut wrong = id.clone();
    wrong.rank = (rank + 1) % 4;
    ensure!(Store::open(root, wrong, budget).is_err(), "wrong rank root");
    let mut s = Store::open(root, id.clone(), budget)?;
    ensure!(
        s.save(&c, true, |_, _, _| Ok(())).is_err(),
        "mixed rejected"
    );
    s.retain_order(&[ckey])?;
    ensure!(s.len() == 1 && s.pages.len() == 1, "intersection retention");
    check_fixture(&mut s, &c)?;
    // Late private corruption is detected even after token/header checks passed.
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(s.path(&ckey, "entry"))?;
    f.seek(SeekFrom::End(-2))?;
    let mut one = [0];
    f.read_exact(&mut one)?;
    one[0] ^= 1;
    f.seek(SeekFrom::End(-2))?;
    f.write_all(&one)?;
    f.sync_all()?;
    drop(f);
    let mut request = c.clone();
    request.push(1);
    ensure!(
        s.load(&ckey, &request, |_, _, _| Ok(())).is_err(),
        "late private corruption"
    );
    s.forget(&ckey)?;
    ensure!(s.bytes() == ROOT_BYTES, "last reference reclamation");
    ensure!(
        save_fixture(&mut s, &c)? == Outcome::Saved,
        "resave after corruption"
    );
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(s.path(&page, "page"))?;
    f.seek(SeekFrom::End(-1))?;
    f.read_exact(&mut one)?;
    one[0] ^= 1;
    f.seek(SeekFrom::End(-1))?;
    f.write_all(&one)?;
    f.sync_all()?;
    drop(f);
    ensure!(
        s.load(&ckey, &request, |_, _, _| Ok(())).is_err(),
        "page corruption"
    );
    // Missing dependencies are removed on reopen, together with orphan payloads.
    fs::remove_file(s.path(&page, "page"))?;
    drop(s);
    let mut s = Store::open(root, id.clone(), budget)?;
    ensure!(
        s.len() == 0 && s.bytes() == ROOT_BYTES,
        "missing page restart cleanup"
    );
    let mut calls = 0;
    let interrupted = s.save(&c, false, |_, _, buf| {
        calls += 1;
        if calls > 45 {
            anyhow::bail!("injected staging failure");
        }
        buf.fill(7);
        Ok(())
    });
    ensure!(
        interrupted.is_err() && !s.healthy,
        "failed write quarantines mutator"
    );
    ensure!(
        fs::read_dir(root.join("entries"))?
            .filter_map(|p| p.ok())
            .any(|p| p.path().extension().is_some_and(|x| x == "tmp")
                && p.metadata().is_ok_and(|m| m.len() > CHUNK as u64)),
        "interrupted body reached disk"
    );
    drop(s);
    let mut s = Store::open(root, id.clone(), budget)?;
    ensure!(
        s.len() == 0 && s.bytes() == ROOT_BYTES,
        "interrupted write/orphan cleanup"
    );
    save_fixture(&mut s, &c)?;
    let final_bytes = check_fixture(&mut s, &c)?;
    let retained = s.bytes();
    let metadata = s.metadata_bytes();
    drop(s);
    // Shrinking the capacity changes retention, not byte compatibility.
    let mut tiny = Store::open(root, id.clone(), ROOT_BYTES)?;
    ensure!(
        tiny.len() == 0 && save_fixture(&mut tiny, &c)? == Outcome::Skipped,
        "capacity shrink/oversize skip"
    );
    drop(tiny);
    let large = fmt::segments(crate::cache_pages::CONTEXT - 64, 1024)?;
    Ok(
        serde_json::json!({"component":"typed_ssd_store_v1","rank":rank,"namespace":hex(&id.namespace),"synthetic_payloads":true,"gpu_integration":false,
        "budget_bytes":budget,"retained_bytes_before_shrink":retained,"catalog_allowance_bytes":metadata,"stage_bytes":CHUNK,
        "full_page_tail_checked_bytes":full,"partial_tail_checked_bytes":partial,"final_reload_checked_bytes":final_bytes,
        "schema_segments":large.len(),"max_prefix_tokens":crate::cache_pages::CONTEXT-64,
        "passed":["complete_schema","duplicate","shared_branch","lru_eviction","exact_lookup","exclusive_owner","restart","mixed_reject","wrong_rank","intersection_retention","private_corruption","page_corruption","missing_page_cleanup","interrupted_write_cleanup","budget_shrink","oversize_skip"],"final_entries":0}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn temporary() -> PathBuf {
        std::env::temp_dir().join(format!(
            "alaska-rust-ssd-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[test]
    fn disk_contract() -> Result<()> {
        let root = temporary();
        let result = probe(&root, 0);
        fs::remove_dir_all(&root)?;
        result?;
        Ok(())
    }
    #[test]
    fn foreground_view_is_exact_bounded_and_never_mutates_shared_files() -> Result<()> {
        let root = temporary();
        let id = fixture_identity(0)?;
        let tokens = vec![19; 320];
        let (key, _) = fmt::keys(&id.namespace, &tokens)?;
        let mut store = Store::open(&root, id.clone(), 128 << 20)?;
        save_fixture(&mut store, &tokens)?;
        store.touch(&key)?;
        let touch = fs::read(store.path(&key, "touch"))?;
        let original = store.digest();
        let mut view = store.reader_view()?;
        assert_eq!(view.metadata_bytes(), store.metadata_bytes());
        assert!(view.metadata_bytes() <= META_BUDGET);
        assert!(view.reader_view().is_err());
        assert!(save_fixture(&mut view, &tokens).is_err());
        assert!(view.retain_order(&[]).is_err());
        check_fixture(&mut view, &tokens)?;
        assert_eq!(fs::read(store.path(&key, "touch"))?, touch);
        assert_eq!(store.digest(), original);
        view.forget(&key)?;
        assert_eq!(view.len(), 0);
        assert!(store.path(&key, "entry").is_file());
        check_fixture(&mut store, &tokens)?;
        drop(view);
        let mut stale = store.reader_view()?;
        store.forget(&key)?;
        let mut request = tokens.clone();
        request.push(20);
        assert_eq!(stale.hint(&request)?, Some(tokens.len()));
        assert_eq!(
            stale.lookup(&request)?,
            None,
            "evicted file must become a checked miss"
        );
        drop(store);
        assert!(
            Store::open(&root, id.clone(), 128 << 20).is_err(),
            "view retains original exclusive lock"
        );
        drop(stale);
        drop(Store::open(&root, id, 128 << 20)?);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn speculative_companion_survives_restart_and_is_charged() -> Result<()> {
        let root = temporary();
        let plain = fixture_identity(0)?;
        let id = plain.clone().with_speculation([7; 32]);
        let ids = vec![13; 2112];
        let (key, _) = fmt::keys(&id.namespace, &ids)?;
        let mut store = Store::open(&root, id.clone(), 96 << 20)?;
        assert_eq!(save_fixture(&mut store, &ids)?, Outcome::Saved);
        let (_, header) = store.header(&key, false)?;
        assert_eq!(
            header.body_bytes - fmt::body_bytes(ids.len(), &plain)?,
            64 + 2047 * 5120
        );
        let charged = store.bytes();
        check_fixture(&mut store, &ids)?;
        drop(store);
        assert!(Store::open(&root, plain, 96 << 20).is_err());
        let mut reopened = Store::open(&root, id, 96 << 20)?;
        assert_eq!(reopened.bytes(), charged);
        check_fixture(&mut reopened, &ids)?;
        drop(reopened);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn reject_foreign_root() -> Result<()> {
        let root = temporary();
        fs::create_dir(&root)?;
        fs::write(root.join("unrelated.tfs"), b"keep")?;
        ensure!(
            Store::open(&root, fixture_identity(0)?, 100 << 20).is_err(),
            "unmarked root"
        );
        ensure!(
            fs::read(root.join("unrelated.tfs"))? == b"keep",
            "foreign file preserved"
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn restart_smaller_budget_preserves_newest_shared_page() -> Result<()> {
        let root = temporary();
        let id = fixture_identity(0)?;
        let mut s = Store::open(&root, id.clone(), 150 << 20)?;
        let a = vec![5; 320];
        let mut b = a.clone();
        b.extend(vec![6; 64]);
        let mut c = b.clone();
        c.extend(vec![7; 64]);
        for ids in [&a, &b, &c] {
            save_fixture(&mut s, ids)?;
        }
        drop(s);
        let mut s = Store::open(&root, id, 55 << 20)?;
        ensure!(s.len() == 1 && s.pages.len() == 1, "restart capacity trim");
        check_fixture(&mut s, &c)?;
        drop(s);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn oversize_preserves_entries_and_exact_tokens_are_checked() -> Result<()> {
        let root = temporary();
        let id = fixture_identity(0)?;
        let mut s = Store::open(&root, id.clone(), 55 << 20)?;
        let ids = vec![5; 320];
        save_fixture(&mut s, &ids)?;
        let before = s.order();
        let bytes = s.bytes();
        let mut called = false;
        ensure!(
            s.save(&vec![6; 16384], false, |_, _, _| {
                called = true;
                Ok(())
            })? == Outcome::Skipped
                && !called,
            "oversize before staging"
        );
        ensure!(
            s.order() == before && s.bytes() == bytes,
            "oversize before eviction"
        );
        let key = before[0];
        let (mut f, mut h) = s.header(&key, false)?;
        f.seek(SeekFrom::Start(h.header_bytes() as u64))?;
        let mut chunk = vec![0; CHUNK];
        f.read_exact(&mut chunk)?;
        drop(f);
        chunk[..4].copy_from_slice(&6i32.to_le_bytes());
        h.sums[0] = Sha256::digest(&chunk).into();
        let mut f = OpenOptions::new().write(true).open(s.path(&key, "entry"))?;
        f.write_all(&h.encode(&id))?;
        f.write_all(&chunk)?;
        f.sync_all()?;
        drop(f);
        let mut request = ids;
        request.push(3);
        let mut consumed = false;
        ensure!(
            s.load(&key, &request, |_, _, _| {
                consumed = true;
                Ok(())
            })
            .is_err()
                && !consumed,
            "rehashed wrong token identity before sink"
        );
        ensure!(
            s.lookup(&request)?.is_none() && s.len() == 0,
            "bad lookup candidate discarded"
        );
        drop(s);
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
