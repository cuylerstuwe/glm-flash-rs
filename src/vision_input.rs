//! Typed internal image rows. Public raw prompts and generated tokens remain vocabulary-only.
//! Image preparation binds these rows to model embeddings through validated virtual IDs.
use anyhow::{Result, ensure};
use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

pub(crate) const BASE: i32 = 1 << 24;
const SPAN: u64 = i32::MAX as u64 - BASE as u64;
pub(crate) const MAX_ROWS: usize = 8000;
const MAX_IMAGES: usize = 8;
const VOCAB: i32 = 154880;
const IMAGE: i32 = 154854;
const BEGIN: i32 = 154830;
const END: i32 = 154831;

/// Public tokenization uses repeated vocabulary image markers, never private
/// virtual embedding IDs. Counting needs no GPU tower evaluation.
pub(crate) fn counted(text: &[i32], counts: &[usize]) -> Result<Arc<[i32]>> {
    ensure!(
        counts.len() <= MAX_IMAGES && counts.iter().all(|n| (1..=MAX_ROWS).contains(n)),
        "image count bound"
    );
    ensure!(
        text.len() <= crate::cache_pages::CONTEXT && text.iter().all(|id| (0..VOCAB).contains(id)),
        "image text bounds"
    );
    ensure!(
        text.iter().filter(|&&id| id == IMAGE).count() == counts.len(),
        "image placeholder count"
    );
    let total = text.len() - counts.len() + counts.iter().sum::<usize>();
    ensure!(
        total <= crate::cache_pages::CONTEXT,
        "expanded image prompt bound"
    );
    let mut ids = Vec::with_capacity(total);
    let mut next = 0;
    for (pos, &id) in text.iter().enumerate() {
        if id == IMAGE {
            ensure!(
                pos > 0 && text.get(pos - 1) == Some(&BEGIN) && text.get(pos + 1) == Some(&END),
                "unframed image placeholder"
            );
            ids.extend(std::iter::repeat_n(IMAGE, counts[next]));
            next += 1;
        } else {
            ids.push(id);
        }
    }
    Ok(ids.into())
}

/// A prompt token: a vocabulary ID or an image row's virtual ID.
pub(crate) fn prompt_token(id: i32) -> bool {
    (0..VOCAB).contains(&id) || id >= BASE
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct ImageKey(pub(crate) [u8; 32]);

/// Hash layout: SHA256(digest || salt_le32), BLAKE2b8(base || row_le32).
/// BLAKE2b8 is a configured8-byte digest, not truncated BLAKE2b512.
pub(crate) fn derive(key: ImageKey, rows: usize, salt: u32) -> Result<Vec<i32>> {
    ensure!((1..=MAX_ROWS).contains(&rows), "image row bound");
    let mut h = Sha256::new();
    Digest::update(&mut h, key.0);
    Digest::update(&mut h, salt.to_le_bytes());
    let base = h.finalize();
    Ok((0..rows)
        .map(|row| {
            let mut h = Blake2bVar::new(8).expect("fixed BLAKE2b length");
            h.update(&base);
            h.update(&(row as u32).to_le_bytes());
            let mut bytes = [0u8; 8];
            h.finalize_variable(&mut bytes)
                .expect("fixed BLAKE2b output");
            BASE + (u64::from_le_bytes(bytes) % SPAN) as i32
        })
        .collect())
}

#[derive(Clone, Debug)]
pub(crate) struct ImageRows {
    key: ImageKey,
    ids: Arc<[i32]>,
}
impl ImageRows {
    pub(crate) fn key(&self) -> ImageKey {
        self.key
    }
    pub(crate) fn ids(&self) -> &[i32] {
        &self.ids
    }
}

/// Bounded collision/salt registry. Recency indexes remain bounded too;
/// repeated hits never append an unbounded queue of stale LRU records.
pub(crate) struct Registry {
    owners: HashMap<i32, (ImageKey, u64)>,
    recency: BTreeMap<u64, i32>,
    salts: HashMap<ImageKey, (u32, u64)>,
    salt_recency: BTreeMap<u64, ImageKey>,
    cap: usize,
    tick: u64,
}
impl Registry {
    pub(crate) fn new(cap: usize) -> Result<Self> {
        ensure!((16..=262144).contains(&cap), "image ID registry bound");
        Ok(Self {
            owners: HashMap::new(),
            recency: BTreeMap::new(),
            salts: HashMap::new(),
            salt_recency: BTreeMap::new(),
            cap,
            tick: 0,
        })
    }
    pub(crate) fn register(&mut self, key: ImageKey, rows: usize) -> Result<ImageRows> {
        ensure!((1..=MAX_ROWS).contains(&rows), "image row bound");
        // Fail before mutation on the practically unreachable clock exhaustion.
        ensure!(
            self.tick < u64::MAX - rows as u64,
            "image registry clock exhausted"
        );
        let mut salt = self.salts.get(&key).map_or(0, |&(s, _)| s);
        let ids = loop {
            let ids = derive(key, rows, salt)?;
            let mut unique = ids.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() == ids.len()
                && ids
                    .iter()
                    .all(|id| self.owners.get(id).is_none_or(|&(owner, _)| owner == key))
            {
                break ids;
            }
            salt = salt
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("image ID salt exhausted"))?;
        };
        self.tick += 1;
        if let Some((_, old)) = self.salts.insert(key, (salt, self.tick)) {
            self.salt_recency.remove(&old);
        }
        self.salt_recency.insert(self.tick, key);
        while self.salts.len() > self.cap / 16 {
            let (_, old) = self.salt_recency.pop_first().expect("salt index");
            self.salts.remove(&old);
        }
        for &id in &ids {
            self.tick += 1;
            if let Some((_, old)) = self.owners.insert(id, (key, self.tick)) {
                self.recency.remove(&old);
            }
            self.recency.insert(self.tick, id);
        }
        while self.owners.len() > self.cap {
            let (_, old) = self.recency.pop_first().expect("owner index");
            self.owners.remove(&old);
        }
        Ok(ImageRows {
            key,
            ids: ids.into(),
        })
    }
}

/// Construction binds every non-vocabulary prompt token to its exact image and
/// row. It cannot be constructed from user-supplied virtual token IDs.
pub(crate) struct ImagePrompt {
    tokens: Arc<[i32]>,
    rows: BTreeMap<i32, (ImageKey, u32)>,
}
impl ImagePrompt {
    pub(crate) fn expand(text_tokens: &[i32], images: &[ImageRows]) -> Result<Self> {
        ensure!(images.len() <= MAX_IMAGES, "image count bound");
        ensure!(
            text_tokens.len() <= crate::cache_pages::CONTEXT,
            "image prompt bound"
        );
        ensure!(
            text_tokens.iter().all(|&id| (0..VOCAB).contains(&id)),
            "only vocabulary input may be expanded"
        );
        let count = text_tokens.iter().filter(|&&id| id == IMAGE).count();
        ensure!(count == images.len(), "image placeholder count");
        let total = text_tokens.len() - count + images.iter().map(|i| i.ids.len()).sum::<usize>();
        ensure!(
            total <= crate::cache_pages::CONTEXT,
            "expanded image prompt bound"
        );
        let mut rows = BTreeMap::new();
        for image in images {
            for (row, &id) in image.ids.iter().enumerate() {
                let value = (image.key, row as u32);
                if let Some(old) = rows.insert(id, value) {
                    ensure!(old == value, "image row ID collision in request");
                }
            }
        }
        let mut tokens = Vec::with_capacity(total);
        let mut next = 0;
        for (pos, &id) in text_tokens.iter().enumerate() {
            if id == IMAGE {
                ensure!(
                    pos > 0
                        && text_tokens.get(pos - 1) == Some(&BEGIN)
                        && text_tokens.get(pos + 1) == Some(&END),
                    "unframed image placeholder"
                );
                tokens.extend_from_slice(&images[next].ids);
                next += 1;
            } else {
                tokens.push(id);
            }
        }
        Ok(Self {
            tokens: tokens.into(),
            rows,
        })
    }
    pub(crate) fn tokens(&self) -> &[i32] {
        &self.tokens
    }
    pub(crate) fn row(&self, id: i32) -> Option<(ImageKey, u32)> {
        self.rows.get(&id).copied()
    }
    /// Cache identity includes the full image digest even for a prefix ending
    /// after only one image row; a31-bit row collision cannot alias two images.
    /// Encoder identity must pin weights/precision/layout when integrated.
    pub(crate) fn prefix_identity(&self, prefix: usize, encoder: [u8; 32]) -> Result<[u8; 32]> {
        ensure!(prefix <= self.tokens.len(), "image prefix bound");
        let mut h = Sha256::new();
        Digest::update(&mut h, b"alaska-image-prefix/1\0");
        Digest::update(&mut h, encoder);
        Digest::update(&mut h, (prefix as u64).to_le_bytes());
        for &id in &self.tokens[..prefix] {
            Digest::update(&mut h, id.to_le_bytes());
            if let Some((key, row)) = self.rows.get(&id) {
                Digest::update(&mut h, key.0);
                Digest::update(&mut h, row.to_le_bytes());
            }
        }
        Ok(h.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_counts_preserve_framing_and_hide_virtual_rows() {
        assert_eq!(
            counted(&[9, BEGIN, IMAGE, END, 10, BEGIN, IMAGE, END], &[2, 3])
                .unwrap()
                .as_ref(),
            &[
                9, BEGIN, IMAGE, IMAGE, END, 10, BEGIN, IMAGE, IMAGE, IMAGE, END
            ]
        );
        assert!(counted(&[IMAGE], &[1]).is_err());
        assert!(counted(&[BEGIN, IMAGE, END], &[0]).is_err());
        assert!(counted(&[BEGIN, IMAGE, END], &[MAX_ROWS + 1]).is_err());
        assert!(counted(&[BASE], &[]).is_err());
        assert!(counted(&[BEGIN, IMAGE, END], &[]).is_err());
    }
    #[test]
    fn blake2b8_byte_goldens() {
        let key = ImageKey(std::array::from_fn(|i| i as u8));
        // Node SHA256 + coreutils b2sum -l64, preserved in evidence/vision-audit.
        for (salt, expected) in [
            (0, [19188257, 1166730586, 1770947517, 2061782932]),
            (1, [82164248, 1051710326, 1717334600, 1929339862]),
            (u32::MAX, [779110398, 1109780407, 857987515, 1273283414]),
        ] {
            let ids = derive(key, 8000, salt).unwrap();
            for (row, value) in [0, 1, 15, 7999].into_iter().zip(expected) {
                assert_eq!(ids[row], value);
            }
        }
    }
    #[test]
    fn registry_collision_recovery_recency_and_bounds() {
        let mut r = Registry::new(32).unwrap();
        let key = ImageKey([1; 32]);
        let first = derive(key, 16, 0).unwrap()[0];
        r.owners.insert(first, (ImageKey([2; 32]), 0));
        r.recency.insert(0, first);
        let img = r.register(key, 16).unwrap();
        assert_eq!(img.ids(), derive(key, 16, 1).unwrap());
        for _ in 0..100 {
            assert_eq!(r.register(key, 16).unwrap().ids(), img.ids());
        }
        assert_eq!(r.owners.len(), r.recency.len());
        assert_eq!(r.salts.len(), r.salt_recency.len());
        for i in 3..20 {
            r.register(ImageKey([i; 32]), 16).unwrap();
        }
        assert!(
            r.owners.len() <= 32
                && r.recency.len() <= 32
                && r.salts.len() <= 2
                && r.salt_recency.len() <= 2
        );
        let before = (r.owners.len(), r.tick);
        assert!(r.register(key, 8001).is_err());
        assert_eq!(before, (r.owners.len(), r.tick));
    }
    #[test]
    fn typed_prompt_identity_and_raw_input_separation() {
        let mut r = Registry::new(256).unwrap();
        let a = r.register(ImageKey([1; 32]), 16).unwrap();
        let b = r.register(ImageKey([2; 32]), 16).unwrap();
        let prompt = [19, BEGIN, IMAGE, END, 20];
        let p = ImagePrompt::expand(&prompt, std::slice::from_ref(&a)).unwrap();
        assert_eq!(p.tokens().len(), 20);
        assert_eq!(p.row(a.ids()[0]), Some((a.key(), 0)));
        assert!(ImagePrompt::expand(&[a.ids()[0]], &[]).is_err());
        assert!(ImagePrompt::expand(&[IMAGE], std::slice::from_ref(&a)).is_err());
        assert!(ImagePrompt::expand(&prompt, &[]).is_err());
        let q = ImagePrompt::expand(&prompt, &[b]).unwrap();
        assert_eq!(
            p.prefix_identity(2, [0; 32]).unwrap(),
            q.prefix_identity(2, [0; 32]).unwrap()
        );
        assert_ne!(
            p.prefix_identity(3, [0; 32]).unwrap(),
            q.prefix_identity(3, [0; 32]).unwrap()
        );
        assert_ne!(
            p.prefix_identity(3, [0; 32]).unwrap(),
            p.prefix_identity(3, [1; 32]).unwrap()
        );
        assert!(p.prefix_identity(21, [0; 32]).is_err());
        assert!(
            ImagePrompt::expand(&[BEGIN, IMAGE, END, BEGIN, IMAGE, END], &[a.clone(), a]).is_ok()
        );
    }
    #[test]
    fn full_image_digest_prevents_short_prefix_row_collision() {
        let a = ImageRows {
            key: ImageKey([1; 32]),
            ids: vec![BASE + 1].into(),
        };
        let b = ImageRows {
            key: ImageKey([2; 32]),
            ids: vec![BASE + 1].into(),
        };
        let p = ImagePrompt::expand(&[BEGIN, IMAGE, END], std::slice::from_ref(&a)).unwrap();
        let q = ImagePrompt::expand(&[BEGIN, IMAGE, END], std::slice::from_ref(&b)).unwrap();
        assert_eq!(p.tokens(), q.tokens());
        assert_ne!(
            p.prefix_identity(2, [0; 32]).unwrap(),
            q.prefix_identity(2, [0; 32]).unwrap()
        );
        assert!(ImagePrompt::expand(&[BEGIN, IMAGE, END, BEGIN, IMAGE, END], &[a, b]).is_err());
    }
}
