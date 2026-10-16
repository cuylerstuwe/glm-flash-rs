//! CPU policy and ABI bounds for the optional all-layer KDA commit.
use anyhow::{Result, ensure};

pub(crate) const LAYERS: usize = 34;
pub(crate) const TABLES: usize = 8; // Four retained model slots, two parities.
pub(crate) const MAX_ROWS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy(bool);
impl Policy {
    pub(crate) fn parse(value: Option<&str>) -> Result<Self> {
        ensure!(
            matches!(value, None | Some("0") | Some("1")),
            "ALASKA_VERIFY_BATCH_KDA_COMMIT must be 0/1"
        );
        Ok(Self(value == Some("1")))
    }
    pub(crate) fn environment() -> Result<Self> {
        match std::env::var("ALASKA_VERIFY_BATCH_KDA_COMMIT") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(error.into()),
        }
    }
    pub(crate) fn enabled(self) -> bool {
        self.0
    }
    pub(crate) fn identity(self) -> [u8; 16] {
        let mut bytes = *b"ALKBTC01\0\0\0\0\0\0\0\0";
        bytes[8] = u8::from(self.0);
        bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    Unchanged,
    Proposal,
    Replay,
}
impl Source {
    pub(crate) fn new(rows: usize, keep: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && keep <= rows,
            "batched KDA commit row bounds"
        );
        Ok(if keep == 0 {
            Self::Unchanged
        } else if keep == rows {
            Self::Proposal
        } else {
            Self::Replay
        })
    }
}

/// `replay_slots_kernel` int64[8]: original state, separate replay destination,
/// k/v/g/b saves, and two zero layer strides. Every CUDA grid-y slot is one
/// independently allocated model layer, so grid-x has only the sixteen heads.
pub(crate) fn table(pointers: &[[u64; 6]]) -> Result<Vec<u8>> {
    ensure!(pointers.len() == LAYERS, "batched KDA layer count");
    for (layer, p) in pointers.iter().enumerate() {
        ensure!(
            pointers.iter().all(|other| other[0] != p[1])
                && pointers[..layer].iter().all(|other| other[1] != p[1]),
            "batched KDA destination overlaps input/another destination"
        );
    }
    let mut bytes = Vec::with_capacity(LAYERS * 8 * 8);
    for p in pointers {
        ensure!(
            p.iter().all(|ptr| *ptr != 0 && ptr % 2 == 0)
                && [p[0], p[1], p[2], p[4], p[5]]
                    .into_iter()
                    .all(|ptr| ptr % 4 == 0)
                && p[0] != p[1],
            "batched KDA pointer alignment/alias"
        );
        for value in p.iter().copied().chain([0, 0]) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_prefix_uses_the_correct_transaction_source() {
        for rows in 1..=MAX_ROWS {
            for keep in 0..=rows {
                let source = Source::new(rows, keep).unwrap();
                assert_eq!(source == Source::Unchanged, keep == 0);
                assert_eq!(source == Source::Proposal, keep == rows);
                assert_eq!(source == Source::Replay, keep > 0 && keep < rows);
                for start in 0..2 {
                    // Zero accepted rows leave the exact original parity live;
                    // accepted rows select parity from the *accepted* count.
                    let destination = (start + keep) % 2;
                    assert_eq!(destination == start, keep % 2 == 0);
                }
            }
        }
        for (rows, keep) in [(0, 0), (17, 0), (1, 2), (16, 17), (usize::MAX, 0)] {
            assert!(Source::new(rows, keep).is_err());
        }
    }

    #[test]
    fn disconnected_layer_addresses_preserve_the_pinned_abi() {
        let pointers: Vec<_> = (0..LAYERS as u64)
            .map(|i| std::array::from_fn(|j| 0x1000 + i * 0x11000 + j as u64 * 0x100))
            .collect();
        let packed = table(&pointers).unwrap();
        assert_eq!(packed.len(), LAYERS * 64);
        for (actual, expected) in packed.as_chunks::<64>().0.iter().zip(&pointers) {
            let actual: Vec<_> = actual
                .as_chunks::<8>()
                .0
                .iter()
                .map(|x| u64::from_le_bytes(*x))
                .collect();
            assert_eq!(&actual[..6], expected);
            assert_eq!(&actual[6..], &[0, 0]);
        }
        assert!(table(&pointers[..LAYERS - 1]).is_err());
        let mut bad = pointers.clone();
        bad[17][1] = bad[17][0];
        assert!(table(&bad).is_err());
        bad[17][1] = 0;
        assert!(table(&bad).is_err());
        bad[17][1] = 0x1001;
        assert!(table(&bad).is_err());
        bad[17][1] = bad[5][1];
        assert!(table(&bad).is_err());
        bad[17][1] = bad[5][0];
        assert!(table(&bad).is_err());
    }

    #[test]
    fn default_off_and_strict_environment_identity() {
        let off = Policy::parse(None).unwrap();
        assert!(!off.enabled());
        assert_eq!(off, Policy::parse(Some("0")).unwrap());
        let on = Policy::parse(Some("1")).unwrap();
        assert!(on.enabled());
        assert_ne!(off.identity(), on.identity());
        for bad in ["", "true", "2", " 1", "1 "] {
            assert!(Policy::parse(Some(bad)).is_err());
        }
    }
}
