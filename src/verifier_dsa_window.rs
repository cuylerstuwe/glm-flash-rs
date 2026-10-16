//! Conservative verifier scheduling: only wholly sparse windows are batched.
//! Existing dense decode kernels have a fixed one-row ABI and remain untouched.
use anyhow::{Result, ensure};
const CONTEXT: usize = 1_048_576;
const MAX_ROWS: usize = 16;
const FIRST_SPARSE: usize = 2051;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy(bool);
impl Policy {
    pub(crate) fn parse(value: Option<&str>) -> Result<Self> {
        ensure!(
            matches!(value, None | Some("0") | Some("1")),
            "ALASKA_VERIFY_SPARSE_WINDOW must be 0/1"
        );
        Ok(Self(value == Some("1")))
    }
    pub(crate) fn environment() -> Result<Self> {
        match std::env::var("ALASKA_VERIFY_SPARSE_WINDOW") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(error.into()),
        }
    }
    pub(crate) fn identity(self) -> [u8; 16] {
        let mut bytes = *b"ALDSAW01\0\0\0\0\0\0\0\0";
        bytes[8] = u8::from(self.0);
        bytes
    }
    pub(crate) fn enabled(self) -> bool {
        self.0
    }
    pub(crate) fn window(self, position: usize, rows: usize) -> Result<bool> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && position.checked_add(rows).is_some_and(|end| end <= CONTEXT),
            "verifier sparse window bounds"
        );
        Ok(self.0 && position >= FIRST_SPARSE)
    }
}

#[derive(Default, serde::Serialize)]
pub(crate) struct Stats {
    pub(crate) window_rounds: u64,
    pub(crate) window_rows: u64,
    pub(crate) fallback_rounds: u64,
    pub(crate) fallback_rows: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_mixed_and_sparse_windows_choose_a_safe_abi() -> Result<()> {
        let on = Policy::parse(Some("1"))?;
        let off = Policy::parse(None)?;
        for position in [0, 128, 2036, 2047, 2050, 2051, 2052, 8190, CONTEXT - 16] {
            for rows in 1..=16 {
                let every_sparse =
                    (position..position + rows).all(|absolute| (absolute + 1) / 4 > 512);
                assert_eq!(on.window(position, rows)?, every_sparse);
                assert!(!off.window(position, rows)?);
            }
        }
        assert!(!on.window(2050, 16)?);
        assert!(on.window(2051, 1)?);
        for (position, rows) in [
            (0, 0),
            (0, 17),
            (CONTEXT, 1),
            (CONTEXT - 15, 16),
            (usize::MAX, 1),
        ] {
            assert!(on.window(position, rows).is_err());
        }
        Ok(())
    }

    #[test]
    fn window_pool_publication_cannot_expose_future_keys() {
        // The window writes all new rows before pooling, but scoring sees only
        // complete pools at each row's own position. Compare that visible set
        // against a serial prefix, including pool/bucket/ring/context edges.
        for position in [2051, 2052, 4093, 4094, 8190, 8191, 16382, CONTEXT - 16] {
            for rows in 1..=16 {
                let window_end = position + rows;
                let completed = window_end / 4;
                let first_new_pool = position / 4;
                for pool in first_new_pool..completed {
                    for key in pool * 4..pool * 4 + 4 {
                        assert!(key + 3 >= position && key < window_end);
                    }
                }
                for row in 0..rows {
                    // Earlier complete pools are identical in both schedules;
                    // compare only pools this window can newly complete.
                    let serial_visible = (first_new_pool..completed)
                        .filter(|pool| pool * 4 + 3 <= position + row)
                        .collect::<Vec<_>>();
                    let window_visible =
                        (first_new_pool..(position + row + 1) / 4).collect::<Vec<_>>();
                    assert_eq!(window_visible, serial_visible);
                    // Every new key and up to three previous keys fit in the
                    // minimum 256-row ring without aliasing another needed row.
                    let start = position - position % 4;
                    let slots: std::collections::BTreeSet<_> =
                        (start..window_end).map(|key| key % 256).collect();
                    assert_eq!(slots.len(), window_end - start);
                }
            }
        }
    }

    #[test]
    fn settings_are_default_off_and_have_distinct_agreed_identities() -> Result<()> {
        let off = Policy::parse(None)?;
        assert!(!off.enabled());
        assert_eq!(off, Policy::parse(Some("0"))?);
        assert_ne!(off.identity(), Policy::parse(Some("1"))?.identity());
        for bad in ["", "yes", "false", "2", " 1", "1 "] {
            assert!(Policy::parse(Some(bad)).is_err());
        }
        Ok(())
    }
}
