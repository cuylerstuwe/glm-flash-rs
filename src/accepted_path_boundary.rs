//! CPU decision record for the existing sampled-verifier publication boundary.
//! A discard proof exists only after all ranks reached the same uncommitted
//! target decision and reported either success or an explicitly typed parser
//! failure. CUDA/transport errors and invalid clipping never yield this proof.
use anyhow::{Result, ensure};

#[derive(Debug)]
pub(in crate::model) struct ParserFailure;
impl std::fmt::Display for ParserFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("accepted output parser rejected the request")
    }
}
impl std::error::Error for ParserFailure {}

#[derive(Debug)]
pub(in crate::model) struct DiscardProof {
    slot: usize,
    cursor: usize,
}
impl std::fmt::Display for DiscardProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("all ranks rejected uncommitted output parsing")
    }
}
impl std::error::Error for DiscardProof {}
impl DiscardProof {
    pub(in crate::model) fn matches(&self, slot: usize, cursor: usize) -> bool {
        self.slot == slot && self.cursor == cursor
    }
}
pub(super) enum Decision {
    Keep(usize),
    Discard(DiscardProof),
}
pub(super) fn record(digest: [u8; 32], clipped: &Result<usize>) -> [u8; 64] {
    let mut record = [0; 64];
    record[..32].copy_from_slice(&digest);
    record[32..36].copy_from_slice(&820u32.to_le_bytes());
    match clipped {
        Ok(keep) if (1..=16).contains(keep) => {
            record[36] = 1;
            record[37] = *keep as u8;
        }
        Err(error) if error.is::<ParserFailure>() => record[36] = 2,
        _ => {}
    }
    record
}
pub(super) fn decide(
    all: &[[u8; 64]],
    local: &[u8; 64],
    slot: usize,
    cursor: usize,
) -> Result<Decision> {
    ensure!(
        all.len() == 4
            && all
                .iter()
                .all(|r| r[..36] == local[..36] && r[38..] == [0; 26]),
        "accepted-path identity disagreement"
    );
    ensure!(
        all.iter().all(|r| match r[36] {
            1 => (1..=16).contains(&r[37]),
            2 => r[37] == 0,
            _ => false,
        }),
        "accepted-path invalid clipping or unclassified failure"
    );
    if all.iter().any(|r| r[36] == 2) {
        return Ok(Decision::Discard(DiscardProof { slot, cursor }));
    }
    ensure!(
        all.iter().all(|r| r[37] == all[0][37]),
        "accepted-path keep disagreement"
    );
    Ok(Decision::Keep(all[0][37] as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn common_success_and_explicit_parser_failures() -> Result<()> {
        let ok = record([7; 32], &Ok(4));
        assert!(matches!(decide(&[ok; 4], &ok, 2, 8190)?, Decision::Keep(4)));
        let failed = record([7; 32], &Err(ParserFailure.into()));
        for count in [1, 4] {
            let mut all = [ok; 4];
            all[..count].fill(failed);
            let Decision::Discard(proof) = decide(&all, &ok, 2, 8190)? else {
                panic!("missing proof")
            };
            assert!(proof.matches(2, 8190));
            assert!(!proof.matches(1, 8190) && !proof.matches(2, 8191));
        }
        Ok(())
    }
    #[test]
    fn malformed_disagreement_and_arbitrary_errors_never_authorize_discard() {
        let ok = record([7; 32], &Ok(4));
        for bad in [
            record([8; 32], &Ok(4)),
            record([7; 32], &Ok(0)),
            record([7; 32], &Ok(17)),
            record([7; 32], &Ok(3)),
            record([7; 32], &Err(anyhow::anyhow!("native failure"))),
        ] {
            let mut all = [ok; 4];
            all[2] = bad;
            assert!(decide(&all, &ok, 0, 128).is_err());
        }
        let mut all = [record([7; 32], &Err(ParserFailure.into())); 4];
        all[1] = record([7; 32], &Err(anyhow::anyhow!("native failure")));
        assert!(decide(&all, &ok, 0, 128).is_err());
        assert!(decide(&[ok; 3], &ok, 0, 128).is_err());
    }
}
