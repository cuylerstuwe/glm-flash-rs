//! Bounded serial-exact input batches for off-grid mixed continuations. Reuses
//! the target verifier's existing row-invariant kernels and full-accept commit.
//! This is forced prompt input, never a speculative output or a sampling draw.
use super::*;

fn rows(cursor: usize, tail: &[i32], backlog: usize) -> Result<usize> {
    ensure!(
        cursor < CONTEXT
            && !cursor.is_multiple_of(64)
            && !tail.is_empty()
            && tail.len() <= CONTEXT - cursor
            && backlog <= crate::accepted_features::ROWS,
        "exact prompt plan bounds"
    );
    let available = crate::accepted_features::ROWS - backlog;
    // Serial appends flush only when the 64-row backlog is already full.
    // Never cross that boundary or flush early: preserve drafter grouping.
    let room = if available == 0 {
        crate::accepted_features::ROWS
    } else {
        available
    };
    let n = (64 - cursor % 64)
        .min(crate::kda_window::MAX_ROWS)
        .min(room);
    // Image overwrites stay on the qualified eager single-token path. A later
    // image also terminates this text batch before its virtual ID is consumed.
    Ok(tail
        .iter()
        .take(n)
        .take_while(|&&t| (0..154880).contains(&t))
        .count()
        .max(1))
}

impl Long<'_> {
    pub(in crate::model) fn exact_prompt_rows(&self, tail: &[i32]) -> Result<usize> {
        self.ready_active()?;
        let s = &self.sessions[self.active];
        ensure!(
            s.mixed && s.prompt_end.checked_sub(s.tokens.len()) == Some(tail.len()),
            "exact mixed prompt plan"
        );
        rows(
            s.tokens.len(),
            tail,
            self.speculation_backlog_rows(self.active)?,
        )
    }

    pub(in crate::model) fn exact_prompt_batch(
        &mut self,
        control: &Control,
        dispatch: [u8; 32],
        tokens: &[i32],
    ) -> Result<Option<i32>> {
        let mut hash = Sha256::new();
        hash.update(b"exact-mixed-prompt-batch-v1");
        hash.update(dispatch);
        hash.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hash.update(token.to_le_bytes());
        }
        // Flush is local drafter work. As for normal speculative dispatch,
        // retain any error until every rank reaches the common preparation
        // fence before target collectives can start.
        let prepared = (|| {
            self.ready_active()?;
            let s = &self.sessions[self.active];
            ensure!(
                s.mixed
                    && (2..=crate::kda_window::MAX_ROWS).contains(&tokens.len())
                    && s.tokens.len() + tokens.len() <= s.prompt_end
                    && tokens.iter().all(|t| (0..154880).contains(t)),
                "exact prompt batch input"
            );
            let allowed = rows(
                s.tokens.len(),
                tokens,
                self.speculation_backlog_rows(self.active)?,
            )?;
            ensure!(
                allowed == tokens.len(),
                "exact prompt batch crosses arithmetic boundary"
            );
            self.healthy = false;
            self.with_speculation(|owner, long| {
                let slot = long.active;
                owner.bind(long, slot)?;
                ensure!(
                    owner.slots[slot].status().end == long.sessions[slot].tokens.len(),
                    "exact prompt backlog cursor"
                );
                if owner.slots[slot].available(owner.slots[slot].lease())? == 0 {
                    owner.flush(long, slot)?;
                }
                Ok(())
            })?
            .context("exact prompt owner unavailable")
        })();
        self.agree_speculation_work(control, 872, hash.finalize().into(), prepared)?;
        let result = self.with_speculation(|owner, long| {
            let slot = long.active;
            let position = long.sessions[slot].tokens.len();
            let outcome = owner.verifier.execute_with_eos(
                long,
                tokens,
                Decision::Forced(tokens.len()),
                false,
                |_| anyhow::bail!("forced prompt must not invoke output clipping"),
            )?;
            long.healthy = false;
            ensure!(
                outcome.keep == tokens.len() && outcome.sampled.is_empty(),
                "forced prompt unexpectedly sampled"
            );
            let taps = owner.verifier.accepted_taps()?;
            let backlog = &mut owner.slots[slot];
            backlog.append(
                &long.engine,
                backlog.lease(),
                position,
                &taps.rows,
                0,
                tokens.len(),
            )?;
            Ok(long.sessions[slot].pending)
        });
        // The enclosing scheduler's normal model-work agreement drains the
        // appended features and checks cursor/owner agreement before publication.
        self.healthy = false;
        result?.context("exact prompt owner unavailable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planning_preserves_grid_backlog_and_image_boundaries() -> Result<()> {
        for cursor in 1..256 {
            if cursor % 64 == 0 {
                continue;
            }
            for backlog in 0..=64 {
                for length in [1, 2, 15, 16, 17, 37, 63, 64, 80] {
                    let tail = vec![17; length];
                    let n = rows(cursor, &tail, backlog)?;
                    assert!(n > 0 && n <= 16 && n <= length && n <= 64 - cursor % 64);
                    assert!(backlog == 64 || n <= 64 - backlog);
                }
            }
        }
        assert_eq!(rows(25, &[1 << 24, 17, 18], 0)?, 1);
        assert_eq!(rows(25, &[17, 18, 1 << 24, 19], 0)?, 2);
        assert_eq!(rows(25, &[17; 37], 63)?, 1);
        assert_eq!(rows(25, &[17; 37], 64)?, 16);
        assert!(rows(64, &[17], 0).is_err());
        assert!(rows(25, &[17], 65).is_err());
        Ok(())
    }
}
