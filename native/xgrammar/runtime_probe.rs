#[path = "../../src/grammar_native.rs"]
#[allow(dead_code)]
mod grammar_native;
#[path = "../../src/grammar_runtime.rs"]
mod grammar_runtime;
use anyhow::{Result, ensure};
use grammar_native::{Kind, Native};
use grammar_runtime::{Constraint, Pool, WORDS};
use std::{path::Path, sync::Arc, time::Duration};
fn main() -> Result<()> {
    let path = std::env::var("ALASKA_XGRAMMAR_LIBRARY")?;
    let sha = std::env::var("ALASKA_XGRAMMAR_SHA256")?;
    let native = Native::open(Path::new(&path), &sha)?;
    // The canonical tokenizer-view proof is separate; these small explicit IDs
    // make every temporary accept/rollback/think/EOS boundary easy to audit.
    let mut vocab = vec![Vec::new(); grammar_runtime::VOCAB];
    for byte in 1..128 {
        vocab[byte as usize] = vec![byte as u8];
    }
    let eos = 128;
    let mut compiler = native.compiler(&vocab, false, false, &[eos])?;
    let grammar = compiler.compile(Kind::Ebnf, "root ::= \"abcdef\"")?;
    let never: Arc<[bool]> = vocab
        .iter()
        .enumerate()
        .map(|(id, v)| v.is_empty() && id != eos as usize)
        .collect::<Vec<_>>()
        .into();
    let make = |active| {
        Constraint::new(
            grammar.matcher()?,
            active,
            129,
            never.clone(),
            131072,
            [42; 32],
        )
    };
    let mut serial = make(true)?;
    serial.advance(&[97])?;
    let mut expected = Vec::new();
    for token in 97..=102 {
        let win = serial.cut(&[token])?;
        let masks = serial.fill(win)?;
        let bits = masks.row(0).unwrap().to_vec();
        ensure!(bits.len() == WORDS, "mask width");
        expected.push(bits);
        serial.advance(&[if token == 102 { eos } else { token + 1 }])?;
    }
    ensure!(
        serial.finished()? && serial.emitted() == 7,
        "serial terminal progress"
    );
    let mut pool = Pool::new(4)?;
    let mut jobs = Vec::new();
    for _ in 0..4 {
        let mut constraint = make(true)?;
        constraint.advance(&[97])?;
        let before = constraint.digest();
        let win = constraint.cut(&[97, 98, 99, 100, 101, 102, eos, 120])?;
        ensure!(
            win.tokens() == [97, 98, 99, 100, 101, 102]
                && win.rows() == [0, 1, 2, 3, 4, 5]
                && win.cut() == 2,
            "native EOS cut"
        );
        let job = pool
            .start(constraint, win)
            .map_err(|_| anyhow::anyhow!("native mask queue rejected"))?;
        jobs.push((job, before));
    }
    for (mut job, before) in jobs {
        let result = job
            .poll(Duration::from_secs(10))?
            .ok_or_else(|| anyhow::anyhow!("native worker stalled"))?;
        let mut constraint = result.constraint;
        let masks = result.masks?;
        ensure!(
            constraint.digest() == before && constraint.emitted() == 1,
            "native fill changed retained progress"
        );
        for (row, bits) in expected.iter().enumerate() {
            ensure!(
                masks.row(row) == Some(bits.as_slice()),
                "native serial/speculative masks differ"
            );
        }
        ensure!(
            masks.window().cut() == 2 && masks.digest() != [0; 32],
            "window proof missing"
        );
        constraint.advance(&[98, 99, 100, 101, 102, eos])?;
        ensure!(
            constraint.finished()? && constraint.digest() == serial.digest(),
            "native accepted state differs"
        );
    }
    let mut thinking = make(false)?;
    let win = thinking.cut(&[120, 121, 129, 97, 98, 99])?;
    ensure!(win.rows() == [2, 3, 4, 5], "native thought transition rows");
    let masks = thinking.fill(win)?;
    ensure!(
        masks.row(0).is_none() && masks.row(1).is_none() && !thinking.active(),
        "thinking mask leaked"
    );
    thinking.advance(&[120, 121, 129, 97])?;
    ensure!(
        thinking.active() && thinking.emitted() == 4,
        "forced closing token progress"
    );
    let win = thinking.cut(&[97, 98, 120, 99])?;
    ensure!(
        win.tokens() == [97, 98] && win.cut() == 2,
        "invalid native draft not cut"
    );
    let rejected = pool
        .start(thinking, win)
        .map_err(|_| anyhow::anyhow!("native queue rejected"))?;
    let mut rejected = rejected;
    let mut complete = rejected
        .poll(Duration::from_secs(10))?
        .ok_or_else(|| anyhow::anyhow!("native worker timeout"))?;
    complete.masks?;
    ensure!(
        complete.constraint.advance(&[120]).is_err() && complete.constraint.cut(&[97]).is_err(),
        "invalid chosen token did not poison owner"
    );
    pool.close();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !pool.retired()? {
        ensure!(
            std::time::Instant::now() < deadline,
            "native worker retirement timeout"
        );
        std::thread::yield_now();
    }
    ensure!(pool.in_flight() == 0, "native worker not drained");
    let mut restored = make(true)?;
    let win = restored.cut(&[0])?;
    let rejected = match pool.start(restored, win) {
        Err(value) => value,
        Ok(_) => anyhow::bail!("closed worker pool accepted"),
    };
    restored = rejected.constraint;
    ensure!(
        restored.fill(rejected.window).is_ok(),
        "rejected owner lost"
    );
    println!(
        "{{\"all\":true,\"native_workers\":4,\"vocab_slots\":154880,\"mask_words\":4840,\"serial_vs_chain_rows\":24,\"eos_cut\":true,\"thinking_gating\":true,\"accepted_progress\":true,\"failure_isolated\":true,\"workers_retired\":true,\"scope\":\"native C++ matcher with explicit ASCII vocabulary; no model/CUDA/HTTP\"}}"
    );
    Ok(())
}
