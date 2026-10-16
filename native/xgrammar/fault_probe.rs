#[path = "../../src/grammar_native.rs"]
#[allow(dead_code)]
mod grammar_native;
use anyhow::{Result, ensure};
use grammar_native::{Kind, Native};
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let path = std::env::var("ALASKA_XGRAMMAR_LIBRARY")?;
    let sha = std::env::var("ALASKA_XGRAMMAR_SHA256")?;
    let mode = std::env::var("ALASKA_GRAMMAR_QUALIFICATION_FAULT")?;
    ensure!(
        [
            "none",
            "compile-error",
            "compile-delay",
            "mask-error",
            "accept-error"
        ]
        .contains(&mode.as_str()),
        "fault probe mode"
    );
    let native = Native::open(Path::new(&path), &sha)?;
    let mut vocab: Vec<Vec<u8>> = (0..128).map(|b| vec![b]).collect();
    vocab.push(Vec::new());
    let mut compiler = native.compiler(&vocab, false, false, &[128])?;
    let begin = Instant::now();
    let first = compiler.compile(Kind::Ebnf, "root ::= \"a\"");
    let first_ms = begin.elapsed().as_secs_f64() * 1000.;
    let grammar = if mode == "compile-error" {
        ensure!(first.is_err(), "compile fault absent");
        compiler.compile(Kind::Ebnf, "root ::= \"a\"")?
    } else {
        first?
    };
    if mode == "compile-delay" {
        ensure!(first_ms >= 2900., "native delay absent");
    }
    // A second compilation always forwards; the shim injects exactly once.
    let repeat = compiler.compile(Kind::Ebnf, "root ::= \"a\"")?;
    ensure!(
        repeat.bytes()? == grammar.bytes()?,
        "forwarding changed grammar"
    );
    let mut matcher = grammar.matcher()?;
    let mut mask = vec![-1; matcher.mask_words()];
    let first_mask = matcher.fill(&mut mask);
    if mode == "mask-error" {
        ensure!(
            first_mask.is_err() && mask.iter().all(|v| *v == 0),
            "mask fault absent/unsafe"
        );
        matcher.fill(&mut mask)?;
    } else {
        first_mask?;
    }
    ensure!(
        mask[97 / 32] as u32 & (1 << (97 % 32)) != 0,
        "forwarded mask wrong"
    );
    let accepted = matcher.accept(97);
    if mode == "accept-error" {
        ensure!(accepted.is_err(), "accept fault absent");
        ensure!(matcher.accept(97)?, "one-shot fault changed matcher");
    } else {
        ensure!(accepted?, "forwarded accept wrong");
    }
    ensure!(
        matcher.accept(128)? && matcher.terminated()?,
        "forwarded EOS wrong"
    );
    println!(
        "{{\"passed\":true,\"mode\":\"{mode}\",\"first_compile_ms\":{first_ms},\"one_shot\":true,\"scope\":\"native fault shim only; no distributed service qualification\"}}"
    );
    Ok(())
}
