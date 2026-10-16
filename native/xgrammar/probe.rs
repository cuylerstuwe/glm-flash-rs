#[path = "../../src/cache_pages.rs"]
#[allow(dead_code)]
mod cache_pages;
#[path = "../../src/chat_json.rs"]
#[allow(dead_code)]
mod chat_json;
#[path = "../../src/grammar_native.rs"]
mod grammar_native;
#[path = "../../src/grammar_request.rs"]
mod grammar_request;
mod request_probe;
use anyhow::{Context, Result, ensure};
use grammar_native::{Kind, Matcher, Native};
use std::path::Path;
fn allowed(mask: &[i32], token: i32) -> bool {
    mask[token as usize / 32] as u32 & (1u32 << (token as usize % 32)) != 0
}
fn mask(m: &mut Matcher) -> Result<Vec<i32>> {
    let mut bits = vec![0; m.mask_words()];
    m.fill(&mut bits)?;
    Ok(bits)
}
fn accept(m: &mut Matcher, text: &str) -> Result<()> {
    for token in text.bytes() {
        ensure!(
            allowed(&mask(m)?, token as i32),
            "mask rejected expected byte {token}"
        );
        ensure!(
            m.accept(token as i32)?,
            "matcher rejected expected byte {token}"
        );
    }
    Ok(())
}
fn main() -> Result<()> {
    let path = std::env::var("ALASKA_XGRAMMAR_LIBRARY").context("library path required")?;
    let sha = std::env::var("ALASKA_XGRAMMAR_SHA256").context("library SHA256 required")?;
    ensure!(
        Native::open(Path::new(&path), &"0".repeat(64)).is_err(),
        "wrong library identity accepted"
    );
    let native = Native::open(Path::new(&path), &sha)?;
    let mut vocab: Vec<Vec<u8>> = (0..128).map(|b| vec![b]).collect();
    vocab.push(vec![]); // EOS
    vocab.push(vec![]); // reserved added token
    let eos = 128;
    let mut compiler = native.compiler(&vocab, false, false, &[eos])?;
    request_probe::run(&mut compiler, eos)?;
    ensure!(
        compiler.compile(Kind::Ebnf, "root ::= [").is_err(),
        "malformed grammar accepted"
    );
    let grammar = compiler.compile(Kind::Ebnf, "root ::= \"a\" (\"b\" | \"c\")")?;
    ensure!(
        grammar.bytes()? > 0 && compiler.cache_bytes()? > 0,
        "native byte accounting missing"
    );
    let mut a = grammar.matcher()?;
    let initial = mask(&mut a)?;
    ensure!(
        allowed(&initial, 97)
            && !allowed(&initial, 98)
            && !allowed(&initial, eos)
            && !allowed(&initial, 129),
        "initial mask wrong"
    );
    ensure!(
        !a.accept(120)? && mask(&mut a)? == initial,
        "rejected token changed state"
    );
    ensure!(a.accept(97)?, "prefix rejected");
    let after_a = mask(&mut a)?;
    let mut fork = a.fork()?;
    ensure!(a.accept(98)? && fork.accept(99)?, "forked branch rejected");
    ensure!(
        allowed(&mask(&mut a)?, eos) && !a.terminated()?,
        "completion/termination conflated"
    );
    ensure!(a.accept(eos)? && a.terminated()?, "EOS not accepted");
    a.rollback(1)?;
    ensure!(
        !a.terminated()? && allowed(&mask(&mut a)?, eos),
        "EOS rollback wrong"
    );
    a.rollback(1)?;
    ensure!(mask(&mut a)? == after_a, "draft rollback mask changed");
    a.rollback(0)?;
    ensure!(mask(&mut a)? == after_a, "zero rollback changed state");
    // Cross-thread transfer is exclusive, not concurrent access to one matcher.
    let a = std::thread::spawn(move || -> Result<Matcher> {
        ensure!(a.accept(99)?, "worker token rejected");
        Ok(a)
    })
    .join()
    .map_err(|_| anyhow::anyhow!("grammar worker panicked"))??;
    drop(a);
    let regex = compiler.compile(Kind::Regex, "[0-9]{2}")?;
    let mut regex_m = regex.matcher()?;
    accept(&mut regex_m, "42")?;
    ensure!(regex_m.accept(eos)?, "regex EOS rejected");
    let json = compiler.compile(Kind::JsonSchema, r#"{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}"#)?;
    let mut json_m = json.matcher()?;
    accept(&mut json_m, r#"{"x":1}"#)?;
    ensure!(json_m.accept(eos)?, "JSON EOS rejected");
    let mut whitespace = json.matcher()?;
    accept(&mut whitespace, "{")?;
    accept(&mut whitespace, &" ".repeat(32))?;
    ensure!(
        !allowed(&mask(&mut whitespace)?, 32) && !whitespace.accept(32)?,
        "33rd JSON blank accepted"
    );
    let structural = compiler.compile(Kind::StructuralTag, r#"{"type":"structural_tag","format":{"type":"tag","begin":"<tool_call>f","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false},"style":"glm_xml","any_order":false,"max_whitespace_cnt":32},"end":"</tool_call>"}}"#)?;
    let mut tools = structural.matcher()?;
    accept(
        &mut tools,
        "<tool_call>f<arg_key>x</arg_key><arg_value>42</arg_value></tool_call>",
    )?;
    ensure!(tools.accept(eos)?, "GLM XML EOS rejected");
    // Rust catches incorrect shapes/IDs before native mutation; state survives.
    let mut short = [0i32; 1];
    ensure!(
        fork.fill(&mut short).is_err() && fork.accept(-1).is_err() && fork.accept(130).is_err(),
        "Rust bounds missing"
    );
    ensure!(
        allowed(&mask(&mut fork)?, eos),
        "Rust argument rejection corrupted state"
    );
    let mut poisoned = grammar.matcher()?;
    ensure!(
        poisoned.rollback(1).is_err() && poisoned.terminated().is_err(),
        "native failure not retained"
    );
    ensure!(
        allowed(&mask(&mut fork)?, eos),
        "one matcher failure affected another"
    );
    // Native shared_ptr plus Arc<Library> keep all ancestors alive as needed.
    drop(compiler);
    drop(grammar);
    drop(native);
    ensure!(
        fork.accept(eos)? && fork.terminated()?,
        "parent destruction invalidated matcher"
    );
    println!(
        "{{\"all\":true,\"source\":\"97787376faee5ed8466cfad57c99855e4ce2f6aa\",\"checks\":[\"library_sha_pin\",\"malformed_compile_recovery\",\"native_byte_accounting\",\"mask_before_accept\",\"rejected_token_no_mutation\",\"fork_independence\",\"EOS_and_rollback\",\"zero_rollback\",\"exclusive_worker_transfer\",\"regex\",\"json_schema\",\"json_32_blank_limit\",\"glm_xml_structural_tag\",\"Rust_argument_bounds\",\"request_local_native_failure\",\"ancestor_lifetimes\"]}}"
    );
    Ok(())
}
