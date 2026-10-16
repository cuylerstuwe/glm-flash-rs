#[path = "../../src/cache_pages.rs"]
#[allow(dead_code)]
mod cache_pages;
#[path = "../../src/chat_json.rs"]
#[allow(dead_code)]
mod chat_json;
#[path = "../../src/grammar_native.rs"]
#[allow(dead_code)]
mod grammar_native;
#[path = "../../src/grammar_request.rs"]
#[allow(dead_code)]
mod grammar_request;
#[path = "../../src/grammar_vocab.rs"]
mod grammar_vocab;
#[path = "../../src/output_decode.rs"]
#[allow(dead_code)]
mod output_decode;
#[path = "../../src/tokenizer.rs"]
#[allow(dead_code)]
mod tokenizer;
use anyhow::{Context, Result, ensure};
use grammar_native::{Kind, Matcher, Native};
use grammar_request::CompileKind;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;
const TOKENIZER_SHA: &str = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d";
fn mask(matcher: &mut Matcher) -> Result<Vec<i32>> {
    let mut bits = vec![0; matcher.mask_words()];
    matcher.fill(&mut bits)?;
    Ok(bits)
}
fn allowed(bits: &[i32], id: u32) -> bool {
    bits[id as usize / 32] as u32 & (1 << (id % 32)) != 0
}
fn compare_decoded(view: &[Vec<u8>], table: &output_decode::Table, path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let mut offset = 0;
    for (id, token) in view.iter().enumerate() {
        let width = u32::from_le_bytes(
            bytes
                .get(offset..offset + 4)
                .context("short oracle length")?
                .try_into()?,
        ) as usize;
        offset += 4;
        let native = bytes
            .get(offset..offset + width)
            .context("short oracle token")?;
        offset += width;
        let rust = if token.is_empty() {
            &[][..]
        } else {
            table.get(id as u32)?
        };
        ensure!(native == rust, "decoded vocabulary differs at {id}");
    }
    ensure!(offset == bytes.len(), "trailing oracle vocabulary bytes");
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}
fn main() -> Result<()> {
    let path = std::env::var("ALASKA_TOKENIZER_JSON")?;
    let bytes = std::fs::read(&path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == TOKENIZER_SHA,
        "canonical tokenizer hash mismatch"
    );
    let source = serde_json::from_slice(&bytes)?;
    let tok = tokenizer::load(Path::new(&path))?;
    let table = output_decode::Table::new(&tok)?;
    let views = grammar_vocab::Views::new(&tok, &source)?;
    let oracle = std::env::var("ALASKA_GRAMMAR_ORACLE_PREFIX")?;
    let text_sha = compare_decoded(
        &views.text,
        &table,
        Path::new(&format!("{oracle}-text.bin")),
    )?;
    let tools_sha = compare_decoded(
        &views.tools,
        &table,
        Path::new(&format!("{oracle}-tools.bin")),
    )?;
    ensure!(
        views.text.len() == grammar_vocab::VOCAB && views.tools.len() == grammar_vocab::VOCAB,
        "view bounds"
    );
    ensure!(
        views.text.iter().filter(|v| v.is_empty()).count() == 60
            && views.tools.iter().filter(|v| v.is_empty()).count() == 54,
        "added/padded counts"
    );
    for (id, marker) in grammar_vocab::TOOL_MARKERS {
        ensure!(
            views.text[id as usize].is_empty() && views.tools[id as usize] == marker.as_bytes(),
            "tool view differs"
        );
        let encoded = tok
            .encode(marker, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            encoded.get_ids() == [id],
            "tool marker does not encode singly"
        );
    }
    let library = std::env::var("ALASKA_XGRAMMAR_LIBRARY")?;
    let sha = std::env::var("ALASKA_XGRAMMAR_SHA256")?;
    let native = Native::open(Path::new(&library), &sha)?;
    let mut text = native.compiler(&views.text, true, views.add_prefix_space, &views.stop)?;
    let mut tools = native.compiler(&views.tools, true, views.add_prefix_space, &views.stop)?;
    let function = json!({"type":"function","function":{"name":"echo","parameters":{"type":"object","properties":{"text":{"type":"string"},"n":{"type":"integer"}},"required":["text","n"],"additionalProperties":false},"strict":true}});
    let mut results = Vec::new();
    for (name, request, output) in [
        (
            "json_object",
            json!({"response_format":{"type":"json_object"}}),
            r#"{"message":"Hello é 中文 😀","count":42,"items":[true,null,1.25]}"#,
        ),
        (
            "json_schema",
            json!({"guided_json":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}),
            r#"{"x":42}"#,
        ),
        (
            "choice",
            json!({"guided_choice":["café","中文","alpha"]}),
            "中文",
        ),
        (
            "regex",
            json!({"guided_regex":"[0-9]{2}-[a-z]{3}"}),
            "42-abc",
        ),
        (
            "ebnf",
            json!({"guided_grammar":"root ::= \"Hello\" \" world!\""}),
            "Hello world!",
        ),
        (
            "named_tool",
            json!({"tools":[function.clone()],"tool_choice":{"type":"function","function":{"name":"echo"}}}),
            "<tool_call>echo<arg_key>text</arg_key><arg_value>café 中文 😀</arg_value><arg_key>n</arg_key><arg_value>42</arg_value></tool_call>",
        ),
        (
            "strict_auto_text",
            json!({"tools":[function.clone()]}),
            "Here is a normal answer in 中文.",
        ),
        (
            "required_tool",
            json!({"tools":[function],"tool_choice":"required","parallel_tool_calls":false}),
            "<tool_call>echo<arg_key>text</arg_key><arg_value>ok</arg_value><arg_key>n</arg_key><arg_value>1</arg_value></tool_call>",
        ),
    ] {
        let spec = grammar_request::parse(&request)?.context("fixture missing grammar")?;
        let is_tools = spec.kind() == grammar_request::Kind::Tools;
        let view = if is_tools { &views.tools } else { &views.text };
        let compiler = if is_tools { &mut tools } else { &mut text };
        let (kind, source) = spec.compile_input()?;
        let kind = match kind {
            CompileKind::JsonSchema => Kind::JsonSchema,
            CompileKind::Regex => Kind::Regex,
            CompileKind::Ebnf => Kind::Ebnf,
            CompileKind::StructuralTag => Kind::StructuralTag,
        };
        let compiled = compiler.compile(kind, &source)?;
        let mut matcher = compiled.matcher()?;
        let tokens = tok
            .encode(output, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = tokens.get_ids();
        let cut = ids.len() / 2;
        let mut checkpoint = vec![];
        for (position, &id) in ids.iter().enumerate() {
            let bits = mask(&mut matcher)?;
            if position == cut {
                checkpoint = bits.clone();
            }
            for (special, token) in view.iter().enumerate().filter(|(_, v)| v.is_empty()) {
                let _ = token;
                if !views.stop.contains(&(special as i32)) {
                    ensure!(
                        !allowed(&bits, special as u32),
                        "{name}: reserved ID {special} allowed"
                    );
                }
            }
            ensure!(
                allowed(&bits, id) && matcher.accept(id as i32)?,
                "{name}: canonical encoded token rejected at {position}: {id}"
            );
        }
        for eos in views.stop {
            let mut branch = matcher.fork()?;
            ensure!(
                allowed(&mask(&mut branch)?, eos as u32)
                    && branch.accept(eos)?
                    && branch.terminated()?,
                "{name}: EOS {eos} failed"
            );
        }
        matcher.rollback(ids.len() - cut)?;
        ensure!(
            mask(&mut matcher)? == checkpoint,
            "{name}: rollback mask differs"
        );
        for &id in &ids[cut..] {
            ensure!(matcher.accept(id as i32)?, "{name}: replay rejected");
        }
        let decoded = tok.decode(ids, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(decoded == output, "{name}: canonical decode mismatch");
        results.push(json!({"name":name,"tokens":ids.len(),"grammar_bytes":compiled.bytes()?,"four_eos":true,"rollback":true}));
    }
    let views_identity = views
        .identity
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!(
        "{}",
        json!({"all":true,"tokenizer_sha256":TOKENIZER_SHA,"decoded_entries_compared":2*grammar_vocab::VOCAB,"text_decoded_sha256":text_sha,"tools_decoded_sha256":tools_sha,"views_identity":views_identity,"add_prefix_space":views.add_prefix_space,"view_empty_counts":[60,54],"stop_ids":views.stop,"cases":results,"text_cache_bytes":text.cache_bytes()?,"tools_cache_bytes":tools.cache_bytes()?})
    );
    Ok(())
}
