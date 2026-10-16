//! Actual Rust request parser/builder -> pinned native compiler/matcher proof.
use crate::grammar_native::{Compiler, Kind, Matcher};
use crate::grammar_request::{self, CompileKind};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

fn compile(compiler: &mut Compiler, body: Value) -> Result<Matcher> {
    crate::chat_json::check(&body)?;
    let spec = grammar_request::parse(&body)?.expect("fixture must request grammar");
    ensure!(
        !spec.field().is_empty(),
        "missing grammar field attribution"
    );
    ensure!(
        spec.kind() == grammar_request::Kind::Json || !spec.text().is_empty(),
        "missing canonical grammar text"
    );
    let (kind, text) = spec.compile_input()?;
    let kind = match kind {
        CompileKind::JsonSchema => Kind::JsonSchema,
        CompileKind::Regex => Kind::Regex,
        CompileKind::Ebnf => Kind::Ebnf,
        CompileKind::StructuralTag => Kind::StructuralTag,
    };
    compiler.compile(kind, &text)?.matcher()
}
fn tool(strict: Option<Value>) -> Value {
    let mut function = json!({"name":"f","parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}});
    if let Some(strict) = strict {
        function["strict"] = strict;
    }
    json!({"type":"function","function":function})
}
pub(super) fn run(compiler: &mut Compiler, eos: i32) -> Result<()> {
    for (request, output) in [
        (json!({"response_format":{"type":"json_object"}}), "{}"),
        (json!({"guided_regex":"[0-9]{2}"}), "42"),
        (json!({"guided_choice":["a","b"]}), "b"),
        (json!({"guided_grammar":"root ::= \"ok\""}), "ok"),
        (
            json!({"structured_outputs":{"json":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}}),
            r#"{"x":2}"#,
        ),
    ] {
        let mut matcher = compile(compiler, request)?;
        super::accept(&mut matcher, output)?;
        ensure!(matcher.accept(eos)?, "request format EOS rejected");
    }
    let mut object = compile(compiler, json!({"response_format":{"type":"json_object"}}))?;
    ensure!(!object.accept(b'[' as i32)?, "json_object permitted array");
    let choice = json!({"type":"function","function":{"name":"f"}});
    let mut named = compile(compiler, json!({"tools":[tool(None)],"tool_choice":choice}))?;
    super::accept(&mut named, "<tool_call>f<arg_key>x</arg_key><arg_value>")?;
    ensure!(
        !named.accept(b'a' as i32)?,
        "named schema accepted a string as integer"
    );
    super::accept(&mut named, "42</arg_value></tool_call>")?;
    ensure!(named.accept(eos)?, "named call EOS rejected");
    let mut auto = compile(compiler, json!({"tools":[tool(Some(true.into()))]}))?;
    super::accept(&mut auto, "An ordinary answer.")?;
    ensure!(auto.accept(eos)?, "strict auto required a call");
    let mut required = compile(
        compiler,
        json!({"tools":[tool(None)],"tool_choice":"required","parallel_tool_calls":false}),
    )?;
    ensure!(!required.accept(eos)?, "required allowed EOS before a call");
    super::accept(
        &mut required,
        "<tool_call>f<arg_key>x</arg_key><arg_value>42</arg_value></tool_call>",
    )?;
    ensure!(
        !required.accept(b'<' as i32)? && required.accept(eos)?,
        "parallel false did not end after one call"
    );
    let mut non_strict = compile(
        compiler,
        json!({"tools":[tool(Some(false.into()))],"tool_choice":"required"}),
    )?;
    super::accept(
        &mut non_strict,
        "<tool_call>f<arg_key>x</arg_key><arg_value>abc</arg_value></tool_call>",
    )?;
    ensure!(
        non_strict.accept(eos)?,
        "explicit strict false unexpectedly enforced integer"
    );
    ensure!(
        grammar_request::parse(&json!({"tools":[tool(None)]}))?.is_none(),
        "ordinary auto acquired grammar"
    );
    ensure!(
        !grammar_request::think_active(&[7, 3], 7, 8)
            && grammar_request::think_active(&[7, 3, 8], 7, 8),
        "prompt think activation wrong"
    );
    println!(
        "{{\"request_transform\":true,\"groups\":11,\"python_oracle\":false,\"vocabulary\":\"synthetic_ascii\"}}"
    );
    Ok(())
}
