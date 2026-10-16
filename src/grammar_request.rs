//! Compatible grammar request policy and XGrammar 0.2.8 GLM transformation.
//! No native compilation or model execution here. Ordinary auto tools are
//! unconstrained; explicit grammar selection is validated before native compilation.
//! See docs/API.md for the serving contract.
use crate::chat_json::{self, json, truthy};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Json,
    JsonSchema,
    Regex,
    Choice,
    Grammar,
    Tools,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompileKind {
    JsonSchema,
    Regex,
    Ebnf,
    StructuralTag,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Spec {
    kind: Kind,
    text: String,
    field: &'static str,
}
impl Spec {
    fn new(kind: Kind, text: String, field: &'static str) -> Self {
        Self { kind, text, field }
    }
    pub(crate) fn kind(&self) -> Kind {
        self.kind
    }
    pub(crate) fn text(&self) -> &str {
        &self.text
    }
    pub(crate) fn field(&self) -> &'static str {
        self.field
    }
    /// Converts already-normalized request semantics into native compiler input.
    /// Validate named-tool selection before compiling its grammar.
    pub(crate) fn compile_input(&self) -> Result<(CompileKind, String)> {
        let result: Result<(CompileKind, String)> = (|| {
            Ok(match self.kind() {
                Kind::Json => (CompileKind::JsonSchema, r#"{"type": "object"}"#.into()),
                Kind::JsonSchema => (CompileKind::JsonSchema, self.text().to_owned()),
                Kind::Regex => (CompileKind::Regex, self.text().to_owned()),
                Kind::Grammar => (CompileKind::Ebnf, self.text().to_owned()),
                Kind::Choice => {
                    let choices: Vec<String> = serde_json::from_str(self.text())?;
                    let options = choices
                        .iter()
                        .map(serde_json::to_string)
                        .collect::<std::result::Result<Vec<_>, _>>()?
                        .join(" | ");
                    (CompileKind::Ebnf, format!("root ::= {options}"))
                }
                Kind::Tools => (CompileKind::StructuralTag, glm_tools(self.text())?),
            })
        })();
        let (kind, text): (CompileKind, String) =
            result.with_context(|| format!("{}: the grammar cannot be enforced", self.field()))?;
        ensure!(
            text.len() <= chat_json::BODY_BYTES,
            "grammar compiler input exceeds 16 MiB"
        );
        Ok((kind, text))
    }
}

fn object(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}
fn schema(value: &Value, where_: &str) -> Result<String> {
    let parsed;
    let value = if let Some(s) = value.as_str() {
        parsed = chat_json::parse(s.as_bytes())
            .with_context(|| format!("{where_} is not valid JSON"))?;
        &parsed
    } else {
        value
    };
    if value == &Value::Bool(true) {
        return Ok("{}".into());
    }
    ensure!(value.is_object(), "{where_} must be a JSON schema object");
    json(value, false, false)
}
fn text(value: &Value, where_: &str) -> Result<String> {
    value
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .with_context(|| format!("{where_} must be a non-empty string"))
}
fn choices(value: &Value, where_: &str) -> Result<String> {
    ensure!(
        value.as_array().is_some_and(
            |a| !a.is_empty() && a.iter().all(|v| v.as_str().is_some_and(|s| !s.is_empty()))
        ),
        "{where_} must be a non-empty list of non-empty strings"
    );
    json(value, false, false)
}
fn read(kind: Kind, value: &Value, where_: &str) -> Result<String> {
    match kind {
        Kind::JsonSchema => schema(value, where_),
        Kind::Choice => choices(value, where_),
        Kind::Regex | Kind::Grammar => text(value, where_),
        _ => bail!("invalid grammar reader"),
    }
}
fn output(body: &Value) -> Result<Option<Spec>> {
    let rf = &body["response_format"];
    if !rf.is_null() {
        ensure!(
            rf.is_object(),
            "response_format must be an object such as {{\"type\": \"json_object\"}}"
        );
        match rf["type"].as_str() {
            Some("json_object") => {
                return Ok(Some(Spec::new(
                    Kind::Json,
                    String::new(),
                    "response_format",
                )));
            }
            Some("json_schema") => {
                let js = &rf["json_schema"];
                ensure!(
                    js.is_object() && !js["schema"].is_null(),
                    "response_format json_schema needs json_schema.schema (a JSON schema object)"
                );
                return Ok(Some(Spec::new(
                    Kind::JsonSchema,
                    schema(&js["schema"], "response_format json_schema.schema")?,
                    "response_format",
                )));
            }
            Some("text") => {}
            _ => bail!("response_format type must be text, json_object or json_schema"),
        }
    }
    for (field, kind) in [
        ("guided_json", Kind::JsonSchema),
        ("guided_regex", Kind::Regex),
        ("guided_choice", Kind::Choice),
        ("guided_grammar", Kind::Grammar),
    ] {
        if !body[field].is_null() {
            return Ok(Some(Spec::new(
                kind,
                read(kind, &body[field], field)?,
                field,
            )));
        }
    }
    let so = &body["structured_outputs"];
    if !so.is_null() {
        let so = so
            .as_object()
            .context("structured_outputs must be an object such as {\"json\": {...}}")?;
        let mut given: Vec<&str> = so
            .iter()
            .filter(|(_, v)| !v.is_null() && v != &&Value::Bool(false))
            .map(|(k, _)| k.as_str())
            .collect();
        given.sort_unstable();
        let unknown: Vec<_> = given
            .iter()
            .copied()
            .filter(|k| !["json", "regex", "choice", "grammar", "json_object"].contains(k))
            .collect();
        ensure!(
            unknown.is_empty(),
            "structured_outputs {} is not supported: use json, json_object, regex, choice or grammar",
            unknown.join(", ")
        );
        for name in given {
            let kind = match name {
                "json" => Kind::JsonSchema,
                "regex" => Kind::Regex,
                "choice" => Kind::Choice,
                "grammar" => Kind::Grammar,
                _ => continue,
            };
            return Ok(Some(Spec::new(
                kind,
                read(kind, &so[name], &format!("structured_outputs.{name}"))?,
                "structured_outputs",
            )));
        }
        if so.get("json_object").is_some_and(truthy) {
            return Ok(Some(Spec::new(
                Kind::Json,
                String::new(),
                "structured_outputs",
            )));
        }
    }
    Ok(None)
}
fn function(tool: &Value) -> Option<&Map<String, Value>> {
    let tool = tool.as_object()?;
    if tool
        .get("type")
        .is_some_and(|v| v.as_str() != Some("function"))
    {
        return None;
    }
    tool.get("function")?.as_object()
}
fn tools(body: &Value) -> Result<Option<Spec>> {
    let tools = &body["tools"];
    if !truthy(tools) {
        return Ok(None);
    }
    let tools = tools.as_array().context("tools must be a list")?;
    let choice = &body["tool_choice"];
    let choice = if choice.is_null() || choice.as_str() == Some("auto") {
        if !tools
            .iter()
            .filter_map(function)
            .any(|f| f.get("strict") == Some(&Value::Bool(true)))
        {
            return Ok(None);
        }
        Value::String("auto".into())
    } else if choice.as_str() == Some("none") {
        return Ok(None);
    } else if choice.as_str() == Some("required") {
        choice.clone()
    } else {
        ensure!(
            choice.is_object()
                && choice["type"].as_str() == Some("function")
                && choice["function"].is_object()
                && choice["function"]["name"].is_string(),
            "tool_choice must be \"none\", \"auto\", \"required\" or {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"
        );
        object([
            ("type", "function".into()),
            (
                "function",
                object([("name", choice["function"]["name"].clone())]),
            ),
        ])
    };
    let mut kept = Vec::new();
    for f in tools.iter().filter_map(function) {
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .context("every function tool needs a name")?;
        let mut entry = Map::new();
        entry.insert("name".into(), name.into());
        if let Some(parameters) = f.get("parameters").filter(|v| !v.is_null()) {
            ensure!(
                parameters.is_object(),
                "tool {name}: parameters must be a JSON schema object"
            );
            entry.insert("parameters".into(), parameters.clone());
        }
        if let Some(strict) = f.get("strict") {
            entry.insert("strict".into(), truthy(strict).into());
        }
        kept.push(object([
            ("type", "function".into()),
            ("function", Value::Object(entry)),
        ]));
    }
    if kept.is_empty() {
        return Ok(None);
    }
    let normalized = object([
        ("tools", kept.into()),
        ("tool_choice", choice),
        (
            "parallel_tool_calls",
            (body["parallel_tool_calls"] != Value::Bool(false)).into(),
        ),
    ]);
    Ok(Some(Spec::new(
        Kind::Tools,
        json(&normalized, false, false)?,
        "tools",
    )))
}

/// Call on a body already validated by chat_json::parse/check. Parse both sides:
/// malformed active tools can error even when response_format otherwise wins.
pub(crate) fn parse(body: &Value) -> Result<Option<Spec>> {
    ensure!(body.is_object(), "grammar request must be an object");
    let out = output(body)?;
    let tools = tools(body)?;
    if let (Some(out), Some(_)) = (&out, &tools) {
        ensure!(
            body["tool_choice"].as_str() != Some("required") && !body["tool_choice"].is_object(),
            "{} cannot be combined with tool_choice \"required\" or a named function: send one",
            out.field
        );
    }
    Ok(out.or(tools))
}

/// Actual prompt state wins over a request's reasoning hint. Raw prompts may
/// also end inside a think block; closing markers activate the next output row.
pub(crate) fn think_active(prompt: &[i32], open: i32, end: i32) -> bool {
    for &token in prompt.iter().rev() {
        if token == end {
            return true;
        }
        if token == open {
            return false;
        }
    }
    true
}

fn glm_tools(text: &str) -> Result<String> {
    let body: Value = serde_json::from_str(text)?;
    let tools = body["tools"]
        .as_array()
        .context("missing normalized tools")?;
    let choice = &body["tool_choice"];
    let functions: Vec<_> = tools.iter().filter_map(function).collect();
    let forced = choice.is_object();
    let functions = if forced {
        let name = choice["function"]["name"]
            .as_str()
            .context("missing forced name")?;
        let found: Vec<_> = functions
            .into_iter()
            .filter(|f| f["name"].as_str() == Some(name))
            .collect();
        ensure!(
            !found.is_empty(),
            "The tool with name '{name}' is not found in the tools list."
        );
        ensure!(
            found.len() == 1,
            "Forced tool choice must resolve to exactly one tool."
        );
        found
    } else {
        functions
    };
    ensure!(!functions.is_empty(), "empty normalized tools");
    let mut tags = Vec::with_capacity(functions.len());
    for function in functions {
        let schema = if function.get("strict") == Some(&Value::Bool(false)) {
            Value::Bool(true)
        } else {
            function
                .get("parameters")
                .cloned()
                .unwrap_or(Value::Bool(true))
        };
        let content = object([
            ("type", "json_schema".into()),
            ("json_schema", schema),
            ("style", "glm_xml".into()),
            ("any_order", false.into()),
            ("max_whitespace_cnt", 32.into()),
            ("excludes", Value::Array(vec![])),
        ]);
        let begin = format!(
            "<tool_call>{}",
            function["name"].as_str().context("missing tool name")?
        );
        tags.push(object([
            ("type", "tag".into()),
            ("begin", begin.into()),
            ("content", content),
            ("end", "</tool_call>".into()),
        ]));
    }
    let format = if forced {
        tags.pop().unwrap()
    } else {
        object([
            ("type", "triggered_tags".into()),
            ("triggers", vec![Value::from("<tool_call>")].into()),
            ("tags", tags.into()),
            ("at_least_one", (choice.as_str() == Some("required")).into()),
            (
                "stop_after_first",
                (body["parallel_tool_calls"] == Value::Bool(false)).into(),
            ),
            (
                "excludes",
                [
                    "<think>",
                    "</think>",
                    "</tool_call>",
                    "<arg_key>",
                    "</arg_key>",
                    "<arg_value>",
                    "</arg_value>",
                ]
                .into_iter()
                .map(Value::from)
                .collect::<Vec<_>>()
                .into(),
            ),
        ])
    };
    // Pydantic model_dump_json emits compact JSON; insertion order above follows
    // the pinned StructuralTag/format model declarations, including defaults.
    json(
        &object([("type", "structural_tag".into()), ("format", format)]),
        false,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json as value;
    fn spec(body: Value) -> Spec {
        chat_json::check(&body).unwrap();
        parse(&body).unwrap().unwrap()
    }
    fn tool(strict: Option<Value>) -> Value {
        let mut f = value!({"name":"f", "description":"discard", "parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}});
        if let Some(strict) = strict {
            f["strict"] = strict;
        }
        value!({"type":"function", "function":f})
    }
    fn built(body: Value) -> Value {
        serde_json::from_str(&spec(body).compile_input().unwrap().1).unwrap()
    }
    #[test]
    fn no_grammar_for_plain_and_ordinary_auto_tools() {
        for body in [
            value!({}),
            value!({"tools":[tool(None)]}),
            value!({"tools":[tool(Some(value!(false)))]}),
            value!({"tools":[tool(Some(value!(1)))]}),
            value!({"tools":[{"function":{}}]}),
            value!({"tools":[tool(Some(value!(true)))],"tool_choice":"none"}),
            value!({"tools":[],"tool_choice":"invalid"}),
        ] {
            assert!(parse(&body).unwrap().is_none(), "{body}");
        }
        for empty in [
            value!(null),
            value!(false),
            value!(0),
            value!(""),
            value!({}),
        ] {
            assert!(
                parse(&value!({"tools":empty,"tool_choice":"required"}))
                    .unwrap()
                    .is_none()
            );
        }
        assert!(parse(&value!({"tools":42})).is_err());
    }
    #[test]
    fn format_precedence_and_canonical_text() {
        let s = spec(value!({"response_format":{"type":"json_object"},"guided_regex":false}));
        assert_eq!(
            (s.kind(), s.text(), s.field()),
            (Kind::Json, "", "response_format")
        );
        assert_eq!(
            s.compile_input().unwrap(),
            (CompileKind::JsonSchema, r#"{"type": "object"}"#.into())
        );
        let s = spec(
            value!({"response_format":{"type":"text"},"guided_json":"{\"title\":\"é\",\"type\":\"object\"}","guided_regex":"ignored"}),
        );
        assert_eq!(
            (s.kind(), s.text(), s.field()),
            (
                Kind::JsonSchema,
                r#"{"title": "é", "type": "object"}"#,
                "guided_json"
            )
        );
        let s = spec(value!({"guided_regex":"  ","guided_choice":["ignored"]}));
        assert_eq!(
            s.compile_input().unwrap(),
            (CompileKind::Regex, "  ".into())
        );
        let s = spec(value!({"guided_grammar":"root ::= \"a\""}));
        assert_eq!(
            s.compile_input().unwrap(),
            (CompileKind::Ebnf, "root ::= \"a\"".into())
        );
    }
    #[test]
    fn schema_true_false_string_and_errors() {
        for schema in [value!(true), value!("true"), value!({})] {
            assert_eq!(spec(value!({"guided_json":schema})).text(), "{}");
        }
        for schema in [
            value!(false),
            value!("false"),
            value!([]),
            value!("[]"),
            value!(2),
            value!("broken"),
        ] {
            assert!(parse(&value!({"guided_json":schema})).is_err());
        }
        for rf in [
            value!(false),
            value!("json"),
            value!({}),
            value!({"type":"image"}),
            value!({"type":"json_schema"}),
            value!({"type":"json_schema","json_schema":{"schema":null}}),
        ] {
            assert!(parse(&value!({"response_format":rf})).is_err());
        }
        let s =
            spec(value!({"response_format":{"type":"json_schema","json_schema":{"schema":true}}}));
        assert_eq!(s.text(), "{}");
    }
    #[test]
    fn choices_are_literal_ebnf_and_nonempty() {
        let s = spec(value!({"guided_choice":["é","a\"b","x\ny"]}));
        assert_eq!(s.text(), "[\"é\", \"a\\\"b\", \"x\\ny\"]");
        assert_eq!(
            s.compile_input().unwrap(),
            (
                CompileKind::Ebnf,
                "root ::= \"é\" | \"a\\\"b\" | \"x\\ny\"".into()
            )
        );
        for choices in [value!([]), value!([""]), value!([1]), value!("a")] {
            assert!(parse(&value!({"guided_choice":choices})).is_err());
        }
        for field in ["guided_regex", "guided_grammar"] {
            assert!(parse(&value!({field:""})).is_err());
        }
    }
    #[test]
    fn structured_outputs_sort_keys_and_ignore_only_false_and_null() {
        let s = spec(
            value!({"structured_outputs":{"regex":"ignored","json":{},"grammar":"ignored","choice":["a"],"json_object":true}}),
        );
        assert_eq!((s.kind(), s.field()), (Kind::Choice, "structured_outputs"));
        for ignored in [value!(false), value!(null)] {
            assert!(
                parse(&value!({"structured_outputs":{"unknown":ignored}}))
                    .unwrap()
                    .is_none()
            );
        }
        for retained in [value!(0), value!(""), value!([]), value!({})] {
            assert!(parse(&value!({"structured_outputs":{"unknown":retained}})).is_err());
        }
        assert!(
            parse(&value!({"structured_outputs":{"json":false}}))
                .unwrap()
                .is_none()
        );
        assert!(parse(&value!({"structured_outputs":{"json":0}})).is_err());
        assert_eq!(
            spec(value!({"structured_outputs":{"json_object":1}})).kind(),
            Kind::Json
        );
        assert!(
            parse(&value!({"structured_outputs":{"json_object":0}}))
                .unwrap()
                .is_none()
        );
        assert!(parse(&value!({"structured_outputs":{"choice":["a"],"unknown":true}})).is_err());
        assert!(parse(&value!({"structured_outputs":[]})).is_err());
    }
    #[test]
    fn both_sides_validate_before_output_wins() {
        let body =
            value!({"response_format":{"type":"json_object"},"tools":[tool(Some(value!(true)))]});
        assert_eq!(spec(body.clone()).kind(), Kind::Json);
        for choice in [
            value!("required"),
            value!({"type":"function","function":{"name":"f"}}),
        ] {
            let mut body = body.clone();
            body["tool_choice"] = choice;
            assert!(parse(&body).is_err());
        }
        let mut bad = body;
        bad["tools"][0]["function"]["name"] = value!("");
        assert!(parse(&bad).is_err());
        assert_eq!(spec(value!({"response_format":{"type":"json_object"},"tools":[],"tool_choice":"required"})).kind(),Kind::Json);
    }
    #[test]
    fn normalized_tool_golden_and_filtered_entries() {
        let s = spec(
            value!({"tools":[null,{"type":"builtin"},{"function":{"name":"é","description":"discard","parameters":{"type":"object"},"strict":"yes"}}],
            "tool_choice":{"type":"function","function":{"name":"é","extra":"discard"},"extra":1},"parallel_tool_calls":false}),
        );
        assert_eq!(
            s.text(),
            r#"{"tools": [{"type": "function", "function": {"name": "é", "parameters": {"type": "object"}, "strict": true}}], "tool_choice": {"type": "function", "function": {"name": "é"}}, "parallel_tool_calls": false}"#
        );
        assert!(
            parse(&value!({"tools":[{"type":"builtin"}],"tool_choice":"required"}))
                .unwrap()
                .is_none()
        );
        assert!(parse(&value!({"tools":[tool(None)],"tool_choice":{"type":"function","function":{"name":1}}})).is_err());
        for bad in [value!(true), value!([]), value!("{}")] {
            let mut t = tool(None);
            t["function"]["parameters"] = bad;
            assert!(parse(&value!({"tools":[t],"tool_choice":"required"})).is_err());
        }
    }
    #[test]
    fn required_and_auto_tag_flags_follow_literal_false_only() {
        for parallel in [value!(null), value!(0), value!(""), value!(true)] {
            let tag = built(
                value!({"tools":[tool(None)],"tool_choice":"required","parallel_tool_calls":parallel}),
            );
            assert_eq!(tag["format"]["at_least_one"], true);
            assert_eq!(tag["format"]["stop_after_first"], false);
        }
        let tag = built(value!({"tools":[tool(Some(value!(true)))],"parallel_tool_calls":false}));
        assert_eq!(tag["format"]["at_least_one"], false);
        assert_eq!(tag["format"]["stop_after_first"], true);
        assert_eq!(
            tag["format"]["excludes"],
            value!([
                "<think>",
                "</think>",
                "</tool_call>",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>"
            ])
        );
    }
    #[test]
    fn named_tag_complete_golden_and_case_sensitive_uniqueness() {
        let body = value!({"tools":[{"function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"f"}}});
        assert_eq!(spec(body.clone()).compile_input().unwrap(), (CompileKind::StructuralTag,
            r#"{"type":"structural_tag","format":{"type":"tag","begin":"<tool_call>f","content":{"type":"json_schema","json_schema":{"type":"object"},"style":"glm_xml","any_order":false,"max_whitespace_cnt":32,"excludes":[]},"end":"</tool_call>"}}"#.into()));
        let mut unknown = body.clone();
        unknown["tool_choice"]["function"]["name"] = "F".into();
        assert!(spec(unknown).compile_input().is_err());
        let mut duplicate = body;
        duplicate["tools"]
            .as_array_mut()
            .unwrap()
            .push(value!({"function":{"name":"f"}}));
        assert!(spec(duplicate).compile_input().is_err());
    }
    #[test]
    fn explicit_non_strict_ignores_schema_but_missing_strict_preserves_it() {
        for strict in [value!(false), value!(null), value!(0), value!("")] {
            let tag = built(value!({"tools":[tool(Some(strict))],"tool_choice":"required"}));
            assert_eq!(tag["format"]["tags"][0]["content"]["json_schema"], true);
        }
        for strict in [None, Some(value!(true)), Some(value!("x"))] {
            let tag = built(value!({"tools":[tool(strict)],"tool_choice":"required"}));
            assert_eq!(
                tag["format"]["tags"][0]["content"]["json_schema"]["type"],
                "object"
            );
        }
        let tag = built(value!({"tools":[{"function":{"name":"f"}}],"tool_choice":"required"}));
        assert_eq!(tag["format"]["tags"][0]["content"]["json_schema"], true);
    }
    #[test]
    fn thinking_uses_last_marker_in_actual_prompt() {
        for (ids, active) in [
            (vec![], true),
            (vec![3, 4], true),
            (vec![7], false),
            (vec![7, 3, 8, 4], true),
            (vec![8, 3, 7, 4], false),
            (vec![7, 7, 8], true),
        ] {
            assert_eq!(think_active(&ids, 7, 8), active);
        }
    }
}
