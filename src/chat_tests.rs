use super::*;
fn renderer() -> Renderer {
    Renderer::new(Policy {
        thinking: true,
        effort: Some("high"),
        clear: Some(false),
    })
    .unwrap()
}
fn render(r: &mut Renderer, body: Value) -> Result<Rendered> {
    r.render(&serde_json::to_vec(&body).unwrap())
}
#[test]
fn canonical_text_goldens() {
    assert!(qualification::render_cases().unwrap().len() >= 25);
}
#[test]
fn both_effort_fields_share_aliases_defaults_and_exact_rendering() {
    let mut r = renderer();
    for (asked, effective) in [
        ("minimal", Some("Low")),
        ("low", Some("Low")),
        ("medium", Some("High")),
        ("high", Some("High")),
        ("xhigh", Some("Max")),
        ("max", Some("Max")),
        ("  MEDIUM  ", Some("High")),
        ("none", None),
    ] {
        for nested in [false, true] {
            let mut body = json!({"messages":[{"role":"user","content":"audit"}]});
            if nested {
                body["chat_template_kwargs"] = json!({"reasoning_effort":asked});
            } else {
                body["reasoning_effort"] = json!(asked);
            }
            let got = render(&mut r, body).unwrap();
            let expected = match effective {
                Some(level) => format!(
                    "[gMASK]<sop><|system|>Reasoning Effort: {level}<|user|>audit<|assistant|><think>"
                ),
                None => "[gMASK]<sop><|user|>audit<|assistant|><think></think>".to_owned(),
            };
            assert_eq!(got.text, expected, "{asked}, nested={nested}");
            assert_eq!(got.thinking, effective.is_some());
            assert_eq!(render(&mut r, got.body).unwrap().text, expected);
        }
    }
    for fields in [
        json!({}),
        json!({"reasoning_effort":null}),
        json!({"chat_template_kwargs":{"reasoning_effort":null}}),
    ] {
        let mut body = json!({"messages":[{"role":"user","content":"audit"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        assert!(
            render(&mut r, body)
                .unwrap()
                .text
                .contains("Reasoning Effort: High")
        );
    }
}
#[test]
fn invalid_effort_is_rejected_even_when_shadowed_or_thinking_is_disabled() {
    let mut r = renderer();
    for invalid in [
        json!("unsupported"),
        json!(""),
        json!("off"),
        json!(4),
        json!(true),
        json!([]),
        json!({}),
    ] {
        for nested in [false, true] {
            for enabled in [false, true] {
                let mut body = json!({"messages":[{"role":"user","content":"audit"}],
                    "reasoning_effort":"high","chat_template_kwargs":{"reasoning_effort":"low","enable_thinking":enabled}});
                if nested {
                    body["chat_template_kwargs"]["reasoning_effort"] = invalid.clone();
                } else {
                    body["reasoning_effort"] = invalid.clone();
                }
                assert!(render(&mut r, body).is_err(), "{invalid}, nested={nested}");
            }
        }
    }
}
#[test]
fn explicit_template_effort_and_thinking_switch_keep_precedence() {
    let mut r = renderer();
    let got = render(&mut r, json!({"messages":[{"role":"user","content":"audit"}],
        "reasoning_effort":"max","chat_template_kwargs":{"reasoning_effort":"minimal","enable_thinking":false}})).unwrap();
    assert!(got.text.contains("Reasoning Effort: Low"));
    assert!(got.text.ends_with("<think></think>"));
    assert!(!got.thinking);
    assert_eq!(render(&mut r, got.body).unwrap().text, got.text);
    let got = render(
        &mut r,
        json!({"messages":[{"role":"user","content":"audit"}],
        "chat_template_kwargs":{"reasoning_effort":"none","enable_thinking":true}}),
    )
    .unwrap();
    assert!(got.thinking);
    assert_eq!(render(&mut r, got.body).unwrap().text, got.text);
}
#[test]
fn unknown_roles_media_and_template_overrides_are_explicit_errors() {
    let mut r = renderer();
    for body in [
        json!({"messages":[{"role":"developer","content":"rules"}]}),
        json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///private"}}]}]}),
        json!({"messages":[{"role":"user","content":[{"type":"input_image","url":"data:image/png;base64,AA=="}]}]}),
        json!({"messages":[],"chat_template_kwargs":{"messages":[]}}),
        json!({"messages":[],"tools":[{"name":"f"}],"tool_choice":{"type":"function","function":{"name":"missing"}}}),
        json!({"messages":[],"reasoning_effort":"invalid"}),
    ] {
        assert!(render(&mut r, body).is_err());
    }
}
#[test]
fn internal_images_render_canonical_placeholders_without_mutating_history() {
    let mut r = renderer();
    let calls = json!([{"id":"image-call","function":{"name":"f","arguments":{"a":2}}}]);
    r.memory
        .remember("ask", calls.as_array().unwrap(), "retained reasoning")
        .unwrap();
    let body = json!({"messages":[
        {"role":"user","content":[{"type":"text","text":"ask"},{"type":"image_url","image_url":{"url":"data:,A","detail":null}}]},
        {"role":"assistant","tool_calls":calls},
        {"role":"tool","tool_call_id":"image-call","content":"done"},
        {"role":"user","content":[{"type":"input_image","url":"data:,B"}]}]});
    let bytes = serde_json::to_vec(&body).unwrap();
    assert!(r.render(&bytes).is_err());
    let (first, refs) = r.render_kind(&bytes, true).unwrap();
    let (second, _) = r.render_kind(&bytes, true).unwrap();
    assert_eq!(first.text, second.text); // Random nonces never reach tokenization.
    assert!(first.text.contains("<think>retained reasoning</think>"));
    assert_eq!(
        first
            .text
            .matches("<|begin_of_image|><|image|><|end_of_image|>")
            .count(),
        2
    );
    assert!(!first.text.contains("tf-image-") && !first.text.contains("data:,"));
    assert_eq!(
        refs.iter().map(|r| r.url.as_str()).collect::<Vec<_>>(),
        ["data:,A", "data:,B"]
    );
    assert_eq!(
        first.body["messages"][0]["content"],
        body["messages"][0]["content"]
    );
    assert!(r.render_kind(&serde_json::to_vec(&json!({"messages":[{"role":"user","content":[{"type":"video","video":"private"}]}]})).unwrap(), true).is_err());
    assert!(r.render_kind(&serde_json::to_vec(&json!({"messages":[{"role":"user","content":[{"type":"wrapper","output":[{"type":"image_url","image_url":"private"}]}]}]})).unwrap(), true).is_err());
}
#[test]
fn internal_image_path_preserves_text_only_render_exactly() {
    let mut r = renderer();
    let bytes = br#"{"messages":[{"role":"user","content":"hello"}],"chat_template_kwargs":{"enable_thinking":null}}"#;
    let plain = r.render(bytes).unwrap();
    let (image, refs) = r.render_kind(bytes, true).unwrap();
    assert_eq!(plain.text, image.text);
    assert_eq!(plain.body, image.body);
    assert!(refs.is_empty());
}
#[test]
fn reasoning_id_then_signature_restore_and_explicit_text_wins() {
    let mut r = renderer();
    let calls = json!([{"id":"call_0123456789abcdef01234567","function":{"name":"f","arguments":{"a":2,"z":1}}}]);
    r.memory
        .remember("ask", calls.as_array().unwrap(), "stored")
        .unwrap();
    let mut body = json!({"messages":[{"role":"user","content":"ask"},{"role":"assistant","tool_calls":calls}]});
    assert!(
        render(&mut r, body.clone())
            .unwrap()
            .text
            .contains("<think>stored</think>")
    );
    body["messages"][1]["tool_calls"][0]["id"] = json!("client-renumbered");
    assert!(
        render(&mut r, body.clone())
            .unwrap()
            .text
            .contains("<think>stored</think>")
    );
    body["messages"][1]["reasoning_content"] = json!("");
    assert!(
        !render(&mut r, body.clone())
            .unwrap()
            .text
            .contains("stored")
    );
    body["messages"][1]
        .as_object_mut()
        .unwrap()
        .shift_remove("reasoning_content");
    body["messages"][1]["content"] = json!("<think>explicit</think>answer");
    assert!(
        render(&mut r, body.clone())
            .unwrap()
            .text
            .contains("<think>explicit</think>answer")
    );
    body["messages"][1]["content"] = json!("");
    body["messages"][0]["content"] = json!("other user");
    assert!(!render(&mut r, body).unwrap().text.contains("stored"));
}
#[test]
fn preparation_is_idempotent_and_invalid_calls_drop_only_their_block() {
    let mut r = renderer();
    let body = json!({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"bad","function":{"name":"f","arguments":"[1]"}}]},
        {"role":"tool","tool_call_id":"bad","content":"drop"},{"role":"user","content":"next"},
        {"role":"tool","tool_call_id":"bad","content":"keep unrelated"}]});
    let first = render(&mut r, body).unwrap();
    let second = render(&mut r, first.body.clone()).unwrap();
    assert_eq!(first.text, second.text);
    assert!(!first.text.contains("drop"));
    assert!(first.text.contains("keep unrelated"));
}
#[test]
fn work_and_expansion_are_bounded() {
    let mut r = renderer();
    r.env.set_fuel(Some(50));
    assert!(render(&mut r, json!({"messages":[{"role":"user","content":"x"}]})).is_err());
    let mut out = Limited {
        bytes: Vec::new(),
        max: 3,
    };
    out.write_all(b"abc").unwrap();
    assert!(out.write_all(b"d").is_err());
    assert_eq!(out.bytes, b"abc");
}
#[test]
fn sorted_manifest_metadata_stays_byte_compatible() {
    let mut value = j::parse(br#"{"z":{"y":2,"a":1},"a":[{"b":3,"a":4}]}"#).unwrap();
    crate::prepared::canonical_metadata(&mut value);
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        r#"{"a":[{"a":4,"b":3}],"z":{"a":1,"y":2}}"#
    );
}

#[test]
fn manifest_float_spelling_remains_canonical() {
    let mut value = j::parse(br#"{"z":1e-05,"a":[1e-06,-5.0,10.0,2.5]}"#).unwrap();
    crate::prepared::canonical_metadata(&mut value);
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        r#"{"a":[1e-6,-5.0,10.0,2.5],"z":0.00001}"#
    );
}

#[test]
fn argument_limits_are_errors_instead_of_silent_history_repair() {
    let mut r = renderer();
    assert!(render(&mut r, json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"n\":18446744073709551616}"}}]}]})).is_err());
    // A parameter named items must not shadow the Python mapping method.
    let result = render(&mut r, json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"items":"kept","keys":1}}}]}]})).unwrap();
    assert!(
        result
            .text
            .contains("<arg_key>items</arg_key><arg_value>kept</arg_value>")
    );
}
