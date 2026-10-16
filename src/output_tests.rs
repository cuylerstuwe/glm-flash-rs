use super::*;
use crate::output_policy::STOP_BYTES;
use crate::output_tools as tools;

pub(super) fn fixtures() -> Vec<Value> {
    serde_json::from_str(include_str!("../fixtures/chat/output-text.json")).unwrap()
}
pub(super) fn tool_fixtures() -> Vec<Value> {
    serde_json::from_str(include_str!("../fixtures/chat/output-tools.json")).unwrap()
}
pub(super) fn text_case(case: &Value, chunks: &[&str]) -> Result<()> {
    let mut text = Text::new(
        case["thinking"].as_bool().unwrap(),
        case["tools"].as_bool().unwrap(),
        &case["stop"],
    )?;
    let mut streamed = Delta::default();
    for chunk in chunks {
        if text.stopped {
            break;
        }
        let d = text.push(chunk)?;
        streamed.reasoning += &d.reasoning;
        streamed.content += &d.content;
        ensure!(
            case["reasoning"]
                .as_str()
                .unwrap()
                .starts_with(&streamed.reasoning),
            "reasoning retract: {}",
            case["name"]
        );
        ensure!(
            case["visible"]
                .as_str()
                .unwrap()
                .starts_with(&streamed.content),
            "content retract: {}",
            case["name"]
        );
    }
    let d = text.finish()?;
    streamed.reasoning += &d.reasoning;
    streamed.content += &d.content;
    ensure!(
        streamed.reasoning == case["reasoning"] && text.reasoning == streamed.reasoning,
        "reasoning: {}",
        case["name"]
    );
    ensure!(
        streamed.content == case["visible"] && text.visible == streamed.content,
        "visible: {} actual {:?}",
        case["name"],
        text.visible
    );
    ensure!(
        text.raw_answer == case["raw_answer"],
        "raw: {} {:?}",
        case["name"],
        text.raw_answer
    );
    ensure!(
        text.stopped == case["stopped"].as_bool().unwrap(),
        "stop: {}",
        case["name"]
    );
    Ok(())
}
#[test]
fn text_goldens_every_character_split() -> Result<()> {
    for case in fixtures() {
        let raw = case["input"].as_str().unwrap();
        for cut in raw.char_indices().map(|(i, _)| i).chain([raw.len()]) {
            text_case(&case, &[&raw[..cut], &raw[cut..]])?;
        }
        let chunks: Vec<_> = raw
            .char_indices()
            .map(|(i, c)| &raw[i..i + c.len_utf8()])
            .collect();
        text_case(&case, &chunks)?;
    }
    Ok(())
}
#[test]
fn tools_goldens() -> Result<()> {
    for case in tool_fixtures() {
        let mut n = 0;
        let mut ids = || {
            n += 1;
            Ok(format!("call_{n:024x}"))
        };
        let (content, calls) = tools::parse(
            case["input"].as_str().unwrap(),
            case["tools"].as_array().unwrap(),
            &mut ids,
        )?;
        assert_eq!(content, case["content"], "{}", case["name"]);
        assert_eq!(
            &calls,
            case["calls"].as_array().unwrap(),
            "{}",
            case["name"]
        );
    }
    Ok(())
}
#[test]
fn typed_schema_rules() -> Result<()> {
    for (raw, schema, want) in [
        ("null", json!({"type":"string"}), json!("null")),
        (" null ", json!({"type":["string","null"]}), Value::Null),
        ("\"5\"", json!({"type":"string"}), json!("\"5\"")),
        ("\"5\"", json!({"type":"integer"}), json!(5)),
        ("5.0", json!({"type":"integer"}), json!(5)),
        ("5.5", json!({"type":"integer"}), json!("5.5")),
        ("\"true\"", json!({"type":"boolean"}), json!(true)),
        ("True", json!({"enum":[false]}), json!(true)),
        ("None", json!({"const":null}), Value::Null),
        ("1", json!({"type":"boolean"}), json!("1")),
        (
            "[1,{\"x\":2",
            json!({"anyOf":[{"type":"array"},{"type":"null"}]}),
            json!([1,{"x":2}]),
        ),
        ("[1", json!({}), json!("[1")),
        ("[1]", json!({}), json!([1])),
        (
            " false ",
            json!({"oneOf":[{},{"type":"string"}]}),
            json!(false),
        ),
        ("True", json!({"type":["boolean","string"]}), json!("True")),
        ("2", json!({"type":"number"}), json!(2)),
        ("2.0", json!({"type":"number"}), json!(2.0)),
    ] {
        assert_eq!(tools::typed(raw, &schema)?, want, "{raw}: {schema}");
    }
    assert!(tools::typed("18446744073709551616", &Value::Null).is_err());
    assert!(tools::typed("1e400", &Value::Null).is_err());
    assert!(tools::typed("18446744073709551616.0", &json!({"type":"integer"})).is_err());
    Ok(())
}
#[test]
fn closure_and_trailing_calls() -> Result<()> {
    for (s, want) in [
        ("[", Some("[]")),
        ("{\"x\":[1", Some("{\"x\":[1]}")),
        ("{\"x\":\"[\"", Some("{\"x\":\"[\"}")),
        ("[}", None),
        ("\"[", None),
        ("[\"a\\\"", None),
        ("[1]", None),
    ] {
        assert_eq!(tools::closed_json(s).as_deref(), want);
    }
    let text = "Reason.\n<tool_call>f</tool_call>\n<tool_call>g</tool_call>\n</think> ";
    assert_eq!(
        tools::trailing(text)?,
        (
            "Reason.",
            "<tool_call>f</tool_call>\n<tool_call>g</tool_call>"
        )
    );
    assert_eq!(
        tools::trailing("x<tool_call>f</tool_call>tail")?,
        ("x<tool_call>f</tool_call>tail", "")
    );
    assert_eq!(
        tools::trailing("<TOOL_CALL>f</TOOL_CALL>")?,
        ("<TOOL_CALL>f</TOOL_CALL>", "")
    );
    Ok(())
}
#[test]
fn stops_match_independent_bruteforce() -> Result<()> {
    // Exhaust all 8-char binary inputs against overlapping/failure-link patterns.
    for value in 0..256u32 {
        let input: String = (0..8)
            .map(|i| if value & (1 << i) == 0 { 'a' } else { 'b' })
            .collect();
        for patterns in [
            vec!["aaaab", "aab"],
            vec!["abba", "bba"],
            vec!["ab", "aba"],
            vec!["bbb", "aaa"],
        ] {
            let mut matcher = Stops::new(&json!(patterns))?;
            let mut actual = String::new();
            let mut cut = None;
            let mut expected = None;
            for (i, ch) in input.char_indices() {
                let prior = actual.len();
                let mut delta = String::new();
                matcher.push(ch, i, &mut delta)?;
                let hit = matcher.boundary(prior, &mut delta)?;
                actual += &delta;
                if let Some(raw) = hit {
                    cut = Some(raw);
                    break;
                }
            }
            for end in 1..=input.len() {
                let start = patterns.iter().filter_map(|p| input[..end].find(p)).min();
                if start.is_some() {
                    expected = start;
                    break;
                }
            }
            assert_eq!(cut, expected);
            if cut.is_none() {
                matcher.finish(&mut actual)?;
            }
            assert_eq!(actual, input[..expected.unwrap_or(input.len())]);
        }
    }
    Ok(())
}
#[test]
fn earliest_stop_start_within_one_decoded_token() -> Result<()> {
    let mut t = Text::new(false, false, &json!(["abcd", "bc"]))?;
    assert_eq!(t.push("abcd")?.content, "");
    assert!(t.stopped);
    assert_eq!(t.raw_answer, "");
    let mut t = Text::new(false, false, &json!(["abcd", "bc"]))?;
    assert_eq!(t.push("abc")?.content, "a");
    assert!(t.stopped);
    assert_eq!(t.raw_answer, "a");
    Ok(())
}
#[test]
fn stop_and_tool_bounds() -> Result<()> {
    for bad in [json!(true), json!(5), json!([null])] {
        assert!(Stops::new(&bad).is_err());
    }
    assert!(Stops::new(&json!(vec!["x"; 33])).is_err());
    assert!(Stops::new(&json!("x".repeat(STOP_BYTES + 1))).is_err());
    let raw = "<tool_call>f</tool_call>".repeat(257);
    assert!(
        tools::parse(&raw, &[json!({"name":"f"})], &mut || Ok(
            "call_000000000000000000000000".into()
        ))
        .is_err()
    );
    Ok(())
}

fn token_stream(
    tokens: &[&[u8]],
    thinking: bool,
    tools: bool,
    stops: Value,
    limit: usize,
) -> Result<Stream> {
    Stream::new(
        Arc::new(Table::test_bytes(tokens)),
        thinking,
        tools,
        &stops,
        limit,
    )
}

#[test]
fn accepted_window_clips_cross_token_stop_without_consuming_suffix() -> Result<()> {
    let mut s = token_stream(
        &[b"prior:", b"aST", b"O", b"Pignored"],
        false,
        false,
        json!("STOP"),
        128,
    )?;
    assert_eq!(
        s.clip_accepted(&[0])?,
        Clipped {
            keep: 1,
            ended: false
        }
    );
    assert_eq!(s.text.visible, "prior:");
    // The invalid padded ID would fail if the clipped suffix were even read.
    assert_eq!(
        s.clip_accepted(&[1, 2, 3, u32::MAX])?,
        Clipped {
            keep: 3,
            ended: true
        }
    );
    assert_eq!(s.ids, [0, 1, 2, 3]);
    assert_eq!(s.text.visible, "prior:a");
    assert_eq!(s.text.raw_answer, "prior:a");
    assert!(s.text.stopped);
    assert_eq!(s.finish()?, Delta::default());
    Ok(())
}

#[test]
fn accepted_window_clips_every_eos_and_partial_utf8_exactly() -> Result<()> {
    for eos in decode::EOS {
        let mut s = token_stream(&[b"ok", &[0xf0, 0x9f]], false, false, Value::Null, 128)?;
        assert_eq!(
            s.clip_accepted(&[0, 1, eos, u32::MAX])?,
            Clipped {
                keep: 3,
                ended: true
            }
        );
        assert_eq!(s.ids, [0, 1, eos]);
        assert_eq!(s.text.visible, "ok");
        // EOS preserves the same final incomplete-byte replacement as serial.
        assert_eq!(s.finish()?.content, "�");
        assert_eq!(s.text.visible, "ok�");
    }
    Ok(())
}

#[test]
fn ignored_eos_hidden_bytes_preserve_utf8_stops_ids_and_length() -> Result<()> {
    for eos in decode::EOS {
        let mut stream = token_stream(
            &[b"A", b"B", &[0xf0, 0x9f], &[0xa6, 0x80]],
            false,
            false,
            Value::Null,
            7,
        )?
        .with_ignore_eos(true);
        assert!(!stream.push(0)?.1);
        assert_eq!(
            stream.clip_accepted(&[eos, 2, eos])?,
            Clipped {
                keep: 3,
                ended: false
            }
        );
        assert_eq!(stream.text.visible, "A");
        assert_eq!(
            stream.clip_accepted(&[3, eos, 1, u32::MAX])?,
            Clipped {
                keep: 3,
                ended: true
            }
        );
        assert_eq!(stream.text.visible, "A🦀B");
        assert_eq!(stream.ids, [0, eos, 2, eos, 3, eos, 1]);
        let reply = stream.reply(
            &json!({}),
            &mut crate::chat_history::Memory::new(4, 100, 1000),
            &mut || Ok("unused".into()),
        )?;
        assert_eq!(reply["content"], "A🦀B");
        assert_eq!(reply["completion_tokens"], 7);
        assert_eq!(reply["finish"], "length");
        assert_eq!(reply["token_ids"], json!([0, eos, 2, eos, 3, eos, 1]));

        let mut stopped =
            token_stream(&[b"A", b"B"], false, false, json!("AB"), 8)?.with_ignore_eos(true);
        stopped.clip_accepted(&[0, eos])?;
        assert_eq!(
            stopped.clip_accepted(&[1, u32::MAX])?,
            Clipped {
                keep: 1,
                ended: true
            }
        );
        assert!(stopped.text.stopped);
        assert_eq!(stopped.ids, [0, eos, 1]);
        assert_eq!(stopped.text.visible, "");

        let mut capped = token_stream(&[b"A"], false, false, Value::Null, 2)?.with_ignore_eos(true);
        capped.clip_accepted(&[0, eos])?;
        let reply = capped.reply(
            &json!({}),
            &mut crate::chat_history::Memory::new(4, 100, 1000),
            &mut || Ok("unused".into()),
        )?;
        assert_eq!(reply["content"], "A");
        assert_eq!(reply["completion_tokens"], 2);
        // Frozen CUDA wire labels a final EOS as stop even under ignore_eos.
        assert_eq!(reply["finish"], "stop");
    }
    Ok(())
}

#[test]
fn accepted_window_output_limit_includes_only_remaining_tokens() -> Result<()> {
    let mut s = token_stream(&[b"a", b"b"], false, false, Value::Null, 2)?;
    s.push(0)?;
    assert_eq!(
        s.clip_accepted(&[1, u32::MAX])?,
        Clipped {
            keep: 1,
            ended: true
        }
    );
    assert_eq!(s.ids, [0, 1]);
    assert_eq!(s.text.visible, "ab");
    assert_eq!(s.finish()?, Delta::default());
    let mut s = token_stream(&[b"x"], false, false, Value::Null, 17)?;
    assert_eq!(
        s.clip_accepted(&[0; 16])?,
        Clipped {
            keep: 16,
            ended: false
        }
    );
    assert_eq!(
        s.clip_accepted(&[0, u32::MAX])?,
        Clipped {
            keep: 1,
            ended: true
        }
    );
    assert_eq!(s.ids.len(), 17);
    Ok(())
}

#[test]
fn accepted_window_preserves_unicode_thinking_and_tool_visibility() -> Result<()> {
    let tokens: &[&[u8]] = &[
        b"hidden STOP</thi",
        b"nk>\n\nA",
        &[0xf0],
        &[0x9f, 0xa6],
        &[0x80],
        b"<tool_",
        b"call>STOP",
        b"</tool_call>B",
        b"ST",
        b"OPtail",
    ];
    let mut s = token_stream(tokens, true, true, json!("STOP"), 128)?;
    assert_eq!(
        s.clip_accepted(&[0, 1, 2, 3])?,
        Clipped {
            keep: 4,
            ended: false
        }
    );
    assert_eq!(s.text.reasoning, "hidden STOP");
    assert_eq!(s.text.visible, "A");
    // The reasoning/tool occurrences are invisible and must not clip the window.
    assert_eq!(
        s.clip_accepted(&[4, 5, 6, 7, 8, 9, u32::MAX])?,
        Clipped {
            keep: 6,
            ended: true
        }
    );
    assert_eq!(s.ids, (0..10).collect::<Vec<_>>());
    assert_eq!(s.text.visible, "A🦀B");
    assert_eq!(s.text.raw_answer, "A🦀<tool_call>STOP</tool_call>B");
    assert_eq!(s.finish()?, Delta::default());
    Ok(())
}

#[test]
fn accepted_window_keeps_serial_token_boundaries_for_overlapping_stops() -> Result<()> {
    for (tokens, keep, visible) in [
        (vec![b"ab".as_slice(), b"cd".as_slice()], 2, ""),
        (vec![b"abc".as_slice(), b"d".as_slice()], 1, "a"),
    ] {
        let mut s = token_stream(&tokens, false, false, json!(["abcd", "bc"]), 128)?;
        assert_eq!(
            s.clip_accepted(&[0, 1, u32::MAX])?,
            Clipped { keep, ended: true }
        );
        assert_eq!(s.text.visible, visible);
        assert_eq!(s.ids.len(), keep);
    }
    Ok(())
}

#[test]
fn accepted_window_errors_poison_owner_without_replaying_consumed_prefix() -> Result<()> {
    let mut s = token_stream(&[b"a", b"b"], false, false, Value::Null, 128)?;
    s.push(0)?;
    assert!(s.clip_accepted(&[1, u32::MAX, 0]).is_err());
    assert_eq!(s.ids, [0, 1]);
    assert_eq!(s.text.visible, "ab");
    assert!(s.failed && s.ended);
    assert!(s.finish().is_err());
    assert!(s.clip_accepted(&[0]).is_err());
    assert_eq!(s.ids, [0, 1]);
    // A failure after token lookup and UTF-8 advancement is also terminal.
    let mut partial = token_stream(&[b"x"], false, false, Value::Null, 128)?;
    partial.text.bytes = decode::MAX_BYTES;
    assert!(partial.clip_accepted(&[0]).is_err());
    assert_eq!(partial.ids, [0]);
    assert!(partial.failed && partial.ended);
    assert!(partial.finish().is_err());
    for invalid in [vec![], vec![0; 17]] {
        let mut s = token_stream(&[b"a"], false, false, Value::Null, 128)?;
        s.push(0)?;
        assert!(s.clip_accepted(&invalid).is_err());
        assert_eq!(s.ids, [0]);
        assert_eq!(s.text.visible, "a");
        assert!(s.failed && s.ended);
        assert!(s.push(0).is_err());
        assert!(s.finish().is_err());
    }
    Ok(())
}

#[test]
fn assistant_turn_eos_ends_once_omits_marker_and_counts_terminal_token() -> Result<()> {
    assert_eq!(decode::EOS, [154820, 154827, 154829, 154828]);
    let mut s = token_stream(&[b"answer", b"second turn"], false, false, Value::Null, 128)?;
    assert_eq!(
        s.clip_accepted(&[0, 154828, 1])?,
        Clipped {
            keep: 2,
            ended: true
        }
    );
    assert_eq!(s.ids, [0, 154828]);
    assert_eq!(s.text.visible, "answer");
    assert_eq!(s.text.raw_answer, "answer");
    let mut memory = crate::chat_history::Memory::new(16, 1 << 20, 2 << 20);
    let reply = s.reply(&json!({}), &mut memory, &mut || Ok("unused".into()))?;
    assert_eq!(reply["content"], "answer");
    assert_eq!(reply["final"]["content"], "");
    assert_eq!(reply["completion_tokens"], 2);
    assert_eq!(reply["token_ids"], json!([0, 154828]));
    assert_eq!(reply["finish"], "stop");
    Ok(())
}
