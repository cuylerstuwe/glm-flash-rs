//! Frozen image content extraction around the canonical text template. Call
//! `mark` on the template's private messages AFTER history preparation. Original
//! request/history objects must never retain these per-render random sentinels.
use crate::chat_json;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::io::Read;

const PREFIX: &str = "\0tf-image-";
const PLACEHOLDER: &str = "<|begin_of_image|><|image|><|end_of_image|>";
const MAX_IMAGES: usize = 8;

#[allow(dead_code)] // Image source metadata used during native preparation.
pub(crate) struct Reference {
    pub(crate) url: String,
    // Only the lowercase string "low" changes the frozen token budget. All
    // other JSON detail values (including null, booleans and containers) use
    // the ordinary budget; do not accidentally reinterpret their truthiness.
    low: bool,
}
impl Reference {
    #[allow(dead_code)] // Canonical image budget used when binding ordered source jobs.
    pub(crate) fn token_budget(&self, cap: usize, low_cap: usize) -> usize {
        let mut budget = 8000;
        if cap != 0 {
            budget = budget.min(cap);
        }
        if self.low && low_cap != 0 {
            budget = budget.min(low_cap);
        }
        budget.max(16)
    }
}
fn reference(part: &Value) -> Result<Option<Reference>> {
    let Some(kind @ ("image_url" | "input_image" | "image")) = part["type"].as_str() else {
        return Ok(None);
    };
    let mut value = if kind == "image" {
        // Python `a or b or c` selects c even when c is falsey. Malformed
        // selected references are then rejected, never silently dropped.
        part.get("image")
            .filter(|v| chat_json::truthy(v))
            .or_else(|| part.get("image_url").filter(|v| chat_json::truthy(v)))
            .or_else(|| part.get("url"))
    } else {
        part.get("image_url")
    };
    if kind == "input_image" && value.is_none_or(Value::is_null) {
        value = part.get("url");
    }
    let mut detail = part.get("detail");
    if let Some(object) = value.and_then(Value::as_object) {
        if object.contains_key("detail") {
            detail = object.get("detail");
        }
        value = object.get("url");
    }
    let url = value
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .context("image part needs a nonempty URL string")?;
    // URL bytes and inline source syntax are validated by the fetching owner,
    // after rendering decides which image references the template actually kept.
    Ok(Some(Reference {
        url: url.to_owned(),
        low: detail
            .and_then(Value::as_str)
            .is_some_and(|s| s.to_lowercase() == "low"),
    }))
}

pub(crate) struct Markup {
    nonce: String,
    references: Vec<Reference>,
}
pub(crate) struct Rendered {
    pub(crate) text: String,
    pub(crate) references: Vec<Reference>,
}
impl Markup {
    pub(crate) fn mark(messages: &mut Value) -> Result<Self> {
        let mut nonce = [0; 6];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut nonce)?;
        Self::with_nonce(messages, nonce)
    }
    fn with_nonce(messages: &mut Value, nonce: [u8; 6]) -> Result<Self> {
        chat_json::check(messages)?;
        let messages = messages
            .as_array_mut()
            .context("image messages must be an array")?;
        ensure!(messages.len() <= 8192, "image message count bound");
        let mut result = Self {
            nonce: nonce.iter().map(|b| format!("{b:02x}")).collect(),
            references: Vec::new(),
        };
        for message in messages {
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if let Some(reference) = reference(part)? {
                    ensure!(result.references.len() < MAX_IMAGES, "image count bound");
                    *part = json!({"type":"text", "text":result.sentinel(result.references.len())});
                    result.references.push(reference);
                }
            }
        }
        Ok(result)
    }
    fn sentinel(&self, index: usize) -> String {
        format!("{PREFIX}{}-{index}\0", self.nonce)
    }
    pub(crate) fn splice(self, mut text: String) -> Result<Rendered> {
        ensure!(text.len() <= chat_json::RENDER_BYTES, "image render bound");
        let mut order = Vec::new();
        for index in 0..self.references.len() {
            let tag = self.sentinel(index);
            let mut occurrences = text.match_indices(&tag);
            if let Some((position, _)) = occurrences.next() {
                // Frozen expand would reject a duplicated placeholder without a
                // corresponding second image. Reject before fetching any bytes.
                ensure!(occurrences.next().is_none(), "duplicate image sentinel");
                order.push((position, index));
            }
        }
        for index in 0..self.references.len() {
            let tag = self.sentinel(index);
            text = text.replace(&tag, PLACEHOLDER);
            ensure!(text.len() <= chat_json::RENDER_BYTES, "image render bound");
        }
        ensure!(!text.contains(PREFIX), "image sentinel survived rendering");
        order.sort_unstable_by_key(|&(position, _)| position);
        let mut references: Vec<_> = self.references.into_iter().map(Some).collect();
        Ok(Rendered {
            text,
            references: order
                .into_iter()
                .map(|(_, index)| references[index].take().expect("unique image order"))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_aliases_falsey_fallback_and_nested_null() {
        let cases = [
            (
                json!({"type":"image_url","image_url":"data:,A","detail":"LOW"}),
                "data:,A",
                true,
            ),
            (
                json!({"type":"input_image","image_url":null,"url":"data:,B"}),
                "data:,B",
                false,
            ),
            (
                json!({"type":"image","image":false,"image_url":[],"url":"data:,C"}),
                "data:,C",
                false,
            ),
            (
                json!({"type":"image","image":{"url":"data:,D","detail":null},"detail":"low"}),
                "data:,D",
                false,
            ),
            (
                json!({"type":"image_url","image_url":{"url":"data:,E"},"detail":"low"}),
                "data:,E",
                true,
            ),
            (
                json!({"type":"image_url","image_url":"data:,F","detail":["low"]}),
                "data:,F",
                false,
            ),
        ];
        for (part, url, low) in cases {
            let r = reference(&part).unwrap().unwrap();
            assert_eq!((&*r.url, r.low), (url, low));
            assert_eq!(r.token_budget(0, 0), 8000);
            assert_eq!(r.token_budget(4000, 128), if low { 128 } else { 4000 });
            assert_eq!(r.token_budget(1, 1), 16);
        }
        for part in [
            json!({"type":"image_url","url":"private"}),
            json!({"type":"input_image","image_url":"","url":"private"}),
            json!({"type":"image","image":{"url":null},"url":"private"}),
            json!({"type":"image_url","image_url":42}),
        ] {
            assert!(
                !reference(&part)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("private")
            );
        }
        assert!(reference(&json!("text")).unwrap().is_none());
    }
    #[test]
    fn image_order_follows_rendered_order_and_discarded_images_are_not_fetched() {
        let mut messages = json!([{"role":"user","content":[
            {"type":"text","text":"prompt"},
            {"type":"image_url","image_url":"data:,A"},
            {"type":"image_url","image_url":"data:,B"},
            {"type":"image_url","image_url":"data:,C"}]}]);
        let marked = Markup::with_nonce(&mut messages, [1; 6]).unwrap();
        assert_eq!(messages[0]["content"][0]["text"], "prompt");
        let text = format!("{} then {}", marked.sentinel(2), marked.sentinel(0));
        let rendered = marked.splice(text).unwrap();
        assert_eq!(rendered.text, format!("{PLACEHOLDER} then {PLACEHOLDER}"));
        assert_eq!(
            rendered
                .references
                .iter()
                .map(|r| r.url.as_str())
                .collect::<Vec<_>>(),
            ["data:,C", "data:,A"]
        );
    }
    #[test]
    fn only_direct_content_parts_are_marked_and_ninth_image_is_refused() {
        let nested =
            json!({"type":"something","output":[{"type":"image_url","image_url":"data:,A"}]});
        let mut messages = json!([{"content":[nested.clone()]}, {"content":"unchanged"}]);
        let marked = Markup::mark(&mut messages).unwrap();
        assert_eq!(messages[0]["content"][0], nested);
        assert!(marked.splice("plain".into()).unwrap().references.is_empty());
        let mut messages =
            json!([{"content":vec![json!({"type":"image_url","image_url":"data:,A"});9]}]);
        assert!(Markup::mark(&mut messages).is_err());
    }
    #[test]
    fn duplicate_or_surviving_sentinels_never_fetch_or_publish() {
        for text in [
            "\0tf-image-other-0\0".to_owned(),
            "\0tf-image-020202020202-0\0".repeat(2),
        ] {
            let mut messages = json!([{"content":[{"type":"image_url","image_url":"data:,A"}]}]);
            let marked = Markup::with_nonce(&mut messages, [2; 6]).unwrap();
            assert!(marked.splice(text).is_err());
        }
    }
}
