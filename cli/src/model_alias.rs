//! Resolve short names like Ollama (`llama3:8b`) to Hugging Face ids.
//! Does not change `org/model` semantics (single `/`, no alias resolution).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

static MERGED: OnceLock<BTreeMap<String, String>> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct AliasesFile {
    #[serde(default)]
    aliases: BTreeMap<String, String>,
}

/// Pattern: exactly one `:`, no `/` (avoids `org/model` and paths).
/// Excludes pure version tags (`ubuntu:22.04`) so `docker pull` can handle them.
pub fn looks_like_short_tag(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || s.contains('/') {
        return false;
    }
    if s.matches(':').count() != 1 {
        return false;
    }
    let (a, b) = s.split_once(':').unwrap();
    if b.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return false;
    }
    !a.is_empty()
        && !b.is_empty()
        && a.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        && b.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
}

/// Resolve a short name to an HF id if present in the merged table (bundled + ~/.neuronbox).
pub fn resolve_short_name(s: &str) -> Option<String> {
    if !looks_like_short_tag(s) {
        return None;
    }
    let key = s.trim().to_lowercase();
    merged_aliases().get(&key).cloned()
}

/// Hugging Face id for the store (resolves short aliases, leaves `org/model` unchanged).
pub fn resolve_for_hf_id(s: &str) -> String {
    resolve_short_name(s).unwrap_or_else(|| s.trim().to_string())
}

fn merged_aliases() -> &'static BTreeMap<String, String> {
    MERGED.get_or_init(load_merged)
}

fn load_merged() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();

    if let Ok(bundled) = serde_json::from_str::<AliasesFile>(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/specs/model-aliases.json"
    ))) {
        for (k, v) in bundled.aliases {
            out.insert(k.to_lowercase(), v);
        }
    }

    let path = user_aliases_json_path();
    {
        if path.exists() {
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(f) = serde_json::from_str::<AliasesFile>(&raw) {
                    for (k, v) in f.aliases {
                        out.insert(k.to_lowercase(), v);
                    }
                }
            }
        }
    }

    let path = user_aliases_toml_path();
    {
        if path.exists() {
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(v) = toml::from_str::<toml::Value>(&raw) {
                    if let Some(table) = v.get("aliases").and_then(|x| x.as_table()) {
                        for (k, val) in table {
                            if let Some(s) = val.as_str() {
                                out.insert(k.to_lowercase(), s.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

fn user_aliases_json_path() -> std::path::PathBuf {
    crate::paths::neuronbox_home().join("model-aliases.json")
}

fn user_aliases_toml_path() -> std::path::PathBuf {
    crate::paths::neuronbox_home().join("aliases.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_tag_detection() {
        assert!(looks_like_short_tag("llama3:8b"));
        assert!(looks_like_short_tag("mistral:7b"));
        assert!(!looks_like_short_tag("meta-llama/Llama-3-8B"));
        assert!(!looks_like_short_tag("ubuntu:22.04"));
        assert!(!looks_like_short_tag("./weights"));
        assert!(!looks_like_short_tag("registry.io:5000/img:tag"));
    }

    #[test]
    fn bundled_llama_alias() {
        let r = resolve_short_name("llama3:8b").expect("alias");
        assert_eq!(r, "meta-llama/Meta-Llama-3-8B-Instruct");
    }

    #[test]
    fn org_model_unchanged() {
        assert_eq!(
            resolve_for_hf_id("mistralai/Mistral-7B-v0.1"),
            "mistralai/Mistral-7B-v0.1"
        );
    }
}
