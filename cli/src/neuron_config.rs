//! Parsed `neuron.yaml` (supported declarative subset).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NeuronConfig {
    pub name: String,
    #[serde(default)]
    pub version: String,
    pub model: ModelSection,
    pub runtime: RuntimeSection,
    #[serde(default)]
    pub gpu: GpuSection,
    /// Container isolation (`host` = venv on the machine, `oci` = Docker with mounts).
    #[serde(default)]
    pub container: ContainerSection,
    pub entrypoint: String,
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSection {
    pub name: String,
    #[serde(default = "default_hf")]
    pub source: String,
    #[serde(default)]
    pub quantization: Option<String>,
}

fn default_hf() -> String {
    "huggingface".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeSection {
    pub python: String,
    #[serde(default)]
    pub cuda: Option<String>,
    #[serde(default)]
    pub packages: Vec<String>,
    /// `host` (default) or `oci` (Docker / runc per `container.executor`).
    #[serde(default = "default_runtime_mode")]
    pub mode: String,
}

fn default_runtime_mode() -> String {
    "host".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ContainerSection {
    /// Base image for `runtime.mode: oci` (e.g. CUDA Ubuntu).
    #[serde(default)]
    pub image: Option<String>,
    /// `docker` (default) or `runc` (bundle under `~/.neuronbox/bundles/...`).
    #[serde(default)]
    pub executor: Option<String>,
}

impl ContainerSection {
    pub fn image_resolved(&self) -> String {
        self.image
            .clone()
            .unwrap_or_else(|| "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04".to_string())
    }

    pub fn executor_resolved(&self) -> String {
        self.executor
            .clone()
            .unwrap_or_else(|| "docker".to_string())
            .to_lowercase()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GpuSection {
    /// e.g. "16gb" / "8GB"
    #[serde(default)]
    pub min_vram: Option<String>,
    #[serde(default = "default_strategy")]
    pub strategy: String,
}

impl Default for GpuSection {
    fn default() -> Self {
        Self {
            min_vram: None,
            strategy: default_strategy(),
        }
    }
}

fn default_strategy() -> String {
    "single".to_string()
}

impl NeuronConfig {
    pub fn oci_mode(&self) -> bool {
        self.runtime.mode.eq_ignore_ascii_case("oci")
    }

    pub fn load_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut cfg: NeuronConfig = serde_yaml::from_str(&raw)?;
        cfg.model.name = expand_env_vars(&cfg.model.name);
        expand_env_in_map(&mut cfg.env);
        Ok(cfg)
    }

    /// Approximate minimum VRAM in MB from `gpu.min_vram` (soft check).
    pub fn min_vram_mb(&self) -> Option<u64> {
        self.gpu.min_vram.as_ref().and_then(|s| parse_vram_to_mb(s))
    }
}

fn parse_vram_to_mb(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase().replace(' ', "");
    if let Some(n) = s.strip_suffix("gb") {
        return n.parse::<f64>().ok().map(|g| (g * 1024.0) as u64);
    }
    if let Some(n) = s.strip_suffix("mb") {
        return n.parse::<u64>().ok();
    }
    s.parse::<u64>().ok()
}

fn expand_env_in_map(map: &mut std::collections::BTreeMap<String, String>) {
    for v in map.values_mut() {
        *v = expand_env_vars(v);
    }
}

/// Expands `${VAR}` using `std::env::var` (missing → empty string).
fn expand_env_vars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '$' && chars.peek() == Some(&'{') {
            chars.next();
            let mut key = String::new();
            for c2 in chars.by_ref() {
                if c2 == '}' {
                    break;
                }
                key.push(c2);
            }
            out.push_str(&std::env::var(&key).unwrap_or_default());
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vram_gb() {
        assert_eq!(parse_vram_to_mb("16gb"), Some(16 * 1024));
        assert_eq!(parse_vram_to_mb("8 GB"), Some(8 * 1024));
    }
}
