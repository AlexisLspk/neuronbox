//! Resolve model path: Hugging Face (store) vs local directory / file.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::neuron_config::NeuronConfig;
use crate::store::model_store;

/// Final location exposed to the script (HF or local disk).
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// Root directory (HF snapshot, weights folder, or parent of a single weights file).
    pub dir: PathBuf,
    /// Weights file when `model.name` points at a file (.gguf, .pt, …).
    pub weights_file: Option<PathBuf>,
}

pub fn is_local_source(source: &str) -> bool {
    matches!(
        source.trim().to_lowercase().as_str(),
        "local" | "path" | "filesystem" | "disk"
    )
}

pub fn is_hf_source(source: &str) -> bool {
    matches!(
        source.trim().to_lowercase().as_str(),
        "huggingface" | "hf" | "hub"
    )
}

/// Detect a local path even when `source` is still huggingface (ergonomics).
fn name_looks_like_filesystem_path(name: &str) -> bool {
    let n = name.trim();
    if n.starts_with("./") || n.starts_with("../") || n.starts_with('~') {
        return true;
    }
    Path::new(n).is_absolute()
}

pub fn use_local_filesystem(cfg: &NeuronConfig) -> bool {
    if cfg.model.source.eq_ignore_ascii_case("s3") {
        return false;
    }
    if is_local_source(&cfg.model.source) {
        return true;
    }
    if is_hf_source(&cfg.model.source) {
        return name_looks_like_filesystem_path(&cfg.model.name);
    }
    name_looks_like_filesystem_path(&cfg.model.name)
}

/// Resolve the model for `neuron run` (after optional `ensure_hf_model`).
/// `hf_repo_id`: resolved HF id (alias `llama3:8b` → `org/model`).
pub fn resolve_model_for_run(
    cfg: &NeuronConfig,
    project_dir: &Path,
    store: &Path,
    hf_repo_id: &str,
) -> Result<ResolvedModel> {
    if cfg.model.source.eq_ignore_ascii_case("s3") {
        anyhow::bail!(
            "source: s3 is not implemented yet. Use source: local or huggingface, or mount S3 on disk."
        );
    }

    if use_local_filesystem(cfg) {
        return resolve_local(&cfg.model.name, project_dir);
    }

    let dir = model_store::model_path(store, hf_repo_id);
    if !dir.exists() {
        anyhow::bail!(
            "HF model missing from store: {}. Run `neuron pull {}` or check model.name.",
            dir.display(),
            hf_repo_id
        );
    }
    validate_model_dir_or_file(&dir, true)?;
    Ok(ResolvedModel {
        dir,
        weights_file: None,
    })
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    if let Some(rest) = path.to_str().and_then(|s| s.strip_prefix("~/")) {
        return dirs::home_dir().map(|h| h.join(rest)).unwrap_or(path);
    }
    path
}

fn resolve_local(name: &str, project_dir: &Path) -> Result<ResolvedModel> {
    let raw = name.trim();
    if raw.is_empty() {
        anyhow::bail!("model.name is empty for a local model.");
    }

    let mut p = PathBuf::from(raw);
    p = expand_tilde(p);
    let p = if p.is_absolute() {
        p
    } else {
        project_dir.join(p)
    };

    let canon = p
        .canonicalize()
        .with_context(|| format!("local model not found: {:?}", p))?;

    if canon.is_file() {
        validate_weights_file(&canon)?;
        let dir = canon
            .parent()
            .map(Path::to_path_buf)
            .context("parent of model file")?;
        return Ok(ResolvedModel {
            dir,
            weights_file: Some(canon),
        });
    }

    if canon.is_dir() {
        validate_model_dir_or_file(&canon, true)?;
        return Ok(ResolvedModel {
            dir: canon,
            weights_file: None,
        });
    }

    anyhow::bail!("invalid local model path: {:?}", canon)
}

fn validate_weights_file(p: &Path) -> Result<()> {
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if matches!(
        ext.as_str(),
        "gguf" | "ggml" | "pt" | "pth" | "bin" | "safetensors" | "onnx" | "mlmodel"
    ) {
        return Ok(());
    }
    anyhow::bail!(
        "unrecognized model file extension: {:?} (expected .gguf, .safetensors, .pt, .onnx, …)",
        p
    )
}

/// Check for common artifacts (HF, LoRA, GGUF, PyTorch checkpoints, etc.).
pub fn validate_model_dir_or_file(path: &Path, is_dir: bool) -> Result<()> {
    if !is_dir {
        return validate_weights_file(path);
    }

    if path.join("config.json").exists() {
        return Ok(());
    }
    if path.join("adapter_config.json").exists() {
        return Ok(());
    }
    if path.join("pytorch_model.bin").exists() || path.join("model.safetensors").exists() {
        return Ok(());
    }

    let mut has_weight_like = false;
    if let Ok(rd) = std::fs::read_dir(path) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_lowercase();
            if n.ends_with(".safetensors")
                || n.ends_with(".gguf")
                || n.ends_with(".ggml")
                || n.ends_with(".bin")
                || n.ends_with(".pt")
                || n.ends_with(".pth")
                || n.ends_with(".onnx")
            {
                has_weight_like = true;
                break;
            }
        }
    }

    if has_weight_like {
        return Ok(());
    }

    anyhow::bail!(
        "local model directory has no recognized artifact in {:?} (expected config.json, adapter_config.json, *.safetensors, *.gguf, pytorch_model.bin, …)",
        path
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_looks_path() {
        assert!(name_looks_like_filesystem_path("./weights"));
        assert!(name_looks_like_filesystem_path("/tmp/m"));
        assert!(name_looks_like_filesystem_path("~/models/x"));
        assert!(!name_looks_like_filesystem_path("meta-llama/Llama-3-8B"));
    }
}
