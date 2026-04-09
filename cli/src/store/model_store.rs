//! Hugging Face download to `store/models/org--model/` via local cache.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;

use super::registry::{IndexEntry, StoreIndex};

/// Layout mode for model files: copy (default) or symlink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfLayout {
    Copy,
    Symlink,
}

impl HfLayout {
    /// Read from NEURONBOX_HF_LAYOUT environment variable.
    /// Default is Copy. Symlink only works on Unix.
    pub fn from_env() -> Self {
        match std::env::var("NEURONBOX_HF_LAYOUT")
            .ok()
            .as_deref()
            .map(|s| s.to_lowercase())
            .as_deref()
        {
            Some("symlink" | "link" | "sym") => {
                #[cfg(unix)]
                {
                    HfLayout::Symlink
                }
                #[cfg(not(unix))]
                {
                    eprintln!(
                        "neuron: warning — NEURONBOX_HF_LAYOUT=symlink is not supported on this platform, using copy"
                    );
                    HfLayout::Copy
                }
            }
            _ => HfLayout::Copy,
        }
    }
}

pub fn model_dir_name(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// Model directory name including revision if specified.
pub fn model_dir_name_with_revision(repo_id: &str, revision: Option<&str>) -> String {
    match revision {
        Some(rev) if !rev.is_empty() && rev != "main" => {
            // Shorten long commit hashes for directory names
            let short_rev = if rev.len() > 12 { &rev[..12] } else { rev };
            format!("{}@{}", repo_id.replace('/', "--"), short_rev)
        }
        _ => repo_id.replace('/', "--"),
    }
}

pub fn models_root(store_root: &Path) -> PathBuf {
    store_root.join("models")
}

pub fn model_path(store_root: &Path, repo_id: &str) -> PathBuf {
    models_root(store_root).join(model_dir_name(repo_id))
}

/// Model path with optional revision.
pub fn model_path_with_revision(
    store_root: &Path,
    repo_id: &str,
    revision: Option<&str>,
) -> PathBuf {
    models_root(store_root).join(model_dir_name_with_revision(repo_id, revision))
}

/// Download the repo if missing (presence of `config.json` or `model.safetensors`).
/// Backward compatible: no revision = default branch.
#[allow(dead_code)]
pub fn ensure_hf_model(store_root: &Path, repo_id: &str, token: Option<&str>) -> Result<PathBuf> {
    ensure_hf_model_with_revision(store_root, repo_id, token, None)
}

/// Download the repo with optional revision.
pub fn ensure_hf_model_with_revision(
    store_root: &Path,
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<PathBuf> {
    let dest = model_path_with_revision(store_root, repo_id, revision);

    // Check if already present
    if dest.join("config.json").exists() || dest.join("model.safetensors").exists() {
        return Ok(dest);
    }
    std::fs::create_dir_all(&dest).with_context(|| format!("mkdir {:?}", dest))?;

    let cache_dir = store_root.join(".hf-cache");
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .with_token(token.map(String::from))
        .with_progress(true)
        .build()
        .context("hf ApiBuilder")?;

    // Use revision if specified
    let repo = if let Some(rev) = revision {
        api.repo(Repo::with_revision(
            repo_id.to_string(),
            hf_hub::RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.model(repo_id.to_string())
    };

    let info = repo.info().context("hf model info")?;

    let layout = HfLayout::from_env();

    for s in &info.siblings {
        let file = &s.rfilename;
        if file.is_empty() {
            continue;
        }
        let src = repo
            .download(file)
            .with_context(|| format!("download {file}"))?;
        let target = dest.join(file);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        match layout {
            HfLayout::Copy => {
                std::fs::copy(&src, &target)
                    .with_context(|| format!("copy {:?} -> {:?}", src, target))?;
            }
            #[cfg(unix)]
            HfLayout::Symlink => {
                // Create symlink to the cached file
                if target.exists() || target.symlink_metadata().is_ok() {
                    std::fs::remove_file(&target).ok();
                }
                std::os::unix::fs::symlink(&src, &target)
                    .with_context(|| format!("symlink {:?} -> {:?}", src, target))?;
            }
            #[cfg(not(unix))]
            HfLayout::Symlink => {
                // Fallback to copy on non-Unix
                std::fs::copy(&src, &target)
                    .with_context(|| format!("copy {:?} -> {:?}", src, target))?;
            }
        }
    }

    // Index key includes revision if specified
    let index_key = match revision {
        Some(rev) if !rev.is_empty() && rev != "main" => format!("{}@{}", repo_id, rev),
        _ => repo_id.to_string(),
    };

    let mut idx = StoreIndex::load(store_root)?;
    idx.models.insert(
        index_key,
        IndexEntry {
            path: dest.clone(),
            revision: Some(info.sha.clone()),
            source: "huggingface".to_string(),
        },
    );
    idx.save(store_root)?;

    Ok(dest)
}
