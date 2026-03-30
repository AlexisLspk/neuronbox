//! Hugging Face download to `store/models/org--model/` via local cache.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;

use super::registry::{IndexEntry, StoreIndex};

pub fn model_dir_name(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

pub fn models_root(store_root: &Path) -> PathBuf {
    store_root.join("models")
}

pub fn model_path(store_root: &Path, repo_id: &str) -> PathBuf {
    models_root(store_root).join(model_dir_name(repo_id))
}

/// Download the repo if missing (presence of `config.json` or `model.safetensors`).
pub fn ensure_hf_model(store_root: &Path, repo_id: &str, token: Option<&str>) -> Result<PathBuf> {
    let dest = model_path(store_root, repo_id);
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

    let repo = api.model(repo_id.to_string());
    let info = repo.info().context("hf model info")?;

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
        std::fs::copy(&src, &target).with_context(|| format!("copy {:?} -> {:?}", src, target))?;
    }

    let mut idx = StoreIndex::load(store_root)?;
    idx.models.insert(
        repo_id.to_string(),
        IndexEntry {
            path: dest.clone(),
            revision: Some(info.sha.clone()),
            source: "huggingface".to_string(),
        },
    );
    idx.save(store_root)?;

    Ok(dest)
}
