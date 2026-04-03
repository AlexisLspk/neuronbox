use std::path::Path;

use anyhow::{Context, Result};

use crate::model_alias::{self, looks_like_short_tag};
use crate::paths::store_root;
use crate::store::model_store;
use crate::store::registry::{IndexEntry, StoreIndex};

pub fn looks_like_hf_repo(s: &str) -> bool {
    let s = s.trim();
    if s.starts_with("./") || s.starts_with('/') {
        return false;
    }
    if s.contains(':') {
        return false;
    }
    s.matches('/').count() == 1
}

fn unsupported_pull_target(target: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "unsupported target for `neuron pull`: {target:?}\n\
         \n\
         `neuron pull` only fetches ML weights into the NeuronBox store:\n\
           - Hugging Face-style id: org/model (one slash, no colon)\n\
           - short alias: e.g. llama3:8b if listed in bundled aliases\n\
           - local path: ./weights or /path/to/folder\n\
         \n\
         OCI image tags like ubuntu:22.04 are not ML models here.\n\
         To pull a container image, run: docker pull {target}\n\
         To build a runc bundle from an image (requires Docker): neuron oci prepare --image {target}\n\
         \n\
         For containerized project runs, use `neuron run --oci` with runtime.mode: oci in neuron.yaml."
    )
}

pub fn pull_model(target: &str) -> Result<()> {
    let store = store_root();
    std::fs::create_dir_all(&store).context("mkdir store")?;

    if target.starts_with("./") || target.starts_with('/') {
        return pull_local_path(Path::new(target));
    }

    let resolved = model_alias::resolve_for_hf_id(target);
    if looks_like_hf_repo(&resolved) {
        let token = std::env::var("HF_TOKEN").ok();
        let path = model_store::ensure_hf_model(&store, &resolved, token.as_deref())
            .with_context(|| format!("HF download {}", resolved))?;
        if resolved != target {
            println!("Alias {} → {}", target, resolved);
        }
        println!("Model ready: {}", path.display());
        return Ok(());
    }

    if looks_like_short_tag(target) {
        anyhow::bail!(
            "unknown alias: {}. Add it in ~/.neuronbox/model-aliases.json or cli/specs/model-aliases.json (from source).",
            target
        );
    }

    Err(unsupported_pull_target(target))
}

fn pull_local_path(src: &Path) -> Result<()> {
    let store = store_root();
    let name = src
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("local-model");
    let dest = model_store::models_root(&store).join(name.replace('/', "--"));
    if dest.exists() {
        println!("Already present: {}", dest.display());
        return Ok(());
    }
    if !src.exists() {
        anyhow::bail!("local path not found: {:?}", src);
    }
    copy_dir(src, &dest)?;
    let canon = src.canonicalize().unwrap_or_else(|_| src.to_path_buf());
    let mut idx = StoreIndex::load(&store)?;
    let key = format!("local:{}", canon.display());
    idx.models.insert(
        key,
        IndexEntry {
            path: dest.clone(),
            revision: None,
            source: "local".to_string(),
        },
    );
    idx.save(&store)?;
    println!("Local import: {}", dest.display());
    Ok(())
}

fn copy_dir(src: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    copy_dir_inner(src, src, dest)
}

fn copy_dir_inner(root: &Path, src: &Path, dest_root: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src).with_context(|| format!("read_dir {:?}", src))? {
        let entry = entry?;
        let p = entry.path();
        let rel = p.strip_prefix(root)?;
        let out = dest_root.join(rel);
        if p.is_dir() {
            std::fs::create_dir_all(&out)?;
            copy_dir_inner(root, &p, dest_root)?;
        } else {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&p, &out)?;
        }
    }
    Ok(())
}
