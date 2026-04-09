use std::path::Path;

use anyhow::{Context, Result};
use clap::Subcommand;

use crate::paths::store_root;
use crate::store::model_store::models_root;
use crate::store::registry::StoreIndex;

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List models registered in index.json.
    List {
        /// Show disk usage for each model.
        #[arg(long)]
        sizes: bool,
    },
    /// Show disk usage for all models in the store.
    Du,
    /// Remove a model from the store.
    Prune {
        /// Model ID to remove (as shown in `neuron model list`).
        model_id: String,
        /// Actually delete the model (default is dry-run).
        #[arg(long)]
        execute: bool,
    },
}

pub fn model(cmd: ModelCommands) -> Result<()> {
    match cmd {
        ModelCommands::List { sizes } => list_models(sizes),
        ModelCommands::Du => disk_usage(),
        ModelCommands::Prune { model_id, execute } => prune_model(&model_id, execute),
    }
}

fn list_models(show_sizes: bool) -> Result<()> {
    let store = store_root();
    let idx = StoreIndex::load(&store)?;
    if idx.models.is_empty() {
        println!("No models in the store. Use `neuron pull org/model`.");
        return Ok(());
    }
    for (id, e) in &idx.models {
        if show_sizes {
            let size = dir_size(&e.path).unwrap_or(0);
            println!("{id} -> {} ({})", e.path.display(), format_size(size));
        } else {
            println!("{id} -> {}", e.path.display());
        }
    }
    Ok(())
}

fn disk_usage() -> Result<()> {
    let store = store_root();
    let idx = StoreIndex::load(&store)?;

    if idx.models.is_empty() {
        println!("No models in the store.");
        return Ok(());
    }

    let mut total: u64 = 0;
    let mut entries: Vec<(String, u64)> = Vec::new();

    for (id, e) in &idx.models {
        let size = dir_size(&e.path).unwrap_or(0);
        entries.push((id.clone(), size));
        total += size;
    }

    // Sort by size descending
    entries.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Model store disk usage:\n");
    for (id, size) in &entries {
        println!("  {:>10}  {}", format_size(*size), id);
    }
    println!(
        "\n  {:>10}  TOTAL ({} model(s))",
        format_size(total),
        entries.len()
    );

    // Also show HF cache if present
    let hf_cache = store.join(".hf-cache");
    if hf_cache.exists() {
        let cache_size = dir_size(&hf_cache).unwrap_or(0);
        println!(
            "  {:>10}  .hf-cache (can be cleaned with `rm -rf`)",
            format_size(cache_size)
        );
    }

    Ok(())
}

fn prune_model(model_id: &str, execute: bool) -> Result<()> {
    let store = store_root();
    let mut idx = StoreIndex::load(&store)?;

    let entry = idx.models.get(model_id).cloned();
    let Some(entry) = entry else {
        anyhow::bail!(
            "Model '{}' not found in index. Use `neuron model list` to see available models.",
            model_id
        );
    };

    // Safety: ensure the path is under the store's models directory
    let models_dir = models_root(&store);
    let canonical_path = entry
        .path
        .canonicalize()
        .unwrap_or_else(|_| entry.path.clone());
    let canonical_models = models_dir
        .canonicalize()
        .unwrap_or_else(|_| models_dir.clone());

    if !canonical_path.starts_with(&canonical_models) {
        anyhow::bail!(
            "Safety check failed: model path {:?} is not under {:?}. Refusing to delete.",
            entry.path,
            models_dir
        );
    }

    let size = dir_size(&entry.path).unwrap_or(0);

    if !execute {
        println!("Dry run — would remove:");
        println!("  Model ID: {}", model_id);
        println!("  Path: {}", entry.path.display());
        println!("  Size: {}", format_size(size));
        println!(
            "\nTo actually delete, run: neuron model prune {} --execute",
            model_id
        );
        return Ok(());
    }

    // Actually delete
    if entry.path.exists() {
        std::fs::remove_dir_all(&entry.path)
            .with_context(|| format!("failed to remove {:?}", entry.path))?;
    }

    idx.models.remove(model_id);
    idx.save(&store)?;

    println!("Removed: {} ({})", model_id, format_size(size));
    Ok(())
}

/// Calculate total size of a directory recursively.
fn dir_size(path: &Path) -> Result<u64> {
    let mut total: u64 = 0;
    if path.is_file() {
        return Ok(path.metadata().map(|m| m.len()).unwrap_or(0));
    }
    if !path.is_dir() {
        return Ok(0);
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() {
            total += p.metadata().map(|m| m.len()).unwrap_or(0);
        } else if p.is_dir() {
            total += dir_size(&p).unwrap_or(0);
        }
    }
    Ok(total)
}

/// Format bytes as human-readable size.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
