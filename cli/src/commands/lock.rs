use std::path::Path;

use anyhow::{Context, Result};

use crate::commands::init::default_yaml_path;
use crate::env_hash;
use crate::neuron_config::NeuronConfig;
use crate::paths::store_root;

/// Generate `store/envs/<hash>/requirements.lock` via `uv pip compile` (reproducibility).
pub fn generate_lock(yaml: Option<&Path>) -> Result<()> {
    let path = yaml
        .map(Path::to_path_buf)
        .unwrap_or_else(default_yaml_path);
    let cfg = NeuronConfig::load_path(&path)?;
    let store = store_root();
    let env_root = env_hash::env_path(&store, &cfg);
    std::fs::create_dir_all(&env_root).context("mkdir env")?;

    let req_in = env_root.join("requirements.in");
    let body: String = cfg
        .runtime
        .packages
        .iter()
        .map(|p| format!("{p}\n"))
        .collect();
    std::fs::write(&req_in, body).context("write requirements.in")?;

    let lock_path = env_root.join("requirements.lock");
    let status = std::process::Command::new("uv")
        .arg("pip")
        .arg("compile")
        .arg(&req_in)
        .arg("-o")
        .arg(&lock_path)
        .status()
        .context("uv pip compile — install uv")?;
    if !status.success() {
        anyhow::bail!("uv pip compile failed");
    }
    println!("Lock written: {}", lock_path.display());
    Ok(())
}
