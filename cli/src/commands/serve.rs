use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::daemon_spawn;
use crate::env_hash;
use crate::model_resolve::use_local_filesystem;
use crate::paths::{neuronbox_home, project_dir_from_yaml, store_root};
use crate::sdk_path;
use crate::{model_alias, model_resolve};

use super::init::default_yaml_path;
use crate::commands::pull::pull_model;
use crate::neuron_config::NeuronConfig;

const WORKER_REL: &str = "scripts/serve_worker.py";

fn pytorch_alloc_env(cfg: &NeuronConfig) -> Option<String> {
    cfg.min_vram_mb().map(|mb| {
        let split = (mb / 512).clamp(128, 4096);
        format!("max_split_size_mb:{split},expandable_segments:True")
    })
}

/// Start the daemon and a Python worker that reacts to `swap_signal.json` and HF swaps.
/// Uses the **same venv** as `neuron run` for this `neuron.yaml`.
pub async fn serve(yaml: Option<PathBuf>) -> Result<()> {
    daemon_spawn::ensure_daemon_running().await?;

    let path = yaml.unwrap_or_else(default_yaml_path);
    let cfg = NeuronConfig::load_path(&path)?;
    let cwd = project_dir_from_yaml(&path);

    let hf_id = model_alias::resolve_for_hf_id(&cfg.model.name);
    if !model_resolve::use_local_filesystem(&cfg) {
        pull_model(&hf_id)?;
    }
    let store = store_root();
    let resolved = model_resolve::resolve_model_for_run(&cfg, &cwd, &store, &hf_id)?;

    let env_root = env_hash::env_path(&store, &cfg);
    let venv_python = env_hash::ensure_python_env(&env_root, &cfg)?;

    let worker_py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(WORKER_REL);
    if !worker_py.exists() {
        anyhow::bail!("worker not found: {:?}", worker_py);
    }

    let hf_home = neuronbox_home().join("hf-cache");
    std::fs::create_dir_all(&hf_home).ok();

    let mut cmd = tokio::process::Command::new(&venv_python);
    cmd.arg(&worker_py);
    cmd.current_dir(&cwd);
    cmd.env("NEURONBOX_HOME", neuronbox_home());
    cmd.env("NEURONBOX_MODEL_DIR", &resolved.dir);
    cmd.env(
        "NEURONBOX_MODEL_SOURCE",
        if use_local_filesystem(&cfg) {
            "local"
        } else {
            "huggingface"
        },
    );
    if let Some(ref f) = resolved.weights_file {
        cmd.env("NEURONBOX_MODEL_PATH", f);
    }
    cmd.env("HF_HOME", &hf_home);
    if let Some(s) = pytorch_alloc_env(&cfg) {
        cmd.env("PYTORCH_CUDA_ALLOC_CONF", s);
    }

    // Enable automatic throughput hooks for ML frameworks (same as neuron run)
    if !sdk_path::autohook_disabled() {
        cmd.env("NEURONBOX_AUTOHOOK", "1");
        if let Some(sdk) = sdk_path::get_sdk_path() {
            let existing = std::env::var("PYTHONPATH").unwrap_or_default();
            let new_pythonpath = if existing.is_empty() {
                sdk.display().to_string()
            } else {
                format!("{}:{}", sdk.display(), existing)
            };
            cmd.env("PYTHONPATH", new_pythonpath);
        } else if !cfg.env.contains_key("PYTHONPATH") {
            cmd.env_remove("PYTHONPATH");
        }
    } else if !cfg.env.contains_key("PYTHONPATH") {
        cmd.env_remove("PYTHONPATH");
    }

    for (k, v) in &cfg.env {
        cmd.env(k, v);
    }
    cmd.stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());

    let status = cmd.status().await.context("python serve worker")?;
    if !status.success() {
        anyhow::bail!("worker exited with {:?}", status.code());
    }
    Ok(())
}
