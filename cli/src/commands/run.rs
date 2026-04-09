use std::path::PathBuf;

use anyhow::{Context, Result};
use neuronbox_runtime::gpu::soft_vram_check;
use neuronbox_runtime::host::HostProbe;
use neuronbox_runtime::protocol::DaemonRequest;

use crate::commands::init::default_yaml_path;
use crate::commands::pull::{looks_like_hf_repo, pull_model, pull_model_with_revision};
use crate::daemon_client;
use crate::daemon_spawn;
use crate::env_hash;
use crate::model_alias;
use crate::model_resolve::{self, use_local_filesystem};
use crate::neuron_config::NeuronConfig;
use crate::oci;
use crate::paths::{neuronbox_home, project_dir_from_yaml, store_root};
use crate::sdk_path;

pub struct RunArgs {
    pub yaml: Option<PathBuf>,
    /// `CUDA_VISIBLE_DEVICES`: `0`, `0,1`, etc.
    pub gpu_devices: Option<String>,
    pub vram_limit_mb: Option<u64>,
    pub script_args: Vec<String>,
    pub oci: bool,
}

fn pytorch_alloc_env(cfg: &NeuronConfig) -> Option<String> {
    cfg.min_vram_mb().map(|mb| {
        let split = (mb / 512).clamp(128, 4096);
        format!("max_split_size_mb:{split},expandable_segments:True")
    })
}

/// Warn if gpu.strategy suggests multi-GPU but only one GPU is visible.
fn gpu_strategy_warning(cfg: &NeuronConfig, gpu_devices: Option<&str>) -> Option<String> {
    let strategy = cfg.gpu.strategy.to_lowercase();
    if strategy != "pipeline" && strategy != "tensor" {
        return None;
    }

    // Count visible GPUs from CUDA_VISIBLE_DEVICES or --gpu flag
    let visible_count = match gpu_devices.or_else(|| {
        std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .as_deref()
            .map(|_| "")
    }) {
        Some(devices) if !devices.is_empty() => {
            // Parse comma-separated device IDs
            devices.split(',').filter(|s| !s.trim().is_empty()).count()
        }
        _ => {
            // No explicit device selection, check actual GPU count
            let snap = HostProbe::snapshot();
            snap.gpus.len()
        }
    };

    if visible_count <= 1 {
        return Some(format!(
            "gpu.strategy: {} typically expects multiple GPUs, but {} GPU(s) are visible. See docs/MULTI_GPU.md for torchrun/DDP setup.",
            cfg.gpu.strategy, visible_count
        ));
    }
    None
}

fn ask_confirmation(prompt: &str) -> Result<bool> {
    use std::io::{self, Write};
    print!("{prompt} [y/N]: ");
    io::stdout().flush().ok();
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    let a = s.trim().to_ascii_lowercase();
    Ok(matches!(a.as_str(), "y" | "yes"))
}

/// `neuron run` with `neuron.yaml` in the current directory.
pub async fn run_project(args: RunArgs) -> Result<()> {
    let yaml_path = args.yaml.clone().unwrap_or_else(default_yaml_path);
    if !yaml_path.exists() {
        anyhow::bail!("{:?} not found.", yaml_path);
    }
    let cfg = NeuronConfig::load_path(&yaml_path)?;
    let cwd = project_dir_from_yaml(&yaml_path);

    let hf_repo_id = model_alias::resolve_for_hf_id(&cfg.model.name);
    if !use_local_filesystem(&cfg) {
        pull_model_with_revision(&hf_repo_id, cfg.model.revision.as_deref())?;
    }

    let mut pre_run_warnings: Vec<String> = Vec::new();
    if let Some(req) = cfg.min_vram_mb() {
        let snap = HostProbe::snapshot();
        if let Some(avail) = snap.primary_vram_mb() {
            if avail > 0 {
                if let Err(msg) = soft_vram_check(avail, req, &cfg.name) {
                    pre_run_warnings.push(msg);
                }
            }
        }
    }

    if let Some(w) = gpu_strategy_warning(&cfg, args.gpu_devices.as_deref()) {
        pre_run_warnings.push(w);
    }

    if !pre_run_warnings.is_empty() {
        eprintln!("\nneuron: pre-run checks found potential issues:");
        for w in &pre_run_warnings {
            eprintln!("\n{w}\n");
        }
        if !ask_confirmation("Continue anyway and start the run?")? {
            anyhow::bail!("run cancelled by user after pre-run checks");
        }
    }

    let store = store_root();
    let resolved = model_resolve::resolve_model_for_run(&cfg, &cwd, &store, &hf_repo_id)?;

    let model_dir = resolved.dir.clone();

    let env_root = env_hash::env_path(&store, &cfg);
    let venv_python = env_hash::ensure_python_env(&env_root, &cfg)?;

    let entry = cwd.join(&cfg.entrypoint);
    if !entry.exists() {
        anyhow::bail!("entrypoint not found: {:?}", entry);
    }

    let hf_home = neuronbox_home().join("hf-cache");
    std::fs::create_dir_all(&hf_home).ok();

    if oci::oci_enabled(&cfg, args.oci) {
        if cfg.container.executor_resolved() == "runc" {
            anyhow::bail!(
                "container.executor: runc is not supported for `neuron run`. Use executor: docker, or `neuron oci prepare` then `neuron oci runc`."
            );
        }
        if !cfg!(target_os = "linux") {
            anyhow::bail!(
                "runtime.mode: OCI with Docker GPU targets Linux + NVIDIA Container Toolkit."
            );
        }
        let mut extra: Vec<(String, String)> = Vec::new();
        for (k, v) in &cfg.env {
            extra.push((k.clone(), v.clone()));
        }
        extra.push((
            "NEURONBOX_MODEL_SOURCE".into(),
            if use_local_filesystem(&cfg) {
                "local".into()
            } else {
                "huggingface".into()
            },
        ));
        if let Some(ref f) = resolved.weights_file {
            extra.push(("NEURONBOX_MODEL_PATH".into(), f.display().to_string()));
        }
        if let Some(ref s) = pytorch_alloc_env(&cfg) {
            extra.push(("PYTORCH_CUDA_ALLOC_CONF".into(), s.clone()));
        }
        let st = oci::run_docker_isolated(oci::DockerRunParams {
            image: &cfg.container.image_resolved(),
            project_dir: &cwd,
            model_dir: &model_dir,
            venv_root: &env_root,
            hf_home: &hf_home,
            store_root: &store,
            entrypoint_rel: &cfg.entrypoint,
            script_args: &args.script_args,
            extra_env: &extra,
            gpu_devices: args.gpu_devices.as_deref(),
        })?;
        if !st.success() {
            anyhow::bail!("docker run (OCI mode) exited with code {:?}", st.code());
        }
        return Ok(());
    }

    daemon_spawn::ensure_daemon_running().await.ok();

    let est_mb = args.vram_limit_mb.or(cfg.min_vram_mb()).unwrap_or(8192);

    let mut cmd = tokio::process::Command::new(&venv_python);
    cmd.current_dir(&cwd);
    cmd.arg(&entry);
    for a in &args.script_args {
        cmd.arg(a);
    }
    cmd.env("NEURONBOX_MODEL_DIR", &model_dir);
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
    if let Some(ref g) = args.gpu_devices {
        cmd.env("CUDA_VISIBLE_DEVICES", g);
    }
    if let Some(s) = pytorch_alloc_env(&cfg) {
        cmd.env("PYTORCH_CUDA_ALLOC_CONF", s);
    }
    cmd.env("NEURONBOX_SESSION_NAME", &cfg.name);
    cmd.env("NEURONBOX_SESSION_VRAM_MB", format!("{est_mb}"));

    // Enable automatic throughput hooks for ML frameworks
    if !sdk_path::autohook_disabled() {
        cmd.env("NEURONBOX_AUTOHOOK", "1");
        if let Some(sdk) = sdk_path::get_sdk_path() {
            // Prepend SDK to PYTHONPATH so hooks are available
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

    let mut child = cmd.spawn().context("spawn Python")?;
    let pid = child.id().expect("pid");

    if let Err(e) = daemon_client::request(DaemonRequest::RegisterSession {
        name: cfg.name.clone(),
        estimated_vram_mb: est_mb,
        pid,
        tokens_per_sec: None,
        model_dir: Some(model_dir.display().to_string()),
    })
    .await
    {
        eprintln!(
            "neuron: warning — RegisterSession failed (dashboard may not show this run): {e:#}"
        );
    }

    let status = child.wait().await.context("wait for process")?;

    if let Err(e) = daemon_client::request(DaemonRequest::UnregisterSession { pid }).await {
        eprintln!("neuron: warning — UnregisterSession failed: {e:#}");
    }

    if !status.success() {
        anyhow::bail!("process exited with code {:?}", status.code());
    }
    Ok(())
}

/// `neuron run org/model` or `neuron run llama3:8b` — download and prepare.
pub async fn run_direct_model(
    model_ref: &str,
    _gpu: Option<String>,
    _vram: Option<u64>,
) -> Result<()> {
    let resolved = model_alias::resolve_for_hf_id(model_ref);
    if !looks_like_hf_repo(&resolved) {
        anyhow::bail!(
            "unrecognized direct model: {} (expected Hugging Face org/model or a short alias like llama3:8b).",
            model_ref
        );
    }
    pull_model(model_ref)?;
    println!(
        "Model {} (→ {}) is ready in the store. Create a neuron.yaml with this model and an entrypoint to train.",
        model_ref,
        resolved
    );
    Ok(())
}
