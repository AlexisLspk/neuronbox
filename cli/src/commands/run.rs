use std::path::PathBuf;

use anyhow::{Context, Result};
use neuronbox_runtime::gpu::soft_vram_check;
use neuronbox_runtime::host::HostProbe;
use neuronbox_runtime::protocol::DaemonRequest;

use crate::commands::init::default_yaml_path;
use crate::commands::pull::{looks_like_hf_repo, pull_model};
use crate::daemon_client;
use crate::daemon_spawn;
use crate::env_hash;
use crate::model_alias;
use crate::model_resolve::{self, use_local_filesystem};
use crate::neuron_config::NeuronConfig;
use crate::oci;
use crate::paths::{neuronbox_home, project_dir_from_yaml, store_root};

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
        pull_model(&hf_repo_id)?;
    }

    if let Some(req) = cfg.min_vram_mb() {
        let snap = HostProbe::snapshot();
        if let Some(avail) = snap.primary_vram_mb() {
            if avail > 0 {
                soft_vram_check(avail, req, &cfg.name).map_err(anyhow::Error::msg)?;
            }
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
    if !cfg.env.contains_key("PYTHONPATH") {
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
