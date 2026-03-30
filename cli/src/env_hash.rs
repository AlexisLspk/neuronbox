//! Hash of the `runtime` section to reuse venvs under `store/envs/`.
//! If `requirements.lock` exists, `uv pip sync --frozen` is used (reproducibility).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use neuronbox_runtime::host::{HostProbe, TrainingBackend};
use sha2::{Digest, Sha256};

use crate::neuron_config::NeuronConfig;

pub fn env_dir_name(cfg: &NeuronConfig) -> String {
    let payload = serde_json::json!({
        "python": cfg.runtime.python,
        "cuda": cfg.runtime.cuda,
        "packages": cfg.runtime.packages,
    });
    let canonical = serde_json::to_string(&payload).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    let hex = hex::encode(h.finalize());
    format!("py-{}", &hex[..16])
}

pub fn env_path(store_root: &Path, cfg: &NeuronConfig) -> PathBuf {
    store_root.join("envs").join(env_dir_name(cfg))
}

fn training_backend() -> TrainingBackend {
    HostProbe::snapshot().training_backend
}

fn has_rocm_gpu() -> bool {
    matches!(training_backend(), TrainingBackend::Rocm)
}

fn use_cuda_pytorch_index(cfg: &NeuronConfig) -> bool {
    let cuda = cfg.runtime.cuda.as_deref().unwrap_or("");
    if cuda.eq_ignore_ascii_case("none") || cuda.is_empty() {
        return matches!(training_backend(), TrainingBackend::Cuda);
    }
    true
}

/// Create a venv; sync from `requirements.lock` if present, else `uv pip install` / `pip`.
pub fn ensure_python_env(env_root: &Path, cfg: &NeuronConfig) -> Result<PathBuf> {
    let py_bin = find_python(&cfg.runtime.python)?;
    std::fs::create_dir_all(env_root)?;

    let venv_python = if cfg!(target_os = "windows") {
        env_root.join("Scripts").join("python.exe")
    } else {
        env_root.join("bin").join("python")
    };

    if !venv_python.exists() {
        let status = Command::new(&py_bin)
            .args(["-m", "venv"])
            .arg(env_root)
            .status()
            .context("python -m venv")?;
        if !status.success() {
            anyhow::bail!("python -m venv failed");
        }
    }

    let lock_path = env_root.join("requirements.lock");
    if lock_path.exists() && which::which("uv").is_ok() {
        let status = Command::new("uv")
            .args(["pip", "sync", "--frozen", "--python"])
            .arg(&venv_python)
            .arg(&lock_path)
            .status()
            .context("uv pip sync")?;
        if !status.success() {
            anyhow::bail!("uv pip sync --frozen failed — regenerate the lock with `neuron lock`");
        }
        return Ok(venv_python);
    }

    let mut extra_index: Vec<String> = Vec::new();
    if has_rocm_gpu() {
        extra_index.push("https://download.pytorch.org/whl/rocm6.0".to_string());
    } else if use_cuda_pytorch_index(cfg) {
        let cuda = cfg
            .runtime
            .cuda
            .clone()
            .unwrap_or_else(|| "12.1".to_string());
        let tag = cuda.replace('.', "");
        if !tag.is_empty() {
            extra_index.push(format!("https://download.pytorch.org/whl/cu{tag}"));
        }
    }

    let pip_exe = if cfg!(target_os = "windows") {
        env_root.join("Scripts").join("pip.exe")
    } else {
        env_root.join("bin").join("pip")
    };

    if which::which("uv").is_ok() {
        let mut cmd = Command::new("uv");
        cmd.arg("pip").arg("install");
        for url in &extra_index {
            cmd.arg("--extra-index-url").arg(url);
        }
        for p in &cfg.runtime.packages {
            cmd.arg(p);
        }
        if !extra_index.is_empty() {
            cmd.args(["torch", "torchvision", "torchaudio"]);
        }
        cmd.arg("--python").arg(&venv_python);
        let status = cmd.status().context("uv pip install")?;
        if !status.success() {
            anyhow::bail!("uv pip install failed — check CUDA/ROCm/Python versions");
        }
    } else {
        let mut cmd = Command::new(&pip_exe);
        cmd.arg("install");
        for url in &extra_index {
            cmd.arg("--extra-index-url").arg(url);
        }
        for p in &cfg.runtime.packages {
            cmd.arg(p);
        }
        if !extra_index.is_empty() {
            cmd.args(["torch", "torchvision", "torchaudio"]);
        }
        let status = cmd.status().context("pip install")?;
        if !status.success() {
            anyhow::bail!("pip install failed — install `uv` for more reliable resolution");
        }
    }

    Ok(venv_python)
}

fn find_python(spec: &str) -> Result<PathBuf> {
    let candidates = [
        format!("python{spec}"),
        "python3".to_string(),
        "python".to_string(),
    ];
    for c in candidates {
        if let Ok(path) = which::which(&c) {
            return Ok(path);
        }
    }
    anyhow::bail!(
        "Python interpreter not found (expected: {}). Install Python or adjust runtime.python.",
        spec
    );
}
