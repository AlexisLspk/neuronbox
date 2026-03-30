use std::path::Path;
use std::process::Stdio;

use anyhow::{Context, Result};

pub struct DockerRunParams<'a> {
    pub image: &'a str,
    pub project_dir: &'a Path,
    pub model_dir: &'a Path,
    pub venv_root: &'a Path,
    pub hf_home: &'a Path,
    pub store_root: &'a Path,
    pub entrypoint_rel: &'a str,
    pub script_args: &'a [String],
    pub extra_env: &'a [(String, String)],
    /// `Some("0")`, `Some("0,1")`, or None for `--gpus all`.
    pub gpu_devices: Option<&'a str>,
}

/// Run `docker run` with store / venv / project mounts (practical OCI-style runtime).
pub fn run_docker_isolated(params: DockerRunParams<'_>) -> Result<std::process::ExitStatus> {
    let mut cmd = std::process::Command::new("docker");
    cmd.arg("run").arg("--rm").arg("-i");
    cmd.arg("-w").arg("/work");
    if cfg!(target_os = "linux") {
        match params.gpu_devices {
            Some(dev) if !dev.is_empty() => {
                cmd.arg("--gpus").arg(format!("device={dev}"));
            }
            _ => {
                cmd.arg("--gpus").arg("all");
            }
        }
    }
    cmd.arg("-v")
        .arg(format!("{}:/work:rw", params.project_dir.display()));
    cmd.arg("-v").arg(format!(
        "{}:/opt/neuron-venv:ro",
        params.venv_root.display()
    ));
    cmd.arg("-v")
        .arg(format!("{}:/models:rw", params.model_dir.display()));
    cmd.arg("-v")
        .arg(format!("{}:/hf:rw", params.hf_home.display()));
    cmd.arg("-v").arg(format!(
        "{}:/root/.neuronbox/store:rw",
        params.store_root.display()
    ));

    for (k, v) in params.extra_env {
        cmd.arg("-e").arg(format!("{k}={v}"));
    }
    cmd.arg("-e").arg("NEURONBOX_MODEL_DIR=/models");
    cmd.arg("-e").arg("HF_HOME=/hf");

    cmd.arg(params.image);
    cmd.arg("/opt/neuron-venv/bin/python");
    cmd.arg(format!("/work/{}", params.entrypoint_rel));
    for a in params.script_args {
        cmd.arg(a);
    }
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let st = cmd.status().context("docker run")?;
    Ok(st)
}
