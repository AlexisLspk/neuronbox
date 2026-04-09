//! `neuron doctor` — diagnostic checks for the NeuronBox environment.

use anyhow::Result;
use neuronbox_runtime::host::HostProbe;

use crate::daemon_client;
use crate::neuron_config::NeuronConfig;
use crate::paths::{neuronbox_home, store_root};
use crate::sdk_path;

use super::init::default_yaml_path;

/// Arguments for `neuron doctor`.
pub struct DoctorArgs {
    /// Exit with non-zero code on any warning (for CI).
    pub strict: bool,
}

/// Result of a single check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckResult {
    Ok,
    Warning,
    Error,
}

impl CheckResult {
    fn symbol(&self) -> &'static str {
        match self {
            CheckResult::Ok => "✓",
            CheckResult::Warning => "⚠",
            CheckResult::Error => "✗",
        }
    }
}

struct DoctorContext {
    warnings: usize,
    errors: usize,
}

impl DoctorContext {
    fn new() -> Self {
        Self {
            warnings: 0,
            errors: 0,
        }
    }

    fn report(&mut self, result: CheckResult, label: &str, detail: &str) {
        match result {
            CheckResult::Ok => println!("  {} {}: {}", result.symbol(), label, detail),
            CheckResult::Warning => {
                self.warnings += 1;
                eprintln!("  {} {}: {}", result.symbol(), label, detail);
            }
            CheckResult::Error => {
                self.errors += 1;
                eprintln!("  {} {}: {}", result.symbol(), label, detail);
            }
        }
    }
}

/// Run all diagnostic checks.
pub async fn doctor(args: DoctorArgs) -> Result<()> {
    println!("NeuronBox Doctor\n");

    let mut ctx = DoctorContext::new();

    // 1. Check neuronbox home directory
    check_home(&mut ctx);

    // 2. Check store directory
    check_store(&mut ctx);

    // 3. Check SDK resolution
    check_sdk(&mut ctx);

    // 4. Check Python availability
    check_python(&mut ctx);

    // 5. Check uv/pip
    check_pip(&mut ctx);

    // 6. Check daemon socket
    check_daemon(&mut ctx).await;

    // 7. Check host/GPU info
    check_host(&mut ctx);

    // 8. Check neuron.yaml (optional)
    check_yaml(&mut ctx);

    // Summary
    println!();
    if ctx.errors > 0 {
        eprintln!(
            "Doctor found {} error(s) and {} warning(s).",
            ctx.errors, ctx.warnings
        );
        std::process::exit(1);
    } else if ctx.warnings > 0 {
        eprintln!("Doctor found {} warning(s).", ctx.warnings);
        if args.strict {
            std::process::exit(1);
        }
    } else {
        println!("All checks passed.");
    }

    Ok(())
}

fn check_home(ctx: &mut DoctorContext) {
    let home = neuronbox_home();
    if home.exists() {
        ctx.report(
            CheckResult::Ok,
            "NeuronBox home",
            &home.display().to_string(),
        );
    } else {
        ctx.report(
            CheckResult::Warning,
            "NeuronBox home",
            &format!("{} (will be created on first use)", home.display()),
        );
    }
}

fn check_store(ctx: &mut DoctorContext) {
    let store = store_root();
    if store.exists() {
        let models_dir = store.join("models");
        let model_count = if models_dir.exists() {
            std::fs::read_dir(&models_dir)
                .map(|d| d.count())
                .unwrap_or(0)
        } else {
            0
        };
        ctx.report(
            CheckResult::Ok,
            "Model store",
            &format!("{} ({} model(s))", store.display(), model_count),
        );
    } else {
        ctx.report(
            CheckResult::Warning,
            "Model store",
            &format!("{} (will be created on first pull)", store.display()),
        );
    }
}

fn check_sdk(ctx: &mut DoctorContext) {
    if sdk_path::autohook_disabled() {
        ctx.report(
            CheckResult::Warning,
            "SDK auto-hooks",
            "disabled via NEURONBOX_DISABLE_AUTOHOOK",
        );
        return;
    }

    match sdk_path::get_sdk_path() {
        Some(path) => {
            ctx.report(
                CheckResult::Ok,
                "SDK path",
                &format!("{} (auto-hooks enabled)", path.display()),
            );
        }
        None => {
            ctx.report(
                CheckResult::Warning,
                "SDK path",
                "not found — auto-hooks will not work. Set NEURONBOX_SDK or install the SDK.",
            );
        }
    }
}

fn check_python(ctx: &mut DoctorContext) {
    // Check for python3 or python
    let python = which::which("python3")
        .or_else(|_| which::which("python"))
        .ok();

    match python {
        Some(path) => {
            // Get version
            let version = std::process::Command::new(&path)
                .args(["--version"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .unwrap_or_default()
                .trim()
                .to_string();

            ctx.report(
                CheckResult::Ok,
                "Python",
                &format!("{} ({})", path.display(), version),
            );
        }
        None => {
            ctx.report(
                CheckResult::Error,
                "Python",
                "not found in PATH — required for neuron run",
            );
        }
    }
}

fn check_pip(ctx: &mut DoctorContext) {
    // Prefer uv
    if which::which("uv").is_ok() {
        let version = std::process::Command::new("uv")
            .args(["--version"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string();

        ctx.report(
            CheckResult::Ok,
            "Package installer",
            &format!("uv ({})", version),
        );
        return;
    }

    // Fallback to pip
    let pip = which::which("pip3").or_else(|_| which::which("pip")).ok();
    match pip {
        Some(path) => {
            ctx.report(
                CheckResult::Ok,
                "Package installer",
                &format!("pip ({})", path.display()),
            );
        }
        None => {
            ctx.report(
                CheckResult::Warning,
                "Package installer",
                "neither uv nor pip found — venv creation may fail",
            );
        }
    }
}

async fn check_daemon(ctx: &mut DoctorContext) {
    match daemon_client::request(neuronbox_runtime::protocol::DaemonRequest::Ping).await {
        Ok(resp) => {
            if matches!(resp, neuronbox_runtime::protocol::DaemonResponse::Pong) {
                ctx.report(CheckResult::Ok, "Daemon", "running and responsive");
            } else {
                ctx.report(
                    CheckResult::Warning,
                    "Daemon",
                    &format!("unexpected response: {:?}", resp),
                );
            }
        }
        Err(_) => {
            let socket = std::env::var("NEURONBOX_SOCKET")
                .unwrap_or_else(|_| neuronbox_home().join("neuron.sock").display().to_string());
            ctx.report(
                CheckResult::Warning,
                "Daemon",
                &format!(
                    "not reachable at {} — run `neuron daemon` or it will start on `neuron run`",
                    socket
                ),
            );
        }
    }
}

fn check_host(ctx: &mut DoctorContext) {
    let snap = HostProbe::snapshot();

    // OS info
    ctx.report(
        CheckResult::Ok,
        "Host",
        &format!("{} ({})", snap.platform.os, snap.platform.arch),
    );

    // Training backend
    let backend = format!("{:?}", snap.training_backend);
    ctx.report(CheckResult::Ok, "Training backend", &backend);

    // GPUs
    if snap.gpus.is_empty() {
        ctx.report(
            CheckResult::Warning,
            "GPUs",
            "none detected (CPU-only mode)",
        );
    } else {
        for gpu in &snap.gpus {
            let vram = if gpu.memory_total_mb > 0 {
                format!("{} MB", gpu.memory_total_mb)
            } else {
                "unknown VRAM".to_string()
            };
            ctx.report(
                CheckResult::Ok,
                &format!("GPU {}", gpu.index),
                &format!("{} ({})", gpu.name, vram),
            );
        }
    }
}

fn check_yaml(ctx: &mut DoctorContext) {
    let yaml_path = default_yaml_path();
    if !yaml_path.exists() {
        ctx.report(
            CheckResult::Ok,
            "neuron.yaml",
            "not present in current directory (optional)",
        );
        return;
    }

    match NeuronConfig::load_path(&yaml_path) {
        Ok(cfg) => {
            ctx.report(
                CheckResult::Ok,
                "neuron.yaml",
                &format!(
                    "valid — project '{}', entrypoint '{}'",
                    cfg.name, cfg.entrypoint
                ),
            );

            // Check entrypoint exists
            let entry = std::env::current_dir()
                .unwrap_or_default()
                .join(&cfg.entrypoint);
            if !entry.exists() {
                ctx.report(
                    CheckResult::Warning,
                    "Entrypoint",
                    &format!("{} not found", cfg.entrypoint),
                );
            }
        }
        Err(e) => {
            ctx.report(
                CheckResult::Error,
                "neuron.yaml",
                &format!("parse error: {}", e),
            );
        }
    }
}
