//! Surveillance « soft » : si l’usage VRAM rapporté par `nvidia-smi` pour un PID dépasse
//! ~115 % de l’estimation enregistrée, envoi de SIGKILL (Linux uniquement).

use std::process::Command;
use std::time::Duration;

use tokio::time::sleep;

use crate::gpu_manager::GpuManager;
use crate::host::compute_apps_pid_memory_mb;

pub async fn run_soft_vram_enforcement(gm: GpuManager) {
    loop {
        sleep(Duration::from_secs(5)).await;
        let usage =
            match tokio::task::spawn_blocking(|| compute_apps_pid_memory_mb().ok_or(())).await {
                Ok(Ok(m)) => m,
                _ => continue,
            };
        let sessions = gm.list().await;
        for s in sessions {
            let Some(used_mb) = usage.get(&s.pid).copied() else {
                continue;
            };
            let limit = s.estimated_vram_mb.saturating_mul(115) / 100;
            if used_mb > limit && limit > 0 {
                tracing::warn!(
                    "VRAM soft limit: pid {} ({}) used {} MiB > {} MiB — SIGKILL",
                    s.pid,
                    s.name,
                    used_mb,
                    limit
                );
                kill_pid_hard(s.pid);
                gm.unregister(s.pid).await;
            }
        }
    }
}

#[cfg(unix)]
fn kill_pid_hard(pid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .status();
}

#[cfg(not(unix))]
fn kill_pid_hard(_pid: u32) {}
