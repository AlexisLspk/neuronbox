use anyhow::Result;
use neuronbox_runtime::protocol::{DaemonRequest, DaemonResponse};

use crate::daemon_client;
use crate::daemon_spawn;

pub async fn stats() -> Result<()> {
    daemon_spawn::ensure_daemon_running().await?;

    let resp = daemon_client::request(DaemonRequest::Stats).await?;

    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ NeuronBox stats                                                          │");
    println!("├──────────────┬──────┬─────────┬─────────┬──────────────────────────────┤");
    println!("│ Session      │ PID  │ Est MiB │ NVIDIA  │ Tokens/s                     │");
    println!("├──────────────┼──────┼─────────┼─────────┼──────────────────────────────┤");

    match resp {
        DaemonResponse::Stats {
            sessions,
            gpu_lines,
            note,
            active_model,
            vram_used_by_pid,
        } => {
            if sessions.is_empty() {
                println!(
                    "│ (no registered sessions)                                                 │"
                );
            }
            for s in &sessions {
                let tok = s
                    .tokens_per_sec
                    .map(|t| format!("{t:.1}"))
                    .unwrap_or_else(|| "—".to_string());
                let name = truncate(&s.name, 12);
                let nv = vram_used_by_pid
                    .get(&s.pid)
                    .map(|m| format!("{m}"))
                    .unwrap_or_else(|| "—".into());
                println!(
                    "│ {:<12} │ {:>4} │ {:>7} │ {:>7} │ {:<28} │",
                    name,
                    s.pid,
                    s.estimated_vram_mb,
                    nv,
                    truncate(&tok, 28),
                );
            }
            println!("└──────────────┴──────┴─────────┴─────────┴──────────────────────────────┘");
            let mut any_path = false;
            for s in &sessions {
                if let Some(p) = &s.model_dir {
                    if !any_path {
                        println!("\nModel paths:");
                        any_path = true;
                    }
                    println!("  {} (pid {}): {}", s.name, s.pid, p);
                }
            }
            if let Some(am) = active_model {
                let q = am.quantization.as_deref().unwrap_or("(default)");
                println!("\nActive model (swap): {} [{}]", am.model_ref, q);
            }
            if !gpu_lines.is_empty() {
                println!("\nnvidia-smi (processes):");
                for l in gpu_lines {
                    println!("  {l}");
                }
            }
            if let Some(n) = note {
                println!("\n{n}");
            }
        }
        DaemonResponse::Error { message } => {
            println!("└──────────────┴──────┴─────────┴─────────┴──────────────────────────────┘");
            anyhow::bail!("{message}");
        }
        other => {
            println!("└──────────────┴──────┴─────────┴─────────┴──────────────────────────────┘");
            anyhow::bail!("unexpected daemon response: {other:?}");
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars()
        .take(max.saturating_sub(2))
        .chain("..".chars())
        .collect()
}
