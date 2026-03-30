use anyhow::Result;
use neuronbox_runtime::protocol::{DaemonRequest, DaemonResponse};

use crate::daemon_client;
use crate::daemon_spawn;

pub async fn stats() -> Result<()> {
    daemon_spawn::ensure_daemon_running().await?;

    let resp = daemon_client::request(DaemonRequest::Stats).await?;

    println!("┌─────────────────────────────────────────┐");
    println!("│ NeuronBox Stats                         │");
    println!("├──────────────┬───────┬──────────────────┤");
    println!("│ Session      │ VRAM  │ Tokens/s         │");
    println!("├──────────────┼───────┼──────────────────┤");

    match resp {
        DaemonResponse::Stats {
            sessions,
            gpu_lines,
            note,
        } => {
            if sessions.is_empty() {
                println!("│ (no registered sessions)                │");
            }
            for s in sessions {
                let tok = s
                    .tokens_per_sec
                    .map(|t| format!("{t:.1}"))
                    .unwrap_or_else(|| "—".to_string());
                let name = truncate(&s.name, 12);
                println!(
                    "│ {:<12} │ {:>5} │ {:<16} │",
                    name,
                    format!("{}/?", s.estimated_vram_mb),
                    tok
                );
            }
            println!("└──────────────┴───────┴──────────────────┘");
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
            println!("└──────────────┴───────┴──────────────────┘");
            anyhow::bail!("{message}");
        }
        _ => {
            println!("└──────────────┴───────┴──────────────────┘");
            println!("unexpected daemon response");
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
