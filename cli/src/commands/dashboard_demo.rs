//! Background task for `neuron dashboard --demo`: fake sessions + swap (Unix only).

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use neuronbox_runtime::protocol::DaemonRequest;
use tokio::process::Command;
use tokio::sync::watch;

use crate::daemon_client::DaemonSession;

/// Runs until `shutdown` is true: registers many sessions, animates tok/s, then cleans up.
pub async fn run(shutdown: watch::Receiver<bool>) -> Result<()> {
    #[cfg(not(unix))]
    {
        let _ = shutdown;
        anyhow::bail!("neuron dashboard --demo is only supported on Unix (uses /usr/bin/sleep)");
    }

    #[cfg(unix)]
    {
        run_unix(shutdown).await
    }
}

#[cfg(unix)]
async fn run_unix(mut shutdown: watch::Receiver<bool>) -> Result<()> {
    let mut session = DaemonSession::connect()
        .await
        .context("demo worker: connect daemon")?;

    let _ = session
        .request(DaemonRequest::SwapModel {
            model_ref: "acme/MegaMock-7B-instruct".to_string(),
            quantization: Some("Q5_K_M".to_string()),
        })
        .await;

    let specs = build_demo_specs();
    let mut sleepers: Vec<tokio::process::Child> = Vec::with_capacity(specs.len());
    let mut meta: Vec<(String, u32, u64, f64, f64)> = Vec::new();

    for (name, vram, base, drift) in specs {
        let child = Command::new("sleep")
            .arg("999999")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("demo: spawn sleep helper")?;
        let pid = child
            .id()
            .ok_or_else(|| anyhow::anyhow!("demo: no PID from sleep child"))?;

        session
            .request(DaemonRequest::RegisterSession {
                name: name.clone(),
                estimated_vram_mb: vram,
                pid,
                tokens_per_sec: Some(base),
            })
            .await
            .context("demo: initial register")?;

        sleepers.push(child);
        meta.push((name, pid, vram, base, drift));
    }

    let t0 = Instant::now();
    let mut tick: u64 = 0;

    loop {
        if *shutdown.borrow() {
            break;
        }

        let elapsed = t0.elapsed().as_secs_f64();
        for (i, (name, pid, vram, base, drift)) in meta.iter().enumerate() {
            let tok = if i < DEMO_HERO_COUNT {
                demo_hero_tok(elapsed, tick, i, *base, *drift, *pid)
            } else {
                demo_other_tok(elapsed, tick, i, *base, *drift, *pid)
            };

            let _ = session
                .request(DaemonRequest::RegisterSession {
                    name: name.clone(),
                    estimated_vram_mb: *vram,
                    pid: *pid,
                    tokens_per_sec: Some(tok),
                })
                .await;
        }

        tick = tick.wrapping_add(1);

        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
        }
    }

    for (_, pid, _, _, _) in &meta {
        let _ = session
            .request(DaemonRequest::UnregisterSession { pid: *pid })
            .await;
    }
    for mut c in sleepers {
        let _ = c.kill().await;
    }

    Ok(())
}

/// The first five sessions have high, very variable tok/s (locked chart cohort).
const DEMO_HERO_COUNT: usize = 5;
/// Tok/s ceiling for all non-headline sessions (stay well below the hero floor).
const DEMO_OTHER_TOK_CEIL: f64 = 32.0;
/// Minimum margin above « others » (avoids overlap if there is a dip).
const DEMO_HERO_MIN_ABOVE_OTHERS: f64 = 12.0;

fn demo_hero_tok(elapsed: f64, tick: u64, i: usize, base: f64, drift: f64, pid: u32) -> f64 {
    // Waves in [0, amp] to avoid long plateaus from a fixed floor (flat-looking traces).
    const RHYTHM: [f64; DEMO_HERO_COUNT] = [0.37, 0.71, 0.53, 0.89, 0.61];
    const HARM2: [f64; DEMO_HERO_COUNT] = [1.63, 1.27, 1.91, 1.49, 1.79];
    const HARM3: [f64; DEMO_HERO_COUNT] = [2.87, 3.41, 2.59, 3.17, 2.23];
    let r = RHYTHM[i];
    let h2 = HARM2[i];
    let h3 = HARM3[i];

    let amp1 = 58.0 + i as f64 * 28.0;
    let amp2 = 52.0 + i as f64 * 18.0;
    let amp3 = 36.0 + i as f64 * 11.0;
    let w1 = amp1 * (0.5 + 0.5 * (elapsed * r + i as f64 * 1.4).sin());
    let w2 = amp2 * (0.5 + 0.5 * (elapsed * r * h2 + 0.8 * i as f64).cos());
    let w3 = amp3 * (0.5 + 0.5 * (elapsed * r * h3 + i as f64 * 0.37).sin());
    // Each hero: slow phased « breathing » (avoids two nearly parallel curves).
    let slow =
        (22.0 + i as f64 * 9.0) * (elapsed * (0.11 + 0.034 * i as f64) + i as f64 * 2.1).sin();
    let ripple = 16.0 * (elapsed * (4.7 + i as f64 * 0.31) + pid as f64 * 0.002).sin();
    let noise = pseudo_noise(elapsed, i, pid, tick) * 1.55;
    let burst = if (tick.wrapping_add(i as u64 * 7)).rem_euclid(29) == 9 {
        58.0 + i as f64 * 14.0
    } else {
        0.0
    };
    let drift_term = 0.72 * drift * (elapsed * 0.088 + i as f64 * 0.25).sin();
    let raw = base + w1 + w2 + w3 + slow + ripple + noise + drift_term + burst;
    raw.max(DEMO_OTHER_TOK_CEIL + DEMO_HERO_MIN_ABOVE_OTHERS)
}

fn demo_other_tok(elapsed: f64, tick: u64, i: usize, base: f64, drift: f64, pid: u32) -> f64 {
    let w = 2.8 * (elapsed * 2.0 + i as f64 * 0.7).sin();
    let noise = pseudo_noise(elapsed, i, pid, tick) * 0.28;
    let drift_term = drift * (elapsed * 0.1).sin() * 0.45;
    (base + w + noise + drift_term).clamp(0.5, DEMO_OTHER_TOK_CEIL)
}

fn pseudo_noise(elapsed: f64, i: usize, pid: u32, tick: u64) -> f64 {
    let u = (elapsed * 1000.0) as u64;
    let x = u
        .wrapping_add(pid as u64)
        .wrapping_mul((i + 1) as u64)
        .wrapping_add(tick.wrapping_mul(7919));
    (x % 10_000) as f64 / 10_000.0 * 24.0 - 12.0
}

fn build_demo_specs() -> Vec<(String, u64, f64, f64)> {
    const PREFIXES: &[&str] = &[
        "infer", "train", "eval", "bench", "serve", "dist", "lora", "quant", "embed", "router",
        "prefill", "decode",
    ];
    const VRAMS: &[u64] = &[
        2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, 8192, 10240, 18432,
    ];
    /// Well-separated initial bases for the five headliners (order = sticky top-5 at startup).
    const HERO_BASES: [f64; DEMO_HERO_COUNT] = [108.0, 292.0, 158.0, 348.0, 218.0];
    const HERO_DRIFTS: [f64; DEMO_HERO_COUNT] = [46.0, -40.0, 55.0, -34.0, 50.0];

    let mut out = Vec::new();
    for (i, p) in PREFIXES.iter().enumerate() {
        let name = format!("{}-{:02}", p, (i * 7 + 3) % 100);
        let vram = VRAMS[i % VRAMS.len()];
        let (base, drift) = if i < DEMO_HERO_COUNT {
            (HERO_BASES[i], HERO_DRIFTS[i])
        } else {
            let slot = i - DEMO_HERO_COUNT;
            let base = 4.5 + (slot % 7) as f64 * 1.65;
            let drift = -2.5 + (slot % 5) as f64 * 1.1;
            (base, drift)
        };
        out.push((name, vram, base, drift));
    }
    out
}
