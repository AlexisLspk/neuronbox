//! Start `neurond` in the background if it is not running.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::net::UnixStream;
use tokio::time::sleep;

use crate::daemon_client::default_socket_path;

/// Path to the `neurond` binary (same directory as `neuron` when possible).
pub fn neurond_exe() -> PathBuf {
    if let Ok(p) = std::env::var("NEUROND_PATH") {
        return PathBuf::from(p);
    }
    let neuron = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("neuron"));
    let name = if cfg!(target_os = "windows") {
        "neurond.exe"
    } else {
        "neurond"
    };
    neuron
        .parent()
        .map(|d| d.join(name))
        .unwrap_or_else(|| PathBuf::from(name))
}

pub async fn ensure_daemon_running() -> Result<()> {
    let sock = default_socket_path();
    if UnixStream::connect(&sock).await.is_ok() {
        return Ok(());
    }
    let exe = neurond_exe();
    if !exe.exists() {
        anyhow::bail!(
            "daemon binary not found at {:?}. Run `cargo build` and add target/debug to PATH, or set NEUROND_PATH.",
            exe
        );
    }
    std::process::Command::new(&exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawn neurond {:?}", exe))?;

    for _ in 0..50 {
        sleep(Duration::from_millis(100)).await;
        if UnixStream::connect(&sock).await.is_ok() {
            return Ok(());
        }
    }
    anyhow::bail!("daemon did not respond on {:?} in time", sock)
}
