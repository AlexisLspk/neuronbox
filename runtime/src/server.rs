//! Unix socket server (newline-delimited JSON).

use std::path::Path;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::gpu_manager::GpuManager;
use crate::host::{compute_apps_display_lines, compute_apps_pid_memory_mb};
use crate::model_loader::ModelLoader;
use crate::protocol::{DaemonRequest, DaemonResponse, SessionInfo, SWAP_SIGNAL_FILE_VERSION};
use crate::vram_watch;

const PROTOCOL_VERSION: u32 = 1;

/// Max bytes per incoming line (excluding the newline). Larger lines are rejected with an error response.
pub const MAX_REQUEST_LINE_BYTES: usize = 256 * 1024;

fn vram_watch_disabled() -> bool {
    std::env::var_os("NEURONBOX_DISABLE_VRAM_WATCH")
        .map(|v| {
            let s = v.to_string_lossy().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes")
        })
        .unwrap_or(false)
}

enum RequestLineError {
    TooLong,
    BadUtf8,
    Io(std::io::Error),
}

async fn read_request_line<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> Result<Option<String>, RequestLineError> {
    let mut line = Vec::new();
    loop {
        if line.len() >= MAX_REQUEST_LINE_BYTES {
            return Err(RequestLineError::TooLong);
        }
        let mut b = [0u8; 1];
        let n = reader.read(&mut b).await.map_err(RequestLineError::Io)?;
        if n == 0 {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err(RequestLineError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "EOF before newline",
                )))
            };
        }
        if b[0] == b'\n' {
            break;
        }
        line.push(b[0]);
    }
    String::from_utf8(line)
        .map(Some)
        .map_err(|_| RequestLineError::BadUtf8)
}

pub async fn run_socket_server(
    socket_path: &Path,
    gpu_manager: GpuManager,
    model_loader: ModelLoader,
) -> Result<()> {
    if socket_path.exists() {
        std::fs::remove_file(socket_path).ok();
    }
    if let Some(dir) = socket_path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create_dir_all {:?}", dir))?;
    }
    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("bind unix socket {:?}", socket_path))?;

    // Restrict socket permissions to owner only (security: local user isolation)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {:?}", socket_path))?;
    }

    if !vram_watch_disabled() {
        let gm_watch = gpu_manager.clone();
        tokio::spawn(vram_watch::run_soft_vram_enforcement(gm_watch));
    }

    loop {
        let (stream, _) = listener.accept().await?;
        let gm = gpu_manager.clone();
        let ml = model_loader.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, gm, ml).await {
                tracing::warn!("connection error: {e:#}");
            }
        });
    }
}

async fn handle_connection(
    stream: UnixStream,
    gpu_manager: GpuManager,
    model_loader: ModelLoader,
) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    loop {
        let line = match read_request_line(&mut reader).await {
            Ok(None) => break,
            Ok(Some(s)) => s,
            Err(RequestLineError::TooLong) => {
                let err = DaemonResponse::Error {
                    message: format!(
                        "request line exceeds maximum size ({MAX_REQUEST_LINE_BYTES} bytes)"
                    ),
                };
                write_response(&mut write_half, &err).await?;
                break;
            }
            Err(RequestLineError::BadUtf8) => {
                let err = DaemonResponse::Error {
                    message: "invalid UTF-8 in request line".to_string(),
                };
                write_response(&mut write_half, &err).await?;
                break;
            }
            Err(RequestLineError::Io(e)) => return Err(e.into()),
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: DaemonRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let err = DaemonResponse::Error {
                    message: format!("invalid JSON request: {e}"),
                };
                write_response(&mut write_half, &err).await?;
                continue;
            }
        };

        let resp = dispatch(req, &gpu_manager, &model_loader).await;
        write_response(&mut write_half, &resp).await?;
    }
    Ok(())
}

async fn write_response(
    w: &mut tokio::net::unix::OwnedWriteHalf,
    resp: &DaemonResponse,
) -> Result<()> {
    let mut s = serde_json::to_string(resp)?;
    s.push('\n');
    w.write_all(s.as_bytes()).await?;
    Ok(())
}

async fn dispatch(
    req: DaemonRequest,
    gpu_manager: &GpuManager,
    model_loader: &ModelLoader,
) -> DaemonResponse {
    match req {
        DaemonRequest::Ping => DaemonResponse::Pong,
        DaemonRequest::Version { v } => {
            if v != PROTOCOL_VERSION {
                DaemonResponse::Error {
                    message: format!("protocol mismatch: client {v}, daemon {PROTOCOL_VERSION}"),
                }
            } else {
                DaemonResponse::VersionInfo {
                    v: PROTOCOL_VERSION,
                }
            }
        }
        DaemonRequest::RegisterSession {
            name,
            estimated_vram_mb,
            pid,
            tokens_per_sec,
        } => {
            gpu_manager
                .register(SessionInfo {
                    name,
                    pid,
                    estimated_vram_mb,
                    tokens_per_sec,
                })
                .await;
            DaemonResponse::Registered { pid }
        }
        DaemonRequest::UnregisterSession { pid } => {
            let ok = gpu_manager.unregister(pid).await;
            if ok {
                DaemonResponse::Unregistered
            } else {
                DaemonResponse::Error {
                    message: format!("pid {pid} not registered"),
                }
            }
        }
        DaemonRequest::ListSessions => {
            let sessions = gpu_manager.list().await;
            DaemonResponse::Sessions { sessions }
        }
        DaemonRequest::Stats => {
            let sessions = gpu_manager.list().await;
            let (gpu_lines, vram_used_by_pid) = nvidia_stats_bundle().await;
            let note = if gpu_lines.is_empty() {
                Some(
                    "tokens/s are shown only when the session reports them (RegisterSession)."
                        .to_string(),
                )
            } else {
                None
            };
            let am = model_loader.get().await;
            let active_model = if am.model_ref.is_empty() {
                None
            } else {
                Some(crate::protocol::ActiveModelInfo {
                    model_ref: am.model_ref,
                    quantization: am.quantization,
                })
            };
            DaemonResponse::Stats {
                sessions,
                gpu_lines,
                note,
                active_model,
                vram_used_by_pid,
            }
        }
        DaemonRequest::SwapModel {
            model_ref,
            quantization,
        } => {
            model_loader
                .swap(model_ref.clone(), quantization.clone())
                .await;
            let swap_path = dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".neuronbox")
                .join("swap_signal.json");
            let payload = serde_json::json!({
                "signal_version": SWAP_SIGNAL_FILE_VERSION,
                "model_ref": model_ref.clone(),
                "quantization": quantization.clone(),
                "ts": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            });
            if let Ok(bytes) = serde_json::to_vec(&payload) {
                let _ = tokio::fs::write(&swap_path, bytes).await;
            }
            DaemonResponse::Swapped {
                model_ref,
                quantization,
            }
        }
    }
}

async fn nvidia_stats_bundle() -> (Vec<String>, std::collections::HashMap<u32, u64>) {
    tokio::task::spawn_blocking(|| {
        let lines = compute_apps_display_lines();
        let map = compute_apps_pid_memory_mb().unwrap_or_default();
        (lines, map)
    })
    .await
    .unwrap_or_else(|_| (Vec::new(), std::collections::HashMap::new()))
}
