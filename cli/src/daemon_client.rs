//! `neurond` client over Unix socket: single request or persistent session (multiple req/response pairs).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use neuronbox_runtime::protocol::{DaemonRequest, DaemonResponse};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::UnixStream;
use tokio::time::timeout;

/// Default timeout to wait for one daemon response line.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);

pub fn default_socket_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".neuronbox")
        .join("neuron.sock")
}

fn socket_path() -> PathBuf {
    std::env::var("NEURONBOX_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_socket_path())
}

/// Open connection to the daemon: multiple `request` calls on the same socket.
pub struct DaemonSession {
    reader: BufReader<OwnedReadHalf>,
    writer: OwnedWriteHalf,
    read_timeout: Duration,
}

impl DaemonSession {
    /// Connect to the default socket or `NEURONBOX_SOCKET`.
    pub async fn connect() -> Result<Self> {
        Self::connect_with_timeout(DEFAULT_READ_TIMEOUT).await
    }

    pub async fn connect_with_timeout(read_timeout: Duration) -> Result<Self> {
        let path = socket_path();
        let stream = UnixStream::connect(&path)
            .await
            .with_context(|| format!("connect daemon at {:?} — start `neuron daemon`", path))?;
        let (read_half, write_half) = stream.into_split();
        Ok(Self {
            reader: BufReader::new(read_half),
            writer: write_half,
            read_timeout,
        })
    }

    /// Send one JSON request line and read one JSON response line.
    pub async fn request(&mut self, req: DaemonRequest) -> Result<DaemonResponse> {
        self.request_with_timeout(req, self.read_timeout).await
    }

    pub async fn request_with_timeout(
        &mut self,
        req: DaemonRequest,
        read_timeout: Duration,
    ) -> Result<DaemonResponse> {
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.writer
            .write_all(line.as_bytes())
            .await
            .context("daemon write")?;

        let read_fut = async {
            let mut out = String::new();
            self.reader.read_line(&mut out).await.map(|n| (n, out))
        };
        let (n, out) = timeout(read_timeout, read_fut)
            .await
            .context("daemon read timeout")??;
        if n == 0 {
            anyhow::bail!("daemon closed connection");
        }
        let resp: DaemonResponse =
            serde_json::from_str(out.trim()).context("parse daemon response")?;
        Ok(resp)
    }
}

/// One request then close the connection (historical CLI behavior).
pub async fn request(req: DaemonRequest) -> Result<DaemonResponse> {
    let mut session = DaemonSession::connect().await?;
    session.request(req).await
}
