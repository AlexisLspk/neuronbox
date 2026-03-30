use serde::{Deserialize, Serialize};

/// Version du fichier `~/.neuronbox/swap_signal.json` (voir `specs/swap-signal.schema.json`).
pub const SWAP_SIGNAL_FILE_VERSION: u32 = 1;

/// JSON messages over the Unix socket (newline-delimited JSON, one object per line).
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum DaemonRequest {
    Ping,
    RegisterSession {
        name: String,
        estimated_vram_mb: u64,
        pid: u32,
        #[serde(default)]
        tokens_per_sec: Option<f64>,
    },
    UnregisterSession {
        pid: u32,
    },
    ListSessions,
    Stats,
    SwapModel {
        model_ref: String,
        #[serde(default)]
        quantization: Option<String>,
    },
    Version {
        v: u32,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "response", rename_all = "snake_case")]
pub enum DaemonResponse {
    Pong,
    Registered {
        pid: u32,
    },
    Unregistered,
    Sessions {
        sessions: Vec<SessionInfo>,
    },
    Stats {
        sessions: Vec<SessionInfo>,
        #[serde(default)]
        gpu_lines: Vec<String>,
        #[serde(default)]
        note: Option<String>,
    },
    Swapped {
        model_ref: String,
        #[serde(default)]
        quantization: Option<String>,
    },
    VersionInfo {
        v: u32,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub name: String,
    pub pid: u32,
    pub estimated_vram_mb: u64,
    #[serde(default)]
    pub tokens_per_sec: Option<f64>,
}
