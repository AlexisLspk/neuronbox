use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Version of `~/.neuronbox/swap_signal.json` (see `specs/swap-signal.schema.json`).
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
        #[serde(default)]
        model_dir: Option<String>,
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

/// Logical active model on the daemon (`neuron swap`), for dashboard / stats display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveModelInfo {
    pub model_ref: String,
    #[serde(default)]
    pub quantization: Option<String>,
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
        /// Logical state after `neuron swap` (hot-swap), for the dashboard.
        #[serde(default)]
        active_model: Option<ActiveModelInfo>,
        /// Real GPU MiB per PID (NVIDIA `nvidia-smi` / NVML compute apps), for dashboard display.
        #[serde(default)]
        vram_used_by_pid: HashMap<u32, u64>,
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
    #[serde(default)]
    pub model_dir: Option<String>,
}
