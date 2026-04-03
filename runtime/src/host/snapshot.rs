//! Serializable schema of machine state (GPU / platform) for CLI debug and client-side decisions.

use serde::{Deserialize, Serialize};

/// `HostSnapshot` schema version (bump if incompatible fields change).
pub const HOST_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeStatus {
    /// `nvidia-smi --query-gpu=...` succeeded.
    pub nvidia_smi_gpu_list: bool,
    /// `nvidia-smi --query-compute-apps=pid,used_gpu_memory` succeeded (VRAM per PID).
    pub nvidia_smi_compute: bool,
    /// NVIDIA GPU list from **NVML** (`nvml` feature, Linux) rather than `nvidia-smi`.
    #[serde(default)]
    pub nvml: bool,
    pub rocm_smi: bool,
    pub apple_system_profiler: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingBackend {
    Cuda,
    Rocm,
    Metal,
    /// GPUs present but backend not classified (rare).
    Cpu,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRecord {
    pub index: u32,
    pub name: String,
    pub memory_total_mb: u64,
    /// E.g. `CUDA (driver 535.x)`, `ROCm`, or `Metal`.
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostSnapshot {
    pub schema_version: u32,
    pub platform: PlatformInfo,
    pub gpus: Vec<GpuRecord>,
    pub training_backend: TrainingBackend,
    pub probes: ProbeStatus,
}

impl HostSnapshot {
    /// Primary VRAM for pre-checks: NVIDIA GPU at index 0, else first ROCm GPU with known VRAM.
    /// `None` if unknown or zero (e.g. Apple with no exposed VRAM size).
    pub fn primary_vram_mb(&self) -> Option<u64> {
        for g in &self.gpus {
            if g.backend.contains("CUDA") && g.index == 0 && g.memory_total_mb > 0 {
                return Some(g.memory_total_mb);
            }
        }
        for g in &self.gpus {
            if g.backend.contains("ROCm") && g.memory_total_mb > 0 {
                return Some(g.memory_total_mb);
            }
        }
        None
    }
}

pub(crate) fn infer_training_backend(gpus: &[GpuRecord]) -> TrainingBackend {
    if gpus.iter().any(|g| g.backend.contains("CUDA")) {
        return TrainingBackend::Cuda;
    }
    if gpus.iter().any(|g| g.backend.contains("ROCm")) {
        return TrainingBackend::Rocm;
    }
    if gpus.iter().any(|g| g.backend.contains("Metal")) {
        return TrainingBackend::Metal;
    }
    if !gpus.is_empty() {
        return TrainingBackend::Cpu;
    }
    TrainingBackend::Unknown
}

pub(crate) fn platform_info() -> PlatformInfo {
    PlatformInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}
