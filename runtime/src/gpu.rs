//! GPU detection and soft VRAM control — delegates probing to [`crate::host`].

use thiserror::Error;

pub use crate::host::GpuRecord as GpuDevice;
use crate::host::HostProbe;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("nvidia-smi failed: {0}")]
    NvidiaSmi(String),
    #[error("could not parse GPU listing")]
    Parse,
}

/// List GPUs. Order: NVIDIA, then AMD ROCm, then Apple Silicon.
pub fn detect_gpus() -> Result<Vec<GpuDevice>, GpuError> {
    Ok(HostProbe::snapshot().gpus)
}

/// Best-effort total VRAM of the primary GPU (NVIDIA index 0, or ROCm).
pub fn primary_gpu_vram_mb() -> Option<u64> {
    HostProbe::snapshot().primary_vram_mb()
}

/// Soft check: returns `Ok(())` if `available_mb >= required_mb`, else an error message for the user.
pub fn soft_vram_check(
    available_mb: u64,
    required_mb: u64,
    project_hint: &str,
) -> Result<(), String> {
    if available_mb >= required_mb {
        return Ok(());
    }
    Err(format!(
        "[neuronbox] Error: {project_hint} needs {required_mb} MB VRAM, only {available_mb} MB available (estimate).\n\
         Try: use a smaller model, enable quantization (e.g. q4_k_m), or free other GPU processes."
    ))
}
