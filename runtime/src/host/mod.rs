//! Single host probe layer: GPU, platform, likely PyTorch backend.

mod apple;
mod nvidia;
#[cfg(all(target_os = "linux", feature = "nvml"))]
mod nvml_linux;
mod probe;
mod rocm;
mod snapshot;

pub use nvidia::{compute_apps_display_lines, compute_apps_pid_memory_mb};
pub use probe::HostProbe;
pub use snapshot::{
    GpuRecord, HostSnapshot, PlatformInfo, ProbeStatus, TrainingBackend,
    HOST_SNAPSHOT_SCHEMA_VERSION,
};
