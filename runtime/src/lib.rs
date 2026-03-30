//! NeuronBox runtime library: GPU detection, daemon protocol, session tracking.

pub mod gpu;
pub mod gpu_manager;
pub mod host;
pub mod model_loader;
pub mod protocol;
pub mod server;
pub mod vram_watch;

pub use gpu::{detect_gpus, primary_gpu_vram_mb, soft_vram_check, GpuDevice, GpuError};
pub use host::{HostProbe, HostSnapshot, TrainingBackend};
pub use protocol::{DaemonRequest, DaemonResponse, SessionInfo};
