//! Build a single `HostSnapshot` for the whole runtime / CLI.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::apple;
use super::nvidia;
use super::rocm;
use super::snapshot::{
    infer_training_backend, platform_info, HostSnapshot, ProbeStatus, HOST_SNAPSHOT_SCHEMA_VERSION,
};

/// Cache TTL to avoid multiple heavy probes in the same CLI flow.
const SNAPSHOT_CACHE_TTL: Duration = Duration::from_secs(2);

static SNAPSHOT_CACHE: Mutex<Option<(Instant, HostSnapshot)>> = Mutex::new(None);

/// Probe the host once (system tools + training-backend heuristic).
pub struct HostProbe;

impl HostProbe {
    pub fn snapshot() -> HostSnapshot {
        let now = Instant::now();
        let mut guard = SNAPSHOT_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((t, ref snap)) = *guard {
            if now.duration_since(t) < SNAPSHOT_CACHE_TTL {
                return snap.clone();
            }
        }
        let snap = Self::snapshot_uncached();
        *guard = Some((now, snap.clone()));
        snap
    }

    /// Bypass cache (tests or forced refresh).
    pub fn snapshot_fresh() -> HostSnapshot {
        Self::snapshot_uncached()
    }

    fn snapshot_uncached() -> HostSnapshot {
        let platform = platform_info();
        let mut probes = ProbeStatus::default();
        let mut gpus = Vec::new();

        let nvidia_list = nvidia::query_gpus();
        probes.nvidia_smi_gpu_list = nvidia_list.probe_ok;
        probes.nvml = nvidia_list.used_nvml;
        if let Some(list) = nvidia_list.gpus {
            if !list.is_empty() {
                gpus = list;
            }
        }

        if gpus.is_empty() {
            let (rocm_gpus, rocm_ok) = rocm::query_gpus();
            probes.rocm_smi = rocm_ok;
            if let Some(list) = rocm_gpus {
                if !list.is_empty() {
                    gpus = list;
                }
            }
        }

        if gpus.is_empty() {
            let (apple_gpu, apple_ok) = apple::query_gpu();
            probes.apple_system_profiler = apple_ok;
            if let Some(g) = apple_gpu {
                gpus.push(g);
            }
        }

        probes.nvidia_smi_compute = nvidia::compute_apps_pid_memory_mb().is_some();

        let training_backend = infer_training_backend(&gpus);

        HostSnapshot {
            schema_version: HOST_SNAPSHOT_SCHEMA_VERSION,
            platform,
            gpus,
            training_backend,
            probes,
        }
    }
}
