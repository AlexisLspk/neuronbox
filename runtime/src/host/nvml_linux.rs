//! NVML (Linux + NVIDIA) : une sonde pour liste GPU, VRAM par PID et lignes d’affichage.
//! Repli : `nvidia.rs` utilise `nvidia-smi` si ce module n’est pas compilé ou si `try_snapshot` échoue.

use std::collections::HashMap;

use nvml_wrapper::enum_wrappers::device::UsedGpuMemory;
use nvml_wrapper::Nvml;

use super::snapshot::GpuRecord;

#[derive(Debug, Clone)]
pub struct NvmlSnapshot {
    pub gpus: Vec<GpuRecord>,
    pub pid_memory_mb: HashMap<u32, u64>,
    pub display_lines: Vec<String>,
}

fn proc_comm(pid: u32) -> String {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .ok()
        .map(|s| s.trim().replace('\n', "").to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "?".to_string())
}

/// `None` si NVML indisponible (pas de pilote, pas de GPU, erreur d’init).
pub fn try_snapshot() -> Option<NvmlSnapshot> {
    let nvml = Nvml::init().ok()?;
    let n = nvml.device_count().ok()?;
    if n == 0 {
        return None;
    }

    let driver = nvml.sys_driver_version().unwrap_or_default();

    let mut gpus = Vec::new();
    for i in 0..n {
        let dev = nvml.device_by_index(i).ok()?;
        let name = dev.name().unwrap_or_else(|_| format!("GPU {i}"));
        let mem = dev.memory_info().ok()?;
        let total_mb = mem.total / (1024 * 1024);
        gpus.push(GpuRecord {
            index: i,
            name,
            memory_total_mb: total_mb,
            backend: format!("CUDA (driver {driver})"),
        });
    }

    let mut pid_memory_mb: HashMap<u32, u64> = HashMap::new();

    for i in 0..n {
        let dev = nvml.device_by_index(i).ok()?;
        let Ok(procs) = dev.running_compute_processes() else {
            continue;
        };
        for p in procs {
            let bytes = match p.used_gpu_memory {
                UsedGpuMemory::Used(b) => b,
                UsedGpuMemory::Unavailable => 0,
            };
            let mb = bytes / (1024 * 1024);
            *pid_memory_mb.entry(p.pid).or_insert(0) += mb;
        }
    }

    let mut display_lines: Vec<String> = pid_memory_mb
        .iter()
        .map(|(pid, mb)| format!("{}, {}, {} MiB", pid, proc_comm(*pid), mb))
        .collect();
    display_lines.sort();

    Some(NvmlSnapshot {
        gpus,
        pid_memory_mb,
        display_lines,
    })
}
