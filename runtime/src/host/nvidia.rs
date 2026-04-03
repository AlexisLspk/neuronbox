//! NVIDIA: NVML on Linux (`nvml` feature), else `nvidia-smi` subprocess.

use std::collections::HashMap;
use std::process::Command;

use super::snapshot::GpuRecord;

#[cfg(all(target_os = "linux", feature = "nvml"))]
use super::nvml_linux;

/// Result of the NVIDIA GPU list probe.
#[derive(Debug, Clone)]
pub struct NvidiaGpuListResult {
    pub gpus: Option<Vec<GpuRecord>>,
    /// NVIDIA tool responded (NVML or `nvidia-smi`).
    pub probe_ok: bool,
    /// Data from NVML (otherwise `nvidia-smi`).
    pub used_nvml: bool,
}

pub fn query_gpus() -> NvidiaGpuListResult {
    #[cfg(all(target_os = "linux", feature = "nvml"))]
    if let Some(snap) = nvml_linux::try_snapshot() {
        if !snap.gpus.is_empty() {
            return NvidiaGpuListResult {
                gpus: Some(snap.gpus),
                probe_ok: true,
                used_nvml: true,
            };
        }
    }

    let (gpus, ok) = query_gpus_nvidia_smi();
    NvidiaGpuListResult {
        gpus,
        probe_ok: ok,
        used_nvml: false,
    }
}

fn query_gpus_nvidia_smi() -> (Option<Vec<GpuRecord>>, bool) {
    let out = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) => o,
        Err(_) => return (None, false),
    };

    if !out.status.success() {
        return (None, false);
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut gpus = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 {
            continue;
        }
        let Ok(index) = parts[0].parse::<u32>() else {
            continue;
        };
        let name = parts[1].to_string();
        let Ok(memory_total_mb) = parts[2].parse::<u64>() else {
            continue;
        };
        let driver = parts.get(3).map(|s| s.to_string()).unwrap_or_default();
        gpus.push(GpuRecord {
            index,
            name,
            memory_total_mb,
            backend: format!("CUDA (driver {driver})"),
        });
    }

    if gpus.is_empty() {
        (None, true)
    } else {
        (Some(gpus), true)
    }
}

/// PID → used MiB (soft VRAM monitoring, stats).
pub fn compute_apps_pid_memory_mb() -> Option<HashMap<u32, u64>> {
    #[cfg(all(target_os = "linux", feature = "nvml"))]
    if let Some(snap) = nvml_linux::try_snapshot() {
        return Some(snap.pid_memory_mb);
    }

    compute_apps_pid_memory_mb_smi()
}

fn compute_apps_pid_memory_mb_smi() -> Option<HashMap<u32, u64>> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|x| x.trim()).collect();
        if parts.len() < 2 {
            continue;
        }
        let Ok(pid) = parts[0].parse::<u32>() else {
            continue;
        };
        let mem = parts[1].replace(" MiB", "");
        let Ok(mb) = mem.trim().parse::<u64>() else {
            continue;
        };
        map.insert(pid, mb);
    }
    Some(map)
}

/// Lines for `DaemonResponse::Stats`.
pub fn compute_apps_display_lines() -> Vec<String> {
    #[cfg(all(target_os = "linux", feature = "nvml"))]
    if let Some(snap) = nvml_linux::try_snapshot() {
        return snap.display_lines;
    }

    compute_apps_display_lines_smi()
}

fn compute_apps_display_lines_smi() -> Vec<String> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        }
        _ => vec![],
    }
}
