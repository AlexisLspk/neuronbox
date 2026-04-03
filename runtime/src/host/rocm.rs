//! AMD detection via `rocm-smi`.

use std::process::Command;

use super::snapshot::GpuRecord;

pub fn query_gpus() -> (Option<Vec<GpuRecord>>, bool) {
    let out = match Command::new("rocm-smi").arg("--showproductname").output() {
        Ok(o) => o,
        Err(_) => return (None, false),
    };
    if !out.status.success() {
        return (None, false);
    }
    let text = String::from_utf8_lossy(&out.stdout);
    if !text.contains("GPU") && !text.contains("Card") {
        return (None, true);
    }
    let mem = rocm_vram_mb().unwrap_or(0);
    let name = text
        .lines()
        .find(|l| l.contains("GPU") || l.contains("Card"))
        .map(|l| l.trim().to_string())
        .unwrap_or_else(|| "AMD GPU".to_string());
    (
        Some(vec![GpuRecord {
            index: 0,
            name,
            memory_total_mb: mem,
            backend: "ROCm".to_string(),
        }]),
        true,
    )
}

fn rocm_vram_mb() -> Option<u64> {
    let out = Command::new("rocm-smi")
        .args(["--showmeminfo", "vram"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if line.contains("VRAM Total Memory") || line.contains("Memory (B)") {
            if let Some(rest) = line.split(':').next_back() {
                let s = rest.split_whitespace().next()?;
                if let Ok(bytes) = s.parse::<u64>() {
                    return Some(bytes / (1024 * 1024));
                }
            }
        }
    }
    None
}
