//! Apple Silicon via `system_profiler`.

use std::process::Command;

use super::snapshot::GpuRecord;

pub fn query_gpu() -> (Option<GpuRecord>, bool) {
    if !cfg!(target_os = "macos") {
        return (None, false);
    }
    let out = match Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return (None, false),
    };
    if !out.status.success() {
        return (None, false);
    }
    let v: serde_json::Value = match serde_json::from_slice(&out.stdout) {
        Ok(v) => v,
        Err(_) => return (None, true),
    };
    let Some(arr) = v["SPDisplaysDataType"].as_array() else {
        return (None, true);
    };
    let Some(first) = arr.first() else {
        return (None, true);
    };
    let name = first["sppci_model"]
        .as_str()
        .unwrap_or("Apple GPU")
        .to_string();
    (
        Some(GpuRecord {
            index: 0,
            name,
            memory_total_mb: 0,
            backend: "Metal".to_string(),
        }),
        true,
    )
}
