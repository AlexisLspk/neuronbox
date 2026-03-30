use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::paths::neuronbox_home;

pub fn oci_bundle_dir(image: &str) -> PathBuf {
    let mut h = Sha256::new();
    h.update(image.as_bytes());
    let hex = hex::encode(h.finalize());
    neuronbox_home()
        .join("bundles")
        .join(format!("oci-{}", &hex[..16]))
}

/// Prepare `rootfs` + minimal `config.json` via `docker create` + `docker export` + `tar`.
pub fn prepare_runc_bundle(image: &str, bundle: &Path) -> Result<()> {
    std::fs::create_dir_all(bundle.join("rootfs")).context("mkdir bundle/rootfs")?;

    let st = Command::new("docker")
        .args(["pull", image])
        .status()
        .context("docker pull")?;
    if !st.success() {
        anyhow::bail!("docker pull {image} failed");
    }

    let out = Command::new("docker")
        .args(["create", image])
        .output()
        .context("docker create")?;
    if !out.status.success() {
        anyhow::bail!("docker create: {}", String::from_utf8_lossy(&out.stderr));
    }
    let cid = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let tar_path = bundle.join("export.tar");
    let st = Command::new("docker")
        .args(["export", "-o"])
        .arg(&tar_path)
        .arg(&cid)
        .status()
        .context("docker export")?;
    let _ = Command::new("docker").args(["rm", "-f", &cid]).status();
    if !st.success() {
        anyhow::bail!("docker export failed");
    }

    let rootfs = bundle.join("rootfs");
    let st = Command::new("tar")
        .args(["xf"])
        .arg(&tar_path)
        .arg("-C")
        .arg(&rootfs)
        .status()
        .context("tar xf export")?;
    if !st.success() {
        anyhow::bail!("tar extract to rootfs failed");
    }
    let _ = std::fs::remove_file(&tar_path);

    write_minimal_runc_config(bundle, image)?;
    Ok(())
}

fn write_minimal_runc_config(bundle: &Path, image: &str) -> Result<()> {
    let cfg = serde_json::json!({
        "ociVersion": "1.0.2",
        "process": {
            "terminal": true,
            "user": { "uid": 0, "gid": 0 },
            "args": ["/bin/bash"],
            "env": [format!("PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"), format!("IMAGE={image}")],
            "cwd": "/"
        },
        "root": { "path": "rootfs", "readonly": false },
        "hostname": "neuronbox",
        "linux": {
            "namespaces": [
                { "type": "pid" },
                { "type": "network" },
                { "type": "ipc" },
                { "type": "uts" },
                { "type": "mount" }
            ]
        }
    });
    let p = bundle.join("config.json");
    std::fs::write(&p, serde_json::to_string_pretty(&cfg)?).context("write config.json")?;
    Ok(())
}

/// Run `runc run` in the bundle (default process: shell — customize config.json for Python).
pub fn run_runc_bundle(bundle: &Path, container_id: &str) -> Result<std::process::ExitStatus> {
    let st = Command::new("runc")
        .args(["run", "-b"])
        .arg(bundle)
        .arg(container_id)
        .status()
        .context("runc run — install runc and a valid bundle (`neuron oci prepare`)")?;
    Ok(st)
}
