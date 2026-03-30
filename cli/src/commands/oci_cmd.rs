use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::commands::init::default_yaml_path;
use crate::neuron_config::NeuronConfig;
use crate::oci::{oci_bundle_dir, prepare_runc_bundle, run_runc_bundle};

pub fn prepare(image_override: Option<String>, yaml: Option<PathBuf>) -> Result<()> {
    let image = if let Some(img) = image_override {
        img
    } else {
        let path = yaml.unwrap_or_else(default_yaml_path);
        let cfg = NeuronConfig::load_path(&path)?;
        cfg.container.image_resolved()
    };
    let bundle = oci_bundle_dir(&image);
    std::fs::create_dir_all(&bundle).ok();
    prepare_runc_bundle(&image, &bundle).with_context(|| format!("prepare bundle {:?}", bundle))?;
    println!("Runc bundle ready: {}", bundle.display());
    println!(
        "Edit config.json for the process command, then: neuron oci runc --bundle {:?}",
        bundle
    );
    Ok(())
}

pub fn runc(bundle: PathBuf, id: String) -> Result<()> {
    let st = run_runc_bundle(&bundle, &id)?;
    if !st.success() {
        anyhow::bail!("runc failed (code {:?})", st.code());
    }
    Ok(())
}
