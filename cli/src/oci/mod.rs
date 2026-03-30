//! OCI execution: Docker (default) or runc with a prepared bundle.
//!
//! **Docker invocation:** only this module tree (`docker_run`, `runc_prepare`) runs the host
//! `docker` binary for NeuronBox. The main CLI path (`neuron pull`, host `neuron run`) does not.

mod docker_run;
mod runc_prepare;

use crate::neuron_config::NeuronConfig;

pub use docker_run::{run_docker_isolated, DockerRunParams};
pub use runc_prepare::{oci_bundle_dir, prepare_runc_bundle, run_runc_bundle};

pub fn oci_enabled(cfg: &NeuronConfig, cli_flag: bool) -> bool {
    cli_flag || cfg.oci_mode()
}
