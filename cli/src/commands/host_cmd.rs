//! `neuron host` subcommand (machine inspection).

use anyhow::Result;
use clap::Subcommand;
use neuronbox_runtime::host::HostProbe;

#[derive(Subcommand)]
pub enum HostCommands {
    /// Print `HostSnapshot` JSON (versioned schema, GPU, training backend).
    Inspect,
}

pub fn host(cmd: HostCommands) -> Result<()> {
    match cmd {
        HostCommands::Inspect => {
            let snap = HostProbe::snapshot();
            println!("{}", serde_json::to_string_pretty(&snap)?);
        }
    }
    Ok(())
}
