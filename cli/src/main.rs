//! NeuronBox CLI — `neuron`.

mod commands;
mod daemon_client;
mod daemon_spawn;
mod env_hash;
mod model_alias;
mod model_resolve;
mod neuron_config;
mod oci;
mod paths;
mod sdk_path;
mod store;
mod ui;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use commands::dashboard;
use commands::doctor::{self, DoctorArgs};
use commands::gpu_cmd::{self, GpuCommands};
use commands::host_cmd::{self, HostCommands};
use commands::init;
use commands::lock;
use commands::model::{self, ModelCommands};
use commands::oci_cmd;
use commands::pull;
use commands::run::{self, RunArgs};
use commands::serve;
use commands::stats;
use commands::swap;
use commands::welcome;

const LONG_ABOUT: &str = "\
NeuronBox runs local ML workflows from a declarative neuron.yaml: Python venv (hashed), \
model store, optional OCI isolation (neuron run --oci / neuron oci — uses Docker on the host when enabled), \
and a Unix-socket daemon (neurond) for sessions, stats, and soft VRAM hints on NVIDIA Linux.\n\
\n\
Environment: NEURONBOX_SOCKET (daemon socket path), NEUROND_PATH (neurond binary if not next to neuron).";

const AFTER_HELP: &str = "\
Quick start:\n  neuron              Open the welcome screen\n  neuron init         Create neuron.yaml\n  neuron pull <id>    Fetch a model\n  neuron run          Run the project entrypoint\n  neuron dashboard    Terminal UI for sessions + GPU summary\n  neuron dashboard --demo   Mock sessions + synthetic VRAM (Unix)\n  neuron help         Full command list";

#[derive(Parser)]
#[command(
    name = "neuron",
    about = "NeuronBox — local ML runner (neuron.yaml, store, daemon)",
    long_about = LONG_ABOUT,
    after_help = AFTER_HELP,
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum OciCommands {
    /// Prepare a runc bundle (rootfs via docker export + minimal config.json).
    Prepare {
        #[arg(long)]
        image: Option<String>,
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Run `runc run` in a prepared bundle.
    Runc {
        #[arg(short, long)]
        bundle: PathBuf,
        #[arg(long, default_value = "neuronbox-job")]
        id: String,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// GPU detection and information.
    Gpu {
        #[command(subcommand)]
        cmd: GpuCommands,
    },
    /// Machine inspection (JSON snapshot for debugging / support).
    Host {
        #[command(subcommand)]
        cmd: HostCommands,
    },
    /// Diagnostic checks for the NeuronBox environment.
    Doctor {
        /// Exit with non-zero code on any warning (for CI).
        #[arg(long)]
        strict: bool,
    },
    /// Create `neuron.yaml` in the current directory.
    Init {
        /// Use a template (inference, finetune, local-model).
        #[arg(long)]
        template: Option<String>,
        /// List available templates.
        #[arg(long)]
        list_templates: bool,
    },
    /// Download a model into the store (HF-style org/model, short alias, or local path).
    Pull {
        model: String,
        /// HF commit SHA or tag to pin (e.g. "main", "v1.0", or a commit hash).
        #[arg(long)]
        revision: Option<String>,
    },
    /// Run the project (`neuron.yaml`), a single HF model id, or pass args to the entrypoint.
    Run {
        #[arg(long, short = 'f', value_name = "PATH")]
        file: Option<PathBuf>,
        /// `CUDA_VISIBLE_DEVICES` (e.g. `0` or `0,1`).
        #[arg(long)]
        gpu: Option<String>,
        #[arg(long, value_name = "12gb")]
        vram: Option<String>,
        /// Force Docker (OCI runtime) even when neuron.yaml is in host mode.
        #[arg(long)]
        oci: bool,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Update the daemon-side “active” model (logical state for workers).
    Swap {
        model: String,
        #[arg(long)]
        quantization: Option<String>,
    },
    /// VRAM and sessions registered with the daemon.
    Stats,
    /// Local terminal dashboard (ratatui); stats ~10 Hz, host/GPU probe ~1 Hz.
    Dashboard {
        /// Fake sessions + synthetic VRAM overlay (Unix). Sets VRAM watch off for spawned neurond.
        #[arg(long)]
        demo: bool,
    },
    /// Model registry in the store.
    Model {
        #[command(subcommand)]
        cmd: ModelCommands,
    },
    /// Start the daemon (blocking).
    Daemon,
    /// Inference / hot-swap worker (reacts to swap_signal.json).
    Serve {
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Generate `requirements.lock` for the hashed env (`uv pip compile`).
    Lock {
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Prepare / run with runc (OCI expert path; prepare uses Docker on the host).
    Oci {
        #[command(subcommand)]
        cmd: OciCommands,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("warn".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        None => welcome::run().map_err(anyhow::Error::from),
        Some(Commands::Doctor { strict }) => doctor::doctor(DoctorArgs { strict }).await,
        Some(Commands::Gpu { cmd }) => gpu_cmd::gpu(cmd),
        Some(Commands::Host { cmd }) => host_cmd::host(cmd),
        Some(Commands::Init {
            template,
            list_templates,
        }) => {
            if list_templates {
                init::list_templates();
                Ok(())
            } else {
                init::init_with_template(template.as_deref())
            }
        }
        Some(Commands::Pull { model, revision }) => {
            pull::pull_model_with_revision(&model, revision.as_deref())
        }
        Some(Commands::Run {
            file,
            gpu,
            vram,
            oci,
            args,
        }) => run_router(file, gpu, vram, oci, args).await,
        Some(Commands::Swap {
            model,
            quantization,
        }) => swap::swap(&model, quantization).await,
        Some(Commands::Stats) => stats::stats().await,
        Some(Commands::Dashboard { demo }) => dashboard::run(demo).await,
        Some(Commands::Model { cmd }) => model::model(cmd),
        Some(Commands::Daemon) => run_daemon_foreground(),
        Some(Commands::Serve { file }) => serve::serve(file).await,
        Some(Commands::Lock { file }) => lock::generate_lock(file.as_deref()),
        Some(Commands::Oci { cmd }) => match cmd {
            OciCommands::Prepare { image, file } => oci_cmd::prepare(image, file),
            OciCommands::Runc { bundle, id } => oci_cmd::runc(bundle, id),
        },
    }
}

fn run_daemon_foreground() -> Result<()> {
    let exe = daemon_spawn::neurond_exe();
    let status = std::process::Command::new(&exe)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "could not start {:?}: {e}. Use `cargo run -p neuronbox-runtime --bin neurond` or set NEUROND_PATH.",
                exe
            )
        })?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

async fn run_router(
    file: Option<PathBuf>,
    gpu: Option<String>,
    vram: Option<String>,
    oci: bool,
    args: Vec<String>,
) -> Result<()> {
    let vram_mb = vram.as_deref().and_then(parse_vram_cli);

    if args.is_empty() {
        return run::run_project(RunArgs {
            yaml: file,
            gpu_devices: gpu,
            vram_limit_mb: vram_mb,
            script_args: vec![],
            oci,
        })
        .await;
    }

    if args.len() == 1
        && (pull::looks_like_hf_repo(&args[0]) || model_alias::looks_like_short_tag(&args[0]))
    {
        return run::run_direct_model(&args[0], gpu, vram_mb).await;
    }

    if file.is_some() || init::default_yaml_path().exists() {
        return run::run_project(RunArgs {
            yaml: file,
            gpu_devices: gpu,
            vram_limit_mb: vram_mb,
            script_args: args,
            oci,
        })
        .await;
    }

    anyhow::bail!(
        "usage: neuron run [--file neuron.yaml] [--gpu 0,1] [--vram 12gb] [--oci]\n\
         or: neuron run org/hf-model\n\
         \n\
         To run a generic container image, use: docker run …\n\
         To run this project inside a container with NeuronBox mounts, use: neuron run --oci (and runtime.mode: oci in neuron.yaml)."
    )
}

fn parse_vram_cli(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase().replace(' ', "");
    if let Some(n) = s.strip_suffix("gb") {
        return n.parse::<f64>().ok().map(|g| (g * 1024.0) as u64);
    }
    if let Some(n) = s.strip_suffix("mb") {
        return n.parse().ok();
    }
    s.parse().ok()
}
