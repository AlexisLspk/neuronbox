//! NeuronBox daemon — Unix socket API for sessions, stats, swap state.

use std::path::PathBuf;

use anyhow::Result;
use neuronbox_runtime::gpu_manager::GpuManager;
use neuronbox_runtime::model_loader::ModelLoader;
use neuronbox_runtime::server::run_socket_server;
use tracing_subscriber::EnvFilter;

fn default_socket_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".neuronbox")
        .join("neuron.sock")
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let path = std::env::var("NEURONBOX_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_socket_path());

    tracing::info!("neurond listening on {}", path.display());

    let gpu_manager = GpuManager::new();
    let model_loader = ModelLoader::new();
    run_socket_server(&path, gpu_manager, model_loader).await
}
