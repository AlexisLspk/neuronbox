use anyhow::Result;
use neuronbox_runtime::protocol::DaemonRequest;

use crate::daemon_client;
use crate::daemon_spawn;

pub async fn swap(model_ref: &str, quantization: Option<String>) -> Result<()> {
    daemon_spawn::ensure_daemon_running().await?;
    let resp = daemon_client::request(DaemonRequest::SwapModel {
        model_ref: model_ref.to_string(),
        quantization: quantization.clone(),
    })
    .await?;
    match resp {
        neuronbox_runtime::protocol::DaemonResponse::Swapped {
            model_ref,
            quantization,
        } => {
            println!(
                "Active model (daemon state): {model_ref} {:?}",
                quantization
            );
            println!("Note: the next `neuron run` uses the store as configured; live model hot-swap requires a long-running worker that reads this daemon state.");
        }
        neuronbox_runtime::protocol::DaemonResponse::Error { message } => {
            anyhow::bail!("{message}");
        }
        _ => anyhow::bail!("unexpected daemon response"),
    }
    Ok(())
}
