use std::path::Path;

use anyhow::Result;
use neuronbox_runtime::protocol::DaemonRequest;

use crate::daemon_client;
use crate::daemon_spawn;
use crate::model_alias;
use crate::paths::{neuronbox_home, store_root};
use crate::store::model_store;

/// Try to resolve model_ref to a local directory path.
/// Returns None if the model is not in the store or not a local path.
fn resolve_model_dir(model_ref: &str) -> Option<std::path::PathBuf> {
    // Check if it's a local path
    let path = Path::new(model_ref);
    if path.exists() && path.is_dir() {
        return path.canonicalize().ok();
    }

    // Try to resolve as HF model in store
    let store = store_root();
    let resolved_id = model_alias::resolve_for_hf_id(model_ref);

    // Try with default revision first
    let model_path = model_store::model_path(&store, &resolved_id);
    if model_path.exists() {
        return Some(model_path);
    }

    None
}

/// Write the swap signal file with resolved_model_dir if available.
fn write_swap_signal(
    model_ref: &str,
    quantization: Option<&str>,
    resolved_dir: Option<&Path>,
) -> Result<()> {
    let signal_path = neuronbox_home().join("swap_signal.json");

    let mut payload = serde_json::json!({
        "signal_version": 1,
        "model_ref": model_ref,
        "ts": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });

    if let Some(q) = quantization {
        payload["quantization"] = serde_json::Value::String(q.to_string());
    }

    if let Some(dir) = resolved_dir {
        payload["resolved_model_dir"] = serde_json::Value::String(dir.display().to_string());
    }

    std::fs::create_dir_all(neuronbox_home())?;
    std::fs::write(&signal_path, serde_json::to_string_pretty(&payload)?)?;

    Ok(())
}

pub async fn swap(model_ref: &str, quantization: Option<String>) -> Result<()> {
    daemon_spawn::ensure_daemon_running().await?;

    // Resolve model to local path if possible
    let resolved_dir = resolve_model_dir(model_ref);

    // Send to daemon (daemon still writes its own signal, but we'll overwrite with resolved path)
    let resp = daemon_client::request(DaemonRequest::SwapModel {
        model_ref: model_ref.to_string(),
        quantization: quantization.clone(),
    })
    .await?;

    match resp {
        neuronbox_runtime::protocol::DaemonResponse::Swapped {
            model_ref: ref_out,
            quantization: quant_out,
        } => {
            // Overwrite signal with resolved_model_dir if we have it
            if let Err(e) =
                write_swap_signal(&ref_out, quant_out.as_deref(), resolved_dir.as_deref())
            {
                eprintln!("neuron: warning — failed to write swap signal: {e}");
            }

            println!("Active model (daemon state): {ref_out} {:?}", quant_out);
            if let Some(dir) = &resolved_dir {
                println!("Resolved model directory: {}", dir.display());
            }
            println!("Note: the next `neuron run` uses the store as configured; live model hot-swap requires a long-running worker that reads this daemon state.");
        }
        neuronbox_runtime::protocol::DaemonResponse::Error { message } => {
            anyhow::bail!("{message}");
        }
        _ => anyhow::bail!("unexpected daemon response"),
    }
    Ok(())
}
