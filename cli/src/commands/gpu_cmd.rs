use anyhow::Result;
use clap::Subcommand;
use neuronbox_runtime::gpu::detect_gpus;

#[derive(Subcommand)]
pub enum GpuCommands {
    /// List detected GPUs (NVIDIA, ROCm, Apple Silicon).
    List,
}

pub fn gpu(cmd: GpuCommands) -> Result<()> {
    match cmd {
        GpuCommands::List => {
            let gpus = detect_gpus()?;
            if gpus.is_empty() {
                println!("No GPU detected (nvidia-smi / rocm-smi / Apple Silicon).");
                return Ok(());
            }
            for g in gpus {
                let mem = if g.memory_total_mb > 0 {
                    format!("{} GB VRAM", g.memory_total_mb / 1024)
                } else {
                    "VRAM n/a".to_string()
                };
                println!("GPU {}: {} — {} — {}", g.index, g.name, mem, g.backend);
            }
            println!(
                "Multi-GPU / DDP: use `neuron run --gpu 0,1` (CUDA_VISIBLE_DEVICES) and your launcher (e.g. torchrun)."
            );
        }
    }
    Ok(())
}
