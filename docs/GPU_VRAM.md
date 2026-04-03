# VRAM, MIG, and NeuronBox monitoring

## Soft check before launch

`neuron run` may refuse to start if `gpu.min_vram` in `neuron.yaml` exceeds the estimated total GPU VRAM (NVIDIA via **NVML** when built with the `nvml` feature on Linux, else `nvidia-smi`; AMD via `rocm-smi` when available).

## Monitoring while running (daemon)

`neurond` may run a loop that reads per-PID memory on NVIDIA GPUs (**NVML** if the `nvml` feature is enabled, else `nvidia-smi`). If a process exceeds about **115%** of the `estimated_vram_mb` value recorded at `RegisterSession`, the daemon sends **SIGKILL** to that PID (Linux).

### Disabling VRAM monitoring (SIGKILL)

To disable this loop (e.g. test environments, machines without GPU, or local policy):

```bash
export NEURONBOX_DISABLE_VRAM_WATCH=1
# accepted values: 1, true, yes (case-insensitive)
neuron daemon
```

Without this variable, default behavior is unchanged.

This is **not** a hardware cap: without **MIG** (Multi-Instance GPU, NVIDIA datacenter) or equivalent partitioning, CUDA does not guarantee a strict per-process VRAM quota like system RAM.

## MIG (NVIDIA)

For **strong isolation** per GPU instance on supported cards (A100, H100, etc.):

1. Configure MIG with NVIDIA system tools (`nvidia-smi mig`).
2. Expose only the desired instance with `CUDA_VISIBLE_DEVICES` (often an instance UUID).
3. Document in your deployment that one NeuronBox job = one MIG instance.

NeuronBox does not configure MIG automatically; this doc is an entry point.

## Injected PyTorch variables

In host mode or via Docker OCI, NeuronBox may set `PYTORCH_CUDA_ALLOC_CONF` (e.g. `max_split_size_mb` derived from `gpu.min_vram`) to reduce fragmentation. This is a **heuristic**, not a guarantee on peak VRAM.

## NVML (Linux NVIDIA)

- **Prerequisite**: NVIDIA driver installed (`libnvidia-ml.so` present at runtime).
- **Build**: `cargo build -p neuronbox-runtime --features nvml` or `cargo build -p neuronbox-cli --features nvml` (enables the same feature on the runtime).
- **`neuron host inspect`**: field `probes.nvml` indicates whether the NVIDIA GPU list came from NVML (`true`) or `nvidia-smi` (`false`).
- If NVML init fails, the code **falls back** to `nvidia-smi` (same as builds without the feature).

On **macOS**, the feature compiles but the NVML path is not used (no NVML in the probe); Apple GPU is handled separately.

## Multi-GPU and DDP

See the dedicated guide: [MULTI_GPU.md](MULTI_GPU.md) (`neuron run --gpu`, `torchrun`, `gpu.strategy` field, roadmap).
