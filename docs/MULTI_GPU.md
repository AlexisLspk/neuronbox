# Multi-GPU with NeuronBox

NeuronBox does **not** yet orchestrate multi-GPU training (no built-in DDP launcher). This document describes the **current contract** and how to chain with PyTorch / the ecosystem.

## Exposing multiple GPUs to the process

- **`neuron run --gpu 0,1`** (or `--gpu 0,1,2`): the CLI sets `CUDA_VISIBLE_DEVICES` for the entrypoint script.
- You can also set **`CUDA_VISIBLE_DEVICES`** yourself in the environment or in `neuron.yaml` under `env`.

IDs are those seen by the NVIDIA driver on the host (outside the container, or inside the container if you use Docker with `--gpus all` / device requests).

## Distributed training (DDP, DeepSpeed, etc.)

**Launch** responsibility (process count, `MASTER_ADDR`, ports, etc.) stays in your **`entrypoint`** (often a shell or Python script that calls `torchrun`).

Minimal example (adapt to your project):

```bash
# After resolving venv / deps from neuron.yaml
torchrun --nproc_per_node=2 train.py
```

Or with explicit variables:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py
```

NeuronBox mainly provides today: **hashed Python environment**, **model store**, **session registration** with the daemon (soft VRAM, stats). It does not replace `torchrun`, Slurm, or Kubernetes.

## `gpu.strategy` field in `neuron.yaml`

The schema accepts `single` | `pipeline` | `tensor`. **No automatic orchestration** is applied today: the field is **documentation / intent** and may later drive launchers or validation.

Possible roadmap:

1. Validate consistency of `strategy` vs visible GPU count.
2. Generate or suggest a documented `torchrun` command line.
3. Optional integration with a scheduler (out of scope for “local only”).

## Strong isolation (MIG, partitions)

For **hardware VRAM quota** per job, see [GPU_VRAM.md](GPU_VRAM.md) (MIG, `CUDA_VISIBLE_DEVICES` with instance UUID).

## macOS and Linux

- **Linux**: typical CUDA / ROCm workflow; multi-GPU NVIDIA via `CUDA_VISIBLE_DEVICES` + your launcher.
- **macOS**: often a single Apple / Metal device; datacenter-style multi-GPU usually does not apply.

## See also

- [GPU_VRAM.md](GPU_VRAM.md) — soft quotas, MIG, monitoring.
- [README.md](../README.md) — overview, tutorial, CLI reference.
