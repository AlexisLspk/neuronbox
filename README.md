# NeuronBox

**Run local AI work (training, fine-tuning, inference, benchmarks) without reinventing glue code every week.**

You describe the project once in **`neuron.yaml`**: where weights live (model hub, a folder on disk, or a single file), which Python stack you need, GPU expectations, and the script to run. NeuronBox builds or reuses a **hashed virtualenv**, exposes paths to your code through **`NEURONBOX_*` env vars**, and can spin an optional **daemon** so you get a **live terminal dashboard** of sessions and GPU activity. When you need hard isolation, flip to **Docker/OCI**: same manifest, containerized execution.

**`neuron`** alone drops you on a short **getting-started** screen; **`neuron help`** lists everything else.

**Scope:** NeuronBox is a **working local stack**: CLI, `neurond` daemon, socket protocol, dashboard, and model store, for **machine-local** AI workflows you run yourself. It is **not** a hosted multi-tenant cloud service. Semver **0.1.x** is early versioning, not a “prototype only” label.

**License:** Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option (see [Cargo.toml](Cargo.toml)).

---

## Why NeuronBox (at a glance)

1. **End of the “infrastructure death march”**: you declare what the job needs (for example **`gpu.min_vram`** in **`neuron.yaml`**) instead of spending cycles on CUDA matrices, driver versions, and ad-hoc Docker volume maps for huge weight trees. The stack stays **environment-aware** for GPU/VRAM and models.

2. **A sane “model store,” not a 50 GB image pull**: Docker layers are a poor transport for multi-gigabyte weights. NeuronBox treats the model as a **first-class artifact**: **shared cache** across projects, paths wired via **`NEURONBOX_*`**, and room for **lazy / progressive** loading patterns, without shipping checkpoints inside images.

3. **Hot-swap for real iteration**: swapping models in plain Docker often means **stop container, new image, cold GPU context**. **`neuron swap`** (plus **`neuron serve`** / workers) keeps the **Python stack and context warm** and swaps **weights** via the daemon and swap signal, which is much faster for research and A/B runs.

4. **GPU abstraction across platforms**: **`neuron host inspect`** and **`neuron gpu list`** summarize **Metal, ROCm, CUDA**, and optional **NVML** so one mental model covers laptops and Linux boxes, not a bespoke script per machine.

---

## Contents

- [Why NeuronBox (at a glance)](#why-neuronbox-at-a-glance)
- [Tutorial: from zero](#tutorial-from-zero)
- [Use cases](#use-cases)
- [NeuronBox vs Docker](#neuronbox-vs-docker)
- [CLI quick reference](#cli-quick-reference)
- [Prerequisites & build](#prerequisites--build)
- [References](#references)
- [Repository layout](#repository-layout)

---

## Tutorial: from zero

**1. Build the workspace**

```bash
cd NeuroBox   # your clone of this repository
cargo build --workspace
```

Use the binary by path (or add `target/debug` to your `PATH`):

```bash
./target/debug/neuron          # welcome screen
./target/debug/neuron help     # full command list
```

**2. Create a project manifest**

```bash
mkdir ~/my-llm-project && cd ~/my-llm-project
/path/to/NeuroBox/target/debug/neuron init
```

This writes **`neuron.yaml`**. Edit `model.name`, `entrypoint`, `gpu.min_vram`, and `runtime.packages` for your machine.

**3. Point NeuronBox at your model**

- **Remote weight trees** via `org/model` ids (the format used by common open hubs):  

  ```bash
  /path/to/NeuroBox/target/debug/neuron pull org/model
  ```

  Downloads land in the **global store** under `~/.neuronbox` by default.

- **Local weights** (folder or `.gguf` / `.safetensors` / etc.): set `model.source: local` and `model.name` in `neuron.yaml` (no pull step).

- **OCI container images** (e.g. `ubuntu:22.04`): **not** handled by `neuron pull`. Use **`docker pull <image>`** directly, or **`neuron oci prepare --image <image>`** if you are building a runc bundle (Docker required for that subcommand only).

See [specs/daemon-sessions.md](specs/daemon-sessions.md) for store layout. See [docs/OCI_AND_DOCKER.md](docs/OCI_AND_DOCKER.md) for when Docker is invoked.

**4. Run your entrypoint**

```bash
/path/to/NeuroBox/target/debug/neuron run
```

NeuronBox creates or reuses a **venv keyed by your `runtime` section**, sets `NEURONBOX_MODEL_DIR` (and related env), and runs your script. **`neuron run`** tries to start **`neurond`** in the background when the socket is down (best effort); if `stats` / `dashboard` still cannot connect, run **`neuron daemon`** in another terminal.

**5. Watch the system (optional)**

**`neuron stats`** and **`neuron dashboard`** need a reachable **`neurond`** on the Unix socket (default `~/.neuronbox/neuron.sock`, or `NEURONBOX_SOCKET`).

In another terminal:

```bash
/path/to/NeuroBox/target/debug/neuron dashboard   # TUI: sessions + host/GPU summary
# or
/path/to/NeuroBox/target/debug/neuron stats       # plain-text snapshot
```

**6. Demo populated dashboard (no real training)**

From the **repo root**:

```bash
./cli/scripts/demo_dashboard.sh
```

In a second terminal: `./target/debug/neuron dashboard`. Stop the demo with `Ctrl+C` in the first terminal.

---

## Use cases

| Scenario | Why NeuronBox |
|----------|----------------|
| **Training, LoRA, eval, batch inference** | One manifest ties **code + weights + Python + GPU**; you run the same flow on your laptop or a beefier box. |
| **Large models & shared disks** | **Central store** for downloaded trees; local paths stay **references**, not copies into every clone. |
| **Reproducible envs** | `runtime` hashes into `store/envs/…`; **`neuron lock`** pins dependencies when you care about bit-for-bit installs. |
| **Visibility while jobs run** | Daemon + **`neuron dashboard`** / **`neuron stats`** show who’s on the GPU and rough throughput where the app reports it. |
| **Optional isolation** | **`neuron run --oci`** / `runtime.mode: oci` when you want container mounts and NVIDIA toolkit without writing `docker run` by hand each time. |
| **Mixed hardware** | **`neuron host inspect`** and **`neuron gpu list`** give a **versioned snapshot** (Metal, ROCm, CUDA) for support and CI notes. |

---

## NeuronBox vs Docker

They solve overlapping but different problems: NeuronBox is not a Docker replacement.

| | **Docker** | **NeuronBox** |
|---|------------|---------------|
| **Primary unit** | Image + container lifecycle | **Project manifest** (`neuron.yaml`) + host paths |
| **Strength** | Portable images, strong isolation, orchestration ecosystems | **Fast iteration on metal**: hashed venvs, **shared model store**, one command to run what `neuron.yaml` declares |
| **ML ergonomics** | You compose `docker run`, volumes, and CUDA yourself | **Wired env vars** (`NEURONBOX_MODEL_DIR`, etc.), native model pull/store on the host; **optional** OCI (`neuron run --oci`) uses Docker only when you opt in |
| **When to prefer Docker alone** | Production deploys, identical bits everywhere, Kubernetes | *n/a* |
| **When NeuronBox helps** | *n/a* | Daily local work, consistent project contract, optional Docker only when you opt in |

**Bottom line:** Use Docker when the image *is* the product. Use NeuronBox when the **repo + YAML + venv + store** are the product and you want one tool to glue them, **without** giving up Docker for the cases where containers are the right tool.

---

## CLI quick reference

| Command | Role |
|---------|------|
| `neuron` | Welcome screen (no subcommand) |
| `neuron help` | Full help + quick start |
| `neuron init` | Create `neuron.yaml` |
| `neuron pull <id>` | ML models only: HF-style id, alias, or local path → store |
| `neuron run` | Project entrypoint on host (venv + optional GPU checks) |
| `neuron run --oci` | Same project via **Docker** OCI mounts (explicit opt-in) |
| `neuron serve` | Long-lived worker + swap signal (same venv as `run`) |
| `neuron swap` | Update daemon state + `~/.neuronbox/swap_signal.json` (for `neuron serve` / workers) |
| `neuron stats` | Text stats (sessions + GPU lines) |
| `neuron dashboard` | Terminal UI (sessions + host/GPU) |
| `neuron host inspect` | JSON `HostSnapshot` |
| `neuron gpu list` | Detected GPUs |
| `neuron model list` | Store index |
| `neuron lock` | Generate `requirements.lock` for hashed env |
| `neuron daemon` | Run `neurond` in foreground |
| `neuron oci prepare` / `oci runc` | Expert OCI bundle path (`prepare` uses Docker on the host) |

### Breaking changes (CLI)

Older workflows that used **`neuron pull`** for Docker image tags, **`neuron ps` / `stop` / `rm`**, or **`neuron run -it image …`** as a `docker run` proxy must switch to the **`docker`** CLI directly. NeuronBox keeps Docker **only** under the OCI commands above.

---

## Prerequisites & build

- **Rust** (this workspace)
- **Python 3** (projects and `neuron serve` worker)
- **GPU tooling** (optional): NVIDIA (`nvidia-smi`), AMD (`rocm-smi`), or Apple Silicon (Metal); see `neuron host inspect`

```bash
cargo build --workspace
```

**Linux + NVIDIA:** NVML build for better GPU reporting:

```bash
cargo build -p neuronbox-cli --features nvml
cargo build -p neuronbox-runtime --features nvml
```

Binaries: `target/debug/neuron` and `target/debug/neurond` (or `release/`).

**Install on PATH (optional):**

```bash
cargo install --path cli
```

---

## References

| Doc | Topic |
|-----|--------|
| [neuronbox-mvp.md](neuronbox-mvp.md) | Product vision, architecture & specification reference |
| [docs/CLI_UX.md](docs/CLI_UX.md) | Welcome screen, terminal theme, dashboard behavior |
| [docs/OCI_AND_DOCKER.md](docs/OCI_AND_DOCKER.md) | When Docker is used (OCI only) vs ML-first host path |
| [specs/neuron.yaml.schema.json](specs/neuron.yaml.schema.json) | `neuron.yaml` schema |
| [specs/swap-signal.schema.json](specs/swap-signal.schema.json) | Swap signal file |
| [specs/daemon-sessions.md](specs/daemon-sessions.md) | Daemon socket & sessions |
| [docs/MULTI_GPU.md](docs/MULTI_GPU.md) | Multi-GPU / DDP (documentation) |
| [docs/GPU_VRAM.md](docs/GPU_VRAM.md) | VRAM, NVML, MIG notes, `NEURONBOX_DISABLE_VRAM_WATCH` |
| [docs/SECURITY.md](docs/SECURITY.md) | Socket trust boundary, line limits, tokens, model trust |
| [specs/examples/](specs/examples/) | Example YAML files |

---

## Repository layout

- **`cli/`**: `neuron` binary and `cli/scripts/` demos
- **`runtime/`**: library + `neurond` daemon
- **`specs/`**: JSON Schema and `neuron.yaml` examples
