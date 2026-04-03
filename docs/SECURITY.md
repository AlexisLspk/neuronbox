# Security notes (NeuronBox)

Short operational and trust boundaries for local use.

## Secrets and tokens

- **`HF_TOKEN`** (and similar) are read from the environment when pulling models. Do not commit tokens; restrict file permissions on shell profiles or CI secrets.
- The daemon socket is a **Unix domain socket** on the host (default under `~/.neuronbox/`). Any process running as the **same user** can connect. It is not intended for remote or multi-tenant exposure.

## Daemon protocol hardening

- Each request line is limited to **256 KiB** (see `MAX_REQUEST_LINE_BYTES` in `neuronbox-runtime`). Oversized or invalid UTF-8 lines get an error response and the connection is closed for that client.
- Disabling VRAM enforcement (`NEURONBOX_DISABLE_VRAM_WATCH`) removes **SIGKILL**-based overshoot handling; use only when you understand the tradeoff (see [GPU_VRAM.md](GPU_VRAM.md)).

## Models and Python workers

- Weights and code from the open hub or arbitrary paths are **untrusted data**. Your entrypoint / `neuron serve` worker should follow the same hygiene as any ML stack: review `trust_remote_code`, custom modules, and serialized formats before execution.

## Reporting

For vulnerabilities, prefer **private** disclosure to **neuronbox.contact@proton.me** (do not open a public issue for undisclosed exploits). Include steps to reproduce and affected versions if known.
