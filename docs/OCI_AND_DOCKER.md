# OCI mode and Docker

NeuronBox is **ML-first**: `neuron pull` and host-mode `neuron run` do **not** call Docker.

## When Docker is used

Docker appears on the host only when you opt into **OCI-style isolation**:

| Entry | What runs `docker` |
|--------|---------------------|
| `neuron run --oci` | `docker run` with project / venv / store mounts ([`cli/src/oci/docker_run.rs`](../cli/src/oci/docker_run.rs)) |
| `runtime.mode: oci` in `neuron.yaml` | Same as above when you run `neuron run` |
| `neuron oci prepare` | `docker pull`, `docker create`, `docker export` to build a runc bundle ([`cli/src/oci/runc_prepare.rs`](../cli/src/oci/runc_prepare.rs)) |

`neuron oci runc` uses the **`runc`** binary on a prepared bundle, not `docker run`.

## If Docker is missing

- **Host workflow** (`neuron pull`, `neuron run` without `--oci`): **unaffected**.
- **OCI workflow**: install Docker (or use only host mode), or prepare bundles on a machine that has Docker and copy the bundle.

## Container images and `neuron pull`

`neuron pull` only targets **ML artifacts** (HF-style ids, aliases, local folders). To pull an OCI **image** (e.g. `ubuntu:22.04`), use `docker pull` directly, or `neuron oci prepare --image …` when building a runc rootfs.
