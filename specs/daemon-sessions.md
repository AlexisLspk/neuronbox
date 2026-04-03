# Daemon contract: sessions and `neuron stats`

The `neurond` daemon listens on a Unix socket (default: `~/.neuronbox/neuron.sock`, override with `NEURONBOX_SOCKET`). Protocol: **one JSON line per request**, one JSON line per response.

Exact types are defined in Rust in `runtime/src/protocol.rs` (`DaemonRequest` / `DaemonResponse`), with `#[serde(tag = "method")]` and `rename_all = "snake_case"`.

## Registering a session (`neuron run`)

After starting the Python process, `neuron run` sends:

```json
{
  "method": "register_session",
  "name": "project-name",
  "estimated_vram_mb": 8192,
  "pid": 12345,
  "tokens_per_sec": null
}
```

- **`estimated_vram_mb`**: estimate used for soft NVIDIA monitoring (Linux), not a hardware reservation.
- **`tokens_per_sec`**: optional; may be omitted or `null` if unknown.

When the process exits, the CLI sends `unregister_session` with the same `pid`.

## Updating tokens/s from your code

The registry is keyed by **`pid`**: a new `register_session` call with the **same** `pid` **replaces** the existing row (same name, new estimate, new throughput).

To show a value in `neuron stats`, send periodically for example:

```json
{
  "method": "register_session",
  "name": "my-inference",
  "estimated_vram_mb": 12000,
  "pid": 12345,
  "tokens_per_sec": 47.3
}
```

### Minimal client

Open the Unix socket in line mode (one JSON request + `\n` per line, one response per line), e.g. with `socat` or a small Python `socket` stdlib script; payloads are as documented here and in `runtime/src/protocol.rs`.

## Other useful methods

| `method`           | Role |
|--------------------|------|
| `ping`             | Daemon health → `pong` |
| `list_sessions`    | List of `SessionInfo` |
| `stats`            | Sessions + NVIDIA compute processes (NVML if built with `nvml`, else `nvidia-smi`) + **`active_model`** (swap) + **`vram_used_by_pid`**: real MiB per PID (compute apps) |
| `version`          | Protocol version negotiation (`v` integer) |

## Persistent session (multiple requests)

The server (`runtime/src/server.rs`) reads JSON lines in a loop on **one connection** until EOF. A client can:

1. Open the socket once;
2. Write one JSON line (request) + `\n`;
3. Read one JSON line (response);
4. Repeat 2–3 without reconnecting;
5. Close the socket when done.

The Rust CLI exposes this via `DaemonSession` in `cli/src/daemon_client.rs` (`connect`, then `request` in a loop). The `request()` helper without a session remains available for a single round trip.

For a Python or other client, keep the socket open and chain lines as above rather than reconnecting for every request.

## Current limitations

- No encryption: local traffic only (Unix socket).
- `tokens_per_sec` is not time-aggregated in the daemon: it is the **last** value recorded for that PID.
- **`neuron dashboard`** chart history is rebuilt **in the CLI** (~1 s samples), not stored in the daemon.
