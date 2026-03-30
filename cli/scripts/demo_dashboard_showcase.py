#!/usr/bin/env python3
"""
Active-service demo to populate `neuron dashboard`: fake sessions + animated tokens/s.

Usage (from repo root):
  ./cli/scripts/demo_dashboard.sh
  # or:
  python3 cli/scripts/demo_dashboard_showcase.py

In **another** terminal: ./target/debug/neuron dashboard

Ctrl+C here: unregisters sessions, stops workers, and stops neurond if this script started it.
"""
from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Repo root: cli/scripts/ -> ../../
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def default_socket_path() -> Path:
    return Path(os.environ.get("NEURONBOX_SOCKET", Path.home() / ".neuronbox" / "neuron.sock"))


def daemon_request(obj: dict[str, Any], sock_path: Path | None = None) -> dict[str, Any]:
    path = sock_path or default_socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(str(path))
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        s.sendall(line.encode("utf-8"))
        buf = bytearray()
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
            if b"\n" in buf:
                break
    text = bytes(buf).decode("utf-8").strip().splitlines()[0]
    return json.loads(text)


def try_ping(sock: Path) -> bool:
    try:
        r = daemon_request({"method": "ping"}, sock)
        return r.get("response") == "pong"
    except OSError:
        return False


def ensure_neurond() -> tuple[subprocess.Popen | None, Path]:
    sock = default_socket_path()
    if try_ping(sock):
        print("[demo] neurond already reachable at", sock)
        return None, sock

    exe = os.environ.get("NEUROND_PATH")
    if exe:
        neurond = Path(exe)
    else:
        neurond = REPO_ROOT / "target" / "debug" / "neurond"
    if not neurond.exists():
        print(
            f"[demo] Binary not found: {neurond}\n"
            "  Build: cargo build -p neuronbox-runtime --bin neurond",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[demo] Starting neurond…")
    proc = subprocess.Popen(
        [str(neurond)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for i in range(60):
        time.sleep(0.1)
        if try_ping(sock):
            print("[demo] neurond ready.")
            return proc, sock
    proc.terminate()
    print("[demo] Timeout: neurond did not respond.", file=sys.stderr)
    sys.exit(1)


def spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(999999)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def register(name: str, pid: int, vram_mb: int, tokens: float | None, sock: Path) -> None:
    body: dict[str, Any] = {
        "method": "register_session",
        "name": name,
        "estimated_vram_mb": vram_mb,
        "pid": pid,
        "tokens_per_sec": tokens,
    }
    r = daemon_request(body, sock)
    if r.get("response") != "registered":
        print(f"[demo] unexpected register response: {r}", file=sys.stderr)


def unregister(pid: int, sock: Path) -> None:
    try:
        daemon_request({"method": "unregister_session", "pid": pid}, sock)
    except OSError:
        pass


def main() -> None:
    neurond_proc, sock = ensure_neurond()

    demos = [
        ("inference-llm", 12288, 42.0),
        ("train-peft", 20480, 12.5),
        ("eval-bench", 4096, 128.0),
    ]

    sleepers: list[subprocess.Popen] = []
    meta: list[tuple[str, int, int, float]] = []

    print("[demo] Spawning fake workers (real PIDs)…")
    for name, vram, base_tok in demos:
        p = spawn_sleeper()
        sleepers.append(p)
        meta.append((name, p.pid, vram, base_tok))
        register(name, p.pid, vram, base_tok, sock)
        print(f"  · {name} pid={p.pid} vram≈{vram} MiB tok/s≈{base_tok}")

    print()
    print("─── Open another terminal and run: ───")
    print(f"  cd {REPO_ROOT} && ./target/debug/neuron dashboard")
    print()
    print("Ctrl+C to stop the demo and clean up.")
    print()

    t0 = time.time()

    def cleanup() -> None:
        print("\n[demo] Cleaning up…")
        for name, pid, _, _ in meta:
            unregister(pid, sock)
        for p in sleepers:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
        if neurond_proc is not None:
            neurond_proc.terminate()
            try:
                neurond_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                neurond_proc.kill()

    def on_sigint(_sig, _frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    try:
        while True:
            elapsed = time.time() - t0
            for i, (name, pid, vram, base) in enumerate(meta):
                # Visible variation on the dashboard (~1 s refresh)
                wobble = 15.0 * math.sin(elapsed * 0.8 + i * 1.7)
                tok = max(0.1, base + wobble)
                register(name, pid, vram, tok, sock)
            time.sleep(0.9)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
