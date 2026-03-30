#!/usr/bin/env python3
"""Exemple de client socket Unix pour neurond : ping et enregistrement de session (tokens/s).

Voir specs/daemon-sessions.md pour le contrat complet.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any


def default_socket_path() -> Path:
    return Path(os.environ.get("NEURONBOX_SOCKET", Path.home() / ".neuronbox" / "neuron.sock"))


def request(obj: dict[str, Any], sock_path: Path | None = None) -> dict[str, Any]:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Client minimal neurond (sessions / stats)")
    p.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Chemin du socket (défaut: ~/.neuronbox/neuron.sock ou NEURONBOX_SOCKET)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping", help="Envoie ping → attend pong")

    reg = sub.add_parser("register", help="Enregistre ou met à jour une session (même pid)")
    reg.add_argument("--name", required=True)
    reg.add_argument("--pid", type=int, required=True)
    reg.add_argument("--vram-mb", type=int, default=8192)
    reg.add_argument("--tokens-per-sec", type=float, default=None)

    args = p.parse_args()
    sock = args.socket
    if args.cmd == "ping":
        out = request({"method": "ping"}, sock)
        print(json.dumps(out, indent=2))
        return

    body: dict[str, Any] = {
        "method": "register_session",
        "name": args.name,
        "estimated_vram_mb": args.vram_mb,
        "pid": args.pid,
    }
    if args.tokens_per_sec is not None:
        body["tokens_per_sec"] = args.tokens_per_sec
    else:
        body["tokens_per_sec"] = None

    out = request(body, sock)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
