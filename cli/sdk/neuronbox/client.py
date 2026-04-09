from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any


def _default_sock() -> Path:
    home = Path.home()
    return home / ".neuronbox" / "neuron.sock"


class DaemonClient:
    """Newline-delimited JSON client for the NeuronBox daemon Unix socket."""

    def __init__(self, socket_path: Path | None = None) -> None:
        raw = os.environ.get("NEURONBOX_SOCKET")
        self.socket_path = (
            Path(raw)
            if raw
            else (socket_path or _default_sock())
        )

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        payload = {"method": method, **params}
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode()

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(self.socket_path))
            s.sendall(data)
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        line = buf.split(b"\n", 1)[0].decode()
        return json.loads(line)

    def ping(self) -> None:
        r = self.call("ping")
        if r.get("response") != "pong":
            raise RuntimeError(f"unexpected: {r}")

    def stats(self) -> dict[str, Any]:
        return self.call("stats")
