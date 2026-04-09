"""Automatic throughput hooks for popular ML frameworks.

This module patches generation methods to report tokens/s to the NeuronBox daemon.
It is loaded automatically when NEURONBOX_AUTOHOOK=1 (set by `neuron run`).

Supported frameworks (when installed):
- transformers (HuggingFace): GenerationMixin.generate
- vllm: LLM.generate
- llama_cpp: Llama.__call__ and Llama.create_completion

The hooks measure wall-clock time and output token count, then push updates to
neurond via the Unix socket. Updates are rate-limited to avoid spamming the daemon.
"""

from __future__ import annotations

import os
import threading
import time
from functools import wraps
from typing import Any, Callable

_MIN_UPDATE_INTERVAL = 0.5  # seconds between daemon updates
_last_update: float = 0.0
_lock = threading.Lock()

# Session info from environment (set by `neuron run`)
_SESSION_NAME: str | None = os.environ.get("NEURONBOX_SESSION_NAME")
_SESSION_VRAM_MB: int = int(os.environ.get("NEURONBOX_SESSION_VRAM_MB", "8192"))
_SESSION_PID: int = os.getpid()

# Optional metrics log file (NDJSON format)
_METRICS_LOG: str | None = os.environ.get("NEURONBOX_METRICS_LOG")


def _log_metrics_to_file(tps: float) -> None:
    """Append metrics to NDJSON log file if NEURONBOX_METRICS_LOG is set."""
    if not _METRICS_LOG:
        return

    try:
        import json
        from pathlib import Path

        log_path = Path(_METRICS_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "ts": time.time(),
            "pid": _SESSION_PID,
            "name": _SESSION_NAME,
            "tokens_per_sec": tps,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Non-blocking: don't crash user code on log failure


def _report_tokens_per_sec(tps: float) -> None:
    """Send throughput update to daemon (rate-limited, non-blocking on failure)."""
    global _last_update
    now = time.monotonic()
    with _lock:
        if now - _last_update < _MIN_UPDATE_INTERVAL:
            return
        _last_update = now

    if not _SESSION_NAME:
        return

    # Log to file if configured
    _log_metrics_to_file(tps)

    # Send to daemon
    try:
        from neuronbox.client import DaemonClient

        DaemonClient().call(
            "register_session",
            name=_SESSION_NAME,
            estimated_vram_mb=_SESSION_VRAM_MB,
            pid=_SESSION_PID,
            tokens_per_sec=tps,
        )
    except Exception:
        pass  # Non-blocking: don't crash user code if daemon is down


def _count_tokens_transformers(output: Any) -> int:
    """Count output tokens from transformers generate() result."""
    try:
        if hasattr(output, "sequences"):
            # GenerateOutput or similar
            seqs = output.sequences
        else:
            # Raw tensor
            seqs = output
        if hasattr(seqs, "shape"):
            # (batch, seq_len)
            return int(seqs.shape[0] * seqs.shape[1])
        elif hasattr(seqs, "__len__"):
            return sum(len(s) for s in seqs)
    except Exception:
        pass
    return 0


def _hook_transformers() -> bool:
    """Patch transformers.GenerationMixin.generate to report throughput."""
    try:
        from transformers import GenerationMixin
    except ImportError:
        return False

    original_generate = GenerationMixin.generate

    @wraps(original_generate)
    def patched_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_generate(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0

        n_tokens = _count_tokens_transformers(result)
        if elapsed > 0 and n_tokens > 0:
            tps = n_tokens / elapsed
            _report_tokens_per_sec(tps)

        return result

    GenerationMixin.generate = patched_generate
    return True


def _hook_vllm() -> bool:
    """Patch vllm.LLM.generate to report throughput."""
    try:
        import vllm
    except ImportError:
        return False

    original_generate = vllm.LLM.generate

    @wraps(original_generate)
    def patched_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_generate(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0

        n_tokens = 0
        try:
            for req_output in result:
                for completion in req_output.outputs:
                    n_tokens += len(completion.token_ids)
        except Exception:
            pass

        if elapsed > 0 and n_tokens > 0:
            tps = n_tokens / elapsed
            _report_tokens_per_sec(tps)

        return result

    vllm.LLM.generate = patched_generate
    return True


def _hook_llama_cpp() -> bool:
    """Patch llama_cpp.Llama to report throughput."""
    try:
        import llama_cpp
    except ImportError:
        return False

    # Patch __call__ (chat-style)
    original_call = llama_cpp.Llama.__call__

    @wraps(original_call)
    def patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_call(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0

        n_tokens = 0
        try:
            if isinstance(result, dict):
                usage = result.get("usage", {})
                n_tokens = usage.get("completion_tokens", 0)
        except Exception:
            pass

        if elapsed > 0 and n_tokens > 0:
            tps = n_tokens / elapsed
            _report_tokens_per_sec(tps)

        return result

    llama_cpp.Llama.__call__ = patched_call

    # Patch create_completion
    original_completion = llama_cpp.Llama.create_completion

    @wraps(original_completion)
    def patched_completion(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_completion(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0

        n_tokens = 0
        try:
            if isinstance(result, dict):
                usage = result.get("usage", {})
                n_tokens = usage.get("completion_tokens", 0)
        except Exception:
            pass

        if elapsed > 0 and n_tokens > 0:
            tps = n_tokens / elapsed
            _report_tokens_per_sec(tps)

        return result

    llama_cpp.Llama.create_completion = patched_completion
    return True


def _hook_openai_local() -> bool:
    """Patch openai client for local endpoints (e.g. vLLM server, llama.cpp server)."""
    try:
        import openai
    except ImportError:
        return False

    # Only hook if using a local base_url (not api.openai.com)
    # This is tricky because the client is instantiated by user code.
    # We'll patch the Completions.create method.

    try:
        original_create = openai.resources.chat.completions.Completions.create
    except AttributeError:
        return False

    @wraps(original_create)
    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_create(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0

        n_tokens = 0
        try:
            if hasattr(result, "usage") and result.usage:
                n_tokens = result.usage.completion_tokens or 0
        except Exception:
            pass

        if elapsed > 0 and n_tokens > 0:
            tps = n_tokens / elapsed
            _report_tokens_per_sec(tps)

        return result

    openai.resources.chat.completions.Completions.create = patched_create
    return True


def install_hooks() -> list[str]:
    """Install all available hooks. Returns list of hooked framework names."""
    hooked: list[str] = []

    if _hook_transformers():
        hooked.append("transformers")
    if _hook_vllm():
        hooked.append("vllm")
    if _hook_llama_cpp():
        hooked.append("llama_cpp")
    if _hook_openai_local():
        hooked.append("openai")

    return hooked


# Auto-install when this module is imported (via PYTHONPATH or sitecustomize)
if os.environ.get("NEURONBOX_AUTOHOOK") == "1":
    _installed = install_hooks()
    if _installed and os.environ.get("NEURONBOX_AUTOHOOK_VERBOSE") == "1":
        import sys
        print(f"[neuronbox] throughput hooks installed: {', '.join(_installed)}", file=sys.stderr)
