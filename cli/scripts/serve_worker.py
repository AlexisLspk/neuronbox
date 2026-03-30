#!/usr/bin/env python3
"""Worker minimal pour `neuron serve` : charge optionnellement un modèle HF et écoute swap_signal.json.

Contrat du fichier signal : specs/swap-signal.schema.json
- signal_version : entier ; version actuelle = 1. Si absent, traité comme 1 (rétrocompat).
- Si signal_version > 1 : le worker journalise et ignore le swap (éviter comportement indéfini).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SUPPORTED_SIGNAL_VERSION = 1


def signal_version_ok(data: dict) -> bool:
    v = data.get("signal_version", 1)
    if not isinstance(v, int):
        print("[neuronbox serve] signal_version invalide, ignoré", flush=True)
        return False
    if v < 1:
        print("[neuronbox serve] signal_version < 1, ignoré", flush=True)
        return False
    if v > SUPPORTED_SIGNAL_VERSION:
        print(
            f"[neuronbox serve] signal_version {v} non prise en charge "
            f"(max {SUPPORTED_SIGNAL_VERSION}), swap ignoré",
            flush=True,
        )
        return False
    return True


def main() -> None:
    home = Path(os.environ.get("NEURONBOX_HOME", Path.home() / ".neuronbox"))
    home.mkdir(parents=True, exist_ok=True)
    sig = home / "swap_signal.json"
    model_dir = os.environ.get("NEURONBOX_MODEL_DIR", "")
    print("[neuronbox serve] NEURONBOX_MODEL_DIR=", model_dir, flush=True)
    model = None
    if model_dir and os.path.isdir(model_dir):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            print("[neuronbox serve] modèle chargé.", flush=True)
        except Exception as e:
            print("[neuronbox serve] import/chargement optionnel échoué:", e, flush=True)

    last_blob = None
    while True:
        if sig.exists():
            try:
                blob = sig.read_text(encoding="utf-8")
                if blob != last_blob:
                    last_blob = blob
                    data = json.loads(blob)
                    if not signal_version_ok(data):
                        time.sleep(1)
                        continue
                    print("[neuronbox serve] signal swap:", data, flush=True)
                    ref = data.get("model_ref")
                    if model is not None and ref:
                        try:
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            import torch

                            tok = AutoTokenizer.from_pretrained(ref, trust_remote_code=True)
                            model = AutoModelForCausalLM.from_pretrained(
                                ref,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto" if torch.cuda.is_available() else None,
                                trust_remote_code=True,
                            )
                            print("[neuronbox serve] rechargé depuis HF:", ref, flush=True)
                        except Exception as e:
                            print("[neuronbox serve] swap HF échoué:", e, flush=True)
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(1)


if __name__ == "__main__":
    main()
