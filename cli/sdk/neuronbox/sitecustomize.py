"""Site customization entry point for NeuronBox auto-hooks.

This file is named sitecustomize.py so Python loads it automatically when the
sdk/ directory is on PYTHONPATH. It installs throughput hooks for ML frameworks.

The hooks only activate when NEURONBOX_AUTOHOOK=1 (set by `neuron run`).
"""

import os

if os.environ.get("NEURONBOX_AUTOHOOK") == "1":
    try:
        from neuronbox._hooks import install_hooks
        install_hooks()
    except Exception:
        pass  # Fail silently; don't break user code
