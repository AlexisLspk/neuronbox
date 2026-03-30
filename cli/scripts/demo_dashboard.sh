#!/usr/bin/env bash
# Dashboard demo: start neurond if needed, register fake sessions.
# Prerequisites: cargo build -p neuronbox-cli && cargo build -p neuronbox-runtime --bin neurond
#
# Terminal 1: ./cli/scripts/demo_dashboard.sh
# Terminal 2: ./target/debug/neuron dashboard

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$ROOT/target/debug:$PATH"
exec python3 "$ROOT/cli/scripts/demo_dashboard_showcase.py" "$@"
