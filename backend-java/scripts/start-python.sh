#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/xunbian-backend/backend.env"
mkdir -p "$HOME/.local/share/xunbian-backend/logs"
echo "$$" > "$HOME/.local/share/xunbian-backend/logs/python.pid"
cd "$(dirname "$0")/../.."
exec /home/halcyon/.venvs/trust-agent-demo/bin/python -u tools/java_backend_bridge.py --port 8787
