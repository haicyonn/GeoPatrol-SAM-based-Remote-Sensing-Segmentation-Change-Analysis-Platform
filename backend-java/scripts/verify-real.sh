#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/xunbian-backend/backend.env"
cd "$(dirname "$0")/../.."
exec /home/halcyon/.venvs/trust-agent-demo/bin/python -u backend-java/scripts/verify_e2e.py
