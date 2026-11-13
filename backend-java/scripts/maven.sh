#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.local/share/xunbian-tools/java-env.sh"
source "$HOME/.config/xunbian-backend/backend.env"
cd "$(dirname "$0")/.."
exec mvn -B -ntp "$@"
