#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.local/share/xunbian-tools/java-env.sh"
source "$HOME/.config/xunbian-backend/backend.env"
cd "$(dirname "$0")/.."
export XB_ARCHIVE_FILE="${XB_ARCHIVE_FILE:-$(cd .. && pwd)/web/app_data.json}"
mkdir -p "$HOME/.local/share/xunbian-backend/logs"
if ! /home/halcyon/.venvs/trust-agent-demo/bin/python -c 'import os,urllib.request; r=urllib.request.Request("http://127.0.0.1:8787/internal/health",headers={"X-Xunbian-Token":os.environ["XB_INTERNAL_TOKEN"]}); urllib.request.urlopen(r,timeout=2)' 2>/dev/null; then
  nohup bash scripts/start-python.sh > "$HOME/.local/share/xunbian-backend/logs/python.log" 2>&1 &
  echo "$!" > "$HOME/.local/share/xunbian-backend/logs/python.pid"
fi
if [ ! -f target/xunbian-backend-0.1.0.jar ]; then bash scripts/maven.sh -DskipTests package; fi
for attempt in {1..20}; do
  if /home/halcyon/.venvs/trust-agent-demo/bin/python -c 'import os,urllib.request; r=urllib.request.Request("http://127.0.0.1:8787/internal/health",headers={"X-Xunbian-Token":os.environ["XB_INTERNAL_TOKEN"]}); urllib.request.urlopen(r,timeout=2)' 2>/dev/null; then break; fi
  if [ "$attempt" = 20 ]; then printf 'Python服务未就绪，请检查本地日志。\n' >&2; exit 1; fi
  sleep 1
done
# A live Java process is insufficient when its Python worker has stopped.
# Recover the worker first, then reuse the existing authenticated backend.
if /home/halcyon/.venvs/trust-agent-demo/bin/python -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:8786/api/v1/health",timeout=2)); assert d.get("application")=="xunbian-backend"' 2>/dev/null; then
  printf '巡变已运行：http://127.0.0.1:8786/\n'; exit 0
fi
# Run a content-addressed copy on Linux storage. A rebuild on the mounted
# Windows drive must not replace the JAR currently open by this JVM.
mkdir -p "$HOME/.local/share/xunbian-backend/app"
digest="$(sha256sum target/xunbian-backend-0.1.0.jar | cut -d' ' -f1)"
runtime="$HOME/.local/share/xunbian-backend/app/backend-${digest}.jar"
if [ ! -f "$runtime" ]; then cp target/xunbian-backend-0.1.0.jar "$runtime"; fi
echo "$$" > "$HOME/.local/share/xunbian-backend/logs/java.pid"
printf '\n巡变 Java 后端：http://127.0.0.1:8786/\n首次问数将按需加载本地模型。\n\n'
exec java -jar "$runtime"
