@echo off
chcp 65001 >nul
echo Starting Xunbian Java backend: http://127.0.0.1:8786/
wsl -d Ubuntu-22.04 -- bash /mnt/e/trust_agent_mainline_clean/backend-java/scripts/start.sh
pause
