@echo off
chcp 65001 >nul
wsl -d Ubuntu-22.04 -- /usr/bin/python3 /mnt/e/trust_agent_mainline_clean/backend-java/scripts/stop.py
pause
