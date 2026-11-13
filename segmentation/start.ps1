param([switch]$NoBrowser,[switch]$Manage)
$ErrorActionPreference='Stop'
& wsl.exe -d Ubuntu-22.04 --cd /mnt/e/trust_agent_mainline_clean --exec /home/halcyon/.venvs/trust-agent-demo/bin/python backend-java/scripts/start-background.py --batch 4270335f-a3c7-46e4-baa9-bb679d4339fa
if($LASTEXITCODE -ne 0){throw 'Unified backend startup failed.'}
& wsl.exe -d Ubuntu-22.04 --cd /mnt/e/trust_agent_mainline_clean --exec /home/halcyon/.venvs/trust-agent-demo/bin/python -m segmentation.control start
if($LASTEXITCODE -ne 0){throw 'Compatibility redirect startup failed.'}
if(-not $NoBrowser){if($Manage){Start-Process 'http://127.0.0.1:8786/manage.html'}else{Start-Process 'http://127.0.0.1:8786/index.html?batch=4270335f-a3c7-46e4-baa9-bb679d4339fa'}}
