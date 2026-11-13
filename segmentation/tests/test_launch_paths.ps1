$ErrorActionPreference='Stop'
$helper=Join-Path (Split-Path -Parent $PSScriptRoot) 'paths.ps1'
if(-not (Test-Path -LiteralPath $helper)){throw 'Cross-platform path conversion is not implemented.'}
. $helper
$values=@('H:\xunbian-segmentation-runs\test\manifest.json','H:\space directory\中文目录\manifest.json')
foreach($value in $values){
    $normalized=ConvertTo-WslArtifactPath $value
    $received=& wsl.exe -d Ubuntu-22.04 --exec /home/halcyon/.venvs/trust-agent-demo/bin/python -c 'import json,sys;print(json.dumps(sys.argv[1],ensure_ascii=True))' $normalized
    if($LASTEXITCODE -ne 0){throw 'WSL argument probe failed.'}
    if(($received | ConvertFrom-Json) -ne $value.Replace('\','/')){throw 'Artifact path changed across the Windows/WSL boundary.'}
}
$rejected=$false
try{ConvertTo-WslArtifactPath 'https://example.com/file'}catch{$rejected=$true}
if(-not $rejected){throw 'Remote file path was accepted.'}
Write-Output 'PASS: backslash path, space/Chinese path and remote-path rejection.'
