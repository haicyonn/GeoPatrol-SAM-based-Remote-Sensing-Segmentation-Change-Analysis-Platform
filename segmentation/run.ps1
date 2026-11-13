param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'start.ps1') -Manage -NoBrowser:$NoBrowser
