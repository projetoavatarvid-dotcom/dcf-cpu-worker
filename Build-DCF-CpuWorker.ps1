# Build-DCF-CpuWorker.ps1
# Requires Docker Desktop / Docker Engine on the machine running this command.

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ImageName
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Building DCF CPU WORKER..." -ForegroundColor Cyan
Write-Host ("Image: " + $ImageName)

docker build `
    --pull `
    -t $ImageName `
    $Root

if ($LASTEXITCODE -ne 0) {
    throw "docker build falhou."
}

Write-Host ""
Write-Host "Local smoke test..." -ForegroundColor Cyan

docker run --rm `
    --entrypoint /usr/local/bin/dcf-worker-healthcheck `
    $ImageName

if ($LASTEXITCODE -ne 0) {
    throw "healthcheck local falhou."
}

Write-Host ""
Write-Host "[OK] Build e healthcheck concluídos." -ForegroundColor Green
Write-Host "Para publicar:"
Write-Host ("  docker push " + $ImageName)
