# DCF.CpuWorker.ps1
# DCF CPU WORKER v1.0 configuration helpers.
#
# This module centralizes the CPU worker image selection so every DCF subsystem
# can migrate from the generic RunPod PyTorch image to the same validated image.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:DCFCpuWorkerVersion = "1.0.0"

function Get-DCFCpuWorkerConfigPath {
    $root = Split-Path -Parent $MyInvocation.MyCommand.Path
    Join-Path $root "DCF.CpuWorker.config.json"
}

function Initialize-DCFCpuWorkerConfig {
    [CmdletBinding()]
    param(
        [string]$ImageName = "REPLACE_WITH_REGISTRY/dcf-cpu-worker:1.0.0"
    )

    $path = Get-DCFCpuWorkerConfigPath

    $config = [ordered]@{
        schema_version = "1.0"
        worker = [ordered]@{
            image_name = $ImageName
            fallback_image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
            use_custom_image = $false
            python_version = "3.12.11"
            network_volume_id = "80zgznqvtq"
            volume_mount_path = "/workspace"
            datacenter_id = "EU-RO-1"
            cpu_flavor = "cpu3c"
            vcpu_count = 2
            container_disk_gb = 10
        }
    }

    $config |
        ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $path -Encoding UTF8

    Get-Item -LiteralPath $path
}

function Get-DCFCpuWorkerConfig {
    [CmdletBinding()]
    param()

    $path = Get-DCFCpuWorkerConfigPath

    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Initialize-DCFCpuWorkerConfig | Out-Null
    }

    Get-Content -LiteralPath $path -Raw -Encoding UTF8 |
        ConvertFrom-Json
}

function Get-DCFCpuWorkerImage {
    [CmdletBinding()]
    param()

    $c = Get-DCFCpuWorkerConfig

    if ([bool]$c.worker.use_custom_image) {
        $image = [string]$c.worker.image_name

        if ([string]::IsNullOrWhiteSpace($image) -or
            $image -like "REPLACE_WITH_REGISTRY/*") {
            throw "DCF CPU Worker custom image ainda nao foi configurada."
        }

        return $image
    }

    return [string]$c.worker.fallback_image
}

function Enable-DCFCpuWorkerImage {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$ImageName
    )

    $path = Get-DCFCpuWorkerConfigPath
    $c = Get-DCFCpuWorkerConfig

    $c.worker.image_name = $ImageName
    $c.worker.use_custom_image = $true

    $c |
        ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $path -Encoding UTF8

    Write-Host ("[DCF CPU WORKER] imagem ativa: " + $ImageName) -ForegroundColor Green
}

function Disable-DCFCpuWorkerImage {
    [CmdletBinding()]
    param()

    $path = Get-DCFCpuWorkerConfigPath
    $c = Get-DCFCpuWorkerConfig
    $c.worker.use_custom_image = $false

    $c |
        ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $path -Encoding UTF8

    Write-Host (
        "[DCF CPU WORKER] fallback ativo: " +
        [string]$c.worker.fallback_image
    ) -ForegroundColor Yellow
}

function Show-DCFCpuWorkerConfig {
    [CmdletBinding()]
    param()

    $c = Get-DCFCpuWorkerConfig

    Write-Host ""
    Write-Host "DCF CPU WORKER v1.0" -ForegroundColor Cyan
    Write-Host ("Custom image....: " + [string]$c.worker.image_name)
    Write-Host ("Enabled.........: " + [string]$c.worker.use_custom_image)
    Write-Host ("Effective image.: " + (Get-DCFCpuWorkerImage))
    Write-Host ("Python..........: " + [string]$c.worker.python_version)
    Write-Host ("Datacenter......: " + [string]$c.worker.datacenter_id)
    Write-Host ("Network Volume..: " + [string]$c.worker.network_volume_id)
    Write-Host ("CPU.............: " + [string]$c.worker.cpu_flavor)
    Write-Host ""
}

Write-Host "[DCF.CpuWorker v$script:DCFCpuWorkerVersion] carregado." -ForegroundColor DarkGray
