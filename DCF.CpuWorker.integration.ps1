# DCF CPU WORKER v1.0 - integration patch guide
#
# Load first:
#   . .\DCF.CpuWorker.ps1
#
# Wherever a DCF CPU Pod spec currently contains:
#
#   imageName = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
#
# replace ONLY the value expression with:
#
#   imageName = Get-DCFCpuWorkerImage
#
# Apply this to:
#   - DCF.ModelInstaller.ps1
#   - DCF.ModelBatchInstaller.ps1
#   - DCF.CustomNodeRepair.ps1
#   - DCF.ComfyCpuValidator.ps1
#
# Keep fallback enabled until the custom image has been pushed and smoke-tested.
#
# After publishing:
#
#   Enable-DCFCpuWorkerImage `
#       -ImageName "docker.io/YOUR_USER/dcf-cpu-worker:1.0.0"
#
# or, for GHCR:
#
#   Enable-DCFCpuWorkerImage `
#       -ImageName "ghcr.io/YOUR_USER/dcf-cpu-worker:1.0.0"
#
# Then:
#
#   Show-DCFCpuWorkerConfig
