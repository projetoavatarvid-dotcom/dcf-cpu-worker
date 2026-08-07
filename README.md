# DCF CPU Worker

CPU preparation and validation worker for the **Digital Clone Framework (DCF)**.

## Purpose

This image is designed for low-cost RunPod CPU Pods used to prepare and validate ComfyUI workflows before allocating a GPU.

It provides:

- Python 3.12.11
- Git, curl, wget, ffmpeg, jq, rsync and SSH tools
- `uv`
- a writable DCF Python virtual environment
- Hugging Face tooling
- helpers for ComfyUI-Blackwell CPU structural validation

Models, workflows, ComfyUI code, custom nodes and jobs remain on the RunPod Network Volume mounted at `/workspace`.

## Image

After the first GitHub release/tag, the image will be published as:

```text
ghcr.io/<github-user>/dcf-cpu-worker:v1.0.0
```

and:

```text
ghcr.io/<github-user>/dcf-cpu-worker:latest
```

## Publish

The repository includes:

```text
.github/workflows/publish.yml
```

Create and push a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will build the Linux AMD64 Docker image and publish it to GitHub Container Registry.

## RunPod

After the package is published, configure the DCF Toolkit:

```powershell
. .\DCF.CpuWorker.ps1

Enable-DCFCpuWorkerImage `
    -ImageName "ghcr.io/<github-user>/dcf-cpu-worker:v1.0.0"

Show-DCFCpuWorkerConfig
```

Keep the generic RunPod image as fallback until the first CPU smoke test succeeds.

## Persistent paths

The image intentionally does not contain:

```text
/workspace/models
/workspace/projects/ComfyUI-Blackwell
/workspace/envs/comfyui-blackwell
/workspace/dcf
/workspace/jobs
```

Those remain on persistent storage.

## DCF CPU validation helper

Inside the worker:

```bash
dcf-comfy-cpu
```

starts ComfyUI-Blackwell in CPU mode while exposing the persisted Blackwell Python packages.

## Version

DCF CPU Worker `1.0.0`
