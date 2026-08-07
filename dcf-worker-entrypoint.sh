#!/usr/bin/env bash
set -euo pipefail

export DCF_WORKSPACE="${DCF_WORKSPACE:-/workspace}"
export DCF_COMFY_ROOT="${DCF_COMFY_ROOT:-/workspace/projects/ComfyUI-Blackwell}"
export DCF_BLACKWELL_ENV="${DCF_BLACKWELL_ENV:-/workspace/envs/comfyui-blackwell}"
export DCF_BLACKWELL_SITE_PACKAGES="${DCF_BLACKWELL_SITE_PACKAGES:-/workspace/envs/comfyui-blackwell/lib/python3.12/site-packages}"

# Use the worker venv for DCF tooling.
export PATH="/opt/dcf-worker/venv/bin:/root/.local/bin:${PATH}"

# Make the persisted Blackwell Python packages importable during CPU structural
# validation without rewriting or recreating the persistent venv.
if [[ -d "${DCF_BLACKWELL_SITE_PACKAGES}" ]]; then
    if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="${DCF_BLACKWELL_SITE_PACKAGES}:${PYTHONPATH}"
    else
        export PYTHONPATH="${DCF_BLACKWELL_SITE_PACKAGES}"
    fi
fi

exec "$@"
