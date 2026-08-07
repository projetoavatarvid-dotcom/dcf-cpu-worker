#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${DCF_COMFY_ROOT:-/workspace/projects/ComfyUI-Blackwell}"
BLACKWELL_SITE="${DCF_BLACKWELL_SITE_PACKAGES:-/workspace/envs/comfyui-blackwell/lib/python3.12/site-packages}"
PORT="${DCF_COMFY_CPU_PORT:-8188}"
LISTEN="${DCF_COMFY_CPU_LISTEN:-127.0.0.1}"

if [[ ! -d "${COMFY_ROOT}" ]]; then
    echo "DCF: ComfyUI root not found: ${COMFY_ROOT}" >&2
    exit 10
fi

if [[ ! -d "${BLACKWELL_SITE}" ]]; then
    echo "DCF: Blackwell site-packages not found: ${BLACKWELL_SITE}" >&2
    exit 11
fi

export PYTHONPATH="${BLACKWELL_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export VIRTUAL_ENV="/opt/dcf-worker/venv"

cd "${COMFY_ROOT}"

exec /opt/dcf-worker/venv/bin/python main.py \
    --cpu \
    --listen "${LISTEN}" \
    --port "${PORT}" \
    "$@"
