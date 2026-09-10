# syntax=docker/dockerfile:1
#
# DCF CPU WORKER v1.0
#
# Purpose:
# - low-cost RunPod CPU worker
# - model/workflow/custom-node installation
# - ComfyUI-Blackwell structural validation
# - DCF job preparation
#
# Persistent project data stays on the RunPod Network Volume mounted at /workspace.
# Models, ComfyUI code, custom nodes and jobs are NOT baked into this image.

FROM python:3.12.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.8.13
ARG S5CMD_VERSION=2.3.0

LABEL org.opencontainers.image.title="DCF CPU Worker"
LABEL org.opencontainers.image.description="CPU preparation and validation worker for Digital Clone Framework"
LABEL org.opencontainers.image.version="1.0.3"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DCF_WORKER_ROOT=/opt/dcf-worker \
    DCF_WORKSPACE=/workspace \
    DCF_COMFY_ROOT=/workspace/projects/ComfyUI-Blackwell \
    DCF_BLACKWELL_ENV=/workspace/envs/comfyui-blackwell \
    DCF_BLACKWELL_SITE_PACKAGES=/workspace/envs/comfyui-blackwell/lib/python3.12/site-packages \
    PATH=/opt/dcf-worker/venv/bin:/root/.local/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        curl \
        ffmpeg \
        git \
        jq \
        openssh-client \
        procps \
        rsync \
        tini \
        unzip \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Pin s5cmd for reproducible S3-compatible transfers.
RUN wget -q "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz" \
    && tar -xzf "s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz" \
    && install -m 0755 s5cmd /usr/local/bin/s5cmd \
    && rm -f s5cmd "s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz"
# Pin uv for reproducibility.
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# A normal writable venv avoids the "externally managed interpreter"
# behavior encountered with a temporary uv-managed Python installation.
RUN python -m venv /opt/dcf-worker/venv \
    && /opt/dcf-worker/venv/bin/python -m pip install --no-cache-dir \
        --upgrade pip setuptools wheel \
    && /opt/dcf-worker/venv/bin/python -m pip install --no-cache-dir \
        requests \
        boto3 \
        huggingface_hub

COPY dcf-worker-entrypoint.sh /usr/local/bin/dcf-worker-entrypoint
COPY dcf-worker-healthcheck.sh /usr/local/bin/dcf-worker-healthcheck
COPY dcf-comfy-cpu.sh /usr/local/bin/dcf-comfy-cpu
COPY dcf_artifact_transfer.py /usr/local/bin/dcf-artifact-transfer

RUN chmod 0755 \
    /usr/local/bin/dcf-worker-entrypoint \
    /usr/local/bin/dcf-worker-healthcheck \
    /usr/local/bin/dcf-artifact-transfer \
    /usr/local/bin/dcf-comfy-cpu

WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/dcf-worker-healthcheck"]

ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/dcf-worker-entrypoint"]
CMD ["sleep","infinity"]
