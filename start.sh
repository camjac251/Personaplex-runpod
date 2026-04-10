#!/usr/bin/env bash
#
# RunPod bootstrap for Personaplex-runpod.
#
# Idempotent: safe to re-run on pod restart. First boot does the heavy
# setup (install uv, clone repo, sync venv, download voice prompts).
# Subsequent boots return to the server launch within seconds because
# uv sync --frozen verifies the lockfile and exits early when nothing
# has changed.
#
# Paths rationale: /workspace is the persistent MooseFS volume that
# survives pod restarts; everything else on / is a ~20 GB overlay wiped
# on restart. /dev/shm is tmpfs and faster, but it is mounted noexec on
# RunPod so we cannot run the venv's python from it. Hence every
# stateful or executable path lives under /workspace.

set -euo pipefail

REPO_URL="https://github.com/camjac251/Personaplex-runpod.git"
REPO_DIR="/workspace/Personaplex-oneclicker"

# HuggingFace + torch caches
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="/workspace/huggingface_cache"
export TORCH_HOME="/workspace/.cache/torch"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR="/workspace/tmp"

# uv binary, wheel cache, and project venv all on /workspace.
# UV_CACHE_DIR path matches runpod/base so other RunPod tooling can
# share the same wheel cache. UV_LINK_MODE=copy because MooseFS does
# not support hardlinks reliably.
export UV_INSTALL_DIR="/workspace/.local/bin"
export UV_CACHE_DIR="/workspace/.cache/uv"
export UV_PROJECT_ENVIRONMENT="/workspace/.venv-personaplex"
export UV_LINK_MODE="copy"
export PATH="$UV_INSTALL_DIR:$PATH"

mkdir -p \
    "$TMPDIR" \
    "$HF_HOME" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$UV_CACHE_DIR" \
    "$UV_INSTALL_DIR"

log() { printf '[boot] %s\n' "$*"; }

log "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo 'none detected')"

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
log "uv version: $(uv --version)"

if [ -d "$REPO_DIR/.git" ]; then
    log "pulling latest from origin into $REPO_DIR"
    git -C "$REPO_DIR" fetch --quiet origin
    git -C "$REPO_DIR" pull --ff-only \
        || log "WARN: pull failed (local edits?); continuing with current checkout"
else
    log "first boot: cloning $REPO_URL into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

log "resolving venv at $UV_PROJECT_ENVIRONMENT"
uv sync --frozen

if [ ! -d "$REPO_DIR/voices" ]; then
    log "fetching voice prompts from Hugging Face (one-time, ~16 GB)"
    uv run --frozen python - <<'PY'
import os
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

archive = hf_hub_download(
    "nvidia/personaplex-7b-v1",
    "voices.tgz",
    token=os.environ.get("HF_TOKEN"),
)
target = Path.cwd() / "voices"
target.mkdir(exist_ok=True)
with tarfile.open(archive, "r:gz") as tf:
    tf.extractall(path=target)
print(f"voices ready at {target}")
PY
fi

log "starting moshi-server on :8998"
exec uv run --frozen moshi-server \
    --host 0.0.0.0 \
    --port 8998 \
    --static none \
    --voice-prompt-dir voices
