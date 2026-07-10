#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${PERSONAPLEX_APP_DIR:-/opt/personaplex-runpod}"
PORT="${PORT:-8998}"
DEFAULT_VOICE_DIR="/workspace/personaplex/voices"
if [ -d "/workspace/Personaplex-oneclicker/voices" ]; then
    DEFAULT_VOICE_DIR="/workspace/Personaplex-oneclicker/voices"
fi
VOICE_DIR="${PERSONAPLEX_VOICE_DIR:-$DEFAULT_VOICE_DIR}"

export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-/workspace/huggingface_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export PYTHONPATH="$APP_DIR/moshi${PYTHONPATH:+:$PYTHONPATH}"

log() { printf '[personaplex] %s\n' "$*"; }

mkdir -p \
    "$HF_HOME" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$TMPDIR" \
    "$(dirname "$VOICE_DIR")"

log "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo 'none detected')"

if [ -z "$HF_TOKEN" ]; then
    log "WARN: HF_TOKEN is not set. Gated model downloads will fail."
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
    log "WARN: GEMINI_API_KEY is not set. Vision features will be disabled."
fi

asset_args=(--voice-dir "$VOICE_DIR")

if [ "${PERSONAPLEX_FETCH_VOICES:-1}" = "0" ]; then
    asset_args+=(--skip-voices)
fi

if [ "${PERSONAPLEX_PREFETCH_MODEL:-1}" = "0" ]; then
    asset_args+=(--skip-model)
fi

log "checking model assets under $HF_HOME and voices under $VOICE_DIR"
"$APP_DIR/.venv/bin/python" "$APP_DIR/docker/prefetch_assets.py" "${asset_args[@]}"

log "starting moshi-server on :$PORT"
exec "$APP_DIR/.venv/bin/python" -m moshi.server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --voice-prompt-dir "$VOICE_DIR"
