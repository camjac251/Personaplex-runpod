#!/usr/bin/env bash
# Supervisor for moshi-server. Run it inside screen/tmux on the GPU host:
#
#   screen -dmS personaplex ~/personaplex-webrtc/scripts/run-personaplex.sh
#
# It restarts the server on any exit (the server hard-exits with code 70 on
# a poisoned CUDA context or a detected GPU pipeline hang), enforces a
# startup deadline (a GPU hang during warmup never crashes; the process
# just never binds the port), and backs off when restarts churn so a dying
# GPU cannot produce a tight crash loop.
set -uo pipefail

APP_DIR="${PERSONAPLEX_DIR:-$HOME/personaplex-webrtc}"
PORT="${PERSONAPLEX_PORT:-8998}"
STARTUP_DEADLINE_SEC=300
BACKOFF_MIN_SEC=3
BACKOFF_MAX_SEC=60
# A run shorter than this counts as churn and doubles the backoff.
HEALTHY_RUN_SEC=300

cd "$APP_DIR" || exit 1
set -a
. ./.env
set +a

backoff=$BACKOFF_MIN_SEC
while true; do
  echo "[supervisor] starting moshi-server at $(date -u +%H:%M:%S)"
  started_at=$(date +%s)
  .venv/bin/moshi-server --host 0.0.0.0 --port "$PORT" \
    --voice-prompt-dir voices --ssl "$APP_DIR/certs" &
  server_pid=$!

  # Startup deadline: kill a launch that never starts serving.
  (
    for _ in $(seq 1 "$STARTUP_DEADLINE_SEC"); do
      sleep 1
      kill -0 "$server_pid" 2>/dev/null || exit 0
      curl -sk --max-time 2 "https://127.0.0.1:$PORT/api/info" \
        >/dev/null 2>&1 && exit 0
    done
    echo "[supervisor] not serving after ${STARTUP_DEADLINE_SEC}s (warmup hang); killing pid $server_pid"
    kill -9 "$server_pid" 2>/dev/null
  ) &
  deadline_pid=$!

  wait "$server_pid"
  code=$?
  kill "$deadline_pid" 2>/dev/null
  wait "$deadline_pid" 2>/dev/null

  ran=$(( $(date +%s) - started_at ))
  if [ "$ran" -ge "$HEALTHY_RUN_SEC" ]; then
    backoff=$BACKOFF_MIN_SEC
  fi
  echo "[supervisor] moshi-server exited (code=$code) after ${ran}s; restarting in ${backoff}s"
  sleep "$backoff"
  backoff=$(( backoff * 2 ))
  if [ "$backoff" -gt "$BACKOFF_MAX_SEC" ]; then
    backoff=$BACKOFF_MAX_SEC
  fi
done
