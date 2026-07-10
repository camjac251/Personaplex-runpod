#!/usr/bin/env bash
set -euo pipefail

if [ "${PERSONAPLEX_START_RUNPOD_SERVICES:-1}" != "0" ] && [ -x /start.sh ]; then
    /start.sh &
fi

exec /opt/personaplex-runpod/docker/app-start.sh
