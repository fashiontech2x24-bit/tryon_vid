#!/usr/bin/env bash
# Restart ONLY the dual-inference FastAPI app (leave the two ComfyUI instances
# running) — e.g. after `git pull`.
#
#   git pull && bash restart_app.sh
#
# Env passthrough (all optional): COMFY_URL_A, COMFY_URL_B, APP_PORT, POSE_DEVICE,
# ASSET_INTERNAL_SECRET, CALLBACK_URL_OVERRIDE. Unset values use the app defaults
# (ComfyUI on 8188/8189; callback override pinned in dual_app.py).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/logs"; mkdir -p "$LOG_DIR"
APP_PORT="${APP_PORT:-8000}"

# pick a python that has uvicorn — prefer the one the running app uses (read from
# the pid file, but only if it really points at our uvicorn dual_app process).
PY="${PY:-}"
if [[ -f "$LOG_DIR/app.pid" ]]; then
  old_pid="$(cat "$LOG_DIR/app.pid")"
  if [[ -r "/proc/$old_pid/cmdline" ]] && \
     tr '\0' ' ' < "/proc/$old_pid/cmdline" | grep -q 'uvicorn dual_app:app'; then
    [[ -z "$PY" ]] && PY="$(tr '\0' '\n' < "/proc/$old_pid/cmdline" | head -1)"
  fi
fi
if [[ -z "$PY" ]] || ! "$PY" -c 'import uvicorn' >/dev/null 2>&1; then
  for cand in /venv/main/bin/python3 /venv/main/bin/python python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import uvicorn' >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
PY="${PY:-python3}"

# stop the running dual app
[[ -f "$LOG_DIR/app.pid" ]] && kill "$(cat "$LOG_DIR/app.pid")" 2>/dev/null || true
pkill -f 'uvicorn dual_app:app' 2>/dev/null || true
sleep 1

echo "Starting dual_app on 0.0.0.0:${APP_PORT}  (PY=$PY)"
( cd "$REPO_DIR/server" && \
  COMFY_URL_A="${COMFY_URL_A:-http://127.0.0.1:8188}" \
  COMFY_URL_B="${COMFY_URL_B:-http://127.0.0.1:8189}" \
  POSE_DEVICE="${POSE_DEVICE:-auto}" \
  ASSET_INTERNAL_SECRET="${ASSET_INTERNAL_SECRET:-}" \
  ${CALLBACK_URL_OVERRIDE:+CALLBACK_URL_OVERRIDE="$CALLBACK_URL_OVERRIDE"} \
  nohup "$PY" -m uvicorn dual_app:app --host 0.0.0.0 --port "$APP_PORT" \
    </dev/null >"$LOG_DIR/app.log" 2>&1 & echo $! >"$LOG_DIR/app.pid" )
sleep 3

if curl -fsS "http://127.0.0.1:${APP_PORT}/api/health" >/dev/null; then
  echo "dual_app restarted (pid $(cat "$LOG_DIR/app.pid"), log: $LOG_DIR/app.log)"
else
  echo "dual_app did not come up — tail $LOG_DIR/app.log" >&2; exit 1
fi
