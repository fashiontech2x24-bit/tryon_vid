#!/usr/bin/env bash
# Restart ONLY the FastAPI app (leave ComfyUI running) — e.g. after `git pull`.
#
#   git pull && bash restart_app.sh
#
# Reuses the running app's COMFY_URL and python interpreter (read from the old
# process), so no arguments are needed. Override via env if desired:
#   COMFY_URL=http://127.0.0.1:8189 APP_PORT=8000 POSE_DEVICE=auto bash restart_app.sh
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/logs"; mkdir -p "$LOG_DIR"
APP_PORT="${APP_PORT:-8000}"

# inherit COMFY_URL + python from the currently running app — but only if the
# pid file actually points at our uvicorn process (PIDs get recycled; a stale
# pid file must not make us inherit from some unrelated process).
if [[ -f "$LOG_DIR/app.pid" ]]; then
  old_pid="$(cat "$LOG_DIR/app.pid")"
  if [[ -r "/proc/$old_pid/cmdline" ]] && \
     tr '\0' ' ' < "/proc/$old_pid/cmdline" | grep -q 'uvicorn app:app'; then
    if [[ -z "${COMFY_URL:-}" && -r "/proc/$old_pid/environ" ]]; then
      COMFY_URL="$(tr '\0' '\n' < "/proc/$old_pid/environ" | sed -n 's/^COMFY_URL=//p')"
    fi
    if [[ -z "${PY:-}" ]]; then
      PY="$(tr '\0' '\n' < "/proc/$old_pid/cmdline" | head -1)"
    fi
  fi
fi
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"

# PY must be a working python; otherwise fall back to common locations
if [[ -z "${PY:-}" ]] || ! "$PY" -c 'import sys' >/dev/null 2>&1; then
  for cand in /venv/main/bin/python3 /venv/main/bin/python python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import uvicorn' >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
PY="${PY:-python3}"

[[ -f "$LOG_DIR/app.pid" ]] && kill "$(cat "$LOG_DIR/app.pid")" 2>/dev/null || true
pkill -f 'uvicorn app:app' 2>/dev/null || true
sleep 1

echo "Starting app on 0.0.0.0:${APP_PORT}  (COMFY_URL=$COMFY_URL, PY=$PY)"
( cd "$REPO_DIR/server" && \
  COMFY_URL="$COMFY_URL" \
  POSE_DEVICE="${POSE_DEVICE:-auto}" \
  nohup "$PY" -m uvicorn app:app --host 0.0.0.0 --port "$APP_PORT" \
    >"$LOG_DIR/app.log" 2>&1 & echo $! >"$LOG_DIR/app.pid" )
sleep 2

if curl -fsS "http://127.0.0.1:${APP_PORT}/api/health"; then
  echo; echo "App restarted (pid $(cat "$LOG_DIR/app.pid"), log: $LOG_DIR/app.log)"
else
  echo "App did not come up — tail $LOG_DIR/app.log" >&2; exit 1
fi
