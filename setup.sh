#!/usr/bin/env bash
#
# setup.sh — one-shot bootstrap for the Wan 2.1 VACE V2V demo on a Vast.ai
#            ComfyUI template instance (A100 80GB / L40S, etc.)
#
#   git clone <this repo> && cd tryon_vid && bash setup.sh
#
# What it does:
#   1. locate the persistent ComfyUI install (+ its python venv)
#   2. install the VideoHelperSuite custom node (provides VHS_LoadVideo)
#   3. download all Wan VACE models  (download_v2v.sh)
#   4. install the demo server's python deps
#   5. (re)start ComfyUI headless on 127.0.0.1:8188 so it picks up the new node
#   6. start the FastAPI demo app on 0.0.0.0:${APP_PORT:-8000}
#
# Flags / env:
#   APP_PORT=8000          port the web demo listens on (expose this in Vast)
#   COMFY_PORT=8188        internal ComfyUI port
#   --skip-models          don't re-run the model download
#   --no-comfy-restart     leave a running ComfyUI alone (still ensures it's up)
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PORT="${APP_PORT:-8000}"
COMFY_PORT="${COMFY_PORT:-8188}"
LOG_DIR="$REPO_DIR/logs"; mkdir -p "$LOG_DIR"

SKIP_MODELS=0
RESTART_COMFY=1
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    --no-comfy-restart) RESTART_COMFY=0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m>> %s\033[0m\n' "$*"; }

# ----------------------------------------------------------------------------
# 1. Locate ComfyUI + its python interpreter
# ----------------------------------------------------------------------------
detect_comfyui() {
  if [[ -n "${COMFYUI_DIR:-}" ]]; then echo "$COMFYUI_DIR"; return; fi
  local pid cwd
  pid="$(pgrep -f 'main.py' | head -n1 || true)"
  if [[ -n "$pid" ]]; then
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    [[ -f "$cwd/main.py" ]] && { echo "$cwd"; return; }
  fi
  for c in /workspace/ComfyUI /opt/workspace-internal/ComfyUI /opt/ComfyUI "$HOME/ComfyUI" /ComfyUI; do
    [[ -f "$c/main.py" ]] && { echo "$c"; return; }
  done
  local found
  found="$(find / -maxdepth 6 -type f -name main.py -path '*ComfyUI*' 2>/dev/null | head -n1 || true)"
  [[ -n "$found" ]] && { dirname "$found"; return; }
  return 1
}

COMFY="$(detect_comfyui || true)"
if [[ -z "$COMFY" || ! -d "$COMFY" ]]; then
  echo "ERROR: could not find ComfyUI. Re-run with COMFYUI_DIR=/path/to/ComfyUI bash setup.sh" >&2
  exit 1
fi
say "ComfyUI root: $COMFY"

# Prefer the interpreter the running ComfyUI uses (its venv); else fall back.
detect_python() {
  local pid exe
  pid="$(pgrep -f 'main.py' | head -n1 || true)"
  if [[ -n "$pid" ]]; then
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    [[ -x "$exe" ]] && { echo "$exe"; return; }
  fi
  for p in "$COMFY/venv/bin/python" "$COMFY/.venv/bin/python"; do
    [[ -x "$p" ]] && { echo "$p"; return; }
  done
  command -v python3 || command -v python
}
PY="$(detect_python)"
say "Python interpreter: $PY"

# ----------------------------------------------------------------------------
# 2. Install the VideoHelperSuite custom node (VHS_LoadVideo)
# ----------------------------------------------------------------------------
VHS_DIR="$COMFY/custom_nodes/ComfyUI-VideoHelperSuite"
if [[ -d "$VHS_DIR/.git" ]]; then
  say "VideoHelperSuite present — updating"
  git -C "$VHS_DIR" pull --ff-only || true
else
  say "Installing VideoHelperSuite"
  git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite "$VHS_DIR"
fi
if [[ -f "$VHS_DIR/requirements.txt" ]]; then
  "$PY" -m pip install -q -r "$VHS_DIR/requirements.txt" || true
fi

# ----------------------------------------------------------------------------
# 3. Download models
# ----------------------------------------------------------------------------
if [[ "$SKIP_MODELS" -eq 1 ]]; then
  say "Skipping model download (--skip-models)"
else
  say "Downloading Wan VACE models"
  COMFYUI_DIR="$COMFY" bash "$REPO_DIR/download_v2v.sh"
fi

# ----------------------------------------------------------------------------
# 4. Install demo server deps
# ----------------------------------------------------------------------------
say "Installing demo server requirements"
"$PY" -m pip install -q -r "$REPO_DIR/server/requirements.txt"

# ----------------------------------------------------------------------------
# 5. (Re)start ComfyUI headless and wait for it
# ----------------------------------------------------------------------------
comfy_up() { curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; }

wait_comfy() {
  local tries="${1:-120}"
  for ((i=0; i<tries; i++)); do comfy_up && return 0; sleep 2; done
  return 1
}

start_comfy() {
  say "Starting ComfyUI on 127.0.0.1:${COMFY_PORT} (log: $LOG_DIR/comfyui.log)"
  ( cd "$COMFY" && nohup "$PY" main.py --listen 127.0.0.1 --port "$COMFY_PORT" \
      >"$LOG_DIR/comfyui.log" 2>&1 & echo $! >"$LOG_DIR/comfyui.pid" )
}

if [[ "$RESTART_COMFY" -eq 1 ]]; then
  if pgrep -f 'main.py' >/dev/null; then
    say "Restarting ComfyUI to load the new custom node"
    pkill -f 'ComfyUI.*main.py' 2>/dev/null || pkill -f 'main.py' 2>/dev/null || true
    sleep 5
  fi
  # a supervisor in the template may auto-respawn ComfyUI; give it a moment.
  if ! wait_comfy 8; then start_comfy; fi
else
  say "Leaving existing ComfyUI as-is (--no-comfy-restart)"
  comfy_up || start_comfy
fi

say "Waiting for ComfyUI to be ready…"
if wait_comfy 150; then
  echo "   ComfyUI is up."
else
  echo "   WARNING: ComfyUI did not respond yet — check $LOG_DIR/comfyui.log" >&2
fi

# ----------------------------------------------------------------------------
# 6. Start the demo web app
# ----------------------------------------------------------------------------
pkill -f 'uvicorn app:app' 2>/dev/null || true
say "Starting demo app on 0.0.0.0:${APP_PORT} (log: $LOG_DIR/app.log)"
( cd "$REPO_DIR/server" && \
  COMFY_URL="http://127.0.0.1:${COMFY_PORT}" \
  nohup "$PY" -m uvicorn app:app --host 0.0.0.0 --port "$APP_PORT" \
    >"$LOG_DIR/app.log" 2>&1 & echo $! >"$LOG_DIR/app.pid" )

sleep 3
cat <<EOF

============================================================
 Setup complete.

 ComfyUI (internal):  http://127.0.0.1:${COMFY_PORT}
 Demo web app:        http://0.0.0.0:${APP_PORT}

 Open the demo in your browser via the Vast.ai mapped address
 for port ${APP_PORT}  (Instance → "Open" / IP:mapped-port).
 Make sure port ${APP_PORT} is exposed in the instance config.

 Logs:   $LOG_DIR/app.log   $LOG_DIR/comfyui.log
 Stop:   kill \$(cat $LOG_DIR/app.pid) \$(cat $LOG_DIR/comfyui.pid)
============================================================
EOF
