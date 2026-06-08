#!/usr/bin/env bash
# Stop the ComfyUI + demo app that setup.sh started (by pid file).
set -uo pipefail
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
for name in app comfyui; do
  f="$LOG_DIR/$name.pid"
  if [[ -f "$f" ]]; then
    pid="$(cat "$f")"
    if kill "$pid" 2>/dev/null; then echo "stopped $name (pid $pid)"; else echo "$name (pid $pid) not running"; fi
    rm -f "$f"
  else
    echo "no pid file for $name"
  fi
done
