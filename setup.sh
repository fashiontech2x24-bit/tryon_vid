#!/usr/bin/env bash
#
# setup.sh — self-contained bootstrap for the Wan 2.1 VACE V2V demo.
#
# Instead of relying on the Vast.ai template's pre-installed ComfyUI, this
# clones our OWN ComfyUI (+ ComfyUI-Manager + VideoHelperSuite) and runs it on
# a port we control. Existing models on the box are reused via
# extra_model_paths.yaml, so nothing is re-downloaded.
#
#   git clone <repo> && cd tryon_vid && bash setup.sh
#
# Env / flags:
#   COMFY_DIR=/workspace/ComfyUI   where to install our ComfyUI
#   COMFY_PORT=8188                preferred internal port (auto-bumps if busy)
#   APP_PORT=8000                  web demo port (expose this in Vast)
#   COMFY_REF=master              git ref to pin ComfyUI to
#   --skip-models                 don't run the model download
#   --skip-comfy-install          don't clone/update ComfyUI or nodes (just (re)start)
#   --fresh                        delete and re-clone ComfyUI
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
COMFY_PORT="${COMFY_PORT:-8188}"
APP_PORT="${APP_PORT:-8000}"
COMFY_REF="${COMFY_REF:-v0.24.0}"   # pinned stable release; override with COMFY_REF=master
LOG_DIR="$REPO_DIR/logs"; mkdir -p "$LOG_DIR"

SKIP_MODELS=0; SKIP_INSTALL=0; FRESH=0
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-comfy-install) SKIP_INSTALL=1 ;;
    --fresh) FRESH=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m>> %s\033[0m\n' "$*"; }

# ----------------------------------------------------------------------------
# 0. Pick a python interpreter that already has CUDA torch
# ----------------------------------------------------------------------------
# Pick the interpreter that ACTUALLY has torch (the one ComfyUI must run with).
# Guessing fixed paths previously chose a bare python3 with no pip/torch, so
# every install silently no-op'd. Test import torch and require pip.
has_torch_and_pip() { "$1" -c 'import importlib.util as u,sys; sys.exit(0 if (u.find_spec("torch") and u.find_spec("pip")) else 1)' 2>/dev/null; }

detect_python() {
  local pid exe c
  # 1. a running ComfyUI's own interpreter wins
  pid="$(pgrep -f 'ComfyUI/main.py' | head -n1 || pgrep -f 'main.py' | head -n1 || true)"
  if [[ -n "$pid" ]]; then
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    [[ -x "$exe" ]] && has_torch_and_pip "$exe" && { echo "$exe"; return; }
  fi
  # 2. first candidate that has both torch and pip (venvs are often /venv/<name>)
  for c in /venv/main/bin/python3.12 /venv/main/bin/python3 /venv/main/bin/python \
           /venv/*/bin/python3.12 /venv/*/bin/python3 \
           /opt/venv/bin/python /opt/conda/bin/python "$COMFY_DIR/venv/bin/python" \
           /usr/local/bin/python3.12 /usr/bin/python3.12 /usr/local/bin/python3 /usr/bin/python3 \
           "$(command -v python3.12 || true)" "$(command -v python3 || true)" "$(command -v python || true)"; do
    [[ -x "$c" ]] || continue
    has_torch_and_pip "$c" && { echo "$c"; return; }
  done
  # 3. last resort: scan real interpreters under any */bin/ (skip dpkg scripts)
  c="$(find / -maxdepth 7 -type f -path '*/bin/*' \( -name 'python3.*' -o -name python3 -o -name python \) 2>/dev/null \
        | while read -r p; do has_torch_and_pip "$p" && { echo "$p"; break; }; done)"
  [[ -n "$c" ]] && { echo "$c"; return; }
  command -v python3 || command -v python
}
# explicit override wins; otherwise auto-detect by torch+pip
if [[ -n "${COMFY_PY:-}" ]]; then
  [[ -x "${COMFY_PY}" ]] || { echo "ERROR: COMFY_PY=$COMFY_PY is not executable" >&2; exit 1; }
  PY="$COMFY_PY"; say "Python (COMFY_PY override): $PY"
else
  PY="$(detect_python)"; say "Python: $PY"
fi
if ! "$PY" -c 'import torch; print("   torch", torch.__version__, "cuda:", torch.cuda.is_available())' 2>/dev/null; then
  echo "   ERROR: '$PY' cannot import torch — ComfyUI will not run with it." >&2
  echo "   Find the right interpreter and pass it explicitly, e.g.:" >&2
  echo "     for p in \$(find / -maxdepth 7 -name 'python3.*' 2>/dev/null); do \$p -c 'import torch' 2>/dev/null && echo \$p; done" >&2
  echo "     COMFY_PY=/that/python bash setup.sh --skip-models" >&2
  exit 1
fi

# install a requirements file, but never touch the working CUDA torch build.
# If the bulk install fails (e.g. one unsatisfiable pin), retry line-by-line so
# one bad entry can't block every other dependency.
pip_install_safe() {
  local req="$1" tmp; tmp="$(mktemp)"
  # Drop torch/torchvision/torchaudio (keep torchsde). NOTE: the boundary class
  # must NOT contain "[=" — that starts a POSIX equivalence class and makes grep
  # error to empty output (which silently installs nothing).
  grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([^[:alnum:]_]|$)' "$req" > "$tmp" || true
  if [[ ! -s "$tmp" ]]; then
    echo "   (filter produced no lines; installing $(basename "$req") unfiltered)"
    cp "$req" "$tmp"
  fi
  if ! "$PY" -m pip install -q -r "$tmp"; then
    echo "   (bulk install of $(basename "$req") failed; retrying line-by-line)"
    while IFS= read -r line; do
      [[ -z "$line" || "$line" == \#* ]] && continue
      "$PY" -m pip install -q "$line" || echo "   (skipped: $line)"
    done < "$tmp"
  fi
  rm -f "$tmp"
}

# ----------------------------------------------------------------------------
# 1. Clone ComfyUI + ComfyUI-Manager + VideoHelperSuite
# ----------------------------------------------------------------------------
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  [[ "$FRESH" -eq 1 ]] && { say "Removing existing $COMFY_DIR (--fresh)"; rm -rf "$COMFY_DIR"; }

  if [[ -d "$COMFY_DIR/.git" ]]; then
    say "Updating ComfyUI at $COMFY_DIR -> $COMFY_REF"
    git -C "$COMFY_DIR" fetch -q --tags --force origin || true
    git -C "$COMFY_DIR" checkout -q "$COMFY_REF" || true
    git -C "$COMFY_DIR" pull -q --ff-only 2>/dev/null || true   # no-op on a detached tag
  else
    say "Cloning ComfyUI -> $COMFY_DIR ($COMFY_REF)"
    mkdir -p "$(dirname "$COMFY_DIR")"
    git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
    git -C "$COMFY_DIR" checkout -q "$COMFY_REF" || true
  fi
  say "Installing ComfyUI requirements (keeping existing torch)"
  pip_install_safe "$COMFY_DIR/requirements.txt"

  mkdir -p "$COMFY_DIR/custom_nodes"
  install_node() {  # <name> <repo-url>
    local dir="$COMFY_DIR/custom_nodes/$1"
    if [[ -d "$dir/.git" ]]; then
      say "Updating custom node $1"; git -C "$dir" pull -q --ff-only || true
    else
      say "Installing custom node $1"; git clone --depth 1 "$2" "$dir"
    fi
    [[ -f "$dir/requirements.txt" ]] && pip_install_safe "$dir/requirements.txt"
  }
  install_node "ComfyUI-Manager"          "https://github.com/ltdrdata/ComfyUI-Manager"
  install_node "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
else
  say "Skipping ComfyUI/node install (--skip-comfy-install)"
fi

# ----------------------------------------------------------------------------
# 2. Reuse models already on the box (no re-download)
# ----------------------------------------------------------------------------
detect_existing_models() {
  for c in /opt/workspace-internal/ComfyUI /workspace/ComfyUI /opt/ComfyUI "$HOME/ComfyUI" /ComfyUI; do
    [[ "$c" == "$COMFY_DIR" ]] && continue
    [[ -d "$c/models/diffusion_models" ]] && { echo "$c"; return; }
  done
  return 1
}

MODELS_COMFY="$COMFY_DIR"
EXISTING_MODELS="$(detect_existing_models || true)"
if [[ -n "$EXISTING_MODELS" ]]; then
  MODELS_COMFY="$EXISTING_MODELS"
  say "Reusing existing models from $EXISTING_MODELS"
  cat > "$COMFY_DIR/extra_model_paths.yaml" <<YAML
tryon_reuse:
    base_path: $EXISTING_MODELS
    checkpoints: models/checkpoints
    vae: models/vae
    text_encoders: models/text_encoders
    clip: models/text_encoders
    diffusion_models: models/diffusion_models
    unet: models/diffusion_models
    loras: models/loras
YAML
  # extra_model_paths.yaml can be flaky across ComfyUI versions; symlink the
  # model folders directly into our ComfyUI/models as the reliable path.
  mkdir -p "$COMFY_DIR/models"
  for sub in diffusion_models text_encoders vae loras checkpoints clip_vision controlnet upscale_models; do
    src="$EXISTING_MODELS/models/$sub"; dst="$COMFY_DIR/models/$sub"
    [[ -d "$src" ]] || continue
    [[ -L "$dst" ]] && continue          # already linked
    [[ -d "$dst" ]] && rmdir "$dst" 2>/dev/null   # remove only if empty (don't clobber real models)
    [[ -e "$dst" ]] || ln -s "$src" "$dst"
  done
  echo "   linked model folders: $(ls -d "$COMFY_DIR"/models/*/ 2>/dev/null | wc -l) present"
else
  say "No pre-existing models found — will download into $COMFY_DIR/models"
fi

# ----------------------------------------------------------------------------
# 3. Download models (skips ones already present)
# ----------------------------------------------------------------------------
if [[ "$SKIP_MODELS" -eq 1 ]]; then
  say "Skipping model download (--skip-models)"
else
  say "Ensuring Wan VACE models are present (target: $MODELS_COMFY)"
  COMFYUI_DIR="$MODELS_COMFY" bash "$REPO_DIR/download_v2v.sh"
fi

# ----------------------------------------------------------------------------
# 4. Demo server deps
# ----------------------------------------------------------------------------
say "Installing demo server + pipeline requirements"
"$PY" -m pip install -q -r "$REPO_DIR/server/requirements.txt"

# boomerang postprocess needs ffmpeg/ffprobe on PATH
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  say "Installing ffmpeg"
  command -v apt-get >/dev/null 2>&1 && (apt-get update -y && apt-get install -y ffmpeg) >/dev/null 2>&1 \
    || echo "   (could not auto-install ffmpeg; install it manually for boomerang)"
fi

# Prefetch the DWPose ONNX models (~350MB) so the first request isn't slow.
say "Prefetching DWPose models for pose retargeting"
"$PY" - <<PYEOF || echo "   (model prefetch skipped/failed; will download on first use)"
import sys; sys.path.insert(0, "$REPO_DIR/server/pipeline")
import pose_retarget
pose_retarget.ensure_dwpose_models("${DWPOSE_MODEL_DIR:-$REPO_DIR/models/dwpose}")
print("   DWPose models ready")
PYEOF

# ALWAYS (re)install ComfyUI's own requirements into the runtime python — even
# with --skip-comfy-install — so the full dependency set (comfy-aimdo,
# comfy-kitchen, blake3, asset-DB libs, ...) is guaranteed present. pip is a
# no-op when already satisfied, so this is cheap on repeat runs.
if [[ -f "$COMFY_DIR/requirements.txt" ]]; then
  say "Ensuring ComfyUI's full requirements are installed in $PY"
  pip_install_safe "$COMFY_DIR/requirements.txt"
fi
"$PY" -m pip install -q blake3 || true   # sometimes optional in older pins
# Verify the import chain that has bitten us before declaring victory.
if "$PY" -c 'import filelock, sqlalchemy, alembic, pydantic_settings, blake3, comfy_aimdo, comfy_kitchen' 2>/dev/null; then
  echo "   ComfyUI core deps import OK"
else
  echo "   WARNING: some ComfyUI deps still not importable by $PY — it may crash." >&2
  "$PY" -c 'import comfy_aimdo' 2>&1 | tail -1 >&2 || true
fi

# ----------------------------------------------------------------------------
# 5. Start OUR ComfyUI on a free port
# ----------------------------------------------------------------------------
port_in_use() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
choose_port() { local p="$1" i; for i in 0 1 2 3 4; do port_in_use $((p+i)) || { echo $((p+i)); return; }; done; echo $((p+10)); }

comfy_up()  { curl -fsS "http://127.0.0.1:${CHOSEN_PORT}/system_stats" >/dev/null 2>&1; }
wait_comfy() { local n="${1:-150}" i; for ((i=0;i<n;i++)); do comfy_up && return 0; sleep 2; done; return 1; }

# stop a ComfyUI WE previously started (leave the template's alone)
[[ -f "$LOG_DIR/comfyui.pid" ]] && kill "$(cat "$LOG_DIR/comfyui.pid")" 2>/dev/null || true
sleep 1

CHOSEN_PORT="$(choose_port "$COMFY_PORT")"
[[ "$CHOSEN_PORT" != "$COMFY_PORT" ]] && say "Port $COMFY_PORT busy — using $CHOSEN_PORT for our ComfyUI"

EMP_ARG=()
[[ -f "$COMFY_DIR/extra_model_paths.yaml" ]] && EMP_ARG=(--extra-model-paths-config "$COMFY_DIR/extra_model_paths.yaml")

say "Starting our ComfyUI on 127.0.0.1:${CHOSEN_PORT} (log: $LOG_DIR/comfyui.log)"
( cd "$COMFY_DIR" && nohup "$PY" main.py --listen 127.0.0.1 --port "$CHOSEN_PORT" "${EMP_ARG[@]}" \
    >"$LOG_DIR/comfyui.log" 2>&1 & echo $! >"$LOG_DIR/comfyui.pid" )

say "Waiting for ComfyUI to be ready…"
if wait_comfy 180; then echo "   ComfyUI is up on :${CHOSEN_PORT}"; else
  echo "   WARNING: ComfyUI not responding — tail $LOG_DIR/comfyui.log" >&2
fi

# ----------------------------------------------------------------------------
# 6. Start the demo web app, pointed at our ComfyUI
# ----------------------------------------------------------------------------
[[ -f "$LOG_DIR/app.pid" ]] && kill "$(cat "$LOG_DIR/app.pid")" 2>/dev/null || true
pkill -f 'uvicorn app:app' 2>/dev/null || true
sleep 1

say "Starting demo app on 0.0.0.0:${APP_PORT} (log: $LOG_DIR/app.log)"
( cd "$REPO_DIR/server" && \
  COMFY_URL="http://127.0.0.1:${CHOSEN_PORT}" \
  nohup "$PY" -m uvicorn app:app --host 0.0.0.0 --port "$APP_PORT" \
    >"$LOG_DIR/app.log" 2>&1 & echo $! >"$LOG_DIR/app.pid" )
sleep 3

cat <<EOF

============================================================
 Setup complete.

 ComfyUI (our own):   http://127.0.0.1:${CHOSEN_PORT}
 Demo web app:        http://0.0.0.0:${APP_PORT}
 ComfyUI install:     $COMFY_DIR
 Models:              $MODELS_COMFY/models

 Open the demo via the Vast.ai mapped address for port ${APP_PORT}.
 Make sure port ${APP_PORT} is exposed in the instance config.

 Logs:   $LOG_DIR/app.log   $LOG_DIR/comfyui.log
 Stop:   bash $REPO_DIR/stop.sh
============================================================
EOF
