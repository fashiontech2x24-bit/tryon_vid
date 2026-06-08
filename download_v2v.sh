#!/usr/bin/env bash
#
# download_v2v.sh — Models for the "Wan VACE 14B V2V" ComfyUI workflow
#                   (video_wan_vace_14B_v2v.json) — Wan 2.1 VACE 1.3B + 14B.
#
# Target box : vast.ai instance, image  vastai/comfy_v0.22.0-cuda-12.9-py312  (L40S)
# Strategy   : aria2c, 16 parallel connections, resume (-c) + INFINITE retries so a
#              dropped connection never aborts a transfer. Falls back to the
#              HuggingFace CLI (hf_transfer) if aria2c is unavailable.
#
# New download size: ~39 GB. (The VAE, both CausVid 14B usage, and umt5 fp8 may
#                    already be present from earlier flows and will be SKIPPED.
#                    umt5_xxl_fp16 is re-fetched if a prior cleanup removed it.)
#
#   Usage:
#     bash download_v2v.sh
#     COMFYUI_DIR=/workspace/ComfyUI bash download_v2v.sh
#     HF_TOKEN=hf_xxx bash download_v2v.sh
#
set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Locate the LIVE / persistent ComfyUI root
# ----------------------------------------------------------------------------
detect_comfyui() {
  if [[ -n "${COMFYUI_DIR:-}" ]]; then echo "$COMFYUI_DIR"; return; fi
  local pid cwd
  pid="$(pgrep -f 'main.py' | head -n1 || true)"
  if [[ -n "$pid" ]]; then
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ -f "$cwd/main.py" ]]; then echo "$cwd"; return; fi
  fi
  local candidates=(
    "/workspace/ComfyUI"
    "/opt/workspace-internal/ComfyUI"
    "/opt/ComfyUI"
    "$HOME/ComfyUI"
    "/ComfyUI"
  )
  for c in "${candidates[@]}"; do
    [[ -f "$c/main.py" ]] && { echo "$c"; return; }
  done
  local found
  found="$(find / -maxdepth 6 -type f -name main.py -path '*ComfyUI*' 2>/dev/null | head -n1 || true)"
  [[ -n "$found" ]] && { dirname "$found"; return; }
  return 1
}

COMFY="$(detect_comfyui || true)"
if [[ -z "$COMFY" || ! -d "$COMFY" ]]; then
  echo "ERROR: Could not locate the ComfyUI directory." >&2
  echo "       Re-run with:  COMFYUI_DIR=/workspace/ComfyUI bash download_v2v.sh" >&2
  exit 1
fi
echo ">> ComfyUI root: $COMFY"

MODELS="$COMFY/models"
DIR_TEXT="$MODELS/text_encoders"
DIR_VAE="$MODELS/vae"
DIR_DIFF="$MODELS/diffusion_models"   # flat — this workflow uses no subfolder
DIR_LORA="$MODELS/loras"             # flat — this workflow uses no subfolder
mkdir -p "$DIR_TEXT" "$DIR_VAE" "$DIR_DIFF" "$DIR_LORA"

# ----------------------------------------------------------------------------
# 2. Robust downloader
# ----------------------------------------------------------------------------
HAVE_ARIA=0
if command -v aria2c >/dev/null 2>&1; then
  HAVE_ARIA=1
else
  echo ">> aria2c not found — attempting to install it..."
  if command -v apt-get >/dev/null 2>&1; then
    (apt-get update -y && apt-get install -y aria2) >/dev/null 2>&1 \
      && HAVE_ARIA=1 && echo ">> aria2c installed." \
      || echo ">> Could not install aria2c (continuing with HF CLI fallback)."
  fi
fi

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    echo ">> Installing huggingface_hub[cli] + hf_transfer..."
    pip install -q -U "huggingface_hub[cli]" hf_transfer >/dev/null 2>&1 || true
  fi
}
export HF_HUB_ENABLE_HF_TRANSFER=1

aria_get() {
  local url="$1" dir="$2" name="$3"
  local header=()
  [[ -n "${HF_TOKEN:-}" ]] && header=(--header="Authorization: Bearer ${HF_TOKEN}")
  aria2c "${header[@]}" \
    -c -x16 -s16 -k1M \
    --max-tries=0 --retry-wait=5 \
    --connect-timeout=60 --timeout=60 \
    --max-connection-per-server=16 \
    --file-allocation=none \
    --console-log-level=warn --summary-interval=15 \
    --auto-file-renaming=false --allow-overwrite=true \
    -d "$dir" -o "$name" "$url"
}

hf_get() {
  local repo="$1" repo_path="$2" dir="$3" name="$4"
  ensure_hf_cli
  local bin="hf"; command -v hf >/dev/null 2>&1 || bin="huggingface-cli"
  local tokenarg=()
  [[ -n "${HF_TOKEN:-}" ]] && tokenarg=(--token "$HF_TOKEN")
  local tmp="$dir/.hfdl"; mkdir -p "$tmp"
  "$bin" download "$repo" "$repo_path" "${tokenarg[@]}" --local-dir "$tmp" >/dev/null
  mv -f "$tmp/$repo_path" "$dir/$name"
  rm -rf "$tmp"
}

# args: <repo> <repo_path> <dest_dir> <dest_name>
fetch() {
  local repo="$1" repo_path="$2" dir="$3" name="$4"
  local url="https://huggingface.co/${repo}/resolve/main/${repo_path}?download=true"
  local target="$dir/$name"
  if [[ -f "$target" ]]; then echo ">> [skip] already present: ${name}"; return; fi
  echo ">> [get ] ${name}"
  echo "          repo: ${repo}  path: ${repo_path}"
  echo "          ->    ${target}"
  if [[ "$HAVE_ARIA" -eq 1 ]]; then
    if aria_get "$url" "$dir" "$name"; then return; fi
    echo ">> aria2c failed for ${name}, falling back to HF CLI..."
  fi
  hf_get "$repo" "$repo_path" "$dir" "$name"
}

# ----------------------------------------------------------------------------
# 3. Model manifest
# ----------------------------------------------------------------------------
echo
echo "========== Wan VACE 14B V2V — model download =========="
echo

# --- Text encoders (two CLIPLoader nodes: fp8 + fp16) ---
fetch "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
      "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
      "$DIR_TEXT" "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

fetch "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
      "split_files/text_encoders/umt5_xxl_fp16.safetensors" \
      "$DIR_TEXT" "umt5_xxl_fp16.safetensors"

# --- VAE (VAELoader) ---
fetch "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
      "split_files/vae/wan_2.1_vae.safetensors" \
      "$DIR_VAE" "wan_2.1_vae.safetensors"

# --- VACE diffusion models: 1.3B + 14B (two UNETLoader nodes) ---
fetch "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
      "split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors" \
      "$DIR_DIFF" "wan2.1_vace_1.3B_fp16.safetensors"

fetch "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
      "split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors" \
      "$DIR_DIFF" "wan2.1_vace_14B_fp16.safetensors"

# --- CausVid LoRAs: 1.3B bidirect2 (new) + 14B (kept from nihal_kollam) ---
fetch "Kijai/WanVideo_comfy" \
      "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors" \
      "$DIR_LORA" "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"

fetch "Kijai/WanVideo_comfy" \
      "Wan21_CausVid_14B_T2V_lora_rank32.safetensors" \
      "$DIR_LORA" "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

# ----------------------------------------------------------------------------
# 4. Summary
# ----------------------------------------------------------------------------
echo
echo "================ Done. Models for video_wan_vace_14B_v2v ================"
printf '%s\n' \
  "$DIR_TEXT/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "$DIR_TEXT/umt5_xxl_fp16.safetensors" \
  "$DIR_VAE/wan_2.1_vae.safetensors" \
  "$DIR_DIFF/wan2.1_vace_1.3B_fp16.safetensors" \
  "$DIR_DIFF/wan2.1_vace_14B_fp16.safetensors" \
  "$DIR_LORA/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors" \
  "$DIR_LORA/Wan21_CausVid_14B_T2V_lora_rank32.safetensors" \
  | while read -r f; do
      if [[ -f "$f" ]]; then
        printf '  [OK]  %8s  %s\n' "$(du -h "$f" | cut -f1)" "$f"
      else
        printf '  [MISS]          %s\n' "$f"
      fi
    done
echo
echo "Refresh ComfyUI in the browser so it picks up the new models."
echo "Note: 'Load Image' (reference image) and 'Load Video' (motion/control video)"
echo "      nodes still need your own inputs uploaded manually."
