# Performance: attention-backend speedups (RTX 6000 Pro / Blackwell)

> Parked for later. The sampler is already lean (CausVid LoRA, 4 steps, cfg 1) —
> we do **not** want to touch sampling params, since that trades quality. The
> levers below are **visually lossless**: they change *how* attention/matmuls are
> computed, not the number of steps or the guidance.

## Hardware
- 1× **RTX 6000 Pro (Blackwell, sm_120)**, 96 GB, 5th-gen tensor cores with
  native fp8/fp4.
- Two ComfyUI instances (A=right, B=left) run **in parallel on the one GPU**, so
  anything that also lowers VRAM gives the two instances more headroom.
- Current launch (`setup.sh` → `start_comfy`) passes **no attention flag** →
  ComfyUI uses PyTorch default SDPA.

## Levers, best first

### 1. SageAttention 2.2 — primary ⭐
Quantizes Q/K to INT8/FP8 inside the attention kernel. Sage 2.2 has an fp8 path
tuned for **sm_120**. Visually lossless on diffusion (an attention approximation,
not a step/cfg cut).
- **Gain**: ~1.5–2× on the attention portion; typically **~20–40% end-to-end** on
  Wan DiT video.
- **Also cuts attention VRAM** → more headroom for the two parallel instances.
- **How**: install the `sageattention` sm_120 wheel, then add
  `--use-sage-attention` to **both** `start_comfy` calls in `setup.sh`.
- **Unverified**: the sm_120 wheel build/load must be checked on the actual
  6000 Pro (dev box here is a 6 GB laptop 3050). ComfyUI logs the chosen
  attention backend at startup — confirm there.

### 2. FlashAttention — fallback only
`--use-flash-attention`. FA2/FA3 Blackwell support is build-dependent and
currently flakier than Sage on sm_120. SDPA (current default) already uses a
flash backend, so this is a smaller delta than Sage. Use only if Sage won't build.

### 3. torch.compile — biggest *non*-attention lever (optional)
Compiles the DiT graph (kernel fusion + Blackwell-tuned codegen). Quality-neutral.
**+10–30%**, stacks on top of Sage. Needs a compile-model node in the workflow (or
ComfyUI `--fast` paths) and pays a first-run warm-up cost. More involved than a
launch flag.

## Recommended first step
Wire `--use-sage-attention` into both instances in `setup.sh` (+ install the
sm_120 `sageattention` wheel in the deps step). Highest gain-per-effort,
Blackwell-native, lossless, frees VRAM. Verify the backend line in
`comfy_A.log` / `comfy_B.log` on the 6000 Pro.

## Explicitly NOT doing (would degrade quality)
- Reducing steps below 4 / changing CausVid LoRA strength.
- Changing cfg, sampler, or scheduler.
- fp8 *weights* — possible later if VRAM ever gets tight, but it's a precision
  change, kept out of the "lossless speedup" bucket for now.
