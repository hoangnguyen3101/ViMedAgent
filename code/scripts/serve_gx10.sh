#!/usr/bin/env bash
# ViMedAgent v2 — GX10 / DGX Spark: run the SGLang server inside the pre-built GB10 container.
# Written for NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, sm121, aarch64), where
# pip wheels for x86 do not apply and the community image already carries a
# CUDA 13 / sgl-kernel build for GB10:  https://hub.docker.com/r/scitrera/dgx-spark-sglang
#
# Usage (inside tmux; stays in the foreground):
#   bash scripts/serve_gx10.sh          # Qwen/Qwen3-32B-AWQ on :30000, ~27GB (18GB weights + ~7GB KV)
#
# Env knobs (same names as serve_sglang.sh where they overlap):
#   IMAGE          default scitrera/dgx-spark-sglang:0.5.17  (0.5.9-t4 = older, forum-tested)
#   MODEL          default Qwen/Qwen3-32B-AWQ   (must match model_id in config.py: sglang-qwen3-32b)
#   PORT           default 30000   (host network → SGLANG_BASE_URL=http://127.0.0.1:$PORT/v1)
#   CONTEXT_LEN    default 8192
#   MEM_GB         default 27 — absolute cap in GB for weights + KV cache (same footprint as
#                  the RTX 8000 validation run: 18GB weights + ~7GB KV ≈ 29k tokens). Raise
#                  (e.g. MEM_GB=60) for a bigger KV pool / more concurrent requests.
#   MEM_FRACTION   alternative to MEM_GB: fraction of the unified pool (unset MEM_GB to use it)
#   MAX_RUNNING    default 32 (matches the 27GB KV pool; raise together with MEM_GB)
#   ATTN_BACKEND   default triton  (recommended on GB10 images; flashinfer on newer tags)
#   HF_CACHE       default ~/.cache/huggingface (mounted into the container)
#   HF_TOKEN       forwarded if set (needed for meta-llama/*)
#
# The server runs as root *inside* the container (image default; its caches live in
# /root), so host tools show the sglang::scheduler process as root. Only the mounted
# HF cache is shared with the host; files the container creates there are root-owned —
# install_gx10.sh downloads the model as your user to avoid that; if needed:
#   sudo chown -R "$(id -u):$(id -g)" ~/.cache/huggingface
#   EXTRA_ARGS     extra flags for sglang.launch_server
set -euo pipefail

IMAGE="${IMAGE:-scitrera/dgx-spark-sglang:0.5.17}"
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-30000}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"
MEM_GB="${MEM_GB-27}"      # MEM_GB= (empty) → fall back to MEM_FRACTION
# MEM_GB → cap SGLang (weights + KV cache) at that many GB regardless of pool size;
# converted to --mem-fraction-static using the total nvidia-smi reports.
if [ -n "${MEM_GB:-}" ]; then
  TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if ! [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    # GB10 unified memory: nvidia-smi prints [N/A]. Ask torch inside the image what
    # SGLang itself will see as total device memory (~5-10 s container start).
    # --privileged giống hệt lệnh serve bên dưới: thiếu nó thì trên DGX OS container
    # không thấy GPU và bước dò này im lặng thất bại.
    TOTAL_MIB="$(docker run --rm --gpus all --privileged "$IMAGE" python3 -c \
      'import torch; print(torch.cuda.mem_get_info()[1] // 2**20)' 2>/dev/null | tail -1 | tr -d ' ')"
  fi
  if ! [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    # last resort: system RAM (unified pool), slightly over-estimates → a bit less KV
    TOTAL_MIB="$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null)"
  fi
  if [ -n "$TOTAL_MIB" ] && [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    MEM_FRACTION="$(awk -v gb="$MEM_GB" -v tot="$TOTAL_MIB" 'BEGIN{f=gb*1024/tot; if(f>0.95)f=0.95; printf "%.3f", f}')"
    echo "[serve-docker] MEM_GB=$MEM_GB of ${TOTAL_MIB}MiB → mem_fraction=$MEM_FRACTION"
  else
    echo "[serve-docker] WARNING: could not read total GPU memory, ignoring MEM_GB (using MEM_FRACTION=$MEM_FRACTION)"
  fi
fi
MAX_RUNNING="${MAX_RUNNING:-32}"
ATTN_BACKEND="${ATTN_BACKEND:-triton}"
# DRY_RUN=1 → in ra tham số sẽ dùng rồi thoát, không khởi động server.
# Dùng để kiểm MEM_GB có ăn không mà không phải chờ vài phút load model.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$HF_CACHE"
echo "[serve-docker] image=$IMAGE  model=$MODEL  port=$PORT  attn=$ATTN_BACKEND"
echo "[serve-docker] ctx=$CONTEXT_LEN  mem_fraction=$MEM_FRACTION  max_running=$MAX_RUNNING"
echo "[serve-docker] clients: export SGLANG_BASE_URL=http://127.0.0.1:$PORT/v1"
if [ -n "${DRY_RUN:-}" ]; then
  if [ -n "${TOTAL_MIB:-}" ] && [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    echo "[serve-docker] DRY_RUN: tổng bộ nhớ dò được ${TOTAL_MIB}MiB → sẽ cấp $(awk -v f="$MEM_FRACTION" -v t="$TOTAL_MIB" 'BEGIN{printf "%.1f", f*t/1024}')GB"
  fi
  echo "[serve-docker] DRY_RUN: thoát, KHÔNG khởi động server."
  exit 0
fi

exec docker run --rm -it \
  --gpus all --privileged \
  --network host --ipc=host \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --context-length "$CONTEXT_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    --attention-backend "$ATTN_BACKEND" \
    --trust-remote-code \
    $EXTRA_ARGS
