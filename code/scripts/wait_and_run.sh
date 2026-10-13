#!/usr/bin/env bash
# Cho SGLang san sang roi chay runner. Dung sau khi reboot hoac khi doi model:
#   tmux new -s run
#   bash scripts/wait_and_run.sh mad_debate sglang-qwen3-32b 32
#
# Doi so (deu co mac dinh):
#   $1 framework   mac dinh mad_debate
#   $2 model       mac dinh sglang-qwen3-32b
#   $3 workers     mac dinh 32
# Bien moi truong:
#   PORT           mac dinh 30000
#   DATASET        mac dinh data/vm14k_clean.jsonl
#   MAX_WAIT       so giay toi da cho server, mac dinh 1800
#   VIMED_PYTHON   interpreter; mac dinh conda env "agent"
#
# LUU Y: /usr/bin/python tren may nay la Python 2.7 — KHONG BAO GIO goi "python"
# tran. Sau reboot, tmux moi khong co conda nen loi nay tai dien ngay.
set -uo pipefail

FW="${1:-mad_debate}"
MODEL="${2:-sglang-qwen3-32b}"
WORKERS="${3:-32}"
PORT="${PORT:-30000}"
DATASET="${DATASET:-data/vm14k_clean.jsonl}"
MAX_WAIT="${MAX_WAIT:-1800}"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Chon interpreter: bien moi truong > conda env "agent" > python3. Khong dung "python".
PY="${VIMED_PYTHON:-}"
[ -z "$PY" ] && [ -x "$HOME/miniconda3/envs/agent/bin/python" ] && PY="$HOME/miniconda3/envs/agent/bin/python"
[ -z "$PY" ] && PY="$(command -v python3 || true)"
[ -x "$PY" ] || { echo "[run] khong tim thay Python 3" >&2; exit 1; }
ver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
case "$ver" in 3.*) ;; *) echo "[run] $PY la Python $ver — can Python 3" >&2; exit 1 ;; esac
echo "[run] python = $PY ($ver)"

URL="http://127.0.0.1:$PORT/v1"
echo "[wait] cho $URL san sang (toi da ${MAX_WAIT}s)..."
t0=$(date +%s)
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$URL/models" 2>/dev/null)" = "200" ]; do
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$MAX_WAIT" ]; then
    echo "[wait] HET GIO sau ${el}s — server khong len. Kiem tmux sglang." >&2
    exit 1
  fi
  printf "\r[wait] %ds..." "$el"
  sleep 10
done
echo
echo "[wait] server san sang sau $(( $(date +%s) - t0 ))s"
curl -s -m 5 "$URL/models" | head -c 200; echo; echo

export SGLANG_BASE_URL="$URL"
# Python ghi stdout ra pipe (| tee) theo KHOI 8KB, khong theo dong. Job cham
# (cot ~5 cau/phut) thi dong tien do nam trong bo dem rat lau, va mat han neu
# tien trinh bi giet — 12/10/2026 log cot trong tron 40 phut vi ly do nay.
export PYTHONUNBUFFERED=1
# 900s thay vi 300: o ~2,75 tok/s moi request, MAX_TOKENS_AGENT=600 mat ~218s.
# Bien 300s qua hep — mot job GPU khac xuat hien la timeout hang loat ngay.
export VIMED_REQUEST_TIMEOUT="${VIMED_REQUEST_TIMEOUT:-900}"

echo "[run] $FW x $MODEL | workers=$WORKERS | timeout=${VIMED_REQUEST_TIMEOUT}s"
echo "[run] resume tu dong — bo qua cac id da xong"
exec "$PY" -m runner.run_experiments \
  --dataset "$DATASET" --framework "$FW" --model "$MODEL" --workers "$WORKERS"
