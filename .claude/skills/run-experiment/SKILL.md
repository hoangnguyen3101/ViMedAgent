---
name: run-experiment
description: Quy trình chạy một thí nghiệm ViMedAgentv2 từ đầu đến cuối — dựng server SGLang đúng ngân sách bộ nhớ cho từng model, khởi chạy runner, theo dõi, và kiểm tính toàn vẹn kết quả. Dùng khi cần chạy/chạy tiếp bất kỳ framework nào, đổi model đang phục vụ, hoặc xác minh một file kết quả đã đủ và đúng schema chưa.
---

# Chạy thí nghiệm ViMedAgentv2

Đọc `CLAUDE.md` trước — nhất là lệnh cấm git và quy tắc đóng băng schema.

## 0. Trước khi làm gì: xem GPU còn bao nhiêu

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read l; do
  pid=$(echo $l|cut -d, -f1); echo "$l  owner=$(ps -o user= -p $pid 2>/dev/null)"; done
```

**GPU dùng chung với người khác.** Tiến trình của user khác (ví dụ `trangmx`) **không được đụng vào**,
kể cả khi nó chiếm nhiều chỗ. Chỉ được dừng tiến trình của `hoangnv`, và chỉ khi cần.

Dừng model Ollama đang giữ VRAM: `OLLAMA_HOST=127.0.0.1:11435 ollama stop <tag>` — **không dùng**
`pkill -f <pattern>` với pattern có trong chính lệnh đang chạy (tự giết shell); nếu buộc phải, dùng
anchor `^sglang::`.

## 1. Dựng server

Chạy trong tmux riêng, server giữ foreground.

```bash
tmux new -s sglang
MODEL=<hf repo id> PORT=30000 MEM_FRACTION=<xem bảng> MAX_RUNNING=32 CONTEXT_LEN=8192 \
  bash scripts/x86_cuda/serve_sglang.sh        # máy s96 (Quadro RTX 8000, Turing sm75)
# hoặc, trên ASUS GX10 (GB10 sm121, 128GB unified):
MODEL=<hf repo id> MEM_GB=<xem bảng> bash scripts/serve_gx10.sh
```

Ngân sách **đã đo thật**, đừng đoán lại:

| Model | Trọng số | s96 (48 GB, dùng chung) | GX10 (128 GB) |
|---|---|---|---|
| `Qwen/Qwen3-32B-AWQ` | 19 GB | `MEM_FRACTION=0.65` → ~28,7 GB | `MEM_GB=27` |
| `casperhansen/llama-3.3-70b-instruct-awq` | 39,8 GB | **KHÔNG VỪA** — xem dưới | `MEM_GB=60` |
| `google/medgemma-4b-it` | 8,0 GB | cần ≳16 GB trống, port 30001 | `MEM_GB=16` |

**70B không chạy được trên s96** khi có tiến trình khác: 49.152 − 8.186 (người khác) = 40.966 MiB,
mà trọng số đã 40.755 MiB → KV cache còn **âm**. Đã thử, SGLang nạp xong trọng số rồi chết với
`RuntimeError: Not enough memory`. Chạy 70B trên GX10.

Chờ dòng `The server is fired up and ready to roll!` rồi mới sang bước sau. Nếu thấy
`Casting torch.bfloat16 to torch.float16` trên sm<80 (họ Gemma) thì **phải kiểm tra chất lượng đầu ra**
trước khi tin kết quả — Gemma có tiền sử tràn số ở fp16.

## 2. Chạy runner

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000/v1
tmux new -s run
python -m runner.run_experiments \
  --dataset data/vm14k_clean.jsonl \
  --framework <fw> --model <key trong MODELS> --workers 32
```

- **Resume tự động**: kết quả append vào JSONL, `load_done_ids()` bỏ qua id đã xong. Ngắt rồi chạy lại
  đúng lệnh cũ là đi tiếp. `--no-resume` mới chạy lại từ đầu.
- `--workers`: SGLang 32, Ollama 4 (trần `OLLAMA_NUM_PARALLEL`). Bỏ trống thì `get_num_workers()` tự chọn.
- Cột `calls=` trong dòng tiến độ là **tích luỹ trong phiên này**, không phải tổng toàn file.

Đối chiếu chi phí mong đợi (nếu lệch nhiều thì có gì đó sai):

| Framework | lời gọi/câu |
|---|---|
| `zero_shot` · `cot` | 1 |
| `voting_ensemble` | 3 |
| `self_refine` | 2–5 |
| `cot_sc` | 5 |
| `mdagents` | 2 / 4 / 7 |
| `mdagents_vi` | 1 / 3 / 6 — **TB 2,69** trên vm14k_clean |
| `medagents_vi` | 8 |
| `medagents` | 9 |
| `mad_debate` | ≤ 9 |

## 3. Theo dõi

Đừng poll bằng `sleep` nối tiếp. Dùng `Monitor` với filter bắt **cả lỗi lẫn tiến độ** — grep chỉ
"thành công" thì một tiến trình crash sẽ im lặng y hệt lúc đang chạy:

```
tail -f <log> | grep -E --line-buffered "\[[0-9]+/|Traceback|ERROR|out of memory|Killed"
```

## 4. Kiểm kết quả — làm mỗi lần chạy xong

```bash
F=results/tier3/<fw>/<fw>__<model>__vm14k_clean__s42.jsonl
python - "$F" <<'PY'
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8") if l.strip()]
print("n =", len(rows), "(kỳ vọng 11094)")
print("api_calls == n_llm_calls :", all(r.get("api_calls")==r.get("n_llm_calls") for r in rows))
print("thiếu usage              :", sum(r.get("n_usage_missing",0) for r in rows))
print("answer None              :", sum(1 for r in rows if not r.get("answer")))
print("trùng id                 :", len(rows)-len({r["id"] for r in rows}))
print("Pass@1                   : %.2f%%" % (100*sum(bool(r["correct"]) for r in rows)/len(rows)))
PY
ls -la "$F.meta.json"
```

Bốn thứ phải đúng: `n` đủ · `api_calls == n_llm_calls` · `n_usage_missing = 0` · không trùng id.
`n_usage_missing > 0` nghĩa là token chỉ là cận dưới → **không được đưa vào bảng chi phí**.

## 5. Sau đó

Bảng và hình sinh ở `../Table_and_fig` (có `CLAUDE.md` và skill `paper-eval-figures` riêng).
`build_matrix.py` là file duy nhất đọc JSONL; mọi script bảng/hình chỉ đọc CSV.

## Những lỗi đã mắc, đừng mắc lại

- **Đổi model đang phục vụ khi có job chạy dở** → job hỏng giữa chừng. Kiểm `ps -eo cmd | grep run_experiments` trước.
- **`runpy.run_module(alter_sys=True)`** thay `sys.modules["__main__"]` làm scheduler con import nhầm file → dùng `alter_sys=False`.
- **Đặt tên module là `_io.py`** → trùng built-in của Python.
- **Sửa schema giữa chừng** → dữ liệu trước và sau không so được. Sửa trước khi chạy.
- **Tin `api_calls` là chi phí** → sai, xem README mục "Bảng tổng hợp chi phí API".
