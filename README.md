# ViMedAgent v2 — cùng benchmark, backend SGLang

> Bản sao của `ViMedAgent` (v1, backend Ollama) với **một thay đổi duy nhất ở tầng hạ tầng**: thêm
> provider **SGLang** (https://github.com/sgl-project/sglang) để chạy inference nhanh hơn nhờ
> *continuous batching* + *RadixAttention prefix cache*. Toàn bộ framework, prompt, seed, định dạng kết quả,
> metric **giữ nguyên** — kết quả v2 so sánh trực tiếp được với v1.
>
> Tài liệu benchmark/framework/metric đầy đủ: [`code/README.md`](code/README.md).
> Log setup + kiểm chứng: [`code/experiment/007-sglang-backend-setup/README.md`](code/experiment/007-sglang-backend-setup/README.md).

---

## Cài trên ASUS Ascent GX10 (NVIDIA GB10) — đích triển khai

Code đã nằm sẵn trên máy, ví dụ `/home/vkist/Datnh/ViMedAgentv2`. **Một lệnh cài:**

```bash
cd /home/vkist/Datnh/ViMedAgentv2/code
bash scripts/install_gx10.sh
```

Lệnh này (chạy lại được, không phá gì): kiểm tra aarch64 + docker + nvidia runtime → `pip install`
client (`openai`, `scipy` — runner chỉ gọi HTTP, không cần torch) → `docker pull`
`scitrera/dgx-spark-sglang:0.5.17` (image dựng sẵn cho GB10: CUDA 13 + sgl-kernel cho sm_121) → tải
trước `Qwen/Qwen3-32B-AWQ` (~19GB) vào `~/.cache/huggingface`.

Chạy server và thí nghiệm:

```bash
tmux new -s sglang
bash scripts/serve_gx10.sh                    # Qwen/Qwen3-32B-AWQ, port 30000, ctx 8192, ~27GB, 32 request song song

# cửa sổ khác
python scripts/sglang_smoke_test.py --model sglang-qwen3-32b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model sglang-qwen3-32b
python scripts/compare_results.py --framework zero_shot --dataset pilot_100 --a ollama2-qwen3-32b --b sglang-qwen3-32b
# nếu Pass@1 ~60–62 % và null=0 → backend đúng (cùng model đã đo trên RTX 8000: 61,33 %), chạy full:
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework self_refine --model sglang-qwen3-32b
```

Chỉ có **một** model Qwen3-32B trong `config.py`: `sglang-qwen3-32b` ↔ `Qwen/Qwen3-32B-AWQ` (int4, cùng
hạng 4-bit với `qwen3:32b` Q4_K_M của Ollama nên bảng v2 so thẳng được với bảng v1). Cố ý không thêm FP8/bf16:
chênh lệch giữa các kiểu lượng tử hoá (vài điểm phần trăm) có thể lớn bằng chênh lệch giữa các framework đang đo.

**Llama-3.3-70B** (đã có trong `config.py`: `sglang-llama3.3-70b` ↔ `casperhansen/llama-3.3-70b-instruct-awq`, int4
AWQ 40GB, cùng hạng 4-bit với `llama3.3-70b-ctx4k` của v1). Weights 40GB nên phải nâng `MEM_GB`; mỗi server một model,
dừng server Qwen (Ctrl-C trong tmux `sglang`) trước khi chạy:

```bash
MODEL=casperhansen/llama-3.3-70b-instruct-awq bash scripts/install_gx10.sh            # tải 40GB một lần
MODEL=casperhansen/llama-3.3-70b-instruct-awq MEM_GB=50 bash scripts/serve_gx10.sh    # 40GB weights + ~10GB KV
python scripts/sglang_smoke_test.py --model sglang-llama3.3-70b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model sglang-llama3.3-70b
python scripts/compare_results.py --framework zero_shot --dataset pilot_100 --a ollama-llama3.3-70b --b sglang-llama3.3-70b
```

(Llama không có thinking nên `enable_thinking` bị bỏ qua, không ảnh hưởng. v1 chạy 70B với ctx 4k vì thiếu VRAM; ở đây
để 8192 như mọi model khác.)

Biến môi trường của `serve_gx10.sh` (tuỳ chọn): `IMAGE` (mặc định `scitrera/dgx-spark-sglang:0.5.17`,
dự phòng `0.5.9-t4` là bản forum đã test), `PORT` (30000), `CONTEXT_LEN` (**8192**, giống tag `*-ctx8k` của v1),
`MEM_GB` (**27** mặc định = weights 18GB + KV ~7GB ≈ 29k token, giống lần kiểm chứng; tăng vd `MEM_GB=60` nếu muốn KV lớn hơn), `MEM_FRACTION` (dùng thay `MEM_GB` khi để `MEM_GB=` rỗng), `MAX_RUNNING` (32, tăng cùng `MEM_GB`), `ATTN_BACKEND` (triton),
`HF_CACHE`, `HF_TOKEN` (cần cho `meta-llama/*`).

**Kỳ vọng tốc độ trên GB10** (chưa đo): decode bị giới hạn băng thông 273 GB/s; với weights int4 18GB và
kernel AWQ-Marlin của image GB10, ước ~15 token/s mỗi sequence, 32 sequence song song → vài trăm token/s,
so với 90 token/s đo trên RTX 8000 (không có kernel int4, phải giải nén mỗi bước). Prefill không phải nút thắt.

Lưu ý GX10: không dùng `scripts/x86_cuda/install_sglang.sh` ở đây (wheel PyPI không có mã cho sm_121;
script đó tự dừng khi thấy aarch64). Cần cài native không Docker → công thức forum NVIDIA ở cuối file.

---

### Cài lại từ đầu trên GX10 (dọn sạch trạng thái cũ)

```bash
# 0. dừng server đang chạy
tmux attach -t sglang      # Ctrl-C; nếu container không thoát: docker ps → docker stop <id>
tmux kill-session -t sglang

# 1. dọn dấu vết của các lần cài trước
conda env remove -n sglang -y 2>/dev/null          # env x86 cài nhầm (nếu có)
sudo rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8 ~/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-32B-FP8
sudo chown -R "$(id -u):$(id -g)" ~/.cache/huggingface   # file do container root tạo
cd /home/vkist/Datnh/ViMedAgentv2/code
rm -f scripts/install_sglang.sh scripts/serve_sglang.sh scripts/serve_sglang_docker.sh \
      scripts/sglang_turing_launch.py scripts/sglang_turing_probe.py scripts/sglang_awq_gemm_bench.py
rm -rf results_test/tier*/*/*__sglang-qwen3-32b-fp8__*                 # kết quả FP8 thử nghiệm (nếu có)

# 2. cập nhật code (tarball từ máy hoangnv) rồi cài
cd /home/vkist/Datnh/ViMedAgentv2 && tar xzf ../ViMedAgentv2_update_2026-09-11.tar.gz && cd code
bash scripts/install_gx10.sh                        # client deps + docker pull + tải Qwen3-32B-AWQ (bỏ qua nếu đã có)

# 3. serve (tmux) — kiểm tra dòng đầu: "MEM_GB=27 of ~117000MiB → mem_fraction=0.23"
tmux new -s sglang
bash scripts/serve_gx10.sh                           # chờ "The server is fired up and ready to roll!"; Ctrl-b d để tách
#   nếu dòng đầu báo WARNING không đọc được bộ nhớ:  MEM_GB= MEM_FRACTION=0.23 bash scripts/serve_gx10.sh

# 4. kiểm chứng (~2 phút)
python scripts/sglang_smoke_test.py --model sglang-qwen3-32b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model sglang-qwen3-32b
python scripts/compare_results.py --framework zero_shot --dataset pilot_100 --a ollama2-qwen3-32b --b sglang-qwen3-32b
#   đạt: OK + has_think_tag=False + null=0; Pass@1 ~60–62 % (RTX 8000 cùng model: 61,33 %)

# 5. chạy full qua đêm (tmux riêng, resume được — Ctrl-C ×2 rồi chạy lại cùng lệnh để tiếp)
tmux new -s vimed
cd /home/vkist/Datnh/ViMedAgentv2/code && mkdir -p ../logs
for fw in zero_shot cot cot_sc self_refine voting_ensemble medagents medagents_vi mdagents mdagents_vi mad_debate; do
  python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework $fw --model sglang-qwen3-32b
done 2>&1 | tee -a ../logs/full_run_gx10.log
```

Sáng hôm sau: `tail -50 ../logs/full_run_gx10.log` (bảng `FINAL`/`Overall` từng framework),
`ls results/tier*/*/ | grep sglang`, và `python scripts/compare_results.py --framework <fw> --dataset vm14k_clean --a ollama2-qwen3-32b --b sglang-qwen3-32b`
cho framework nào v1 đã có đủ 3 seed (`self_refine` sắp xong bên máy hoangnv).

Nếu `pip` trong `base` (python 3.14) không cài được `scipy`: `conda create -n vimed python=3.11 -y && conda activate vimed && pip install -r requirements-client.txt`, rồi chạy runner trong env đó.

## Chạy thí nghiệm (mọi máy)

Giống hệt v1, chỉ đổi tên model sang `sglang-*`:

```bash
# toàn bộ 10 framework cho 1 model (tuần tự, resume được, Ctrl-C an toàn)
for fw in zero_shot cot cot_sc self_refine voting_ensemble medagents medagents_vi mdagents mdagents_vi mad_debate; do
  python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework $fw --model sglang-qwen3-32b
done
```

Kết quả vào `results/tier{N}/<framework>/<framework>__<model>__<dataset>__s<seed>.jsonl` — tên model khác v1
(`ollama2-qwen3-32b`) nên không ghi đè file cũ. `--workers` (hoặc `VIMED_NUM_WORKERS`) là số request đồng thời;
mặc định 32 cho provider sglang. `SGLANG_BASE_URL` (mặc định `http://127.0.0.1:30000/v1`) nếu server ở host/port khác.

So sánh 2 backend trên cùng framework/dataset: `python scripts/compare_results.py --framework <fw> --dataset <stem> --a <modelA> --b <modelB>`.

---

## Gemma-3-4B / MedGemma-4B — cài SGLang và chạy local (không gọi API)

Hai model 4B chạy **trên chính máy có GPU**: runner và server SGLang cùng một máy, runner gọi
`http://127.0.0.1:<port>`. Không qua LiteLLM, không qua Tailscale, nên không bị lỗi mạng như khi gọi
server từ xa.

| Key `--model` | HF repo | Vai trò |
|---|---|---|
| `sglang-gemma3-4b` | `google/gemma-3-4b-it` | đối chứng: cùng nền Gemma 3 4B, **không** tinh chỉnh y khoa |
| `sglang-medgemma-4b` | `google/medgemma-4b-it` | model y khoa |

Hiệu số MedGemma − Gemma3 là tác dụng riêng của tinh chỉnh y khoa, **nên phải chạy hai model trong cùng
điều kiện**: cùng máy, cùng độ chính xác (mặc định bf16), cùng `CONTEXT_LEN`. Ba điểm phải khai trong
Methods: xem `code/README.md` mục 3.3.1.

### 0. Yêu cầu

- **GPU ≥ 16 GB VRAM trống** cho mỗi model: 8 GB trọng số bf16 + KV cache. **Ampere trở lên (sm ≥ 80)**
  để chạy bf16 gốc. Trên Turing (sm75, ví dụ Quadro RTX 8000), SGLang ép bf16 → fp16, mà họ Gemma có tiền
  sử tràn số ở fp16. Phải kiểm tra đầu ra (bước 5) trước khi tin kết quả.
  Máy `ailab` (GTX 1650, 4 GB) **không chạy được**.
- **Quyền tải model.** Cả hai repo đều bị khoá. Đăng nhập Hugging Face, bấm đồng ý điều khoản ở
  <https://huggingface.co/google/gemma-3-4b-it> (`gated=manual`, có thể phải chờ duyệt) và
  <https://huggingface.co/google/medgemma-4b-it>, rồi tạo token đọc ở <https://huggingface.co/settings/tokens>.
  Đừng ghi token vào file nào trong repo.
- **Client:** `openai`, `scipy` (`pip install -r code/requirements-client.txt`, hoặc dùng env `medagent`).

### 1. Cài SGLang

```bash
cd ViMedAgent/code

# Máy x86 + GPU NVIDIA: conda env "sglang" (sglang + torch, ~11 GB, ~6 phút)
bash scripts/x86_cuda/install_sglang.sh
python scripts/x86_cuda/sglang_turing_probe.py      # chỉ cần nếu GPU sm < 80

# ASUS GX10 / DGX Spark (GB10): dùng docker image dựng sẵn, KHÔNG dùng install_sglang.sh
bash scripts/install_gx10.sh                          # docker pull + client deps
```

### 2. Tải model (một lần)

```bash
export HF_TOKEN=hf_xxx            # chỉ export trong shell, không ghi vào file
# x86 (env sglang):
conda run -n sglang huggingface-cli download google/gemma-3-4b-it
conda run -n sglang huggingface-cli download google/medgemma-4b-it
# GX10 (tải bằng container, file thuộc user hiện tại):
MODEL=google/gemma-3-4b-it  bash scripts/install_gx10.sh
MODEL=google/medgemma-4b-it bash scripts/install_gx10.sh
```

### 3. Chạy server (trong tmux, server chạy foreground)

Mỗi server chỉ phục vụ **một** model. Dùng cổng **30001** để không đụng server 30000 đang chạy.

```bash
tmux new -s gemma
cd ViMedAgent/code

# x86: MEM_FRACTION là tỉ lệ VRAM của cả card. Chọn sao cho được ~16 GB
#      (card 24 GB → 0.65; card 48 GB → 0.35)
MODEL=google/gemma-3-4b-it PORT=30001 MEM_FRACTION=0.35 MAX_RUNNING=32 CONTEXT_LEN=8192 \
  bash scripts/x86_cuda/serve_sglang.sh

# GX10: MEM_GB là số GB tuyệt đối
MODEL=google/gemma-3-4b-it PORT=30001 MEM_GB=16 CONTEXT_LEN=8192 HF_TOKEN=$HF_TOKEN \
  bash scripts/serve_gx10.sh
```

Chờ dòng `The server is fired up and ready to roll!`, rồi tách khỏi tmux bằng `Ctrl-b d`. Đổi sang
MedGemma: Ctrl-C server rồi chạy lại với `MODEL=google/medgemma-4b-it`.

⚠️ **Trên GX10 khi server 70B (cổng 30000) đang chạy:** 70B đã giữ ~85 GB. Thêm 16 GB nữa vẫn vừa
128 GB nhưng sát. Hai server dùng chung một GPU nên **cả hai cùng chậm đi**. Nên chạy Gemma khi 70B
không có job.

### 4. Trỏ runner vào server local

```bash
cd ViMedAgent/code
export SGLANG_BASE_URL=http://127.0.0.1:30001/v1
```

`model_id` trong `config.py` phải trùng **chính xác** với `MODEL=` của server (SGLang lấy đường dẫn
model làm tên phục vụ). Bước 5 kiểm tra điều này.

### 5. Kiểm tra trước khi chạy

```bash
python scripts/sglang_smoke_test.py --model sglang-gemma3-4b
#   đạt: [1] có google/gemma-3-4b-it, [2] has_think_tag=False, [3] trả 1 ký tự A–D, [4] null=0

python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model sglang-gemma3-4b
#   ghi vào results_test/. Xem Pass@1 và null: null > 0 hoặc đáp án lệch hẳn về một chữ cái
#   → nghi tràn số fp16 (GPU sm<80) hoặc sai template. Dừng lại kiểm tra, đừng chạy full
```

### 6. Chạy full (tmux riêng; resume được)

```bash
tmux new -s run-gemma
cd ViMedAgent/code && export SGLANG_BASE_URL=http://127.0.0.1:30001/v1 && mkdir -p ../logs
for fw in zero_shot cot cot_sc self_refine voting_ensemble medagents medagents_vi mdagents mdagents_vi mad_debate; do
  python -u -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework $fw \
    --model sglang-gemma3-4b --workers 32
done 2>&1 | tee -a ../logs/full_run_gemma3_4b.log
```

Làm lại y hệt cho `sglang-medgemma-4b`, sau khi đổi server sang MedGemma ở bước 3.

- **Ngắt / chạy tiếp:** Ctrl-C hai lần rồi chạy lại đúng lệnh cũ. Runner bỏ qua các câu đã có trong file.
- **Không trộn model:** tên file kết quả chứa key model (`sglang-gemma3-4b`), nhưng runner **không**
  kiểm tra server đang phục vụ model nào. Đổi model trên server thì phải chạy lại bước 5 trước.
- **Kiểm tra kết quả:** dùng skill `run-experiment` mục 4 (đủ n, `api_calls == n_llm_calls`,
  `n_usage_missing = 0`, không trùng id).

### Ghi chú cho Methods

- **Temperature** do code cố định theo framework (0,3; 0,5 cho CoT-SC; 0,0 cho critique). Các tham số
  sampling khác lấy theo `generation_config.json` của model (SGLang `sampling_defaults='model'`).
  Gemma 3 dùng `top_k=64, top_p=0.95`.
- Gemma 3 **không có thinking**. Cờ `enable_thinking=False` mà code gửi bị bỏ qua, không ảnh hưởng.
- Template Gemma 3 **không có vai `system`**: nó gộp system message vào lượt user đầu. Các prompt
  của repo luôn có dạng `[system, user]` hoặc `[user]` nên chạy được.
- bf16, không lượng tử hoá, khác các model 32B/70B (4-bit). Nếu đổi sang 4-bit thì phải đổi **cả hai** model
  4B và sửa `code/README.md` 3.3.1 + bảng "Ba điểm lệch" trong `CLAUDE.md`.

---

## Vì sao đổi backend

| | Ollama (v1) | SGLang (v2) |
|---|---|---|
| Xử lý song song | `OLLAMA_NUM_PARALLEL=4`, mỗi request 1 slot cố định | continuous batching: 32–64+ sequence trong 1 batch |
| Prefix trùng (system prompt, câu hỏi lặp qua nhiều bước MAS) | tính lại mỗi lần | RadixAttention cache KV theo prefix |
| Tắt thinking Qwen3 | cổng native `/api/chat` (`"think": false`) | `chat_template_kwargs.enable_thinking=false` qua API OpenAI-compatible |
| Context | `num_ctx` bake vào tag `*-ctx8k` | `--context-length 8192` (giới hạn mỗi request; KV pool cấp riêng theo `MEM_FRACTION`) |
| Client concurrency | 4 | 32 mặc định, `--workers` để đổi |

## Kết quả kiểm chứng (11/09/2026, Quadro RTX 8000 48GB — Turing sm75)

Chi tiết: [`code/experiment/007-sglang-backend-setup/README.md`](code/experiment/007-sglang-backend-setup/README.md).

| pilot_100, 3 seed | Ollama `qwen3-32b-ctx8k` (Q4_K_M, 4 worker) | SGLang `Qwen3-32B-AWQ` (32 worker) |
|---|---:|---:|
| zero_shot Pass@1 | 60,33 ± 0,58 % | **61,33 ± 1,15 %** (agreement từng câu 87,7 %) |
| zero_shot thời gian / 100 câu | ~7,5 phút | **12 s** (~35×) |
| cot thời gian / 100 câu (seed 42) | ~25 phút | **~9 phút** (~2,7×) — Pass@1 62 % vs 57 % |

Δ Pass@1 nằm trong sai số giữa 2 kiểu lượng tử hoá 4-bit; null rate 0 → backend đúng. Số đo tốc độ là
trường hợp xấu (GPU Turing không được sgl-kernel hỗ trợ, phải đi đường compat).

## Thay đổi so với v1 (diff)

- `infrastructure/config.py`: provider `sglang`, model `sglang-qwen3-32b` (= `Qwen/Qwen3-32B-AWQ`, cùng hạng 4-bit với v1) + vài model khác cùng họ v1, `SGLANG_BASE_URL`,
  `OLLAMA_PROVIDERS` (1 chỗ duy nhất), `NUM_WORKERS_BY_PROVIDER` + `get_num_workers()`, `REQUEST_TIMEOUT`.
- `infrastructure/llm_client.py`: client OpenAI dùng chung theo `base_url` (thread-safe); nhánh `sglang` truyền
  `extra_body.chat_template_kwargs.enable_thinking` theo `think`; nhánh Ollama giữ nguyên.
- `runner/run_experiments.py`: số worker theo provider, thêm `--workers`.
- `requirements-client.txt`: phía client chỉ cần `openai` + `scipy`.
- `scripts/`: **GX10:** `install_gx10.sh`, `serve_gx10.sh`. **Chung:** `sglang_smoke_test.py`, `compare_results.py`.
  **Server x86 (`scripts/x86_cuda/`):** `install_sglang.sh`, `serve_sglang.sh`, `sglang_turing_launch.py`,
  `sglang_turing_probe.py`, `sglang_awq_gemm_bench.py`.
- Không đụng `frameworks/`, `metrics/`, `analysis/`, `data/`, prompt.

---

## Chạy trên máy hiện tại (s96, Quadro RTX 8000 48GB — Turing sm75)

Máy này **đã cài sẵn** từ 11/09: conda env `sglang` (sglang 0.5.9, torch 2.9.1+cu128, ninja, GCC 13) và
model `Qwen/Qwen3-32B-AWQ` (19GB) trong `~/.cache/huggingface`. Không cần cài lại, chỉ chạy:

```bash
cd /home/hoangnv/Agent_medical/ViMedAgentv2/code

# 1. GPU phải trống phần của mình — unload Ollama nếu đang giữ model 32B
OLLAMA_HOST=127.0.0.1:11435 /storage/hoangnv/ollama-local/bin/ollama ps
OLLAMA_HOST=127.0.0.1:11435 /storage/hoangnv/ollama-local/bin/ollama stop qwen3-32b-ctx8k

# 2. serve (tmux) — GPU dùng chung với người khác nên để MEM_FRACTION=0.65
tmux new -s sglang
cd /home/hoangnv/Agent_medical/ViMedAgentv2/code
MEM_FRACTION=0.65 MAX_RUNNING=32 bash scripts/x86_cuda/serve_sglang.sh
#   chờ "The server is fired up and ready to roll!" (~2 phút: load 7s + capture CUDA graph 22s)
#   Ctrl-b d để tách khỏi tmux

# 3. chạy thí nghiệm (env sglang hoặc env agent đều được — chỉ cần openai + scipy)
conda activate sglang
cd /home/hoangnv/Agent_medical/ViMedAgentv2/code
python scripts/sglang_smoke_test.py --model sglang-qwen3-32b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model sglang-qwen3-32b
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework cot --model sglang-qwen3-32b
```

Cài lại từ đầu (nếu đổi máy x86 khác hoặc env hỏng):

```bash
bash scripts/x86_cuda/install_sglang.sh      # conda env "sglang" + pip "sglang[all]" ~11GB, ~6 phút
#   GPU sm<80 thì script tự cài thêm GCC 13 (cần cho kernel JIT)
python scripts/x86_cuda/sglang_turing_probe.py   # kiểm tra kernel: jit_norm/jit_rope/triton awq phải OK
```

**Đặc thù máy này:** wheel `sgl-kernel` không có mã cho sm75 nên `serve_sglang.sh` tự chạy qua
`scripts/x86_cuda/sglang_turing_launch.py` (JIT cho RMSNorm/RoPE, Triton cho AWQ và attention, PyTorch cho sampling).
Cần `CUDA_HOME` trỏ CUDA 12.x (script tự tìm `/usr/local/cuda-12.9`) và g++ ≥ 10 (env có GCC 13). Nếu lỗi kernel,
chạy probe ở trên trước khi sửa gì khác.

**GPU dùng chung:** người khác thường chiếm ~8GB. `MEM_FRACTION=0.65` → SGLang giữ ~31GB (weights 18GB + KV 7GB),
tổng ~39/48GB. Không chạy SGLang và Ollama cùng lúc với model 32B.

**Tốc độ đo được:** zero_shot pilot_100 3 seed = 54 s (Ollama ~22 phút); cot 100 câu = ~9 phút (Ollama ~25 phút);
decode ~90 token/s tổng ở 32 request (AWQ không có kernel int4 fused trên Turing nên mỗi bước phải giải nén weights).

## Phụ lục — GX10 cài native (không Docker), theo thread NVIDIA forum "Run SGLang in Spark"

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.sglang --python 3.12 && source ~/.sglang/bin/activate
export TORCH_CUDA_ARCH_LIST=12.1a TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
uv pip install sgl-kernel --prerelease=allow --index-url https://docs.sglang.ai/whl/cu130/
uv pip install sglang --prerelease=allow
uv pip install --force-reinstall torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130
uv pip install flashinfer-python openai scipy ninja
python scripts/x86_cuda/sglang_turing_launch.py --model-path Qwen/Qwen3-32B-AWQ --host 127.0.0.1 --port 30000 \
    --context-length 8192 --mem-fraction-static 0.23 --max-running-requests 32 --attention-backend triton --trust-remote-code
```
(launcher quyết định theo probe: nếu sgl-kernel build đúng cho sm_121 thì nó không đổi gì.)
