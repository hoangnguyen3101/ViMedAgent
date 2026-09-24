# Experiment 007 — Backend SGLang cho ViMedAgent v2 (setup + kiểm chứng trên pilot_100)

<!-- theo chuẩn experiment/Template/README.md -->

- **Date:** 11/09/2026 (03:10–04:15 JST)
- **Author:** hoangnv (qua Claude Code)
- **Status:** done — setup + kiểm chứng xong; v2 sẵn sàng chạy trên server khác
- **Base model:** `Qwen/Qwen3-32B-AWQ` (HF, int4 AWQ) qua provider mới `sglang`, đối chiếu với `qwen3-32b-ctx8k` (GGUF Q4_K_M) qua `ollama2`
- **Method:** hạ tầng — thêm backend inference, không đụng framework/prompt/metric

## 1. Motivation / Hypothesis

Job `self_refine × ollama2-qwen3-32b` trên full VM14K (11.094 câu × 3 seed) đang chạy ~26 s/câu
(4 worker, ~4,15 call/câu) → mỗi seed ~3,3 ngày. Ollama giới hạn `OLLAMA_NUM_PARALLEL=4`, không có
prefix cache giữa các bước của cùng 1 câu hỏi, nên GPU phần lớn thời gian chờ.

Giả thuyết: SGLang (continuous batching + RadixAttention) với 32+ request đồng thời sẽ nhanh hơn
nhiều lần **mà không đổi kết quả** (cùng prompt, cùng T, cùng cách tắt thinking). Kiểm chứng bằng
`zero_shot × pilot_100 × 3 seed`: Pass@1 phải nằm trong ±2–3 pp so với Ollama (2 kiểu lượng tử hoá
4-bit khác nhau nên không kỳ vọng trùng từng câu).

## 2. Setup

- **Code:** `ViMedAgentv2/` = copy nguyên `ViMedAgent/` + provider `sglang` (xem [`../../../README.md`](../../../README.md) mục "Thay đổi so với v1").
- **Env:** conda `sglang` (python 3.11) — `sglang 0.5.9`, `sgl-kernel 0.3.21`, `torch 2.9.1+cu128`,
  `triton 3.5.1`, `flashinfer-python`, `openai 2.6.1`, `ninja`, + `gxx_linux-64=13` (chỉ cần cho GPU cũ).
  Cài bằng `uv pip install "sglang[all]"` (~11GB, ~6 phút). Driver 550.163 (CUDA 12.4) chạy wheel cu128 bình thường (minor-version compatibility).
- **Model:** `huggingface_hub.snapshot_download("Qwen/Qwen3-32B-AWQ")` → 19GB vào `~/.cache/huggingface` (~10 phút).
- **Hardware:** Quadro RTX 8000 48GB (**Turing, sm75**), dùng chung — user khác chiếm 8,2GB cố định.
- **Server:** `MEM_FRACTION=0.65 MAX_RUNNING=32 bash scripts/x86_cuda/serve_sglang.sh`
  → `--dtype float16 --attention-backend triton --sampling-backend pytorch --context-length 8192`
  qua `scripts/x86_cuda/sglang_turing_launch.py`. Weights 18,26GB, KV cache 29.635 token (7,2GB), CUDA graph bs 1–32.
- **Client:** `python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model sglang-qwen3-32b` (32 worker).
- **Δ vs v1:** chỉ backend + số worker; prompt, seed, T, `think=False`, parse_answer giữ nguyên.

## 3. What I did (log)

1. Dừng job đang chạy trong tmux `qwen32b` (Ctrl-C ×2 — runner v1 bắt Ctrl-C theo từng seed nên
   1 lần chỉ nhảy sang seed 44; lần 2 mới thoát). Seed 43 dừng ở 9859/11094, seed 44 có 23 dòng lẻ
   (resume tự bỏ qua). `ollama stop qwen3-32b-ctx8k` để trả 28GB VRAM.
2. Lần chạy đầu: server load weights + KV cache OK nhưng chết ở CUDA-graph capture:
   `RMSNorm failed with error code no kernel image is available for execution on the device`.
3. Điều tra bằng `cuobjdump --list-elf` trên `sgl_kernel/sm90/common_ops.abi3.so`: chỉ có
   **sm_80/89/90/90a/100a/120a, không có sm_75, không có PTX** → mọi fused op của sgl-kernel
   không chạy được trên Turing. `flash_ops` cũng chỉ sm_80/86/90a. (sgl-kernel bỏ hỗ trợ Turing
   từ bản 0.3.x.)
4. Rà toàn bộ chỗ SGLang gọi `sgl_kernel` trong đường forward của model dense fp16 + AWQ:
   `layernorm` (rmsnorm/fused_add_rmsnorm), `activation` (silu_and_mul), `rotary_embedding`,
   `sampler` (flashinfer backend), `quantization/awq.py` (`awq_dequantize`). Mỗi chỗ đều có
   đường thay thế sẵn trong package: `forward_native` (torch), `sglang.jit_kernel.*` (nvcc JIT
   biên dịch đúng arch của GPU), `awq_triton.py` (Triton thuần), `--sampling-backend pytorch`,
   `--attention-backend triton`. AWQ-Marlin tự bị loại vì `check_marlin_supported` yêu cầu sm80.
5. Viết `scripts/x86_cuda/sglang_turing_probe.py` để test từng kernel. Phát hiện 3 lỗi môi trường liên tiếp
   khi JIT: (a) thiếu `ninja` → `uv pip install ninja`; (b) `nvcc` trên PATH là CUDA **10.1** (`/usr/bin/nvcc`)
   dù `/usr/local/cuda-12.9` có sẵn → `CUDA_HOME=/usr/local/cuda-12.9`; (c) g++ hệ thống 9.4 (Ubuntu 20.04)
   không có `<concepts>` (C++20) → `conda install -c conda-forge gxx_linux-64=13` + `NVCC_PREPEND_FLAGS="-ccbin <conda g++>"`.
6. Sau khi JIT rmsnorm biên dịch được vẫn báo "no kernel image" → nguyên nhân là **lỗi CUDA tồn dư**
   (`cudaErrorNoKernelImageForDevice`, code 209) từ lần gọi sgl_kernel thất bại trước đó chưa được
   `cudaGetLastError()` tiêu thụ, kernel sau đọc nhầm thành lỗi của mình. Thêm `_clear_cuda_error()`
   (ctypes → libcudart) sau mỗi probe. Probe: JIT rmsnorm/fused_add_rmsnorm/rope OK (maxdiff ≤ 4e-3 fp16), Triton AWQ dequant khớp 100% với bản decomposition.
7. Viết `scripts/x86_cuda/sglang_turing_launch.py`: hook `MultiPlatformOp.dispatch_forward` (điểm duy nhất
   mọi fused layer chọn backend) + thay `awq.awq_dequantize` → probe 1 lần trong process scheduler
   rồi chọn `forward_cuda` (JIT) cho RMSNorm/RotaryEmbedding, `forward_native` cho phần còn lại;
   trên sm80+ là no-op. Lỗi lần 1 của launcher: `runpy.run_module(alter_sys=True)` thay
   `sys.modules["__main__"]` → tiến trình con spawn re-import file của sglang thay vì launcher nên
   hook không được cài; sửa `alter_sys=False`.
8. Server lên (capture CUDA graph 22 s). Smoke test: `think=False` không lộ `<think>`, trả lời 1 ký tự, 32 call song song 12,6 s.
9. Chạy `zero_shot × pilot_100 × 3 seed` → 54 s tổng (Ollama trước đây ~7,5 phút/seed). Kết quả mục 4.
10. Chạy thêm `cot × pilot_100` để đo tốc độ decode dài: seed 42 xong sau ~9 phút (Ollama ~25 phút/seed),
    throughput decode ~90 token/s tổng ở 32 request. Dừng ở seed 43 (xoá file lẻ) vì đủ số liệu.
11. Thử fused int4 Triton GEMM (`awq_gemm_triton`, heuristic vLLM ROCm M≤256) để tăng decode:
    benchmark `scripts/x86_cuda/sglang_awq_gemm_bench.py` trên đúng shape Qwen3-32B → **chậm hơn** dequant+cuBLAS
    ở mọi M (0,6–0,8× tại M≤32, 0,1–0,4× khi M lớn). Giữ code dưới env `SGLANG_TURING_AWQ_GEMM_MAX_M`, mặc định tắt.
12. Tắt SGLang, chạy lại đúng lệnh cũ trong tmux `qwen32b` (resume seed 43 từ 9859).

## 4. Results

**Pass@1 pilot_100, 3 seed (42/43/44), cùng prompt/T/think=False:**

| Framework | Ollama `qwen3-32b-ctx8k` (Q4_K_M) | SGLang `Qwen3-32B-AWQ` (int4 AWQ) | Δ | Agreement từng câu |
|---|---:|---:|---:|---:|
| zero_shot | 60,33 ± 0,58 % (60/61/60) | **61,33 ± 1,15 %** (60/62/62) | +1,00 pp | 87,7 % (300 cặp) |
| cot (chỉ seed 42) | 57,0 % (3 seed: 59,33 ± 2,52) | 62,0 % | +2,67 pp vs mean | 73,0 % (100 cặp) |

Null rate 0/300 cả hai bên. Δ nằm trong sai số giữa 2 kiểu lượng tử hoá 4-bit khác nhau → backend
hoạt động đúng (thinking tắt đúng, prompt/parse không đổi). Không kỳ vọng trùng từng câu vì
weights khác (GGUF Q4_K_M vs AWQ int4) và AWQ chạy fp16 trên Turing.

**Tốc độ trên chính GPU này (Turing, đường compat):**

| | Ollama (4 worker) | SGLang (32 worker) | Nhanh hơn |
|---|---:|---:|---:|
| zero_shot, 100 câu (prefill-bound, 1 token) | ~7,5 phút | **12 s** | ~35× |
| cot, 100 câu (decode ≤512 token) | ~25 phút | **~9 phút** | ~2,7× |
| decode throughput tổng | — | ~90 token/s @ 32 seq | |

Decode trên Turing bị giới hạn bởi AWQ không có kernel int4 fused: mỗi step phải dequantize toàn bộ
weight ra fp16 (đọc int4 + ghi/đọc lại fp16 ≈ 440 GB/s ≈ sát băng thông 672 GB/s của RTX 8000) →
~280 ms/step cho 64 layer. **Trên Ampere+ (A100/A6000/H100) SGLang dùng AWQ-Marlin + FlashInfer
+ bf16, không cần launcher compat, kỳ vọng nhanh hơn thêm 5–10× so với số đo ở đây.**

## 5. Error analysis

- Toàn bộ 4 lỗi gặp phải đều là **môi trường/hạ tầng**, không có lỗi nào từ code ViMedAgent:
  sgl-kernel không có sm75; nvcc cũ trên PATH; g++ 9 không có C++20; CUDA last-error tồn dư.
- Rủi ro còn lại của đường compat: chưa probe `jit_kernel.pos_enc` (chỉ dùng khi head_size ∉ {64,128,256,512}) → launcher ép `forward_native` cho trường hợp đó.
- `pkill -f <pattern>` tự giết shell đang chạy nếu pattern xuất hiện trong chính command line (gặp 2 lần) → dùng anchor `^sglang::`.

## 6. Conclusion

Setup thành công. v2 chạy đúng (Pass@1 zero_shot lệch +1,0 pp, trong sai số lượng tử hoá) và nhanh
hơn rõ ràng ngay cả trên GPU Turing không được sgl-kernel hỗ trợ chính thức. Trên server có GPU
Ampere+ chỉ cần `install_sglang.sh` → `serve_sglang.sh` → chạy runner với `--model sglang-qwen3-32b`.

## 7. Next step

- **Đích triển khai đã chốt: ASUS Ascent GX10 (GB10, sm121, aarch64, 128GB unified).** Đường đi: Docker `scitrera/dgx-spark-sglang` qua `scripts/install_gx10.sh` + `scripts/serve_gx10.sh`, **giữ nguyên model `Qwen/Qwen3-32B-AWQ` và footprint ~27GB (`MEM_GB=27`)** để so thẳng với số đo ở đây và với bảng v1 (cùng hạng 4-bit). Không thêm FP8/bf16 theo quyết định của hoangnv 11/09. Launcher compat đã đổi sang quyết định theo probe.

- Chạy full ma trận 10 framework × `sglang-qwen3-32b` × 3 seed trên server mới (`for fw in ...` trong README gốc), tên model khác nên không ghi đè kết quả `ollama2-*`.
- Nếu muốn so sánh chặt với bảng v1 (cùng quantization): chạy lại các framework đã có trên cùng backend, (đã quyết định không thêm biến thể FP8/bf16 vào config để bảng không bị lẫn kiểu lượng tử hoá).
- Nếu phải chạy lâu trên Turing: thử `--quantization gptq` với model GPTQ-Int4 (kernel gptq riêng của sgl_kernel cũng không có sm75 → vẫn cần đường Triton), hoặc bnb 4-bit; không kỳ vọng vượt dequant+cuBLAS hiện tại.

## 8. Artifacts & links

- Code: [`infrastructure/config.py`](../../infrastructure/config.py), [`infrastructure/llm_client.py`](../../infrastructure/llm_client.py), [`runner/run_experiments.py`](../../runner/run_experiments.py)
- Scripts: [`scripts/x86_cuda/install_sglang.sh`](../../scripts/x86_cuda/install_sglang.sh), [`scripts/x86_cuda/serve_sglang.sh`](../../scripts/x86_cuda/serve_sglang.sh), [`scripts/x86_cuda/sglang_turing_launch.py`](../../scripts/x86_cuda/sglang_turing_launch.py), [`scripts/x86_cuda/sglang_turing_probe.py`](../../scripts/x86_cuda/sglang_turing_probe.py), [`scripts/x86_cuda/sglang_awq_gemm_bench.py`](../../scripts/x86_cuda/sglang_awq_gemm_bench.py), [`scripts/sglang_smoke_test.py`](../../scripts/sglang_smoke_test.py), [`scripts/compare_results.py`](../../scripts/compare_results.py)
- Kết quả: `results_test/tier1/zero_shot/zero_shot__sglang-qwen3-32b__pilot_100__s{42,43,44}.jsonl`, `results_test/tier1/cot/cot__sglang-qwen3-32b__pilot_100__s42.jsonl`
- Logs: [`../../../logs/`](../../../logs/) (`pip_install.log`, `hf_download.log`, `conda_gxx.log`, `sglang_server.log`)
- Related: exp `004-ollama2-instance-setup` (provider ollama2, bug think=False)
