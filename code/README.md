# ViMedAgent — Benchmark Multi-Agent cho Câu hỏi Y khoa Tiếng Việt (VM14K)

> **v2 (SGLang backend):** đây là bản copy của ViMedAgent với provider `sglang`. Cài đặt/chạy server/so sánh backend: xem [`../README.md`](../README.md); log setup: [`experiment/007-sglang-backend-setup/`](experiment/007-sglang-backend-setup/README.md). Phần dưới đây giữ nguyên từ v1.

> Tài liệu song ngữ Việt–Anh. Diễn giải bằng tiếng Việt, giữ nguyên thuật ngữ kỹ thuật/metrics tiếng Anh (Pass@1, NMAR, ICR, CG, CLD...).

---

## 1. Giới thiệu (Introduction)

**ViMedAgent** là một bộ khung (framework) thực nghiệm để **so sánh các phương pháp suy luận LLM và hệ đa tác tử (Multi-Agent Systems)** trên bộ dữ liệu câu hỏi y khoa **tiếng Việt VM14K** — dạng trắc nghiệm 4 lựa chọn (MCQ A/B/C/D).

Câu hỏi nghiên cứu trọng tâm:

> **Các hệ đa tác tử (multi-agent) có thực sự tốt hơn LLM đơn (single-LLM) trên tiếng Việt không, hay chúng khuếch đại các kiểu lỗi hợp tác (collaboration failures) sang một ngôn ngữ ít tài nguyên?**

Để trả lời, dự án chạy **10 framework** thuộc **3 tầng (tier)** trên nhiều model, mỗi cấu hình lặp lại **3 seed**, rồi đo không chỉ độ chính xác (**Pass@1**) mà còn các **metric lỗi hợp tác** (**NMAR**, **ICR**) và so sánh với **giá trị tham chiếu tiếng Anh (EN reference)** để tính **Cross-Lingual Delta (CLD)**.

Điểm nhấn "Cross-Lingual": mỗi framework đa tác tử gốc (`medagents`, `mdagents`) đều có một **biến thể `_vi`** (`medagents_vi`, `mdagents_vi`) — đây là **đóng góp của paper**: thay các bước phải gọi LLM để tự "đoán" (chọn chuyên gia / phân loại độ khó) bằng cách **dùng trực tiếp nhãn có sẵn trong VM14K** (`medical_topic`, `difficulty_level`), vừa tiết kiệm 1 API call vừa ổn định hơn.

**Định dạng một mẫu dữ liệu (record) VM14K:**

```jsonc
{
  "id": "...",                         // ID câu hỏi (dùng cho resume & ghép cặp metric)
  "question": "...",                   // Nội dung câu hỏi tiếng Việt
  "options": ["...", "...", "...", "..."], // 4 lựa chọn (index 0→3 tương ứng A→D)
  "answer": "B",                       // Đáp án đúng (gold)
  "difficulty_level": "Medium",        // Easy | Medium | Hard (Challenging gộp vào Hard)
  "medical_topic": ["Cardiology", ...] // Danh sách chuyên khoa (có thể là list hoặc string)
}
```

---

## 2. Cấu trúc thư mục (Repository layout)

```
code/
├── infrastructure/
│   ├── config.py          # Cấu hình trung tâm: MODELS, SEEDS, tiers, EXPERIMENT_MATRIX, EN_REFERENCE
│   └── llm_client.py      # call_llm() (retry/backoff) + parse_answer() (trích A/B/C/D)
│
├── frameworks/
│   ├── utils.py           # majority_vote(), extract_reasoning(), format_options()
│   ├── tier1/             # Single-LLM: zero_shot, cot, cot_sc, self_refine
│   ├── tier2/             # Ensemble: voting_ensemble
│   └── tier3/             # Multi-Agent: medagents(_vi), mdagents(_vi), mad_debate
│
├── metrics/
│   └── compute.py         # Pass@1, Pass@k, mean±std, CG, NMAR, ICR, CLD, Fisher, McNemar
│
├── runner/
│   └── run_experiments.py # Driver CLI: chạy (framework × model × seed) → ghi JSONL
│
├── analysis/
│   └── analyze_results.py # Đọc JSONL → in Bảng I–VII của paper
│
├── data/
│   └── prepare_data.py    # Lọc VM14K thô → vm14k_clean.jsonl + pilot_100.jsonl
│
├── results/               # Kết quả chạy chính  (*.jsonl)
│   ├── tier1/{zero_shot,cot,cot_sc,self_refine}/
│   ├── tier2/voting_ensemble/
│   └── tier3/{medagents,medagents_vi,mdagents,mdagents_vi,mad_debate}/
└── results_test/          # Kết quả chạy thử   (*.jsonl)
    ├── tier1/{zero_shot,cot,cot_sc,self_refine}/
    ├── tier2/voting_ensemble/
    └── tier3/{medagents,medagents_vi,mdagents,mdagents_vi,mad_debate}/
```

**Pattern tên file kết quả** (do `run_experiments.py` sinh ra), nằm trong `tier{N}/<framework>/`:

```
{framework}__{model}__{dataset_stem}__s{seed}.jsonl
# ví dụ: results_test/tier1/zero_shot/zero_shot__ollama-llama3_1-8b__pilot_100__s42.jsonl
# Lưu ý: dấu "/" và "." trong tên model được thay bằng "_"
# Tier của mỗi framework được tra từ FRAMEWORK_TIER trong infrastructure/config.py
```

---

## 3. Cài đặt & Cấu hình (Setup & Configuration)

### 3.1. Thư viện

```bash
pip install openai scipy pandas openpyxl
```

- `openai` — SDK gọi API (dùng chung cho Ollama và SGLang vì đều tương thích chuẩn OpenAI).
- `scipy` — kiểm định thống kê (`fisher_exact`, `binomtest`).
- `pandas`, `openpyxl` — đọc/ghi `Metric.xlsx`.

### 3.2. Nhà cung cấp model (providers)

Định nghĩa trong [infrastructure/config.py](infrastructure/config.py). **Tất cả đều chạy local** — không có provider cloud nào.

| Provider | Base URL | API key | Vai trò |
|----------|----------|---------|---------|
| **ollama** | `http://localhost:11434/v1` | không cần | Chạy v1; gọi qua `/api/chat` native khi cần `think=False` |
| **ollama2** | `http://127.0.0.1:11435/v1` | không cần | Instance riêng của hoangnv (`/storage/hoangnv/ollama-local`) |
| **sglang** | `$SGLANG_BASE_URL` (mặc định `http://127.0.0.1:30000/v1`) | không cần | Backend của v2 — continuous batching + RadixAttention |

> **Provider `groq` đã bị gỡ bỏ ngày 26/09/2026.** Không file kết quả nào trong repo chạy bằng Groq,
> và 6 khoá Groq từng bị hardcode trong `config.py` phải coi là **đã lộ**. Nếu cần dùng lại cloud,
> thêm provider mới và **chỉ đặt khoá qua biến môi trường** — không bao giờ ghi vào file.

### 3.3. Danh sách model khả dụng (`MODEL_NAMES`)

Chỉ những key **không bị comment** trong dict `MODELS` mới hợp lệ với tham số `--model`:

| Tên (`--model`) | Provider | model_id |
|-----------------|----------|----------|
| `ollama-llama3.1-8b`     | ollama | `llama3.1:8b` |
| `ollama-qwen3-14b`       | ollama | `qwen3:14b` |
| `ollama-qwen3-32b`       | ollama | `qwen3:32b` |
| `ollama-deepseek-r1-14b` | ollama | `deepseek-r1:14b` |
| `sglang-qwen3-32b`       | sglang | `Qwen/Qwen3-32B-AWQ` |
| `sglang-qwen3-14b`       | sglang | `Qwen/Qwen3-14B` |
| `sglang-llama3.1-8b`     | sglang | `meta-llama/Llama-3.1-8B-Instruct` |
| `sglang-llama3.3-70b`    | sglang | `casperhansen/llama-3.3-70b-instruct-awq` |
| `sglang-deepseek-r1-14b` | sglang | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| `sglang-medgemma-4b`     | sglang | `google/medgemma-4b-it` — xem 3.3.1 |
| `sglang-gemma3-4b-awq`   | sglang | `gaunernst/gemma-3-4b-it-int4-awq` — **bản 4-bit chạy được trên s96** (W4A32), xem 3.3.1 |
| `sglang-gemma3-4b-w4a16` | sglang | `RedHatAI/gemma-3-4b-it-quantized.w4a16` — chỉ chạy trên GPU ≥ sm80 (GX10), **chặn cứng trên s96** |
| `sglang-medgemma-4b-w4a16` | sglang | tự lượng tử hoá bằng `scripts/quantize_w4a16.py`, cùng recipe RedHatAI |
| `sglang-gemma3-4b`       | sglang | `google/gemma-3-4b-it` bf16 — chỉ để đo mức sụt do lượng tử hoá |
| `litellm-llama3.3-70b-awq` | litellm | model group `Llama-3.3-70B` trên GX10 (AWQ) |
| `litellm-qwen2.5-72b`    | litellm | model group `Qwen2.5-72B` trên GX10 (GPTQ-Int4) |

#### 3.3.1. Gemma 3 / MedGemma-4B

`sglang-medgemma-4b` và đối chứng `sglang-gemma3-4b*` khác mọi model còn lại ở ba điểm
**bắt buộc khai trong Methods**:

1. **Kiểu lượng tử hoá.** Trên máy s96 chỉ chạy được 4-bit dưới dạng **W4A32** (trọng số
   4-bit, nhân ma trận fp32) — các model khác là W4A16 / AWQ-fp16 / Q4_K_M. Lý do là phần
   cứng, xem bên dưới. Vẫn cùng hạng *"4-bit chỉ-trọng-số"*, nhưng activation khác.
2. **Chuyên biệt y khoa**, các model khác phổ thông. Đứng một mình thì **không tách được**
   ảnh hưởng của tinh chỉnh y khoa khỏi ảnh hưởng của cỡ 4B. Vì vậy chạy kèm
   `google/gemma-3-4b-it` làm đối chứng (cùng cỡ, cùng họ, cùng nền đa ngữ, chỉ khác ở chỗ
   không tinh chỉnh y khoa); hiệu số MedGemma − Gemma3 là tác dụng riêng của y khoa hoá.
3. **Ngôn ngữ.** Tinh chỉnh y khoa của MedGemma chủ yếu trên dữ liệu **tiếng Anh**, còn VM14K
   là **tiếng Việt**.

**Trước khi tải:** `google/medgemma-4b-it` bị khoá (`gated=auto`) — đồng ý điều khoản Health AI
Developer Foundations tại <https://huggingface.co/google/medgemma-4b-it> rồi dùng token có quyền.
`google/gemma-3-4b-it` cũng gated (`manual`).

##### Chạy trên Turing sm75 (máy s96, Quadro RTX 8000) — đã kiểm 12/10/2026, SGLang 0.5.9

Gemma 3 được thiết kế cho **bf16**; Turing (2018) **không có bf16**. Cần **đồng thời ba thứ**,
thiếu một là sập:

| # | Thay đổi | Vì sao |
|---|---|---|
| 1 | Bản vá launcher `install_fp32_passthrough()` | SGLang `model_runner.py:906` ép fp16 **vô điều kiện** trên sm<80, nuốt luôn `--dtype float32`. Lý do nó ghi là "thiếu bf16" — đúng với bf16, nhưng sm75 chạy fp32 bình thường, nên đây là lỗi logic. Bản vá chỉ sửa đúng điều kiện đó; fp16/auto giữ nguyên hành vi. |
| 2 | `DTYPE=float32` | Ở fp16, logits Gemma tràn số → softmax ra `NaN` → sampler assert `probability tensor contains either inf, nan or element < 0` → scheduler chết ngay lời gọi đầu. |
| 3 | `ATTN_BACKEND=torch_native` | Ở fp32, kernel attention triton với `head_dim=256` cần 69.760 B shared memory, Turing chỉ có 65.536 B → `OutOfResources`. |

Để chạy **4-bit** cần thêm:

| # | Thay đổi | Vì sao |
|---|---|---|
| 4 | Bản vá launcher `install_awq_fp32()` | `AWQConfig.get_supported_act_dtypes()` chỉ trả `[float16]`. Đường tính toán thật vốn đã chạy được fp32 (`scales` theo `params_dtype` → dequant ra `scales.dtype` → `matmul`), chỉ có danh sách kiểm tra chặn. Mở ra → **W4A32**. |

Hai bản vá nằm trong `scripts/x86_cuda/sglang_turing_launch.py`, tự cài ở mọi tiến trình
(kể cả scheduler con) và **chỉ có tác dụng khi truyền rõ `float32`** — các run Qwen/Llama fp16
không bị ảnh hưởng.

**Lệnh chạy (4-bit, W4A32):**

```bash
# tmux gsrv
MODEL=gaunernst/gemma-3-4b-it-int4-awq DTYPE=float32 ATTN_BACKEND=torch_native \
MEM_FRACTION=0.35 MAX_RUNNING=32 CONTEXT_LEN=8192 \
  bash scripts/x86_cuda/serve_sglang.sh

# tmux grun — tự chờ server rồi chạy
bash scripts/wait_and_run.sh zero_shot sglang-gemma3-4b-awq 32
```

Bộ nhớ đo được: fp32 gốc **16,14 GB**; AWQ 4-bit **6,14 GB** (bảng embedding 262k từ và tháp
thị giác không lượng tử hoá, mà ở fp32 chúng to gấp đôi so với bf16).

**Đã thử và KHÔNG được — đừng thử lại:** `--enable-fp32-lm-head`,
`--triton-attention-reduce-in-fp32`, `DTYPE=bfloat16` (sm75 không có), `SGLANG_TURING_NO_JIT=1`,
`--disable-cuda-graph`, `torch_native` ở fp16. Bản `RedHatAI/gemma-3-4b-it-quantized.w4a16`
(compressed-tensors) bị **chặn cứng**: `Min capability: 80. Current capability: 75` — không bản vá
nào qua được, chỉ chạy trên GPU ≥ sm80 (GX10).

##### Kết quả đã đo

| Thí nghiệm | Model | Pass@1 | Ghi chú |
|---|---|---|---|
| `zero_shot`, `pilot_100`, fp32 cả hai | MedGemma-4B | **43,0%** [33,7 ; 52,8] | |
| | Gemma3-4B | **32,0%** [23,7 ; 41,7] | |
| | **Chênh lệch** | **+11,0 pp** [+1,4 ; +20,6] | McNemar exact **p = 0,043**, OR 2,57 |
| `zero_shot`, `pilot_100`, AWQ 4-bit | Gemma3-4B | 33,0% | khớp fp32 (32,0%) → lượng tử hoá không làm giảm chất lượng |
| `zero_shot`, `vm14k_clean` full, AWQ 4-bit | Gemma3-4B | **34,18%** [33,30 ; 35,07] | Easy 37,76 · Medium 32,76 · Hard 31,10 |

CI 95% là Wilson trên **trục câu hỏi**. Ở `zero_shot`, 3 seed cho đáp án **giống hệt nhau**
(đáp án 1 token, phân phối rất nhọn ở T=0,3) — nên đơn vị thống kê phải là **câu**, không phải
(câu, seed). Đếm 300 cặp thay vì 100 câu là giả lặp: lần đầu tính ra p = 0,0002, đúng phải là
p = 0,043.

##### Hạn chế còn lại

- **Chưa có MedGemma 4-bit.** Không có bản AWQ MedGemma nào trên HF, và recipe AWQ của
  `gaunernst` không công bố. Để cặp đối chứng 4-bit công bằng phải **tự lượng tử hoá cả hai**
  bằng cùng một recipe. Hiện chỉ so được ở fp32.
- **Framework sinh dài rất chậm.** fp32 không dùng được tensor core fp16 của Turing, và
  `torch_native` không có kernel hợp nhất. `zero_shot` (3 token ra) chạy ~36 phút/seed, nhưng
  `cot` (tới 512 token ra) đo được **~5,5 câu/phút** → cỡ **4 ngày** cho 3 seed. Các framework
  sinh dài nên chạy trên GX10.

### 3.4. Tham số suy luận (Inference settings)

| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `DEFAULT_TEMPERATURE` | `0.3` | nhiệt độ mặc định cho hầu hết framework |
| `COT_SC_TEMPERATURE` | `0.5` | riêng CoT-SC dùng T cao hơn để đa dạng đường suy luận |
| `MAX_RETRIES` | `3` | số lần thử lại khi lỗi/rate-limit |
| `RETRY_DELAY` | `10s` | thời gian chờ cơ bản trước khi retry |
| `N_RUNS` / `SEEDS` | `3` / `[42, 43, 44]` | mỗi cấu hình chạy 3 lần với 3 seed |

Các "token budget" (`MAX_TOKENS_SHORT=16`, `MAX_TOKENS_LONG=512`, `MAX_TOKENS_AGENT=600`...) giới hạn độ dài đầu ra theo từng loại lời gọi (đáp án 1 ký tự, suy luận CoT, phân tích chuyên gia...).

**Nhiệt độ gắn với FRAMEWORK, không gắn với model.** Mọi backbone (8B / 32B / 70B / MedGemma-4B)
đều dùng cùng giá trị, vì trục so sánh của bài là năng lực backbone chứ không phải cách lấy mẫu.
Chỉ có hai ngoại lệ, cả hai đều theo paper gốc:

| Framework | T | Vì sao |
|---|---|---|
| `cot_sc` | **0,5** | k=5 đường phải khác nhau, không thì bỏ phiếu vô nghĩa (Wang et al.) |
| `self_refine` bước `critique` | **0,0** | phán "đúng/sai" phải tất định, không để ngẫu nhiên quyết định có lặp thêm vòng (Madaan et al.) |
| tất cả còn lại | **0,3** | |

> `voting_ensemble` chạy T=0,3 và lấy đa dạng từ **xáo thứ tự đáp án**; `cot_sc` chạy T=0,5 và lấy
> đa dạng từ **lấy mẫu**. Hai cơ chế cùng là "chạy k lần rồi đếm phiếu" nhưng khác cả nguồn đa dạng
> lẫn nhiệt độ — nếu kết quả lệch nhau thì **không quy được về một trong hai**. Phải nói rõ trong Methods.

#### 3.4.1. Cửa sổ context — KHÔNG đồng nhất giữa các model

| Model | Tag / cấu hình | Context |
|---|---|---|
| `ollama-llama3.1-8b` | `llama3.1-8b-ctx8k` | 8192 |
| `ollama-qwen3-32b` | `qwen3-32b-ctx8k` | 8192 |
| **`ollama-llama3.3-70b`** | **`llama3.3-70b-ctx4k`** | **4096** |
| `sglang-*` | `CONTEXT_LEN` của script serve | 8192 |

70B chạy ở **nửa** context của hai model kia. Lý do phần cứng ghi tại `config.py` mục `ollama-llama3.3-70b`:
bản ctx8k phình lên 57 GB trên card 48 GiB → Ollama chia 13% CPU/87% GPU → timeout >300 s mỗi lời gọi;
ctx4k vừa 49 GB, chạy 100% GPU, ~0,3 s/lời gọi. Prompt dài nhất đo được trên `vm14k_clean` là **~2475
token** (MedAgents) nên 4096 vẫn dư — nhưng **phải khai điểm lệch này trong Methods**, và lưu ý con số
2475 đo cho MedAgents, chưa đo riêng `mad_debate` (nhánh này nhồi toàn văn phản hồi của các agent khác
qua 3 vòng, là ứng viên vượt 4096 thật sự).

### 3.5. `results/` vs `results_test/`

Đường dẫn ghi kết quả chọn theo **tên dataset**: stem bắt đầu bằng `pilot` → `results_test/`, còn lại → `results/`.

| | `results/` — CHẠY THẬT | `results_test/` — CHẠY THỬ |
|---|---|---|
| **Dataset** | `vm14k_clean` (**11.094 câu/seed**) | `pilot_100` (**100 câu/seed**) |
| **Mục đích** | số liệu đưa vào paper | kiểm pipeline, đo thời gian trước khi chạy full |

Luôn chạy `pilot_100` trước khi chạy full: nó phát hiện lỗi prompt/parse/OOM trong vài phút
thay vì vài chục giờ.

> Cài & khởi động Ollama trước khi chạy thật:
> ```bash
> # https://ollama.com/download
> ollama pull llama3.1:8b      # ~5GB VRAM
> ollama serve                 # server local ở http://localhost:11434
> ```

---

## 4. Phân tầng Framework — 3 Tiers

Định nghĩa trong `config.py`. Ý tưởng: đi từ đơn giản (1 lời gọi) đến phức tạp (nhiều agent giao tiếp), để đo **chi phí tăng thêm có đổi lại độ chính xác không**.

| Tier | Ý nghĩa | Framework |
|------|---------|-----------|
| **Tier 1** | Single-LLM prompting — không có giao tiếp giữa agent | `zero_shot`, `cot`, `cot_sc`, `self_refine` |
| **Tier 2** | Ensemble — 1 model, đa dạng nhờ shuffle, không giao tiếp | `voting_ensemble` |
| **Tier 3** | Multi-Agent Systems (MAS) — **có giao tiếp giữa các agent** | `medagents`, `medagents_vi`, `mdagents`, `mdagents_vi`, `mad_debate` |

`ALL_FRAMEWORKS = TIER1 + TIER2 + TIER3` (10 framework). `EXPERIMENT_MATRIX = ALL_FRAMEWORKS × MODEL_NAMES`.

> Chỉ **Tier 3** đủ điều kiện tính **NMAR/ICR** (`MAS_FRAMEWORKS = TIER3_FRAMEWORKS`), vì chỉ ở đây mới có "phiếu bầu giữa các agent" (inter-agent votes). Tier 2 dùng shuffle chứ không phải agent nên bị loại khỏi NMAR/ICR.

---

## 5. Giải thích logic TỪNG Framework

Mọi framework nhận `(question, options, model)` và trả về một `dict` có tối thiểu `answer` (A/B/C/D hoặc `None`) và `api_calls`. Đáp án được rút bằng `parse_answer()` (bỏ khối `<think>...</think>` của các model reasoning như Qwen3/DeepSeek-R1, rồi ưu tiên: ký tự đơn → "Final answer: X" / "đáp án: X" → `**X**` → `(X)` → chữ cái đứng một mình).

### Tier 1 — Single-LLM

#### 5.1. `zero_shot` — [tier1/zero_shot.py](frameworks/tier1/zero_shot.py)
- **Nguồn:** tái lập đúng prompt VM14K Figure 6.
- **Luồng:** 1 lời gọi duy nhất — đưa câu hỏi + 4 lựa chọn + yêu cầu *"Only response with 1 character"*.
- **API calls:** **1**.
- Đây là **mốc tái lập (reproducibility anchor)** — kỳ vọng khớp VM14K Table 2. Cũng là **baseline** dùng để tính ICR (Mục 7).

#### 5.2. `cot` — Chain-of-Thought — [tier1/cot.py](frameworks/tier1/cot.py)
- **Nguồn:** Wei et al., NeurIPS 2022.
- **Luồng:** system prompt "medical expert, think step by step" → yêu cầu kết thúc bằng *"Final answer: X"*.
- **API calls:** **1**.

#### 5.3. `cot_sc` — CoT Self-Consistency — [tier1/cot_sc.py](frameworks/tier1/cot_sc.py)
- **Nguồn:** Wang et al., NeurIPS 2023.
- **Luồng:** chạy **k = 5** đường CoT **độc lập** (cùng prompt, nhưng **T = 0.5** để đa dạng) → **majority vote**. Tie-break: lấy đáp án hợp lệ đầu tiên (deterministic fallback).
- **Output đặc biệt:** `individual_votes` = danh sách 5 đáp án → dùng để tính **Pass@5**.
- **API calls:** **5**.

#### 5.4. `self_refine` — [tier1/self_refine.py](frameworks/tier1/self_refine.py)
- **Nguồn:** Madaan et al., NeurIPS 2023.
- **Luồng (vòng lặp tự phê bình):**
  1. **Round 0 — initial:** trả lời CoT ban đầu (T=0.3).
  2. **Round r — critique:** một "reviewer" đánh giá lập luận (**T=0**, deterministic). Nếu kết luận *"CORRECT. No changes needed."* → **early stop**.
  3. Nếu *"INCORRECT"* → **refine:** sửa lại đáp án dựa trên phê bình (T=0.3).
- **API calls:** **2 → 5** (2 nếu dừng sớm ngay round 1; 5 nếu chạy đủ `max_rounds=2`).
- **Output đặc biệt:** `audit_trail` — nhật ký từng bước (initial/critique/refine/early_stop) phục vụ phân tích lỗi.

### Tier 2 — Ensemble

#### 5.5. `voting_ensemble` — [tier2/voting_ensemble.py](frameworks/tier2/voting_ensemble.py)
- **Nguồn:** VM14K Table 3 "Ensemble".
- **Luồng:** chạy **n = 3** lần, **mỗi lần xáo trộn (shuffle) thứ tự 4 lựa chọn** để khử **position bias**. Mỗi lần dùng `reverse_map` để **ánh xạ đáp án về không gian gốc** rồi mới bỏ phiếu → **majority vote** (tie-break = đáp án lần chạy đầu).
- **Tái lập:** nhận `seed` từ runner → `random.seed(seed)` đảm bảo chuỗi shuffle giống nhau giữa các lần chạy cùng seed.
- **Output đặc biệt:** `individual_votes` (đáp án đã map về gốc) → tính **Pass@3**.
- **API calls:** **3**.

### Tier 3 — Multi-Agent Systems

#### 5.6. `medagents` — [tier3/medagents.py](frameworks/tier3/medagents.py)
- **Nguồn:** Tang et al., ACL 2024 — khung **Multi-disciplinary Collaboration (MC) 5 bước**:
  1. **Expert Gathering** — LLM tự đề xuất 3 vai chuyên gia liên quan.
  2. **Analysis Proposition** — mỗi chuyên gia phân tích câu hỏi + từng lựa chọn.
  3. **Report Summarization** — tổng hợp các phân tích thành báo cáo chung.
  4. **Collaborative Consultation** — các chuyên gia đọc báo cáo, cập nhật quan điểm.
  5. **Decision Making** — bác sĩ cấp cao chốt đáp án cuối.
- **API calls:** `1 + N + 1 + N + 1 = 2N + 3`; với **N=3 → 9 calls**.
- **Output:** `round1_answers` (bước 2), `round2_answers`/`individual_votes` (bước 4 — phiếu bầu giữa agent, dùng cho NMAR), `audit_trail`.

#### 5.7. `medagents_vi` — [tier3/medagents_vi.py](frameworks/tier3/medagents_vi.py) — **đóng góp của paper**
- Giống hệt `medagents`, **chỉ thay Bước 1**: thay vì hỏi LLM chọn chuyên gia, **tra thẳng `medical_topic` của VM14K** qua bảng `TOPIC_TO_ROLES` (≈70 chuyên khoa → danh sách vai). Topic ngoài bảng (noise) → `GENERIC_ROLES`.
- **Lợi ích:** bỏ 1 API call **(9 → 8 calls)** và neo việc chọn vai vào **nhãn chuyên khoa đã được kiểm chứng** thay vì để LLM tự đoán.

#### 5.8. `mdagents` — [tier3/mdagents.py](frameworks/tier3/mdagents.py)
- **Nguồn:** Kim et al., NeurIPS 2024 — khung **thích ứng theo độ phức tạp 4 bước**:
  1. **Complexity Check** — moderator LLM phân loại câu hỏi: **Low / Moderate / High**.
  2. **Định tuyến theo độ phức tạp:**
     - **Low** → 1 Primary Care Clinician (PCC) làm CoT → **2 calls**.
     - **Moderate** → Multi-Disciplinary Team (MDT) 3 agent bỏ phiếu độc lập → **4 calls**.
     - **High** → Integrated Care Team (ICT) 5 chuyên gia + 1 synthesizer → **7 calls**.
- **API calls:** **2 / 4 / 7** tùy nhánh.
- `parse_complexity()` dùng word-boundary để tránh khớp nhầm ("not high", "highly").

#### 5.9. `mdagents_vi` — [tier3/mdagents_vi.py](frameworks/tier3/mdagents_vi.py) — **đóng góp của paper**
- Giống `mdagents`, **thay Bước 1**: dùng thẳng `difficulty_level` của VM14K (Easy/Medium/Hard) để định tuyến thay vì để LLM tự phân loại → bỏ 1 call phân loại và ổn định hơn.

#### 5.10. `mad_debate` — Multi-Agent Debate — [tier3/mad_debate.py](frameworks/tier3/mad_debate.py)
- **Nguồn:** Du et al., ICML 2024 — "society of minds", **không có vai chuyên gia** (agent ẩn danh).
  - **Round 1:** N=3 agent đưa đáp án + lập luận độc lập.
  - **Round 2+:** mỗi agent đọc **toàn bộ phản hồi thô** của agent khác rồi phản biện/cập nhật.
  - **Early stop:** khi tất cả đồng thuận. **Chốt:** majority vote đáp án vòng cuối.
- **API calls:** `N × số_vòng_thực_tế` (tối đa `3 × 3 = 9`, thường ít hơn nhờ early stop).

### Bảng tổng hợp chi phí API (đối chiếu Bảng VII)

| Framework | API calls / câu | Ghi chú |
|-----------|-----------------|---------|
| `zero_shot` | 1 | |
| `cot` | 1 | |
| `cot_sc` | 5 | k=5 đường CoT |
| `self_refine` | 2–5 | early stop |
| `voting_ensemble` | 3 | n=3 shuffle |
| `medagents` | 9 | 2N+3, N=3 |
| `medagents_vi` | 8 | 2N+2, bỏ bước gather |
| `mdagents` | 2 / 4 / 7 | 1 (router) + Low 1 / Moderate 3 / High 6 |
| `mdagents_vi` | **1 / 3 / 6** | đọc `difficulty_level` từ dataset → **không tốn lời gọi router** |
| `mad_debate` | ≤ 9 | N × vòng, có early stop |

**Chi phí thực đo** trên `vm14k_clean` (n=11.094; Easy 32,1% / Medium 56,9% / Challenging 10,2% / Hard 0,8%):
`mdagents_vi` trung bình **2,69 lời gọi/câu** — khớp với 2,9–4,8 quan sát được theo cửa sổ 10 câu.

> **`api_calls` KHÔNG phải thước đo chi phí đủ.** Đo trên 5 câu thật: nhánh Low (PCC, 1 lời gọi)
> sinh 140 token ra, nhánh Moderate (MDT, 3 lời gọi) chỉ sinh 6 token ra. Đếm theo lời gọi thì
> Moderate đắt gấp 3; đếm theo token ra thì Moderate **rẻ hơn 23 lần**. Đây là lý do runner ghi
> `tokens_in`/`tokens_out` (xem Mục 6) — bảng chi phí phải dùng token, không dùng số lời gọi.

---

## 6. Cơ chế chạy thí nghiệm (`run_experiments.py`)

[runner/run_experiments.py](runner/run_experiments.py) là driver chính.

- **`dispatch(framework, sample, model, seed)`** — định tuyến tới đúng runner. Gọi `random.seed(seed)` **trước** để mọi ngẫu nhiên ở tầng Python (đặc biệt shuffle của `voting_ensemble`) **tái lập được** theo seed.
  > Lưu ý: lời gọi LLM **không** truyền tham số `seed` xuống server → khác biệt giữa các seed đến từ **sampling ngẫu nhiên ở T>0** và từ `random.seed(seed)` ở tầng Python (shuffle của `voting_ensemble`), không phải seed của model.
- **Resume (chạy tiếp):** kết quả ghi **append** vào JSONL. Khi chạy lại, `load_done_ids()` đọc các `id` đã xong và **bỏ qua** chúng. Dùng `--no-resume` để chạy lại từ đầu.
- **Cấu trúc một record ghi ra:**

```jsonc
{
  "id": "...", "seed": 42, "framework": "medagents", "model": "ollama-llama3.1-8b",
  "gold": "B", "answer": "B", "correct": true,      // correct = (answer == gold)
  "difficulty": "Medium", "topic": ["Cardiology"],
  "api_calls": 8,                     // framework tự đếm
  "tokens_in": 336, "tokens_out": 6,  // server báo thật (SGLang usage / Ollama *_eval_count)
  "n_llm_calls": 8,                   // client đếm — PHẢI bằng api_calls, đây là kiểm chéo
  "n_usage_missing": 0,               // >0 nghĩa là token là cận dưới, đừng báo cáo chi phí
  "individual_votes": ["B","B","C"],  // phiếu agent (Tier3) HOẶC k đường sampling (cot_sc/voting)
  "round1_answers": ["A","B","C"],
  "audit_trail": [ ... ],             // mỗi bước gọi LLM đều có "raw" ĐẦY ĐỦ, không cắt
  "timestamp": "..."
}
```
  > **Kèm theo mỗi file kết quả là `<file>.jsonl.meta.json`** — manifest cấu hình lần chạy:
  > `model_id`, `provider`, `base_url`, `temperature`, toàn bộ `MAX_TOKENS_*`, `n_samples`,
  > host, phiên bản Python, và `runs[]` ghi lại từng lần chạy/chạy tiếp. Đây là thứ để tái lập
  > mà tên file không nói được.
  > Cùng một trường `individual_votes` mang **hai ý nghĩa khác nhau**: với `cot_sc`/`voting_ensemble` nó là k lần thử độc lập (→ Pass@k); với MAS nó là phiếu agent trong **1** lần chạy (→ **không** dùng cho Pass@k).

- **`--all`:** chạy trọn `EXPERIMENT_MATRIX` (10 framework × số model) × `N_RUNS` seed. Ví dụ 10×3 pairs × 3 seed.
- **Tổng hợp:** mỗi seed tính `compute_pass_at_1()`, rồi `compute_mean_std()` gộp thành **mean ± std** và in bảng tóm tắt cuối.

---

## 7. Các METRICS và CÁCH TÍNH (`metrics/compute.py`)

[metrics/compute.py](metrics/compute.py). Helper chung: `_pct(num, den) = num/den*100` (0 nếu den=0); `_norm_diff` gộp `"Challenging" → "Hard"`.

### 7.1. `Pass@1` — Độ chính xác chuẩn
- **Định nghĩa:** tỉ lệ câu có `answer == gold`.
- **Công thức:** `Pass@1 = (số câu đúng / tổng số câu) × 100`.
- Chia nhỏ **theo difficulty** (Easy/Medium/Hard) và **theo topic** (một câu nhiều topic sẽ được cộng vào từng topic).

### 7.2. `Pass@k` — Có ít nhất 1 trong k lần đúng
- **Định nghĩa:** tỉ lệ câu mà **≥1 trong k lần thử độc lập** cho đáp án đúng.
- **Công thức:** với mỗi câu, đúng nếu `any(v == gold for v in individual_votes[:k])`.
- **CHỈ hợp lệ cho:** `cot_sc` (**k=5**) và `voting_ensemble` (**k=3**) — xem `PASS_AT_K_MAP`. Với MAS, `individual_votes` là phiếu agent trong 1 run nên **không được** dùng cho Pass@k (`n_total=0` → hiển thị `---`).

### 7.3. `mean ± std` qua nhiều seed — `compute_mean_std()`
- Dùng **sample standard deviation** (chia **n−1**, không phải n), đúng quy ước báo cáo của MDAgents/MedAgentBoard cho n=3 seed. Khi n=1 thì std=0 (không tính được).
- **Công thức std:** `std = sqrt( Σ(vᵢ − mean)² / (n−1) )`.

### 7.4. `CG` — Collaboration Gain — `compute_collaboration_gain()`
- **Định nghĩa:** lợi ích của việc hợp tác so với phương pháp **không giao tiếp mạnh nhất**.
- **Công thức:** `CG = Acc(MAS) − max(Acc(zero_shot), Acc(cot))`, đơn vị **pp (percentage points)**.
- CG > 0 → hợp tác có lợi; CG < 0 → hợp tác làm **hại**.

### 7.5. `NMAR` — Negative Majority Assimilation Rate — `compute_nmar()`
- **Ý nghĩa:** đo hiện tượng **"đồng hóa theo đa số sai"** — khi đáp án đúng vốn là **thiểu số** trong các phiếu agent, nhưng đáp án cuối lại **chạy theo đa số (sai)**.
- **Điều kiện đủ (eligible):** câu mà `gold` **có mặt** trong phiếu agent **nhưng** đa số (`majority`) **khác** `gold`.
- **Tính NMAR case:** trong các câu eligible, đếm câu mà `answer == majority` **và** `answer != gold`.
- **Công thức:** `NMAR = (nmar_count / eligible) × 100`. Hàm trả về `(rate, nmar_count, eligible)` để phục vụ Fisher test.

### 7.6. `ICR` — Ineffective Collaboration Rate — `compute_icr()`
- **Ý nghĩa:** đo mức độ hợp tác **vô ích** — bao nhiêu câu mà MAS cho ra **đúng y hệt** đáp án `zero_shot` (ghép cặp theo `id`), tức là bao công sức nhiều agent **không thay đổi** kết quả.
- **Công thức:** `ICR = (số câu answer_MAS == answer_zero_shot / số câu ghép cặp được) × 100`.

### 7.7. `CLD` — Cross-Lingual Delta — `compute_cld()`
- **Ý nghĩa:** so sánh metric lỗi trên **tiếng Việt** với **tham chiếu tiếng Anh** (EN reference lấy từ MedAgentAudit, lưu trong `EN_REFERENCE`).
- **Công thức:** `CLD = metric_VI − metric_EN`. CLD > 0 → lỗi bị **khuếch đại** khi sang tiếng Việt.

### 7.8. `Fisher exact test` — `fisher_test_vs_reference()`
- Kiểm định **NMAR_VI có khác EN reference một cách có ý nghĩa** không. Dựng bảng 2×2 tổng hợp (dùng đúng số `eligible` của VI làm cỡ mẫu) rồi chạy `fisher_exact`. **Significant nếu p < 0.05**.

### 7.9. `McNemar test` — `mcnemar_test()`
- So sánh **hai phương pháp trên cùng tập câu** (ghép cặp theo `id`) — dùng để đối chiếu biến thể `_vi` với bản gốc.
- `b` = A sai / B đúng; `c` = A đúng / B sai. Dùng **binomtest chính xác** trên `b` với `n=b+c`, `p=0.5`, hai phía. **Significant nếu p < 0.05**.

---

## 8. Phân tích kết quả — Bảng I–VII (`analyze_results.py`)

[analysis/analyze_results.py](analysis/analyze_results.py) đọc **mọi file JSONL theo seed**, gộp lại và in 7 bảng của paper:

| Bảng | Nội dung |
|------|----------|
| **I** | Hiệu năng chính (Pass@1 mean±std theo Easy/Medium/Hard/Overall) + **Pass@k** (chỉ cot_sc/voting) + **CG** |
| **II** | Biến thể `_vi` so với bản gốc (delta theo độ khó) + **McNemar p** |
| **III** | Phân tích lỗi cross-lingual: **NMAR/ICR** (VI vs EN) + **CLD** + **Fisher p** (chỉ Tier 3) |
| **IV** | Taxonomy lỗi (NMAR, ICR gộp qua seed) |
| **V** | Theo chuyên khoa (top-N specialty, `llama-3.3-70b`: Best-Prompt vs MDAgents-VI + CG) |
| **VI** | Khuyến nghị triển khai theo điều kiện (bảng tĩnh: điều kiện → CG/NMAR risk/Cost/khuyến nghị) |
| **VII** | Chi phí API (số call trung bình / câu, gộp qua seed) |

Hằng số điều khiển: `PASS_AT_K_MAP = {"cot_sc": 5, "voting_ensemble": 3}`; `MAS_FRAMEWORKS = TIER3_FRAMEWORKS` (chỉ Tier 3 đủ điều kiện NMAR/ICR).

**Chạy phân tích:**

```bash
python -m analysis.analyze_results --dataset pilot_100
python -m analysis.analyze_results --dataset vm14k_clean --results-dir results
```

> `--dataset` ở đây là **stem** (không có `.jsonl`, không có `data/`), khớp phần `{dataset_stem}` trong tên file kết quả.

---

## 9. Chuẩn bị dữ liệu (`prepare_data.py`)

[data/prepare_data.py](data/prepare_data.py) lọc VM14K thô → 2 file dùng cho thí nghiệm:

**Bộ lọc:**
1. **Chỉ giữ MCQ 4 lựa chọn** (`len(options) == 4`).
2. **Loại record mà TẤT CẢ `medical_topic` đều là noise** (không phải y khoa — xem `NOISE_TOPICS`: Chemistry, Physics, Sociology, Health Policy...).

**Đầu ra:**
- `vm14k_clean.jsonl` — toàn bộ dữ liệu sạch (dataset chính).
- `pilot_100.jsonl` — mẫu 100 câu phân tầng **Easy 33 / Medium 50 / Hard 17** để chạy thử nhanh.

```bash
python -m data.prepare_data --input /path/to/raw_vm14k.jsonl
```

---

## 10. CÁC CÂU LỆNH CHẠY (chạy thật bằng LOCAL / Ollama)

Các lệnh dưới đây là **luồng chạy thật** trên toàn bộ `vm14k_clean` bằng **model local** — đúng như thực tế đã chạy trong `results/`. Chạy từ **thư mục `code/`**.

> Sheet `Run` gốc trong `Metric.xlsx` ghi tên model theo provider cloud đã gỡ bỏ — khi tái hiện phải **thay `--model` sang `ollama-*` hoặc `sglang-*`** (xem [Mục 3.3](#33-danh-sách-model-khả-dụng-model_names)).

### Bước chuẩn bị (bắt buộc) — khởi động server local

```bash
# https://ollama.com/download
ollama pull llama3.1:8b       # model 8B — luồng chạy thật hiện tại (~5GB VRAM)
ollama serve                  # server local ở http://localhost:11434
```

### Tier 1

```bash
# Zero-shot (1 call/câu)
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework zero_shot --model ollama-llama3.1-8b

# CoT (1 call/câu)
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework cot --model ollama-llama3.1-8b

# CoT-SC (5 calls/câu)
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework cot_sc --model ollama-llama3.1-8b

# Self-Refine (2-5 calls/câu)
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework self_refine --model ollama-llama3.1-8b
```

### Tier 2 — Ensemble

```bash
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework voting_ensemble --model ollama-llama3.1-8b
```

### Tier 3 — Multi-Agent Systems

```bash
# MedAgents (gốc) + MedAgents-VI
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework medagents    --model ollama-llama3.1-8b
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework medagents_vi --model ollama-llama3.1-8b

# MDAgents (gốc) + MDAgents-VI
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework mdagents    --model ollama-llama3.1-8b
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework mdagents_vi --model ollama-llama3.1-8b

# MAD Debate
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework mad_debate --model ollama-llama3.1-8b
```

### Chạy nhanh toàn bộ ma trận (khuyến nghị)

Thay vì gõ từng lệnh, đặt model local mặc định rồi chạy `--all` (10 framework × các model đang bật × 3 seed):

```bash
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --all
```

> Lưu ý: `--all` chạy trọn `EXPERIMENT_MATRIX = ALL_FRAMEWORKS × MODEL_NAMES` = **10 framework × 12 model = 120 cặp** × 3 seed. Muốn thu hẹp thì comment bớt model trong dict `MODELS` của [config.py](infrastructure/config.py).

### Chạy model 70B local trên RTX 8000 (48GB)

Model 70B bản **Q4_K_M (~43GB)** vừa khít RTX 8000. Trước tiên thêm một entry vào dict `MODELS` trong [config.py](infrastructure/config.py):

```python
"ollama-llama3.3-70b": {
    "provider": "ollama",
    "model_id": "llama3.3:70b",   # Q4_K_M ~43GB — vừa 1 card RTX 8000
},
```

rồi:

```bash
ollama pull llama3.3:70b
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework zero_shot --model ollama-llama3.3-70b
```

> ⚠️ 70B local trên RTX 8000 chạy **~8–14 tok/s** → rất chậm với ~11k câu × 3 seed. Nên **test trên `pilot_100` trước** để đo thời gian, ưu tiên Tier 1 (1 call/câu), hoặc giảm còn 1 seed.

### Test nhanh trên pilot_100 trước khi chạy full

```bash
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model sglang-qwen3-32b
```

Ghi vào `results_test/`, không đụng `results/`.

---

## 11. Quy trình end-to-end (Workflow) — LOCAL

```bash
# Bước 0 — (một lần) chuẩn bị dữ liệu từ VM14K thô
python -m data.prepare_data --input /path/to/raw_vm14k.jsonl

# Bước 1 — khởi động server local + kéo model
ollama pull llama3.1:8b
ollama serve

# Bước 2 — test nhanh pipeline trên pilot 100 câu (vẫn dùng local)
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model ollama-llama3.1-8b

# Bước 3 — chạy đầy đủ trên dữ liệu lớn (từng lệnh Mục 10, hoặc --all)
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --all

# Bước 4 — sinh Bảng I–VII
python -m analysis.analyze_results --dataset vm14k_clean --results-dir results
```

> **Mẹo:** vì kết quả ghi append + hỗ trợ resume, bạn có thể ngắt (Ctrl-C) và chạy lại đúng lệnh cũ để tiếp tục từ chỗ dừng. Dùng `--no-resume` khi muốn làm lại từ đầu.

---

## 12. Ghi chú hiệu năng — chạy chậm với Ollama local & cách khắc phục

Khi chạy thật `zero_shot` × `ollama-qwen3-32b` trên toàn bộ `vm14k_clean` (11.094 câu), job **chậm bất thường**: sau ~39 giờ chỉ xong 1830/11094 câu (ETA ước tính ~8 ngày). Điều tra + khắc phục như sau.

### Hạn chế ban đầu

1. **Chạy tuần tự (sequential)** — [runner/run_experiments.py](runner/run_experiments.py) gọi LLM từng câu một trong vòng `for`, dù service Ollama đã bật sẵn `OLLAMA_NUM_PARALLEL=4` (cho phép 4 request song song) — GPU gần như rảnh (~15% util) phần lớn thời gian.
2. **Model tràn sang CPU** — Ollama mặc định load model với **context tối đa của model** (`num_ctx` = 40960 cho `qwen3:32b`), khiến KV cache quá lớn để fit cùng weights trên GPU 48GB → **77% layer chạy trên CPU, chỉ 23% trên GPU** → suy luận rất chậm (~76–90 giây/câu).
3. **Truyền `num_ctx` qua request không có tác dụng** — thử ép `num_ctx` nhỏ hơn (4096) qua `extra_body`/`options` của thư viện `openai` (endpoint `/v1/chat/completions`) nhưng **bị Ollama âm thầm bỏ qua**; chỉ endpoint gốc `/api/chat` mới tôn trọng tham số này.

### Đã khắc phục

1. **Song song hoá client** — [runner/run_experiments.py](runner/run_experiments.py) đổi từ vòng `for` tuần tự sang `ThreadPoolExecutor` với `NUM_WORKERS=4` (khớp `OLLAMA_NUM_PARALLEL`), ghi file JSONL an toàn bằng `threading.Lock`.
2. **Bake `num_ctx` vào Modelfile riêng thay vì truyền qua API** — vì API không đáng tin cậy, tạo **tag model riêng** với `num_ctx=8192` cố định sẵn:
   ```bash
   ollama create qwen3-32b-ctx8k -f infrastructure/ollama_modelfiles/Modelfile.qwen3-32b
   ```
   rồi trỏ `model_id` trong [infrastructure/config.py](infrastructure/config.py) sang tag đó. Áp dụng đồng bộ cho cả 4 model Ollama (`llama3.1-8b`, `qwen3-14b`, `qwen3-32b`, `deepseek-r1-14b`) để đảm bảo **công bằng khi so sánh** giữa các model (cùng `num_ctx`, không model nào bị ưu ái).
3. **Kết quả đo được** (`qwen3-32b-ctx8k`): `100% GPU` (trước: 23%/77% CPU-GPU), **~6–7 giây/câu** (trước: ~76–90s) → **nhanh hơn ~12 lần**, ETA cho 11k câu giảm từ **~8 ngày còn ~19–21 giờ**.

> Model `llama3.1:8b`, `qwen3:14b`, `deepseek-r1:14b` **chưa được pull** trên máy hiện tại. Khi cần dùng, chạy `ollama pull <model>` rồi `ollama create <model>-ctx8k -f infrastructure/ollama_modelfiles/Modelfile.<model>` trước khi thực nghiệm.
