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

- `openai` — SDK gọi API (dùng chung cho cả Groq và Ollama vì đều tương thích chuẩn OpenAI).
- `scipy` — kiểm định thống kê (`fisher_exact`, `binomtest`).
- `pandas`, `openpyxl` — đọc/ghi `Metric.xlsx`.

### 3.2. Hai nhà cung cấp model (providers)

Định nghĩa trong [infrastructure/config.py](infrastructure/config.py):

| Provider | Base URL | API key | Vai trò thực tế trong dự án |
|----------|----------|---------|------------------------------|
| **groq** (cloud) | `https://api.groq.com/openai/v1` | cần `GROQ_API_KEY` | **CHỈ để TEST nhanh** trên `pilot_100` (100 câu). Nhanh nhưng có **rate limit** + chi phí → không kham nổi dữ liệu lớn |
| **ollama** (local) | `http://localhost:11434/v1` | không cần (chuỗi `"ollama"`) | **CHẠY THẬT** trên toàn bộ `vm14k_clean` (~11k câu/seed). Chạy offline, không rate limit, không tốn tiền |

> 🔑 **Groq = test, Ollama local = chạy thật.** Đây là quy ước vận hành thực tế của dự án, thấy rõ qua chính dữ liệu kết quả (xem [Mục 3.5](#35-groq--test-nhanh--ollama-local--chạy-thật-thực-tế-trong-repo)):
> - **`results/`** (chạy thật, dataset `vm14k_clean` **11.094 câu/seed**) → **chỉ dùng `ollama-llama3.1-8b`** (local).
> - **`results_test/`** (chạy thử, dataset `pilot_100` **100 câu**) → dùng model Groq (`llama-3.1-8b`, `llama-3.3-70b`, `qwen3-32b`).
>
> Với khối lượng ~11k câu × 3 seed × nhiều framework, **bắt buộc dùng server local (Ollama)** để tránh rate limit/chi phí của Groq. Groq chỉ dùng cho vòng thử nghiệm nhanh trên 100 câu pilot.

> ⚠️ **Lưu ý:** Trong `config.py`, tất cả các dòng `GROQ_API_KEY = ...` **hiện đang bị comment** — hợp lý vì luồng chạy thật dùng Ollama (không cần key). Chỉ khi muốn test bằng Groq mới phải **bật lại một dòng** (hoặc export biến môi trường `GROQ_API_KEY`), nếu không `get_api_key("groq")` sẽ báo `NameError`.

### 3.3. Danh sách model khả dụng (`MODEL_NAMES`)

Chỉ những key **không bị comment** trong dict `MODELS` mới hợp lệ với tham số `--model`:

| Tên (`--model`) | Provider | model_id |
|-----------------|----------|----------|
| `llama-3.3-70b` | groq | `llama-3.3-70b-versatile` |
| `llama-3.1-8b`  | groq | `llama-3.1-8b-instant` |
| `qwen3-32b`     | groq | `qwen/qwen3-32b` |
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

> `deepseek-r1-70b` (Groq) **đang bị comment** → chưa dùng được (xem Mục 10).

#### 3.3.1. MedGemma-4B — model duy nhất phá ba giả định của bảng

`sglang-medgemma-4b` khác mọi model còn lại. Ba điểm **bắt buộc khai trong Methods**:

1. **Không lượng tử hoá (bf16).** 8B/32B/70B đều 4-bit. Ép một model 4B xuống 4-bit
   mất chất lượng nặng hơn nhiều so với 32B/70B, nên bf16 là lựa chọn đúng — nhưng
   nó phá quy tắc "cùng hạng 4-bit" mà bảng v1/v2 dựa vào để so sánh.
2. **Chuyên biệt y khoa**, các model khác phổ thông. Đứng một mình thì **không tách
   được** ảnh hưởng của tinh chỉnh y khoa khỏi ảnh hưởng của cỡ 4B. Muốn tách phải
   thêm `google/gemma-3-4b-it` làm đối chứng (cùng cỡ, cùng họ, cùng nền đa ngữ,
   chỉ khác ở chỗ không tinh chỉnh y khoa); hiệu số MedGemma − Gemma3 khi đó là
   tác dụng riêng của y khoa hoá.
3. **Ngôn ngữ.** Tinh chỉnh y khoa của MedGemma chủ yếu trên dữ liệu **tiếng Anh**,
   còn VM14K là **tiếng Việt**. Điểm thấp có thể do ngôn ngữ chứ không do y khoa —
   reviewer chắc chắn nêu điểm này.

**Trước khi tải:** repo bị khoá (`gated=auto`). Phải bấm đồng ý điều khoản Health AI
Developer Foundations tại <https://huggingface.co/google/medgemma-4b-it>, rồi dùng
token có quyền (`HF_TOKEN=...` hoặc `~/.cache/huggingface/token`).

**Đã kiểm ngày 24/09/2026** (SGLang 0.5.9):

| Hạng mục | Kết quả |
|---|---|
| Kiến trúc | `Gemma3ForConditionalGeneration`, `model_type=gemma3` |
| SGLang hỗ trợ | có — `gemma3_causal.py` / `gemma3_mm.py` |
| Đa phương thức | SGLang **tự tắt** (`Multimodal is disabled for gemma3`) → chạy text-only, không tốn chỗ cho tháp thị giác |
| `role: "system"` trong prompt | **chạy được** — template Gemma 3 gộp system vào lượt user đầu; `raise_exception` chỉ bắt role không xen kẽ, mà prompt của repo luôn là `[system, user]` |
| Trọng số | 8,01 GB (bf16), nạp thực tế 8,04 GB |
| Context gốc | 131 072 (ta vẫn cap `CONTEXT_LEN=8192` như các model khác) |

**Lệnh chạy** (cổng 30001 để không đụng server 32B ở 30000):

```bash
MODEL=google/medgemma-4b-it PORT=30001 \
MEM_FRACTION=0.35 MAX_RUNNING=32 CONTEXT_LEN=8192 \
bash scripts/x86_cuda/serve_sglang.sh

export SGLANG_BASE_URL=http://127.0.0.1:30001/v1
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl \
  --framework zero_shot --model sglang-medgemma-4b --workers 32
```

**Hai cảnh báo về phần cứng:**

- **Cần ≳16 GB VRAM trống.** 8 GB trọng số + KV cache. Trên máy s96 (Quadro RTX 8000
  48 GB) khi server Qwen3-32B đang giữ 28,7 GB thì **không đủ chỗ** — SGLang nạp xong
  trọng số rồi chết với `RuntimeError: Not enough memory`. Phải dừng server 32B trước,
  hoặc chạy ở máy khác.
- **Trên GPU sm < 80 (Turing/Volta), SGLang tự ép bf16 → fp16**
  (`Casting torch.bfloat16 to torch.float16`). Họ Gemma có tiền sử tràn số ở fp16.
  **Chưa kiểm chứng được** trên máy này vì server không lên nổi do thiếu VRAM.
  Trên GPU Ampere trở lên (GX10 sm121) có bf16 gốc nên không gặp vấn đề này.

### 3.4. Tham số suy luận (Inference settings)

| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `DEFAULT_TEMPERATURE` | `0.3` | nhiệt độ mặc định cho hầu hết framework |
| `COT_SC_TEMPERATURE` | `0.5` | riêng CoT-SC dùng T cao hơn để đa dạng đường suy luận |
| `MAX_RETRIES` | `3` | số lần thử lại khi lỗi/rate-limit |
| `RETRY_DELAY` | `10s` | thời gian chờ cơ bản trước khi retry |
| `N_RUNS` / `SEEDS` | `3` / `[42, 43, 44]` | mỗi cấu hình chạy 3 lần với 3 seed |

Các "token budget" (`MAX_TOKENS_SHORT=16`, `MAX_TOKENS_LONG=512`, `MAX_TOKENS_AGENT=600`...) giới hạn độ dài đầu ra theo từng loại lời gọi (đáp án 1 ký tự, suy luận CoT, phân tích chuyên gia...).

### 3.5. Groq = test nhanh / Ollama local = chạy thật (thực tế trong repo)

Bằng chứng lấy trực tiếp từ các file kết quả có sẵn:

| | `results/` — **CHẠY THẬT** | `results_test/` — **CHẠY THỬ** |
|---|---|---|
| **Dataset** | `vm14k_clean` (**11.094 câu/seed**) | `pilot_100` (**100 câu/seed**) |
| **Provider** | **Ollama (local)** | **Groq (cloud)** |
| **Model** | `ollama-llama3.1-8b` | `llama-3.1-8b`, `llama-3.3-70b`, `qwen3-32b` |
| **Framework đã chạy** | Tier 1 (`zero_shot`, `cot`, `cot_sc`, `self_refine`) | cả Tier 1 + `voting_ensemble` + `medagents` |

**Kết luận vận hành:**
- Khi **chạy thật với dữ liệu lớn** → dùng `--model ollama-llama3.1-8b` (hoặc `ollama-qwen3-*`), sau khi đã cài Ollama và `ollama pull llama3.1:8b`. **Không** dùng Groq cho `vm14k_clean`.
- Khi **thử nghiệm nhanh** để kiểm tra pipeline / so sánh nhiều model mạnh → dùng model Groq trên `pilot_100`.
- Vì vậy các lệnh chép từ sheet `Run` (Mục 10) — vốn ghi model Groq kèm `vm14k_clean` — **cần đổi `--model` sang `ollama-*`** nếu muốn tái hiện đúng luồng chạy thật.

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
| `mdagents` | 2 / 4 / 7 | tùy Low/Moderate/High |
| `mdagents_vi` | 2 / 4 / 7 | định tuyến bằng difficulty_level |
| `mad_debate` | ≤ 9 | N × vòng, có early stop |

---

## 6. Cơ chế chạy thí nghiệm (`run_experiments.py`)

[runner/run_experiments.py](runner/run_experiments.py) là driver chính.

- **`dispatch(framework, sample, model, seed)`** — định tuyến tới đúng runner. Gọi `random.seed(seed)` **trước** để mọi ngẫu nhiên ở tầng Python (đặc biệt shuffle của `voting_ensemble`) **tái lập được** theo seed.
  > Lưu ý: Groq **không** hỗ trợ tham số `seed` cho lời gọi LLM → sự khác biệt giữa các seed đến từ **sampling ngẫu nhiên ở T>0**, không phải seed của model.
- **Resume (chạy tiếp):** kết quả ghi **append** vào JSONL. Khi chạy lại, `load_done_ids()` đọc các `id` đã xong và **bỏ qua** chúng. Dùng `--no-resume` để chạy lại từ đầu.
- **Cấu trúc một record ghi ra:**

```jsonc
{
  "id": "...", "seed": 42, "framework": "medagents", "model": "ollama-llama3.1-8b",
  "gold": "B", "answer": "B", "correct": true,      // correct = (answer == gold)
  "difficulty": "Medium", "topic": ["Cardiology"],
  "api_calls": 8,
  "individual_votes": ["B","B","C"],  // phiếu agent (Tier3) HOẶC k đường sampling (cot_sc/voting)
  "round1_answers": ["A","B","C"],
  "audit_trail": [ ... ],
  "timestamp": "..."
}
```
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

Các lệnh dưới đây là **luồng chạy thật** trên toàn bộ `vm14k_clean` bằng **model local (Ollama)** — đúng như thực tế đã chạy trong `results/`. **Không gọi lên Groq.** Chạy từ **thư mục `code/`**.

> Sheet `Run` gốc trong `Metric.xlsx` ghi model Groq (`llama-3.1-8b/llama-3.3-70b`) chỉ để **test nhanh trên `pilot_100`**. Với dữ liệu lớn ta **thay `--model` sang `ollama-*`** (xem [Mục 3.5](#35-groq--test-nhanh--ollama-local--chạy-thật-thực-tế-trong-repo)).

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

> Lưu ý: `--all` chạy trọn `EXPERIMENT_MATRIX = ALL_FRAMEWORKS × MODEL_NAMES`. Nếu muốn `--all` **chỉ** dùng model local, hãy **comment các model Groq** trong dict `MODELS` của [config.py](infrastructure/config.py) để `MODEL_NAMES` chỉ còn các `ollama-*`.

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

### (Tùy chọn) Test nhanh bằng Groq — KHÔNG dùng cho dữ liệu lớn

Chỉ khi muốn đối chiếu nhanh trên 100 câu pilot với model mạnh (llama-3.3-70b, qwen3-32b). Cần bật `GROQ_API_KEY` trong [config.py](infrastructure/config.py):

```bash
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model llama-3.3-70b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework zero_shot --model qwen3-32b
```

> `deepseek-r1-70b` trong sheet `Run` gốc **đang bị comment** trong `config.py` → không dùng được cho tới khi bỏ comment.

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
