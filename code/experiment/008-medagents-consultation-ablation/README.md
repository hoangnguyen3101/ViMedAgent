# Experiment 008 — MedAgents: tín hiệu bất đồng mất ở đâu khi hội chẩn (mốc, A hội đồng dị thể, B bỏ report, CoT-SC)

<!-- theo chuẩn experiment/Template/README.md -->
<!-- Bản ĐĂNG KÝ TRƯỚC: mục 1–3 viết trước khi chạy bất kỳ lời gọi nào.
     Mục 4–6 để trống, chỉ điền bằng số liệu thật sau khi chạy. -->

- **Date:** 11/11/2026 (viết kế hoạch và code; chưa chạy)
- **Author:** hoangnv (qua Claude Code), theo thiết kế của thầy hướng dẫn
- **Status:** planned
- **Base model:** `sglang-qwen3-32b` (`Qwen/Qwen3-32B-AWQ`). Riêng Thực nghiệm A dùng thêm `sglang-llama3.3-70b`
  (`casperhansen/llama-3.3-70b-instruct-awq`) và `sglang-llama3.1-8b-awq` (`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`).
- **Method:** không huấn luyện. Bốn điều kiện chạy trên cùng 2.000 câu VM14K, 2 seed, cùng một cấu hình.
  A và B là hai biến thể của MedAgents, viết chung trong `frameworks/tier3/medagents_ablation.py`.

---

## 1. Motivation / Hypothesis

Phân tích log hôm 04/11 (`docs/MedAgent_Research_Narrative_2026-11-04.md`, mục Q2b–Q2c) cho thấy **bước hội chẩn của
MedAgents là một phép kéo về đa số**. Một chuyên gia đang đúng mà hai đồng nghiệp sai thì bỏ đáp án đúng khoảng 50–60%
số lần.

Trước hội chẩn, việc 3 chuyên gia **bất đồng** là một tín hiệu tốt rằng câu đó dễ sai. Sau hội chẩn, họ thường
**đồng thuận**, kể cả đồng thuận sai, nên tín hiệu đó mất. Thí nghiệm này gọi hiện tượng trên là *laundering*:
bất đồng bị "rửa" thành đồng thuận.

Thí nghiệm tìm xem laundering nằm ở đâu, bằng cách loại trừ hai giả thuyết:

| | Giả thuyết | Can thiệp | Đọc kết quả, so với mốc |
|---|---|---|---|
| **A** | Tín hiệu mất vì 3 chuyên gia là **cùng một model** | 3 chuyên gia là 3 model khác nhau | Nếu tín hiệu **vẫn mất** thì laundering đến từ cơ chế hội chẩn, không do các agent giống nhau |
| **B** | Tín hiệu mất ở **bước report**, nơi 3 ý kiến bị gộp thành một văn bản | Bỏ report; mỗi người hội chẩn đọc thẳng 3 bài phân tích | Nếu tín hiệu **được giữ** thì bước tổng hợp là nguyên nhân. Nếu **vẫn mất** thì nguyên nhân là việc đọc ý kiến người khác |

**Lưu ý khi đọc kết quả B.** B được chạy theo phương án B2: đọc **toàn văn** các bài phân tích. Vì vậy B khác mốc ở
**hai** điểm cùng lúc, là bỏ bước tóm tắt và được đọc nhiều thông tin hơn (xem mục 2.3).

## 2. Setup

### 2.1 Bốn điều kiện

| Điều kiện | Framework (`--framework`) | Chuyên gia 1 / 2 / 3 | Report | Lời gọi/câu | Tổng lời gọi (2.000 câu × 2 seed) |
|---|---|---|---|---:|---:|
| Mốc | `medagents` (chuẩn, không sửa) | Qwen / Qwen / Qwen | có | 9 | 36.000 |
| A, hội đồng dị thể | `medagents_hetero` | **Qwen / Llama-70B / Llama-8B** | có | 9 | 36.000 (Qwen 20k, 70B 8k, 8B 8k) |
| B, bỏ report | `medagents_noreport` | Qwen / Qwen / Qwen | **không** | 8 | 32.000 |
| CoT-SC | `cot_sc` (chuẩn) | — (5 mẫu, bỏ phiếu đa số) | — | 5 | 20.000 |

Tổng khoảng **124.000 lời gọi**. Ước lượng 110.000 của thầy chưa tính CoT-SC.

Ở cả A và B, bước chọn vai, bước report (nếu có) và bước decision đều do `--model` (Qwen) đảm nhận.

### 2.2 Thực nghiệm A, chi tiết

Luồng chạy và prompt **giữ nguyên** như MedAgents; chỉ đổi model của từng chuyên gia.

| Bước | Model |
|---|---|
| 1. Chọn 3 vai | Qwen |
| 2. Phân tích, chuyên gia i | `HETERO_PANEL[i]` = Qwen / Llama-70B / Llama-8B |
| 3. Report | Qwen |
| 4. Hội chẩn, chuyên gia i | cùng model với bước 2 của chuyên gia đó |
| 5. Decision | Qwen |

- Vai thứ i do Qwen sinh ra giao cho chuyên gia i, **theo thứ tự cố định**. `parse_roles` luôn trả đúng 3 vai: thiếu thì
  thêm vai mặc định, thừa thì cắt bớt.
- Hàm sinh prompt được **import** từ `medagents.py`, không chép lại. Test chứng minh rằng khi cả 3 chuyên gia là Qwen,
  chuỗi lời gọi của A trùng khớp từng chữ với `medagents` (mục 2.6).

### 2.3 Thực nghiệm B, chi tiết

Hội đồng toàn Qwen. **Bỏ bước 3 (report).** Ở bước 4, prompt hội chẩn là đúng `prompt_consult()` của MedAgents. Chỉ khối

```
Consultation report from all specialists:
<report>
```

được thay bằng

```
Analyses from all specialists:
[<role 1>] (Answer: X):
<toàn văn bài phân tích 1>

[<role 2>] (Answer: Y):
...
```

Mọi chữ khác trong prompt giữ nguyên, kể cả đoạn *"Your previous analysis"*. Bước decision không đổi, vì nó vốn không
đọc report.

**Vì sao "toàn văn" (B2) là một can thiệp lớn.** Trong MedAgents chuẩn, mỗi bài phân tích được `extract_reasoning()` cắt
còn khoảng 500 ký tự trước khi đưa sang bước report và hội chẩn. Đo trên log có sẵn:

| | Độ dài trung vị một bài phân tích | Bị cắt | Phần được chuyển đi | Phần chuyển đi có dòng "Final answer" |
|---|---:|---:|---:|---:|
| USMLE × Qwen (trần 1024) | 2.157 ký tự | 100% | ~440 ký tự (20%) | 0% |
| VM14K × Llama-70B (trần 600) | 1.374 ký tự | 100% | ~435 ký tự (32%) | 0% |

Nghĩa là ở mốc và ở A, người viết report chỉ đọc 20–30% phần đầu của mỗi bài, cộng nhãn đáp án. Ở B2, người hội chẩn
đọc 100%. Phương án **B1** (đưa đúng đoạn đã cắt mà bước report nhận, tức chỉ đổi đúng một thứ) có sẵn trong code
dưới dạng `consult_input="analyses_truncated"`, nhưng **không** nằm trong kế hoạch chạy.

### 2.4 Dữ liệu

- **File:** `code/data/vm14k_sub2000.jsonl` (2.000 dòng, chép nguyên từng dòng từ `vm14k_clean.jsonl`),
  `vm14k_sub2000_ids.txt` (danh sách ID cố định), `vm14k_sub2000.manifest.json`.
- **Script:** `code/data/make_vm14k_subset.py`, seed **42**. Chạy lại luôn ra cùng danh sách; script từ chối ghi đè
  một danh sách khác nếu không có `--force`.
- **SHA-256 danh sách ID:** `c6595eeedfd5a202c8f79e8e56562ce464beb003563bc4556a1488acfdf68f99`
- **SHA-256 nguồn** (`vm14k_clean.jsonl`): `a131243274cadb03b6346b42f65d419138518c8b5f725937c28b08fac5c8940d`
- **Quần thể chọn:** 10.139 ID. Mỗi nội dung (`question` + `options`) giữ ID nhỏ nhất; bỏ 50 nhóm trùng có đáp án
  chuẩn mâu thuẫn. Đây đúng là tập sensitivity của `metrics/generate_paper_tables.dataset_audit`.
- **Tầng:** độ khó 3 mức (`metrics.io.DIFFICULTY_MAP`: Medium → Moderate, Challenging/Hard → Hard) × chuyên khoa
  đầu tiên (`medical_topic[0]`), tổng 177 tầng.
- **Phân bổ theo tỉ lệ**, làm tròn bằng phần dư lớn nhất. 138/177 tầng có ít nhất 1 câu. 22 chuyên khoa rất hiếm
  (tổng 36 câu trong quần thể) không có câu nào.
- **Phân bố so với quần thể:** Easy 32,5% (32,5%), Moderate 56,6% (56,5%), Hard 11,0% (11,0%). Chuyên khoa lệch
  nhiều nhất 0,07 điểm phần trăm.
- Runner đưa mọi dataset có tên bắt đầu bằng `vm14k_sub` vào **`code/results_ablation/`**, tách khỏi `results/`.

### 2.5 Cấu hình chung (giống nhau ở cả 4 điều kiện)

| Tham số | Giá trị | Đặt bằng |
|---|---|---|
| Code | commit chứa thí nghiệm này. `code_version` trong manifest **không được** có hậu tố `-dirty` | commit trước khi chạy |
| Parser | `parse_answer` của `3c549a2` | code |
| Engine | SGLang cho mọi model | server |
| Trần token | ROLES, MEDIUM, LONG, AGENT, DEBATE = **1024**; SHORT = 16, COMPLEX = 8 | `VIMED_TOKEN_CAP=1024` |
| Nhiệt độ | **0,3 cho mọi lời gọi**, kể cả CoT-SC (bảng chính dùng 0,5) | `VIMED_FORCE_TEMPERATURE=0.3` |
| Seed | 42 và 43 | `--seeds 42 43` |
| `think` | `False` (giống bảng chính) | code |

Manifest (`.meta.json`) của mỗi file ghi lại các giá trị thực dùng: `max_tokens`, `force_temperature`,
`cot_sc_temperature`, `code_version`. A và B ghi thêm `panel` (model, `model_id` và địa chỉ server của từng chuyên gia)
và `consult_input`.

**Khác biệt so với bảng chính VM14K (Bảng 1), phải khai khi so sánh:**
- trần token 1024, bảng chính là 512 / 600 / 400 / 300;
- nhiệt độ của CoT-SC 0,3, bảng chính là 0,5;
- parser mới;
- Llama-8B ở đây chạy SGLang AWQ, trong khi ở bảng chính khoảng 91% câu của 8B chạy Ollama Q4_K_M
  (`_RENAME_LLAMA8B_2026-11-10.md`).

### 2.6 Code và kiểm thử

| File | Vai trò |
|---|---|
| `frameworks/tier3/medagents_ablation.py` *(mới)* | `run_medagents_ablation(q, opts, model, panel, consult_input)`, cùng hai hàm bọc `run_medagents_hetero` (A) và `run_medagents_noreport` (B). Mỗi bước trong `audit_trail` ghi thêm `model` và `call` (tokens_in, tokens_out, **finish_reason**, temperature, max_tokens) |
| `infrastructure/config.py` | `VIMED_TOKEN_CAP`, `VIMED_FORCE_TEMPERATURE`, `ABLATION_FRAMEWORKS`; mỗi model trỏ được tới server riêng (`SGLANG_BASE_URL__<KEY>`); `results_ablation/` |
| `infrastructure/llm_client.py` | `get_last_call()` (thông tin lời gọi gần nhất của luồng); ép nhiệt độ; gọi theo địa chỉ server riêng của từng model |
| `runner/run_experiments.py` | Thêm 2 framework; `--seeds`; `check_endpoints()` cho A; manifest ghi `panel` và `consult_input`; `code_version` có hậu tố `-dirty` |
| `data/make_vm14k_subset.py` *(mới)* | Chọn tập con (mục 2.4) |
| `tests/test_ablation.py` *(mới)* | 17 test offline, không cần server |

`frameworks/` không sửa file nào có sẵn. `diff -rq ../ViMedAgent/code/frameworks code/frameworks` chỉ báo thêm
`medagents_ablation.py`.

Các test chứng minh:
- Với 3 × Qwen và có report, module chung gửi đi **đúng từng chữ, đúng tham số và đúng thứ tự** như `medagents`.
- Ở A, các lời gọi đi đúng thứ tự model: Qwen, Qwen, 70B, 8B, Qwen, Qwen, 70B, 8B, Qwen.
- Ở B, có 8 lời gọi và không có report. Prompt hội chẩn chỉ khác prompt chuẩn đúng ở khối report, và nhận được
  toàn văn kèm dòng "Final answer".
- Client ghi `finish_reason` và ép nhiệt độ. `check_endpoints` từ chối khi server phục vụ sai model hoặc khi hai model
  dùng chung một địa chỉ.

```bash
cd code && python -m pytest tests/test_ablation.py -q      # 17 passed
```

### 2.7 Hardware

- **Mốc, B và CoT-SC** chỉ cần một server Qwen3-32B-AWQ: `MEM_GB=27` trên GX10, hoặc `MEM_FRACTION=0.65` trên s96.
- **A cần 3 server SGLang cùng lúc:** Qwen (~27 GB), Llama-70B (`MEM_GB=60`), Llama-8B (chưa đo, ước khoảng
  12–16 GB). Tổng khoảng 100 GB, vừa một GX10 128 GB nếu không ai dùng chung; nếu không thì chia ra nhiều máy.
- Lệnh dựng server và ngân sách bộ nhớ đã đo nằm trong skill **`run-experiment`**. Đừng tự chế lệnh serve.

### 2.8 Command

**Biến môi trường, dùng cho mọi đợt:**

```bash
cd code
export VIMED_TOKEN_CAP=1024 VIMED_FORCE_TEMPERATURE=0.3
D=data/vm14k_sub2000.jsonl
```

**Thêm cho A:** mỗi model một server riêng. Runner kiểm `/v1/models` của từng server trước khi chạy và dừng nếu có
gì sai.

```bash
export SGLANG_BASE_URL__SGLANG_QWEN3_32B=http://<máy>:30000/v1
export SGLANG_BASE_URL__SGLANG_LLAMA3_3_70B=http://<máy>:30001/v1
export SGLANG_BASE_URL__SGLANG_LLAMA3_1_8B_AWQ=http://<máy>:30002/v1
```

**Mỗi điều kiện chạy cả hai seed, 42 và 43**, tức 4 điều kiện × 2 seed = 8 file kết quả. Seed 42 trong mục 2.4 là
seed để *chọn* 2.000 câu, khác với seed của lần chạy.

**Thứ tự chạy:** chạy xong **seed 42 cho cả 4 điều kiện trước**, rồi mới tới seed 43. Như vậy nếu phải dừng giữa
chừng thì vẫn có một bộ so sánh đầy đủ ở seed 42. Mỗi lệnh chạy trong một tmux riêng.

| Đợt | Seed | Lệnh | Server |
|---|---|---|---|
| 0, chạy thử | 42 | Như các đợt dưới, nhưng `--dataset data/vm14k_sub_pilot20.jsonl`. File thử tạo bằng `head -n 20 data/vm14k_sub2000.jsonl > data/vm14k_sub_pilot20.jsonl` | cả 3 |
| 1 | **42** | `python -m runner.run_experiments --dataset $D --framework medagents --model sglang-qwen3-32b --seeds 42`<br>`... --framework cot_sc ... --seeds 42` · `... --framework medagents_noreport ... --seeds 42` | Qwen |
| 2 | **42** | `python -m runner.run_experiments --dataset $D --framework medagents_hetero --model sglang-qwen3-32b --seeds 42` | Qwen + 70B + 8B |
| 3 | **43** | Như đợt 1, đổi thành `--seeds 43` | Qwen |
| 4 | **43** | Như đợt 2, đổi thành `--seeds 43` | Qwen + 70B + 8B |

Nếu không cần thứ tự này, mỗi điều kiện chạy luôn cả hai seed bằng một lệnh `--seeds 42 43`. Runner sẽ chạy 42 rồi 43.

Ba điều kiện chỉ dùng Qwen (đợt 1 và 3) chạy song song trên cùng một server Qwen được, vì mỗi lời gọi độc lập với nhau.

**Quy tắc chạy:**
- Giữa các đợt không đổi commit, không đổi phiên bản SGLang, không đổi weights, không đổi tham số khởi động server.
- Bị ngắt thì chạy lại **đúng lệnh cũ**: runner tự bỏ qua các ID đã xong.
- **Không** chạy lại riêng những câu không ra đáp án. Làm vậy là lấy mẫu lại có chọn lọc và làm lệch kết quả. Đây đúng
  là lỗi đã gặp ở MDAgents USMLE.

### 2.9 Kiểm tra sau mỗi đợt

Với mỗi file `results_ablation/tier*/<fw>/<fw>__sglang-qwen3-32b__vm14k_sub2000__s<seed>.jsonl`:

1. **Đủ dữ liệu:** `n = 2000`, không trùng ID, ID khớp `vm14k_sub2000_ids.txt`.
2. **Không sót lời gọi:** `api_calls == n_llm_calls` ở mọi dòng; `n_usage_missing = 0`. Số lời gọi mỗi câu: 9 (mốc, A),
   8 (B), 5 (CoT-SC).
3. **Manifest đúng cấu hình:** `max_tokens` có `MAX_TOKENS_AGENT/DEBATE/LONG/MEDIUM/ROLES = 1024`;
   `force_temperature = 0.3`; `code_version` **không có** `-dirty`. A và B thêm `panel` đúng như mục 2.1.
4. **Trần token thật sự có hiệu lực:** ở A và B, thống kê `call.finish_reason` theo bước. Ghi lại tỉ lệ `"length"`
   (chạm trần) của từng bước. Ở mốc và CoT-SC (code chuẩn, không có `finish_reason`), xác nhận `tokens_out` có giá trị
   vượt 600 hoặc 300. Bước kiểm này để tránh lặp lại chuyện của máy `gx10-4T`, nơi chỉ LONG được nâng.
5. **Đúng model (A):** mỗi bước trong `audit_trail` có `model` đúng như mục 2.2.

### 2.10 Chỉ số (pre-registered)

Định nghĩa phải **giống hệt Bảng 2–3 của chính văn**; đối chiếu lại trước khi viết script.

Ở mốc, A và B:
- **Nhất trí trước → sau:** tỉ lệ câu mà cả 3 chuyên gia cùng một đáp án hợp lệ, ở bước phân tích
  (`round1_answers`) so với sau hội chẩn (`round2_answers`).
- **Accuracy trong nhóm nhất trí:** trong các câu nhất trí, bao nhiêu phần trăm đáp án nhất trí là đúng (trước và sau).
- **Nhất trí sai:** tỉ lệ câu cả 3 cùng một đáp án nhưng sai.
- **NMAR:** trong các câu mà đáp án đúng chỉ thuộc thiểu số, bao nhiêu phần trăm đáp án cuối đi theo đa số sai. Repo có
  hai cài đặt, `metrics/collaboration.nmar` và `metrics/compute.compute_nmar`. Dùng đúng hàm của bảng chính.
- **AUROC / AURC trước và sau:** dùng mức đồng thuận (3/3, 2/3, 1/1/1) làm điểm tin cậy cho đáp án cuối.
- **Accuracy cuối so với CoT-SC** trên cùng 2.000 câu: hiệu ghép cặp theo (câu, seed), khoảng tin cậy 95% bằng cluster
  bootstrap 10.000 lần, theo `generate_paper_tables.paired_row`.

Mọi chỉ số tính trên từng seed, rồi báo trung bình ± SD qua 2 seed. Script tính chỉ số **chưa viết**.

### 2.11 Δ so với thí nghiệm trước

So với Bảng 1 VM14K (toàn bộ 11.094 câu), thí nghiệm này khác ở các điểm sau:
- chạy trên tập con 2.000 câu;
- trần token 1024;
- ép mọi lời gọi về nhiệt độ 0,3;
- dùng parser mới;
- A thay model của chuyên gia; B bỏ report và cho đọc toàn văn.

Mỗi điều kiện A, B chỉ được so với **mốc chạy cùng cấu hình** trong thí nghiệm này, không so với Bảng 1.

## 3. What I did (log)

- **10/11/2026:** thầy nhận xét việc VM14K và USMLE chạy khác cấu hình. Thầy đề ra thiết kế 4 điều kiện trên cùng
  2.000 câu, cùng một cấu hình.
- **11/11/2026:**
  - Chốt cấu hình với thầy: trần 1024, ép nhiệt độ 0,3, A gán vai theo thứ tự cố định, B = B2.
  - Tạo tập con (seed 42).
  - Viết `medagents_ablation.py`, sửa config, client và runner; 17 test offline qua.
  - Toàn bộ test của repo qua (41 test). Chưa có lời gọi model nào.

## 4. Results

*(chưa chạy)*

## 5. Error analysis

*(chưa chạy)*

## 6. Conclusion

*(chưa chạy)*

## 7. Next step

- Viết script tính các chỉ số ở mục 2.10.
- Nếu B cho thấy tín hiệu bất đồng được giữ, cân nhắc chạy **B1** (`analyses_truncated`) để tách ảnh hưởng của việc
  bỏ tóm tắt khỏi ảnh hưởng của việc đọc toàn văn. Thêm khoảng 32.000 lời gọi.
- Cân nhắc A với **gán vai xoay vòng** nếu cần tách ảnh hưởng của model khỏi ảnh hưởng của vai.

## 8. Artifacts & links

- **Dữ liệu:** `code/data/vm14k_sub2000.jsonl`, `vm14k_sub2000_ids.txt`, `vm14k_sub2000.manifest.json`
- **Kết quả:** `code/results_ablation/` (không đưa lên git)
- **Code:** `code/frameworks/tier3/medagents_ablation.py`, `code/tests/test_ablation.py`
- **Liên quan:** `docs/MedAgent_Research_Narrative_2026-11-04.md` (Q2b–Q2c); exp `005` (MedAgents-VI); exp `007`
  (dựng SGLang); `_RENAME_LLAMA8B_2026-11-10.md`
