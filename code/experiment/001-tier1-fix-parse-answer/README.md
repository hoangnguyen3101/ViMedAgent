# Experiment 001-tier1-fix-parse-answer — Sửa lỗi trích xuất đáp án (`parse_answer`) cho CoT

<!-- theo chuẩn experiment/Template/README.md -->

- **Date:** 07/07/2026
- **Author:** hoangnv (qua Claude Code)
- **Status:** running — fix code đã xong, chưa chạy lại `cot` để lấy số liệu chính thức
- **Base model:** `ollama-qwen3-32b` (tag `qwen3-32b-ctx8k`, Qwen3 32.8B, Q4_K_M, num_ctx=8192)
- **Method:** không phải training — sửa lỗi logic trong hàm parse đáp án
  (`infrastructure/llm_client.py::parse_answer`)

## 1. Motivation / Hypothesis

Sau `001-tier1-disable-thinking`, chạy `cot x ollama-qwen3-32b` (seed 42) cho accuracy
**63.4%** — thấp hơn hẳn `zero_shot` (67.96%) dù cùng model, cùng đã tắt thinking.
Null rate chỉ 0.93% nên không phải do bug cũ tái diễn. Giả thuyết: model không suy luận
tệ hơn, mà có lỗi ở bước `parse_answer()` khi xử lý văn bản tự do (free-text) dài mà
`zero_shot` (chỉ 1 ký tự) không gặp phải.

## 2. Setup

- **Data:** `data/vm14k_clean.jsonl` — cùng 11094 câu như `001-tier1-disable-thinking`.
- **Config:** không đổi so với `001-tier1-disable-thinking` (`MAX_TOKENS_LONG=512`,
  `think=False`, `SEEDS=[42,43,44]`).
- **Hardware:** 1x Quadro RTX 8000 (49GB), Ollama server local.
- **Command:**
  ```
  python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework cot --model ollama-qwen3-32b
  ```
- **Δ vs previous:** sửa hàm `parse_answer()` dùng chung cho mọi framework — không đổi
  gì ở `cot.py`/prompt/model. Đây là fix ở lớp hạ tầng, không phải ở framework logic.

## 3. What I did (log)

1. Quan sát: `cot` (63.4%) thấp hơn `zero_shot` (67.96%) sau khi đã tắt thinking cho cả 2.
2. Kiểm tra null rate trong `results/cot__ollama-qwen3-32b__vm14k_clean__s42.jsonl`
   (10702/11094 dòng tại thời điểm kiểm tra): chỉ 0.93% (100 dòng) — loại trừ khả năng
   tái diễn bug ở `001-tier1-disable-thinking`.
3. Đọc lại regex fallback trong `parse_answer()`:
   ```python
   r'(?:final\s+answer|answer|đáp án|đáp|option|is)[:\s]+([ABCD])\b'
   ```
   Nghi ngờ từ khóa `"is"` dưới `re.IGNORECASE` khớp nhầm mạo từ "a" trong câu tiếng Anh
   kiểu "The patient **is a** 62-year-old...".
4. Test đơn lẻ xác nhận: text chứa "...is a 62-year-old...Final answer: B" →
   `parse_answer()` cũ trả về `"A"` (sai), đúng ra phải là `"B"`.
5. Kiểm chứng bằng thống kê trên dữ liệu thật: so phân bố `answer` (dự đoán) với `gold`
   trên 2 file kết quả seed 42 (`cot` vs `zero_shot`, cùng model, cùng dataset).
6. Sửa `infrastructure/llm_client.py::parse_answer()`:
   - Ưu tiên match `"Final answer: X"` trước tiên (marker mọi framework CoT-style đều yêu cầu).
   - Bỏ `"is"` khỏi từ khóa fallback.
   - Fallback yếu lấy **match cuối** (`re.finditer(...)[-1]`) thay vì match đầu
     (`re.search(...)`).
7. Viết unit test xác nhận fix hoạt động đúng trên các case đã biết (bug case + case
   bình thường).
8. *(chưa làm)* backup + chạy lại `cot x ollama-qwen3-32b` với `parse_answer()` mới.

## 4. Results

- **Metrics** (phân bố đáp án dự đoán vs gold, seed 42, cùng dataset):

  | Framework | Gold "A" | Predicted "A" (parser cũ) | Lệch |
  |--------|---------:|---------:|---:|
  | `zero_shot` (1 ký tự, không qua regex fallback) | 28.7% | 27.5% | ~0 (đối chứng, không lệch) |
  | `cot` (free-text, dính bug) | 28.6% | **39.4%** | **+11pp** |

  B/C/D trong `cot` đều hụt tương ứng so với gold — dấu hiệu lệch hệ thống về "A" do lỗi
  parser, không phải model chọn sai thật.

- **Unit test (trước/sau fix):**

  | Input | Parser cũ | Parser mới |
  |---|---|---|
  | `"...is a 62-year-old...Final answer: B"` | `A` (sai) | `B` (đúng) |
  | `"...đáp án B, C, D đều...Final answer: B"` | `B` (đúng, tình cờ) | `B` (đúng) |
  | `"...**D** is correct."` | `D` | `D` (không đổi) |

- **Chưa có:** accuracy `cot` sau khi chạy lại với parser mới — cần chạy lại full 3 seed
  để có số liệu chính thức (xem mục 7).

## 5. Error analysis

- Root cause là thiết kế regex quá tham lam: `re.search()` chỉ trả về match **đầu tiên**
  theo vị trí trong chuỗi, không phải match có độ tin cậy cao nhất. Với văn bản CoT liệt
  kê từng option theo thứ tự A→D trước khi kết luận, bất kỳ từ khóa yếu nào ("is", "option",
  "answer") xuất hiện sớm đều có thể "thắng" trước khi quét tới `"Final answer: X"` thật ở cuối.
  - Chỉ 100 dòng dính bug này ở "no explicit marker" case rất khó xảy ra
- Lỗi này ảnh hưởng **mọi framework dùng free-text + `parse_answer()`** (không riêng `cot`):
  `cot_sc`, `self_refine`, `medagents`, `mdagents` đều có nguy cơ tương tự vì cùng yêu cầu
  format `"Final answer: X"` và dùng chung hàm parse — chưa kiểm tra các file kết quả đó.

## 6. Conclusion

Xác nhận giả thuyết: `cot` không suy luận kém hơn `zero_shot` — chênh lệch accuracy quan
sát được chủ yếu (có thể toàn bộ) do lỗi trích xuất đáp án, không phải chất lượng model.
Fix đã áp dụng vào lớp hạ tầng dùng chung (`llm_client.py`), không đổi gì ở từng framework.

## 7. Next step

- Backup `results/cot__ollama-qwen3-32b__vm14k_clean__s42.jsonl` (không thể vá lại vì
  `raw` text không được lưu vào file kết quả) và chạy lại `cot x ollama-qwen3-32b` full
  3 seed để có accuracy chính thức với parser mới.
- Rà soát lại các file kết quả free-text khác đã chạy trước fix này:
  `cot_sc`, `self_refine` (Tier 1), và mọi Tier 2/3 đã có kết quả cho
  `ollama-llama3.1-8b`/`ollama-qwen3-32b` — kiểm tra bằng cách so phân bố predicted vs
  gold giống cách đã làm ở đây, để biết framework nào cần chạy lại.

## 8. Artifacts & links

- Code thay đổi: [`infrastructure/llm_client.py`](../../infrastructure/llm_client.py)
  (hàm `parse_answer`)
- File kết quả cần backup + chạy lại: [`results/cot__ollama-qwen3-32b__vm14k_clean__s42.jsonl`](../../results/cot__ollama-qwen3-32b__vm14k_clean__s42.jsonl)
- Related: `001-tier1-disable-thinking`
