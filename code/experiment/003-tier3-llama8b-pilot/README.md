# Experiment 003-tier3-llama8b-pilot — Test thử Tier 3 (multi-agent) với Llama-3.1-8B trên pilot_100

<!-- theo chuẩn experiment/Template/README.md -->
<!-- Status: planned — chưa chạy. hoangnv sẽ tự chạy, Claude Code KHÔNG tự thực thi lệnh này. -->

- **Date:** 13/07/2026
- **Author:** hoangnv
- **Status:** planned
- **Base model:** `ollama-llama3.1-8b` (tag `llama3.1-8b-ctx8k`, Llama 3.1 8B Instruct, local Ollama)
- **Method:** không phải training — chạy inference qua các framework multi-agent Tier 3

## 1. Motivation / Hypothesis

Trước giờ mới có kết quả Tier 3 (`medagents`) cho **`llama-3.1-8b` qua Groq** (cloud,
`results_test/medagents__llama-3_1-8b__pilot_100__s*.jsonl`), chưa có lần chạy nào cho
bản **local Ollama** (`ollama-llama3.1-8b`). Cần test thử nhanh trên `pilot_100.jsonl`
(100 câu, không phải full 11094) để:
- Xác nhận model + framework chạy được (setup đúng, không lỗi).
- Có con số sơ bộ trước khi quyết định chạy full `vm14k_clean.jsonl`.

Lưu ý: Llama-3.1-8B **không có** cơ chế hidden thinking (khác Qwen3/DeepSeek-R1) — xem
`001-tier1-disable-thinking` — nên không cần lo về bug null-answer do thinking đã gặp ở
Qwen3. Tuy vậy, `parse_answer()` đã được sửa ở `001-tier1-fix-parse-answer`; các
framework Tier 3 (free-text, nhiều bước) cũng nên được rà soát tương tự sau khi có kết quả.

## 2. Setup

- **Data:** [`data/pilot_100.jsonl`](../../data/pilot_100.jsonl) — 100 câu (tập con của
  `vm14k_clean.jsonl`, dùng để test nhanh).
- **Config:** không đổi so với `infrastructure/config.py` mặc định
  (`DEFAULT_TEMPERATURE=0.3`, `SEEDS=[42,43,44]`, `NUM_WORKERS=4`, các `MAX_TOKENS_*`
  riêng cho Tier 3: `MAX_TOKENS_AGENT=600`, `MAX_TOKENS_DEBATE=400`, `MAX_TOKENS_ROLES=512`).
- **Hardware:** 1x Quadro RTX 8000 (49GB), Ollama server local.
- **Framework:** Tier 3 gồm 5 framework độc lập — chưa rõ bạn muốn chạy hết hay chọn
  1-2 cái, xem mục 3 để chọn lệnh tương ứng.
- **Δ vs previous:** model mới (`ollama-llama3.1-8b`) + dataset nhỏ (`pilot_100`, không
  phải full) — lần đầu tiên chạy tổ hợp này.

### ⚠️ Việc cần làm trước khi chạy (model chưa có trên máy)

Kiểm tra hiện tại (`ollama list`) cho thấy `llama3.1:8b` **chưa được pull**, và tag
`llama3.1-8b-ctx8k` (bắt buộc — xem comment trong `infrastructure/config.py` về việc
context 40960 mặc định sẽ đẩy phần lớn layer sang CPU nếu không dùng tag `-ctx8k`)
**chưa được tạo**. Chạy 2 lệnh sau trước:

```bash
ollama pull llama3.1:8b
ollama create llama3.1-8b-ctx8k -f infrastructure/ollama_modelfiles/Modelfile.llama3.1-8b
```

Kiểm tra lại bằng `ollama list` — phải thấy `llama3.1-8b-ctx8k` trong danh sách trước khi
chạy lệnh ở mục 3.

## 3. Lệnh chạy

Chạy trong tmux session `agent` (đúng conda env + thư mục `code/` đã dùng các lần trước).
Mỗi lệnh chỉ chạy **1 framework**, cả 3 seed (42/43/44), trên `pilot_100.jsonl`:

```bash
# Tier 3 — chọn 1 hoặc chạy lần lượt cả 5:
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework medagents     --model ollama-llama3.1-8b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework medagents_vi  --model ollama-llama3.1-8b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework mdagents      --model ollama-llama3.1-8b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework mdagents_vi   --model ollama-llama3.1-8b
python -m runner.run_experiments --dataset data/pilot_100.jsonl --framework mad_debate    --model ollama-llama3.1-8b
```

Lưu ý:
- `--framework` chỉ nhận **1 giá trị** mỗi lần (không có cú pháp "chạy hết Tier 3" — muốn
  hết thì chạy tuần tự 5 lệnh trên, hoặc dùng `--all` để chạy **toàn bộ** ma trận
  framework×model, không khuyến khích cho 1 lần test thử).
- 100 câu × 3 seed × nhiều API call/câu (Tier 3 là multi-agent, mỗi câu tốn nhiều lượt
  gọi LLM hơn Tier 1 nhiều) — nên sẽ chậm hơn per-câu so với `zero_shot`/`cot`, dù tổng
  số câu nhỏ (pilot).
- Nếu Ctrl-C để dừng: chỉ hủy **seed hiện tại** rồi tự nhảy seed kế tiếp (xem
  `001-tier1-disable-thinking`, mục 3, bước 9) — muốn dừng hẳn phải
  `kill -TERM <pid>` (tìm pid bằng `ps aux | grep run_experiments`).
- Kết quả ghi vào `results_test/<framework>__ollama-llama3_1-8b__pilot_100__s<seed>.jsonl`.
  `runner/run_experiments.py::result_dir()` tự route: dataset có stem bắt đầu bằng
  `"pilot"` → `results_test/`, còn lại (vd `vm14k_clean`) → `results/` như cũ
  (xem `infrastructure/config.py::RESULTS_TEST_DIR`, thêm trong lúc chuẩn bị experiment này).

## 4. What I did (log)

_(để trống — điền lại sau khi chạy: các bước thực tế đã làm, sự cố gặp phải, sửa
giữa chừng nếu có)_

## 5. Results

_(để trống — điền bảng accuracy theo framework/seed sau khi chạy xong)_

| Framework | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|---|---|---|---|---|
| medagents | | | | |
| medagents_vi | | | | |
| mdagents | | | | |
| mdagents_vi | | | | |
| mad_debate | | | | |

## 6. Error analysis

_(để trống — điền sau khi có kết quả: null rate, so phân bố predicted vs gold answer
giống cách đã làm ở `001-tier1-fix-parse-answer` để phát hiện sớm nếu `parse_answer()`
vẫn còn lệch với văn bản multi-agent dài hơn CoT)_

## 7. Conclusion

_(để trống)_

## 8. Next step

_(để trống — ví dụ: nếu ổn thì chạy full `vm14k_clean.jsonl`; nếu phát hiện bug thì mở
experiment mới `003-tier3-<fix-gì-đó>`)_

## 9. Artifacts & links

- Data: [`data/pilot_100.jsonl`](../../data/pilot_100.jsonl)
- Model setup: [`infrastructure/ollama_modelfiles/Modelfile.llama3.1-8b`](../../infrastructure/ollama_modelfiles/Modelfile.llama3.1-8b)
- Kết quả (sau khi chạy): `results_test/medagents__ollama-llama3_1-8b__pilot_100__s*.jsonl`
  (và tương tự cho các framework khác)
- Related: `001-tier1-disable-thinking`, `001-tier1-fix-parse-answer`
