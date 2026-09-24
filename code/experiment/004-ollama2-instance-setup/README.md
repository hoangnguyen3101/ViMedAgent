# Experiment 004-ollama2-instance-setup — Instance Ollama riêng (hoangnv) + fix bug think=False bị bỏ qua

<!-- theo chuẩn experiment/Template/README.md -->

- **Date:** 24-26/07/2026
- **Author:** hoangnv (qua Claude Code)
- **Status:** done (setup + fix) — cần chạy lại các file đã dính bug
- **Base model:** `qwen3-32b-ctx8k` qua provider mới `ollama2`
- **Method:** hạ tầng — cài thêm 1 instance Ollama độc lập, không phải training

## 1. Motivation / Hypothesis

Server Ollama dùng chung (11434, `cuongvt`) có `OLLAMA_MAX_LOADED_MODELS=1` — chỉ giữ
1 model tại 1 thời điểm. Khi cần chạy Qwen3-32B song song với job Llama-3.1-8B (8 tier3
frameworks, full dataset, đã chạy nhiều ngày), 2 bên tranh nhau slot duy nhất → unload/reload
liên tục → timeout 100% (đã thấy lặp lại nhiều lần trong session này với cả Qwen3-32B lẫn
Llama-3.3-70B).

VRAM thực tế còn dư nhiều (Llama-8B chỉ chiếm 11GB/48GB) — nên hướng giải quyết: cài
**1 instance Ollama thứ 2, độc lập hoàn toàn**, do `hoangnv` tự quản lý, bind cổng khác
(11435), quản lý `OLLAMA_MAX_LOADED_MODELS` riêng — để 2 model có thể cùng resident trong
VRAM và chạy song song thật sự, không tranh chấp.

## 2. Setup

- **Vị trí cài:** `/storage/hoangnv/ollama-local/` (**không** phải `/home` — disk `/home`
  chỉ còn 34GB trống lúc quyết định, `/storage` có 119GB trống và đã sẵn thư mục riêng của
  `hoangnv`, không cần quyền gì thêm).
- **Binary:** tải trực tiếp từ GitHub release (`ollama/ollama`), **không** dùng script cài
  đặt hệ thống (không cần sudo/docker — đã kiểm tra `/dev/nvidia*` có quyền `crw-rw-rw-`
  nên user thường vẫn dùng GPU trực tiếp được).
- **Cấu hình server riêng** (chạy trong tmux `ollama-hoangnv`):
  ```bash
  OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_MODELS=/storage/hoangnv/ollama-local/models \
  OLLAMA_NUM_PARALLEL=4 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  LD_LIBRARY_PATH=/storage/hoangnv/ollama-local/lib/ollama:$LD_LIBRARY_PATH \
  /storage/hoangnv/ollama-local/bin/ollama serve
  ```
- **Model:** `ollama pull qwen3:32b` (dùng chính binary/instance riêng để pull — server
  nào pull thì server đó có model, không share được blob giữa 2 instance khác thư mục),
  rồi `ollama create qwen3-32b-ctx8k -f infrastructure/ollama_modelfiles/Modelfile.qwen3-32b`
  (giống hệt tag đã dùng ở instance chung, để so sánh công bằng).
- **Wiring vào `config.py`:** thêm provider mới `"ollama2"` (`get_api_key`, `get_base_url`
  trỏ `http://127.0.0.1:11435/v1`) và model `"ollama2-qwen3-32b"`.

## 3. What I did (log)

1. Lần đầu tải binary bị lỗi — URL `ollama.com/download/ollama-linux-amd64.tgz` redirect
   sang GitHub nhưng asset đó **không tồn tại** (GitHub release đổi định dạng file thành
   `.tar.zst`, không còn `.tgz`). Sửa lại đúng URL từ GitHub API
   (`api.github.com/repos/ollama/ollama/releases/latest`).
2. Cài đặt thành công (bản mới nhất lúc đó: **v0.32.3**), start server, pull `qwen3:32b`
   (20GB), tạo tag `ctx8k`, verify GPU nhận diện đúng (Quadro RTX 8000, 37.5GiB khả dụng
   sau khi trừ phần Llama-8B đang chiếm).
3. Test request đơn + tăng `OLLAMA_NUM_PARALLEL` từ 1 lên 4 — chạy song song với job
   Llama-8B trên instance chung, xác nhận **không tranh chấp** (2 model cùng 100% GPU,
   tổng ~33-40GB/48GB).
4. Chạy `medagents x ollama2-qwen3-32b` full pilot_100 × 3 seed — **hoàn thành nhưng
   0.00% accuracy tuyệt đối** (mọi `answer` đều `null`). Không có `ERROR` nào được log —
   tức các call đều "thành công" nhưng trả về nội dung rỗng ở bước cụ thể.
5. Điều tra: đọc trực tiếp `results_test/medagents__ollama2-qwen3-32b__pilot_100__s42.jsonl`
   → phát hiện bước **"report" (Step 3) luôn `raw: ""`**, làm hỏng toàn bộ chuỗi sau đó
   (consultation → decision đều `null`).
6. Test đơn lẻ qua `curl` trực tiếp (prompt đơn giản) → **không** tái hiện được — ra kết
   quả bình thường. Nghi ngờ ban đầu: bug riêng của bản Ollama mới (0.32.3) so với bản cũ
   (0.17.4) đang chạy ổn định ở instance chung.
7. Tái hiện được bug qua đúng code path thật (`call_llm()` + `prompt_analysis()` +
   `prompt_report()` với dữ liệu câu hỏi thật) — thấy **cả bước "analysis" đầu tiên lẫn
   "report" đều rỗng** trong 1 lần chạy.
8. Theo yêu cầu, hạ instance riêng xuống đúng **v0.17.4** (khớp bản ổn định của
   `cuongvt`) — xoá binary/lib cũ, giữ nguyên `models/` (không tốn công tải lại 20GB).
9. Chạy lại đúng test tái hiện trên v0.17.4 — **vẫn rỗng ở bước "report"** dù các bước
   "analysis" đều ra kết quả tốt. → loại trừ giả thuyết "do version Ollama".
10. Cô lập nguyên nhân: build lại đúng messages thật của bước "report" (từ `prompt_report()`),
    gọi thẳng bằng `urllib.request` tới `/api/chat` (bỏ qua `call_llm()`) → **ra kết quả
    đúng, đầy đủ**. Vậy bug nằm trong code Python (`call_llm()`), không phải server/model.
11. Đọc lại `infrastructure/llm_client.py` dòng 79:
    ```python
    if provider == "ollama" and think is not None:   # <-- thiếu "ollama2"
    ```
    → mọi lời gọi `think=False` qua provider `ollama2` **rơi vào nhánh OpenAI-SDK**
    (endpoint `/v1/chat/completions`), vốn **âm thầm bỏ qua `think`** — tái lập chính xác
    bug thinking gốc (`001-tier1-disable-thinking`), nhưng lần này do chính lỗi thêm
    provider mới của tôi, không phải do Ollama.
12. Sửa: `if provider in ("ollama", "ollama2") and think is not None:`.
13. Verify lại đúng chuỗi test (10) qua `call_llm()` sau khi sửa → bước "report" ra
    683 ký tự nội dung hợp lệ.
14. Rà soát toàn bộ `results_test/*ollama2*` — mọi file (không riêng `medagents`) đều
    dính null rate cao (29-58%), backup hết vào `_pre_ollama2_gate_fix_backup/`.

## 4. Results

**VRAM song song (2 instance cùng lúc):**
```
Instance chung (11434): llama3.1-8b-ctx8k   11GB   100% GPU
Instance riêng (11435): qwen3-32b-ctx8k     28GB   100% GPU  (NUM_PARALLEL=4)
Tổng: ~39GB / 48GiB — chạy song song ổn định, không tranh chấp
```

**Bug think=False bị bỏ qua (trước/sau fix), test qua đúng chuỗi `medagents` step 1-3:**

| | Trước fix | Sau fix |
|---|---|---|
| Bước "report" (step 3) | `raw: ""` (rỗng) | 683 ký tự, nội dung hợp lệ |
| `medagents` full run (100 câu × 3 seed) | Overall **0.00% ± 0.00%** | _(chưa chạy lại)_ |

**File `results_test/` bị nhiễm bug, đã backup** (`_pre_ollama2_gate_fix_backup/`):
`medagents`, `medagents_vi` (s42), `cot`, `cot_sc`, `self_refine`, `voting_ensemble`,
`zero_shot` — tất cả seed đã chạy qua `ollama2-qwen3-32b` trước thời điểm fix (26/07/2026).

## 5. Error analysis

- Root cause thực sự là lỗi lập trình khi thêm provider mới (`ollama2`), không phải lỗi
  Ollama hay model. Bài học: `_call_ollama_native()` cần **danh sách provider dùng chung
  logic native `/api/chat`** ở một chỗ duy nhất (constant/set), tránh phải nhớ sửa mọi nơi
  mỗi khi thêm provider Ollama-based mới — hiện tại đang là 2 chỗ hard-code riêng lẻ
  (`get_api_key`, `get_base_url`, và điều kiện ở `call_llm`) — rủi ro lặp lại nếu thêm
  `ollama3` sau này mà quên 1 trong 3 chỗ.
- Vì sao "phân tích" (analysis) step ít bị lộ hơn "report": `max_tokens` của analysis
  (600) thường đủ dư để thinking-ẩn vẫn còn chừa vài chục token cho câu trả lời thật; còn
  "report" (`max_tokens=300`) sát hơn nên thinking-ẩn dễ ăn hết trước khi model kịp viết
  báo cáo — đúng cơ chế đã ghi trong `001-tier1-disable-thinking`.

## 6. Conclusion

Instance Ollama riêng (`ollama2`) hoạt động đúng như thiết kế (chạy song song thật, không
tranh chấp GPU/model-slot với instance chung) — nhưng **toàn bộ dữ liệu chạy qua nó trước
26/07/2026 không đáng tin**, do bug ở lớp wiring Python, không phải do ý tưởng dùng instance
riêng sai.

## 7. Next step

- Chạy lại toàn bộ 7 file đã backup (`medagents`, `medagents_vi`, `cot`, `cot_sc`,
  `self_refine`, `voting_ensemble`, `zero_shot`) trên `pilot_100` với `ollama2-qwen3-32b`.
- Cân nhắc refactor `llm_client.py`: gom danh sách "Ollama-based provider" (hiện là
  `ollama`, `ollama2`) vào 1 hằng số/tuple dùng chung cho `get_api_key`, `get_base_url`,
  và điều kiện `_call_ollama_native`, thay vì lặp lại chuỗi so sánh ở 3 nơi.

## 8. Artifacts & links

- Code thay đổi: [`infrastructure/config.py`](../../infrastructure/config.py) (provider
  `ollama2` + model `ollama2-qwen3-32b`), [`infrastructure/llm_client.py:79`](../../infrastructure/llm_client.py)
  (fix điều kiện provider)
- Backup kết quả nhiễm bug: [`results_test/_pre_ollama2_gate_fix_backup/`](../../results_test/_pre_ollama2_gate_fix_backup/)
- Related: `001-tier1-disable-thinking` (bug gốc), `003-tier3-disable-thinking`
