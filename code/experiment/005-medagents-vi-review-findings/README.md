# Experiment 005-medagents-vi-review-findings — Rà soát mdagents_vi / medagents_vi (bug + claim sai + simplification chưa ghi rõ)

<!-- theo chuẩn experiment/Template/README.md -->

- **Date:** 16-17/08/2026
- **Author:** hoangnv (qua Claude Code)
- **Status:** review xong, **CHƯA SỬA CODE** — để lại note này làm việc sau họp
- **Base model:** `ollama-llama3.1-8b` (dùng để đo, vì đã có đủ 3 seed hoàn chỉnh cho cả 4 framework liên quan)
- **Method:** không phải training — audit code + đối chiếu dữ liệu đã chạy + đối chiếu source gốc 2 paper (MDAgents NeurIPS'24, MedAgents ACL'24)

## 1. Motivation / Hypothesis

Ý tưởng ban đầu: `mdagents_vi`/`medagents_vi` thay bước LLM tự phân loại (độ khó / chuyên khoa)
bằng nhãn có sẵn trong VM14K (`difficulty_level`, `medical_topic`) để **đỡ tốn API call**.
Trước khi trình bày ở buổi họp, cần kiểm chứng lại: (1) ý tưởng có tiết kiệm call thật không,
(2) code có bug không, (3) có đang lệch khỏi thuật toán gốc theo cách cần ghi rõ trong paper không.

## 2. Setup

Không chạy job mới — dùng lại kết quả đã có sẵn (`results/tier3/{mdagents,mdagents_vi,medagents,medagents_vi}/*ollama-llama3_1-8b*vm14k_clean*s{42,43,44}.jsonl`,
3 seed đầy đủ cho `mdagents`/`mdagents_vi`, 2/3 seed cho `medagents_vi`) + đọc trực tiếp source
code 4 file trong `frameworks/tier3/` + đọc source gốc trên GitHub
(`gersteinlab/MedAgents`: `utils.py`, `prompt_generator.py`).

## 3. What I did (log)

1. Đo tỉ lệ LLM tự phân loại độ khó (`mdagents`, bước complexity_check) khớp với `difficulty_level`
   thật của dataset — parse `audit_trail[0]` của toàn bộ 33.282 dòng (3 seed).
2. Từ đó tính lại số API call trung bình thực tế của `mdagents` vs `mdagents_vi` (field `api_calls`
   có sẵn trong mỗi dòng kết quả) — so với con số lý thuyết ghi trong docstring.
3. Làm tương tự cho `medagents` vs `medagents_vi` (bước expert gathering theo topic).
4. Trong lúc so `roles` field của `medagents_vi`, phát hiện nhiều team có role trùng lặp — viết
   script đếm `len(roles) != len(set(roles))` trên `audit_trail[0]['roles']` của 13.702 dòng
   (seed 42+43) để định lượng.
5. Đọc lại hàm `get_roles_from_topic()` + đoạn pad trong `run_medagents_vi()`
   ([medagents_vi.py:152-156](../../frameworks/tier3/medagents_vi.py#L152-L156)) → xác định
   nguyên nhân: pad thêm `"General Practitioner"` không check đã có trong list chưa.
6. Kiểm tra bản gốc MedAgents (`gersteinlab/MedAgents` trên GitHub) — đọc `fully_decode()` trong
   `utils.py` và `get_question_domains_prompt()`/hằng số `NUM_QD=5`, `NUM_OD=2` trong
   `prompt_generator.py` — xác nhận bản gốc **không** ép số domain cố định (không có logic pad
   kiểu này), nên bug là do phần code viết thêm ở ViMedAgent, không kế thừa từ paper.
7. Đọc lại Algorithm 1 (Collaborative Consultation) trong paper MedAgents, đối chiếu với
   `Step 4` trong `medagents.py`/`medagents_vi.py` — xác nhận code hiện tại **không** implement
   vòng lặp `while nocon_flag and n_try < t` (vote yes/no trên report rồi revise tới khi đồng
   thuận) mà chỉ làm 1 vòng cố định mỗi expert vote thẳng A/B/C/D.
8. Đối chiếu trực tiếp với **nguyên văn paper** (arxiv 2311.10537, không chỉ code) để tránh hiểu
   nhầm là "task khác nhau": paper gốc đánh giá trên 9 dataset, **toàn bộ đều multiple-choice/closed-set**
   (MedQA, MedMCQA, PubMedQA, MMLU 6 subtask y khoa) — cùng loại task với VM14K, không phải
   "final answer tự do". Khác biệt thật sự nằm ở **cơ chế nội bộ của bước tư vấn**, không phải
   loại task — xem chi tiết ở mục 5.

## 4. Results

**(a) LLM tự phân loại độ khó (`mdagents`) so với `difficulty_level` thật — 33.282 câu, model `ollama-llama3.1-8b`:**

```
Ty le khop: 42.2%
Phan bo LLM tu cham:        Low 82.0% / Moderate 16.6% / High 1.4%
Phan bo difficulty_level that: Low 32.0% / Moderate 56.9% / High 11.0%
```
→ Model gần như luôn chấm "Low" bất kể độ khó thật.

**(b) So sánh API call thực tế (đo trực tiếp field `api_calls`, không phải lý thuyết):**

| Framework | avg api_calls/câu | accuracy | Ghi chú |
|---|---:|---:|---|
| `mdagents` (gốc) | 2,401 | 38,19% | |
| `mdagents_vi` | **2,690** (+12%) | 39,61% (+1,4pp) | **Tốn hơn, không rẻ hơn** như docstring ghi |
| `medagents` (gốc) | 9,000 (cố định) | 47,73% | |
| `medagents_vi` | 8,000 (cố định) | 47,31% | Đúng như thiết kế — tiết kiệm thật 1 call, accuracy gần như không đổi |

**(c) Bug pad-trùng-role ở `medagents_vi`:**
```
17,70% cau (2.425/13.702, seed 42+43) co role TRUNG LAP trong team 3 nguoi
Vi du: topic=['Gastroenterology'] -> roles=['Gastroenterologist','General Practitioner','General Practitioner']
```
Kiểm tra chéo bản gốc `medagents.py`: cùng kiểu code pad nhưng **0/33.282 câu** bị trùng (LLM tự
liệt kê role hiếm khi thiếu). Bug chỉ "phát tác" ở `medagents_vi` vì bảng `TOPIC_TO_ROLES` chủ
đích chỉ định nghĩa 2 role/topic.

## 5. Error analysis

- **Root cause (a)+(b):** `mdagents_vi` giả định ngầm rằng thay self-classification bằng ground
  truth chỉ đơn thuần "bớt 1 call". Thực tế self-classification của `llama3.1-8b` bị lệch nặng
  (đa số nói "Low" để tránh nhánh đắt), nên bản gốc "rẻ" là do model lười, không phải do phân loại
  đúng. Khi thay bằng nhãn thật (phân bố khó hơn), hệ thống bị đẩy vào nhánh Moderate/High nhiều
  hơn hẳn → tổng call tăng dù bớt được bước classify.
- **Root cause (c):** `run_medagents_vi()` viết `while len(roles) < n_experts: roles.append("General Practitioner")`
  — literal string cứng, không lọc phần tử đã có. Bản gốc MedAgents không có bug này vì họ không
  ép cố định số domain (dùng đúng số domain LLM liệt kê ra, chỉ có fallback lặp khi *toàn bộ* API
  call lỗi, không phải khi phân tích "thành công nhưng ít hơn kỳ vọng").
- **Điểm chưa ghi rõ trong paper:** cả 2 biến thể `_vi` dùng nhãn dataset (`difficulty_level`,
  `medical_topic`) tại thời điểm suy luận — đây là oracle/privileged info, một hệ thống triển khai
  thật sẽ không có sẵn. Cần khung lại là "oracle-informed routing", không phải "cùng thuật toán,
  bớt 1 bước".
- **Điểm chưa ghi rõ khác:** `Step 4 — Collaborative Consultation` đã đơn giản hoá từ vòng lặp
  đồng thuận thích ứng (Algorithm 1 gốc, chi phí biến động) thành 1 vòng vote cố định (chi phí luôn
  cố định 8-9 call, 0 phương sai đo được trên toàn bộ 33.282 câu) — hợp lý cho bài toán trắc
  nghiệm nhưng là simplification có chủ đích, không phải port trung thành Algorithm 1.
  **Lưu ý quan trọng (đã verify lại bằng nguyên văn paper, không chỉ code):** khác biệt **không
  phải** "paper làm final-answer tự do, mình làm multiple-choice" — cả 2 đều multiple-choice
  (9 dataset gốc: MedQA, MedMCQA, PubMedQA, MMLU). Khác biệt thật sự là **đối tượng được
  vote/sửa bên trong vòng lặp**:
  - Paper (Section 2.4): expert vote **yes/no trên bản báo cáo tổng hợp** —
    `vote = LLM(Repo, r_vote, prompt_vote)`; nếu có "no" thì sinh advice sửa report, lặp lại.
  - Paper (Section 2.5): **chỉ sau khi** report hội tụ đồng thuận (`Repo_f`), mới có **1 lời gọi
    LLM riêng** đóng vai "decision maker" suy ra đáp án trắc nghiệm cuối —
    `ans = LLM(q, op, Repo_f, prompt_dm)`.
  - Code của bạn (`Step 4`): mỗi expert vote **thẳng A/B/C/D** ngay trong vòng consultation,
    không qua report trung gian, không có vòng lặp yes/no-rồi-sửa-report tách biệt khỏi bước
    suy ra đáp án.
  → Cần ghi trong paper là "thay cơ chế đồng thuận trên report bằng vote trực tiếp trên đáp án",
  không phải "thay đổi loại task", để tránh reviewer hiểu nhầm.
- **Cơ chế "thảo luận" thật sự đang chạy trong `medagents.py`/`medagents_vi.py`** (đọc lại
  `prompt_report()` + `prompt_consult()`): KHÔNG phải hội thoại qua lại, mà là **broadcast 1 lần
  rồi mỗi expert tự re-vote độc lập**:
  1. Step 2: n expert phân tích **độc lập**, không thấy nhau.
  2. Step 3: **1 LLM coordinator riêng** (không phải chính các expert) đọc cả n phân tích, viết
     **đúng 1 lần** report 3-5 câu tóm tắt đồng ý/bất đồng — report này **không** được sửa lại
     sau đó.
  3. Step 4: gọi lại từng expert (n call **tách rời, độc lập**), mỗi expert chỉ thấy report tĩnh
     của Step 3 + đúng phân tích của **chính mình** ở Step 2 — **không thấy** câu trả lời Step 4
     của các expert khác, không có vòng 2.
  → Gọi là "Collaborative Consultation" nhưng thực chất các expert không hề "thấy" nhau ở bước
  quyết định lại — khác với `mad_debate.py` trong cùng codebase, nơi agent thực sự đọc **nguyên
  văn câu trả lời của nhau** mỗi vòng (biến `other_args`, [mad_debate.py:111](../../frameworks/tier3/mad_debate.py#L111))
  và lặp lại tới `max_rounds`.

## 6. Conclusion

- `medagents_vi`: ý tưởng đúng, tiết kiệm call đúng như kỳ vọng — nhưng **có bug thật cần sửa**
  (role trùng lặp 17,7% số câu). Dữ liệu seed 42/43 hiện tại không dùng được cho paper tới khi sửa.
- `mdagents_vi`: ý tưởng "tốn call" hoá ra **sai theo hướng ngược lại** trên model 8B — cần đo lại
  bằng model khác (qwen3-32b, llama3.3-70b) trước khi kết luận chung, và sửa lại claim trong
  docstring/paper.
- Cả 2 cần bổ sung đoạn limitation về oracle label + về việc đơn giản hoá Algorithm 1.

## 7. Next step (làm sau họp)

1. **Sửa bug** [medagents_vi.py:152-156](../../frameworks/tier3/medagents_vi.py#L152-L156):
   pad bằng cách loại phần tử đã có trong `roles`, ví dụ dùng `GENERIC_ROLES` làm nguồn pad và
   skip nếu đã tồn tại, thay vì hard-code `"General Practitioner"`.
2. **Chạy lại** `medagents_vi` seed 42, 43 (và seed 44 khi tới lượt) cho mọi model sau khi sửa,
   để số liệu nhất quán trước khi đưa vào paper.
3. **Sửa docstring/claim** của `mdagents_vi.py` (dòng "eliminates 1 API call") — thay bằng số đo
   thực tế theo model, không giả định cố định.
4. Khi các run `mdagents`/`mdagents_vi` trên `ollama2-qwen3-32b` và `ollama-llama3.3-70b` chạy
   xong, chạy lại đúng script so sánh (agreement rate + avg api_calls) để xem pattern "LLM thiên
   lệch Low" có phải đặc thù riêng của model yếu (8B) hay không.
5. Viết thêm đoạn limitation trong paper: (a) `_vi` dùng oracle label tại inference time,
   (b) `Step 4` đã đơn giản hoá so với Algorithm 1 gốc (fixed-cost thay vì adaptive consensus loop).
6. **Cân nhắc sửa `Step 4` để expert thực sự "thấy nhau"** (hiện tại không) — 2 hướng, chọn 1
   tuỳ mức độ trung thành với paper muốn đạt:
   - **Fix nhẹ (khuyến nghị trước mắt):** giữ nguyên cấu trúc 5 bước, chỉ thêm cho mỗi expert ở
     Step 4 thấy **câu trả lời của các expert khác** (không chỉ report tĩnh của coordinator) —
     giống hệt pattern `other_args` đã dùng trong
     [mad_debate.py:111-114](../../frameworks/tier3/mad_debate.py#L111-L114):
     ```python
     # trong prompt_consult(), truyền thêm other_opinions (list các (role, answer, analysis)
     # của n-1 expert còn lại, lấy từ `analyses` ở Step 2), thay vì chỉ dùng report chung chung
     other_opinions_str = "\n\n".join(
         f"[{o['role']}] (Answer: {o['answer']}): {o['analysis']}"
         for o in analyses if o["role"] != role
     )
     ```
     Không tốn thêm API call (vẫn n call như hiện tại), chỉ đổi nội dung prompt — nhưng đúng
     nghĩa "consultation" hơn vì expert phản biện trực tiếp lẫn nhau, không chỉ đọc report đã
     bị coordinator lược giản.
   - **Fix đầy đủ (trung thành với Algorithm 1, tốn thêm call + code phức tạp hơn):** thêm vòng
     `while` giống paper — sau Step 4, kiểm tra nếu chưa đồng thuận (`len(set(votes)) > 1`) thì
     quay lại Step 3 sinh report mới (đưa thêm ý kiến Step 4 vào), lặp tới `max_rounds` hoặc tới
     khi đồng thuận, giống cấu trúc `while nocon_flag and n_try < t` ở Algorithm 1. Chi phí API
     call sẽ **biến động theo câu hỏi** (giống paper thật), khác với chi phí cố định hiện tại —
     cần đo lại toàn bộ bảng `api_calls` sau khi đổi.
   - Dù chọn hướng nào, nhớ cập nhật lại đoạn limitation ở mục 7.5 cho khớp với cơ chế mới.

## 8. Artifacts & links

- Code liên quan: [`frameworks/tier3/medagents_vi.py`](../../frameworks/tier3/medagents_vi.py),
  [`frameworks/tier3/mdagents_vi.py`](../../frameworks/tier3/mdagents_vi.py),
  [`frameworks/tier3/medagents.py`](../../frameworks/tier3/medagents.py),
  [`frameworks/tier3/mdagents.py`](../../frameworks/tier3/mdagents.py)
- Dữ liệu đối chiếu: `results/tier3/{mdagents,mdagents_vi,medagents,medagents_vi}/*ollama-llama3_1-8b*vm14k_clean*s{42,43,44}.jsonl`
- Source gốc tham chiếu: [gersteinlab/MedAgents](https://github.com/gersteinlab/MedAgents)
  (`utils.py::fully_decode`, `prompt_generator.py::get_question_domains_prompt`)
- Paper gốc: [arxiv 2311.10537](https://arxiv.org/abs/2311.10537) — Section 2.4 (Collaborative
  Consultation, vote yes/no trên report) và Section 2.5 (Decision Making, suy đáp án từ report
  đồng thuận)
- Related: `003-tier3-llama8b-pilot`, `004-ollama2-instance-setup`
