# Experiment 006 — `deda_med`: pilot + full-run plan (rule-out audit trên disagreement + blinded arbiter)

<!-- theo chuẩn experiment/Template/README.md -->
<!-- Đây là bản PRE-REGISTRATION: viết TRƯỚC khi chạy bất kỳ API call nào.
     Mục 1-2-3 điền trước (plan); mục 4-5-6 để trống, chỉ điền bằng số liệu thực tế sau khi chạy. -->

- **Date:** 29/08/2026 (viết plan; chưa chạy)
- **Author:** hoangnv (qua Claude Code)
- **Status:** planned
- **Base model:** `ollama-llama3.1-8b` (chính) **và** `ollama2-qwen3-32b` — bắt buộc cả hai, vì critique ability có scaling law (critic ≤8B gần random), kỳ vọng hành vi phân kỳ giữa 2 model.
- **Method:** không phải training — framework mới tier 3 `deda_med` (+ wrapper `deda_med_nogate`), một codepath duy nhất, hard cap 4 calls/câu, không có vòng lặp:
  **Stage 0 SCREEN** (2 CoT samples, cả hai T=0.5) → nếu đồng thuận thì **gate** trả lời ngay (2 calls) → **Stage 1 RULE-OUT AUDIT** (1 call, candidate-blind, verdict per option `RULED_OUT | UNCERTAIN | SUPPORTED` + category counter-evidence bắt buộc) → **Stage 2 BLINDED ARBITER** (1 call, single character) hoặc programmatic fallback.

---

## 1. Motivation / Hypothesis

Các số đo đã có trên VM14K (11.094 câu × 3 seed, `ollama-llama3.1-8b`) chỉ ra một khoảng trống cụ thể:

1. **Critic tự do là kênh phá hoại đã đo được** (exp trước + logs `results/tier2/self_refine/`):
   `self_refine` cho verdict `INCORRECT` trong **99,80%** mọi critique bất kể đúng sai (zero-information contrarian),
   flip correct→wrong 18,84% vs wrong→correct 12,23% (**damage/gain 1,54×**), kéo accuracy 48,72% → 42,11%.
   Khớp literature: Huang et al. (ICLR'24) — intrinsic self-correction làm giảm accuracy; Tyen et al. — LLM
   không *tìm* được lỗi nhưng *sửa* được nếu chỉ đúng chỗ → critic phải có target cụ thể (từng option), không phải "find flaws".
2. **Tín hiệu gating có thật và rẻ**: trên SC@5, tập unanimous (25,4% câu) đạt 74,66% accuracy; càng phân tán càng kém
   (2 distinct 48,16% / 3 distinct 35,28% / 4 distinct 31,99%). Đo trực tiếp trên cặp 2 sample đầu của `cot_sc` (T=0.5):
   pairwise agreement **g = 55,10%** (llama, n=21.856) / **72,84%** (qwen, n=30.011); accuracy tập agreement **58,84% / 75,27%**;
   khi disagreement, P(gold ∈ cặp) = **66,14%** (oracle ceiling của mọi cơ chế chỉ chọn trong cặp).
3. **MAS hiện tại thua vì selection, không phải vì thiếu kiến thức**: `medagents` oracle (≥1 expert đúng) = 65,31%
   vs final 47,73% → **17,6pp selection failure**. Smit et al. (ICML'24): MAD không thắng self-consistency ở cùng
   token budget → **SC@3 là baseline bắt buộc ở cùng chi phí**.
4. **Process-of-Elimination** (+4~7pp trên model 7-9B, POE EMNLP'23) nhưng LLM yếu khi phải *chỉ ra option sai*
   → output elimination phải parse-robust và đo được (metric riêng, xem mục 2.6).
5. **Judge anchoring**: arbiter phải blinded (shuffle order, giấu nguồn), vì CoT/"ignore" instruction không gỡ được anchoring.

**Hypothesis (pre-registered):**

- **H1 (chính, 8B):** `deda_med` ≥ SC@3 về accuracy (paired McNemar theo `(id, seed)`, pooled 3 seed) với chi phí đo được ≤ SC@3
  (kỳ vọng thực tế **+1–3pp**, E[calls] ≈ 2,66–2,88 < 3,0). Không claim nhảy vọt: chỉ 44,04% câu (llama) vào được stage 1,
  slice disagreement của SC@3 chỉ đúng 37,3%, ceiling trong-cặp 66,14%.
- **H2:** audit có **discrimination thật**: P(RULED_OUT | distractor) − P(RULED_OUT | gold) > 0 (CI cluster-bootstrap không chứa 0),
  `elimination_precision` ≥ 85% (base rate ngẫu nhiên đã là 75% vì 3/4 option là distractor — phải nói rõ trong paper),
  `gold_survival` ≥ 90%.
- **H3:** gate + candidate-protection chặn được kênh phá hoại kiểu `self_refine`: accuracy nhánh `gate` ≈ 58,84% (llama)
  không bị suy giảm (không critic nào được chạm vào tập agreement), và `anchor_override_rate` có gain > damage.
- **H4 (32B, model-conditional):** trên `qwen3-32b`, zero_shot 67,96% > cot 64,3% (extra reasoning có hại) → so sánh chính
  của 32B là **vs zero_shot**, kết quả âm được đóng khung là finding model-conditional, không phải thất bại của framework.

Link tiền đề: exp `005-medagents-vi-review-findings` (bug pad role + bài học "phải audit code trước khi tin số liệu")
— `deda_med` **không import gì từ `medagents_vi`** để cách ly bug đó.

---

## 2. Setup

### 2.1 Data

- **Pilot:** [`data/pilot_100.jsonl`](../../data/pilot_100.jsonl) — 100 câu Vietnamese MCQ
  (Medium 50 / Easy 33 / Challenging 16 / Hard 1; fields `id, difficulty_level, medical_topic, question, options, option_map, answer, answer_index`;
  option true/false được pad bằng placeholder `"Không có"` — ảnh hưởng metric, xem 2.6).
- **Pre-pilot (mined):** ~1.000 `id` lấy **offline, 0 call** từ logs `results/tier1/cot_sc/*vm14k_clean*` nơi `individual_votes[:2]` bất đồng.
- **Full run:** `data/vm14k_clean.jsonl` — 11.094 câu, `EXPECTED_N` đã có sẵn trong `metrics/io.py:21`.
- `deda_med` **không đọc nhãn dataset nào** (`medical_topic`/`difficulty_level` không dùng) → không cần disclosure oracle-condition.

### 2.2 Registration bundle (1 commit duy nhất, đúng các điểm sanctioned)

| # | File | Thay đổi |
|---|---|---|
| 1 | [`infrastructure/config.py`](../../infrastructure/config.py) (dòng 157-161) | append `"deda_med"`, `"deda_med_nogate"` vào `TIER3_FRAMEWORKS` (tên không chứa `__` — ràng buộc `io.py:190-193`). Tự lan sang `ALL_FRAMEWORKS`/`FRAMEWORK_TIER`/`EXPERIMENT_MATRIX`. |
| 2 | `infrastructure/config.py` | thêm hằng số `MAX_TOKENS_AUDIT = 800`; `EN_REFERENCE["deda_med"] = EN_REFERENCE["deda_med_nogate"] = {"nmar": None, "icr": None}` (CLD/Fisher degrade về None có chủ đích). |
| 3 | `frameworks/tier3/deda_med.py` (mới) | export `run_deda_med(question, options, model, seed=None, use_gate=True, use_audit=True, use_evidence_constraint=True, use_arbiter=True) -> dict` và `run_deda_med_nogate(...)` = wrapper bake `use_gate=False` (tên framework riêng để resume/result path không bao giờ trộn config). |
| 4 | `frameworks/__init__.py` | export cả 2 runner. |
| 5 | [`runner/run_experiments.py`](../../runner/run_experiments.py) (36-42, 62-89) | thêm vào import block + 2 nhánh elif trong `dispatch()`: `return run_deda_med(q, opts, model, seed=seed)` / `run_deda_med_nogate(q, opts, model, seed=seed)`. |
| 6 | [`metrics/io.py`](../../metrics/io.py) (dòng 29) | `_AGENT_STEPS["deda_med"] = _AGENT_STEPS["deda_med_nogate"] = ("screen",)` — để `pre_synthesis_answers`/`pre_synthesis_reasoning` (Information-Loss TODO) đọc được audit_trail. |
| 7 | [`PROMPTS.md`](../../PROMPTS.md) | 2 P-number mới (audit = class C, arbiter = class F), screen cross-reference sang prompt CoT hiện có, thêm dòng `MAX_TOKENS_AUDIT=800` vào bảng shared-config. Verbatim + file:line cite, không paraphrase. |

**CẢNH BÁO vận hành:** vào `TIER3_FRAMEWORKS` nghĩa là `--all` sẽ chạy `deda_med` trên **mọi** model của `EXPERIMENT_MATRIX`,
kể cả các key groq (hiện `GROQ_API_KEY` bị comment tại `config.py:11-16` → NameError). Mọi run của exp này **chỉ launch chọn lọc
qua `--framework`, không bao giờ `--all`**. `deda_med` chạy full run chỉ trên ollama (Groq chỉ dành cho smoke pilot_100, và chỉ khi key được export).

### 2.3 Kiến trúc codepath (đủ chi tiết để implement — bám đúng bản FINAL đã chốt)

**Nguyên tắc chung:** không module-level mutable state (thread-safe dưới `NUM_WORKERS=4`); mọi randomness qua RNG per-question
`random.Random(int(hashlib.md5(f"{seed}|{question}".encode("utf-8")).hexdigest(), 16))` — deterministic qua resume, miễn nhiễm
PYTHONHASHSEED, **không** dựa vào `random.seed(seed)` racy của `dispatch()`. Mọi call qua `call_llm` với `think=False`.

**Stage 0 — SCREEN (luôn chạy, đúng 2 calls):**
- Prompt: import **byte-identical** `build_prompt` từ [`frameworks/tier1/cot.py`](../../frameworks/tier1/cot.py) (cross-ref trong PROMPTS.md).
- Cả 2 sample: `temperature=COT_SC_TEMPERATURE=0.5`, `max_tokens=MAX_TOKENS_LONG=512`. `a1=parse_answer(raw1)`, `a2=parse_answer(raw2)`.
  Lý do 0.5/0.5: cặp screen **phân bố y hệt 2 sample đầu của SC@3**, nên mọi hằng số đã đo (g=55,10%/72,84%, acc tập pass 58,84%/75,27%, oracle 66,14%) transfer chính xác.
- Nhánh: (a) `a1==a2` valid + `use_gate` → **HARD RETURN** `a1`, `api_calls=2`, path=`gate` (tập agreement không bao giờ lộ cho critic — anti-self_refine cấu trúc; paper chỉ cite con số ĐO ĐƯỢC 58,84%, **không bao giờ** cite 74,66% làm bound);
  (b) đúng 1 letter valid (đo được 0,86%) → RETURN nó, path=`single_valid` (không bao giờ audit một answer không có đối thủ — damage 1,54×);
  (c) cả 2 None (0,09%) → `C=[]`, đi tiếp; (d) bất đồng → `C=[a1,a2]`, đi tiếp.
  Ở config nogate: predicate gate **vẫn được tính và log** (field `agreed`) nhưng không short-circuit → tái dựng rung post-hoc chính xác.

**Stage 1 — RULE-OUT AUDIT (chỉ khi gate fail; nogate: mọi câu; 1 call):**
- **Candidate-blind** (không thấy a1/a2/reasoning — target cụ thể là từng option, giữ metric elimination không nhiễm, làm rung là pure function).
- `T=0.0` (convention critique), `max_tokens=MAX_TOKENS_AUDIT=800`.
- Prompt (P-number mới, class C): scaffold English, question + `format_options` tiếng Việt verbatim. System cấp **quyền abstain** nguyên văn
  ("If you cannot name concrete disqualifying evidence you MUST output UNCERTAIN — saying you cannot rule an option out is a correct and expected answer").
  Mỗi option: ≤2 câu suy nghĩ rồi **1 verdict line ngay sau option đó** — `X: VERDICT | CATEGORY | one-sentence reason` — **không có recap block cuối** (truncation sẽ nuốt mất).
  VERDICT ∈ {RULED_OUT, UNCERTAIN, SUPPORTED}; CATEGORY bắt buộc cho RULED_OUT, ∈ {CONTRAINDICATION, WRONG_STAGE, WRONG_ANATOMY, MECHANISM_MISMATCH, CONTRADICTS_STEM}, còn lại NONE.
  Một one-shot example **mixed-verdict** (1 RULED_OUT có category + 1 UNCERTAIN) trên skeleton phi y khoa (example all-UNCERTAIN đã bị bác — anchor verdict-copying).
- **`parse_audit(raw, evidence_constraint)`** — pure function trong module framework, script ablation offline dùng lại verbatim; `parser_version` log per record:
  1. Strip `<think>` (defensive).
  2. Line binding regex nới cho 8B/tiếng Việt: `^\s*(?:[-*•>]|\d+[.)])?\s*(?:option|đáp án|lựa chọn|phương án)?\s*[\*\(\[]*([ABCD])[\)\]\*]*\s*[:.\-]`; dòng match **cuối cùng** per letter thắng.
  3. Verdict đọc **chỉ từ field pipe-delimited đầu tiên** sau separator, match token enum chính xác có `\b`; **NEGATION-FIRST guard** chạy trước mọi token dương:
     `(cannot|can't|not|no|không(\s+thể)?|chưa)\W+(?:be\s+)?(rul|elim|loại|support)` → UNCERTAIN; field match nhiều verdict class → UNCERTAIN;
     token trần `INCORRECT` **cố tình không map vào đâu cả** (token 99,80% zero-information). Dòng không có pipe → free-text scan cùng thứ tự negation-first.
  4. Category match closed list (chịu space/hyphen/underscore/case); không có → NONE.
  5. Evidence constraint (`use_evidence_constraint`): RULED_OUT không có category hợp lệ → hạ xuống UNCERTAIN (letter log vào `downgraded`);
     parse **không ràng buộc** của CÙNG raw text cũng được lưu (`survivors_unconstrained`) → rung +evidence-constraint là re-parse, không phải rerun.
  6. **FAIL-OPEN mọi nơi:** letter không bind → UNCERTAIN + `parse_ok[L]=False`; 0 letter bind → all-UNCERTAIN + `audit_parse_failed=True`. Parse failure không bao giờ được phép loại option.
- Degeneracy (tính trên verdict RAW trước protection): cả 4 RULED_OUT → `degenerate=True`, hạ hết về UNCERTAIN, matrix không đưa cho arbiter.
- **CANDIDATE-PROTECTION INVARIANT:** `pruned = {L: RULED_OUT valid} − C`; `survivors S = {A,B,C,D} − pruned` → candidate stage-0 hợp lệ **không bao giờ** bị 1 audit đơn độc loại; RULED_OUT nhắm vào candidate được giữ lại làm **"challenge"** đưa cho arbiter như evidence.
- SHORT-CIRCUIT: `|S|==1` (chỉ reachable khi ≤1 candidate valid) → RETURN sole survivor, `api_calls=3`, path=`sole_survivor`.

**Stage 2 — DECISION (0 hoặc 1 call):**
- **Arbiter** (khi `use_arbiter` và `|S|>=2`): `T=0.0`, `max_tokens=MAX_TOKENS_SHORT=16`, format kế thừa MedAgents convention
  "Respond with only 1 character: A, B, C, or D" (P-number mới class F — không sinh convention format thứ 4; 16 token không thể truncate).
  Nội dung fully blinded, render bởi **`render_ballot(options, S, C, matrix)`** riêng — **luôn đủ 4 option, letter canonical A-D, không bao giờ
  `format_options` trên subset, không bao giờ re-letter** (đã bác: corrupt back-mapping): pruned non-candidate liệt kê dưới
  "ELIMINATED (may not be chosen): X — CATEGORY: reason"; candidate bị challenge render "X: challenged — CATEGORY: reason" (không bao giờ nằm dưới ELIMINATED);
  UNCERTAIN → "X: no conclusive counter-evidence"; SUPPORTED kèm reason. **Code-assert invariant:** không letter nào của S xuất hiện dưới ELIMINATED; unit test trên gap ballot ({A,C,D}, {B,D}, {C}).
  Block candidate: rng-shuffle, giấu nguồn, không reasoning, không count ("The following answers were independently proposed: X, Y" / 1 candidate / "No candidate answers are available."); `n_candidates_shown` log.
  Hàng evidence rng-shuffle (thứ tự present log lại cho position-bias diagnostic). Audit degenerate/parse-failed → thay matrix bằng "No option could be reliably eliminated."
- Decision rule: `arb = parse_answer(raw)`; **ACCEPT iff `arb in S`** — arbiter không resurrect được option đã pruned, không invent elimination
  (enforce bằng check code, không phải prompt); **được phép** chọn survivor non-candidate (kênh recovery cho ~34% disagreement mà gold ngoài cặp); log `arbiter_pick ∈ {candidate, non_candidate}`.
- **`decide_programmatic(C, matrix, S)`** — fallback deterministic 0 call, cũng là decision rule cho rung arbiter-off, **không bao giờ trả None**:
  (1) rank candidate valid theo SUPPORTED > UNCERTAIN > challenged; chênh rank strict → chọn candidate rank cao hơn (**bounded overturn**: chỉ về phía sample CoT độc lập còn lại;
  **kill-switch pre-registered:** nếu pre-pilot đo flip damage/gain > 1 → tắt rank-flip, tie luôn về a1);
  (2) else a1 nếu valid (đo được: first-sample tie-break 33,77% vs alphabetical 29,93% trên llama disagreement; khớp convention fallback của `cot_sc`;
  số ngược chiều của qwen — alphabetical 45,5% vs 37,0% — báo cáo là sensitivity analysis, **không** làm rule model-conditional);
  (3) else a2 nếu valid; (4) else survivor SUPPORTED duy nhất; (5) else `min(S)`. Field `answer` **không bao giờ None**.

**Output contract (đúng contract runner `run_experiments.py:191-213`):**
```
{"answer": letter (never None),
 "api_calls": số call THÀNH CÔNG thật (assert khớp path: 2 = gate/single_valid/cả stage1+2 fail;
               3 = sole_survivor hoặc audit chạy + arbiter fail/invalid; 4 = full path),
 "individual_votes": [a1, a2]  (ordered, giữ None — io.py tự filter),
 "round1_answers": cùng list,
 "audit_trail": schema dưới}
```

**`audit_trail` schema (mọi con số paper là pure function của JSONL):**
1-2. `{"step":"screen","agent":1|2,"temperature":0.5,"answer":a_i,"reasoning":extract_reasoning(raw_i,500),"prompt_chars":n,"raw_chars":n}` — step đã đăng ký `_AGENT_STEPS`; index order == `individual_votes` (contract Information-Loss, `metrics/INFORMATION_LOSS_TODO.md`).
3. `{"step":"gate","agreed":bool,"gate_passed":bool,"candidates":[...]}`
4. (nếu audit chạy) `{"step":"ruleout_audit","raw":<FULL verbatim — load-bearing cho re-parse>,"matrix":{L:{"verdict","category","reason"(≤200ch),"valid"}},"parse_ok":{L:bool},"audit_parse_failed":bool,"downgraded":[...],"pruned":[...],"survivors":[...],"survivors_unconstrained":[...],"candidate_challenges":[...],"degenerate":bool,"parser_version":str,"prompt_chars","raw_chars"}` — KHÔNG có key `"answer"`, không nằm trong `_AGENT_STEPS`.
5. (nếu arbiter chạy) `{"step":"arbiter","allowed":[...],"n_candidates_shown":int,"candidates_order":[...],"evidence_order":[...],"answer":L|null,"valid":bool,"arbiter_pick":"candidate"|"non_candidate"|null,"prompt_chars","raw_chars"}`
6. `{"step":"decision","answer":L,"decided_by":"gate"|"single_valid"|"sole_survivor"|"arbiter"|"fallback_rank"|"fallback_a1"|"fallback_a2"|"fallback_supported"|"fallback_min_survivor","flip_damage_guard_active":bool,"config":{4 flags},"api_calls":n}`
7. (lỗi được bắt) `{"step":"error","stage":"ruleout_audit"|"arbiter","error":str[:200]}`
Không có step `expert_gathering`, không có confidence field (+GS/+CW đúng là N/A, disclose trong paper).

**Failure handling:** exception ở stage 0 **PROPAGATE** (cơ chế >10-errors abort của runner là handler dự kiến — server sập thì không được bịa answer);
stage 1 exception → bắt, matrix all-UNCERTAIN + error step, đi tiếp (arbiter suy biến thành tie-breaker blinded giữa 2 candidate — sàn ≈ sample thứ 3 của SC@3);
stage 2 exception/invalid → bắt, `decide_programmatic`. All-ruled-out → xử lý degenerate ở trên. No-ruled-out (toàn UNCERTAIN/SUPPORTED) → outcome abstention hợp lệ kiểu ColMAD, không phạt, báo tần suất.

### 2.4 Config flags & ablation ladder (1 codepath, 2 run vật lý)

Flags: `use_gate`, `use_audit`, `use_evidence_constraint`, `use_arbiter`. **Chỉ 2 framework name vật lý** (`deda_med` full = headline;
`deda_med_nogate` = superset cho ablation) — đã bác kwargs-trên-1-tên (resume trộn config, fatal) và 5 run vật lý (~90h/model lãng phí).
Vì audit candidate-blind ở T=0.0 và raw audit text log verbatim, **mọi rung R1–R4 là pure function của run nogate superset** (cùng stage-0 draws → paired McNemar giữa các rung):

| Rung | Config | Decision |
|---|---|---|
| R1 "stage-0 anchor" | audit/arbiter off | trả `a1` (a2 nếu a1 None) — label thật thà: ≈ CoT@0.5 với 2 calls; baseline matched thật là SC@3 |
| R2 "+rule-out, ungated, unconstrained" | gate=F, audit=T, ec=F, arbiter off | `decide_programmatic` |
| R3 "+gate" | gate=T | agreed → a1; else quyết định R2 (lưu ý gate inert ở R1 vì đằng nào cũng trả a1 — disclose) |
| R4 "+evidence-constraint" | ec=T | re-parse cùng raw audit text |
| R5 "+blinded arbiter" | full | = `deda_med`; side arm full-but-ungated = chính row `deda_med_nogate` |

`analysis/deda_ablation.py` **IMPORT `parse_audit`/`decide_programmatic`/render logic TỪ module framework** — offline recompute không thể drift.

### 2.5 Baseline

- **SC@3 (bắt buộc, cùng seeds):** `majority_vote(cot_sc.individual_votes[:3], fallback=first-valid-of-the-3)` — tái dựng từ logs `cot_sc` sẵn có
  (`results/tier1/cot_sc/`), **0 call mới** trên vm14k; tie-break ghi cố định tại đây để không ai phải đoán. Trên pilot_100: nếu chưa có `cot_sc`
  trong `results_test/` thì chạy (100 × 5 × 3 seed = 1.500 calls/model).
- **32B:** primary comparison pre-registered là **zero_shot (67,96%)** — baseline vận hành thực tế của qwen3-32b (mạnh hơn cot 64,3%); SC@3 báo cạnh bên.
- Pass@k **không bao giờ** báo cho `deda_med` (votes là MAS-votes, `compute.py:49-61`).
- NMAR/CLD: **declared N/A** — với 2 vote, `Counter.most_common` tie 1-1 trả theo insertion order nên "majority" trên mọi disagreement định nghĩa là a1
  (artifact, không phải đo lường; `collaboration.py` KHÔNG sửa global). Metric thay thế: **`anchor_override_rate`** (xem 2.6).

### 2.6 Metric mới — thêm vào `metrics/` (module mới `metrics/deda.py`)

Chỉ đọc record đã normalize qua `metrics/io.py` (single source of truth); mọi hàm là pure function của JSONL; ngưỡng dưới đây là pre-registered.

1. **`gate_rate`** — P(stage-0 đồng thuận và gate trả lời).
   Numerator: record có `audit_trail` step `gate` với `gate_passed==True` (tương đương `decision.decided_by=="gate"`). Denominator: mọi record.
   Báo per model × seed, mean±std qua 3 seed. Kỳ vọng từ số đã đo: ≈55,1% (llama) / ≈72,8% (qwen); lệch >±5pp so với kỳ vọng → điều tra (prompt drift so với cot_sc).

2. **`elimination_precision`** — P(option thực sự là distractor | verdict RULED_OUT valid, sau evidence-constraint).
   Trên mọi record có step `ruleout_audit` với `degenerate==False`: đếm mọi cặp (question, L) có `matrix[L].verdict=="RULED_OUT"` và `valid==True`;
   precision = #(L != gold) / #tổng. **CI 95% bằng cluster bootstrap resample theo QUESTION id** (1.000 reps — verdict trong cùng 1 audit tương quan,
   binomial per-option là sai). **Loại option placeholder `"Không có"`** bằng id-join với dataset tại metric-time (số inclusive để footnote appendix —
   loại placeholder là free win làm phồng precision). Ghi kèm: base rate ngẫu nhiên 75% (3/4 option là distractor) nêu rõ như descriptive context.
   Trên run gated, metric này là **disagreement-conditioned** (disclose); ước lượng unconditional lấy từ run nogate superset.

3. **`gold_ruled_out_rate`** — P(gold ∈ `pruned`) trên các record có audit; complement **`gold_survival`** = P(gold ∈ `survivors`) là
   **metric elimination CHÍNH** (ngưỡng ≥90%). Lưu ý cấu trúc: nhờ candidate-protection, gold chỉ có thể bị pruned khi gold ∉ C.

4. **`per_branch_accuracy`** — group theo `decision.decided_by` ∈ {gate, single_valid, sole_survivor, arbiter, fallback_rank, fallback_a1,
   fallback_a2, fallback_supported, fallback_min_survivor}: báo n, share %, accuracy, avg `api_calls` mỗi nhánh, per model, mean±std qua seed.
   Đây là bảng chẩn đoán trung tâm: nhánh `gate` phải ≈ accuracy tập agreement đã đo; nhánh `arbiter` so với 37,3% (SC@3 trên disagreement slice).

5. **`anchor_override_rate`** (thay NMAR): P(final != a1 | disagreement), tách theo gold==a1 vs gold==a2 → cho ra trực tiếp damage/gain của override;
   tính từ `audit_trail` (candidates + decision), tên riêng, không đi qua đường NMAR.

Phụ trợ: verdict-marginal telemetry (phân bố RULED_OUT/UNCERTAIN/SUPPORTED per option per model — cái analog trực tiếp của chữ ký 99,8%-INCORRECT),
`parse_ok` rate per letter, truncation rate (raw_chars chạm trần ~800 token), arbiter position-bias P(pick first-presented) từ `candidates_order`/`evidence_order`,
token budget đo được (Σ prompt_chars + raw_chars → ước lượng token offline) so với SC@3 — claim "matched budget" là **so sánh đo được, không phải khẳng định từ max_tokens**.
Sau ≥500 record: chạy `validity.py` parser-skew check vs zero_shot cùng model.

### 2.7 PILOT DESIGN — `pilot_100` × 3 seeds (42/43/44) × 2 models, với 3 decision gate

Chạy trên `data/pilot_100.jsonl` → tự ghi vào `results_test/tier3/deda_med*/` (stem `pilot*`). 4 run vật lý:
`deda_med` × {`ollama-llama3.1-8b`, `ollama2-qwen3-32b`} và `deda_med_nogate` × 2 model đó (nogate cho unconditional telemetry).
Chi phí: gated ~810 calls (llama, E≈2,7) + ~720 (qwen, E≈2,4); nogate ~1.140 calls/model; cộng `cot_sc` pilot nếu thiếu (1.500/model) → tổng < 8.000 calls, **< nửa ngày** cả hai server.

**Gate P1 — verdict-entropy / degeneracy check (chống pathology 99,8%-INCORRECT).** Trên toàn bộ audit của pilot (xét riêng từng model):
- marginal của BẤT KỲ verdict đơn nào > 90% → **FAIL** (chữ ký contrarian/parrot y như `self_refine`);
- UNCERTAIN > 90% → **FAIL** (audit là dead weight +1 call);
- tỉ lệ audit có ≥3 option RULED_OUT > 50% → **FAIL** (rule-out-everything);
- kèm theo: `parse_ok` per letter ≥ 95%, truncation < 5% (smoke mức pilot; ngưỡng chính thức xét lại ở pre-pilot).
FAIL → **dừng, redesign prompt audit, chạy lại pilot**; tuyệt đối không sang phase sau.

**Gate P2 — elimination precision (~85%).** Điểm ước lượng `elimination_precision` ≥ 85% VÀ `gold_survival` ≥ 90%.
**Disclose thẳng:** pilot_100 chỉ sinh ~44 audit (llama) → CI ±14–18pp, gate này ở pilot chỉ là **screen bắt fail thô**
(ví dụ precision ≤ 75% base rate → dừng ngay); quyết định thống kê THẬT nằm ở pre-pilot ~1.000 câu (mục 2.8) với discrimination CI.
Ngưỡng 85% được chọn vì phải cao hơn hẳn base rate 75% — ngưỡng 60% từng đề xuất đã bị bác (dưới base rate, contrarian cũng pass).

**Gate P3 — head-to-head vs SC@3, cùng seeds cùng model.** So `deda_med` với SC@3 (định nghĩa 2.5) trên đúng 100 id × 3 seed:
- accuracy(deda_med) ≥ accuracy(SC@3) − 3pp (noise band của n=100 — gate này bắt **lỗi tích hợp làm sập accuracy**, không phải kết luận thống kê);
- mean `api_calls` (run gated) < 3,0 và assert `api_calls` khớp path trên 100% record;
- `gate_rate` trong khoảng kỳ vọng (≈55%/≈73% theo model).
FAIL → debug tích hợp (parser, ballot, fallback) trước khi đốt bất kỳ giờ GPU full-run nào.

Cả 3 gate pass ⇒ sang Phase 2 (pre-pilot). Kết quả từng gate điền vào mục 4.

### 2.8 PRE-PILOT (mined disagreement, ~1.000 câu — go/no-go thống kê thật)

- `scripts/prepilot_deda.py` (mới): mine ~1.000 id từ logs `cot_sc` vm14k nơi `individual_votes[:2]` bất đồng (offline, 0 call);
  candidates lấy từ votes đã lưu; chạy **audit + arbiter fresh** (~2.000 calls/model), import prompt-builder + `parse_audit` từ chính module framework;
  ghi `results_test/` với stem prefix `pilot`.
- **Gates pre-registered, outcome map sẵn:**
  - (a) `parse_ok` per letter ≥ 95% VÀ truncation < 5% — fail → rework prompt, chạy lại pre-pilot;
  - (b) discrimination P(RULED_OUT|distractor) − P(RULED_OUT|gold) > 0 với 95% cluster-bootstrap CI (resample theo QUESTION) không chứa 0,
    VÀ `elimination_precision` ≥ 85%, VÀ `gold_survival` ≥ 90% — fail → **elimination hạ cấp thành evidence-only** (tắt pruning, matrix vẫn đưa arbiter);
  - (c) flip damage/gain của programmatic rank-flip > 1 → **tắt rank-flip** (kill-switch), tie luôn về a1;
  - (d) degeneracy monitor như Gate P1 → **abort + redesign**.
- **Parser FREEZE** tại thời điểm kết thúc pre-pilot; `parser_version` log per record; mọi thay đổi lexicon sau đó = re-parse có disclose kèm số before/after.
- Overlap của 1.000 id mined (và 100 id pilot) với tập headline: **disclose + sensitivity analysis loại trừ chúng**.

### 2.9 FULL-RUN PLAN + COST ESTIMATE (11.094 câu × 3 seed = 33.282 record/model/config)

Công thức chi phí: `E[calls] = 2 + P(stage1) × (1 + P(arbiter|audited))`; đo được P(stage1): llama 0,4404 / qwen ≈0,26. Hard cap 4. SC@3 = 3,0 flat.

| Run | Model | Server | Seeds | E[calls]/câu | Tổng calls | Wall-clock ước tính (4 workers, ~5–7s/call) |
|---|---|---|---|---:|---:|---|
| `deda_med` (gated — **headline**) | `ollama-llama3.1-8b` | :11434 | 42/43/44 | 2,66–2,88 | ~93–96k | **~33–40h** |
| `deda_med_nogate` (superset ablation) | `ollama-llama3.1-8b` | :11434 | 42/43/44 | ~3,7–3,8 | ~125k | **~61h** |
| `deda_med` (gated) | `ollama2-qwen3-32b` | :11435 | 42/43/44 | 2,31–2,52 | ~77–84k | **~50h+** (32B chậm hơn; chạy song song với llama nhờ server riêng) |
| `deda_med_nogate` | `ollama2-qwen3-32b` | :11435 | **chỉ seed 42** (disclose) | ~3,7 | ~41k | ~20–25h |
| SC@3 baseline | cả 2 | — | 42/43/44 | 0 (tái dựng từ logs `cot_sc` sẵn có) | 0 | 0 (nếu `cot_sc` qwen vm14k chưa đủ 3 seed → phải chạy bù, disclose) |

- Headline `api_calls` báo cáo lấy **duy nhất từ run gated**, không bao giờ từ superset.
- Reportability: mỗi cell cần đủ 3 seed × 11.094 record (`io.py:155-157`); resume mặc định per `(framework, model, dataset, seed)`; seed abort khi >10 lỗi.
- 32B dùng tag ctx-capped sẵn có (`qwen3-32b-ctx8k` — full-context KV cache spill CPU: 76s/call vs 6–7s/call).
- Token budget: quy đổi Σ`prompt_chars`/`raw_chars` → token offline, báo "matched budget vs SC@3" bằng **số đo**, vì `call_llm` không trả usage stats.
- Tổng ngân sách xin duyệt: **~95h GPU (:11434, llama) + ~75h GPU (:11435, qwen)** — open question cho owner: có mua đủ 3 seed nogate cho 32B không (thêm ~40–50h)?

### 2.10 GO/NO-GO cho paper claim (pre-registered — trả lời ở mục 6)

**Primary endpoint (pre-registered):** accuracy `deda_med` vs SC@3, `ollama-llama3.1-8b`, `vm14k_clean`,
**paired McNemar theo `(id, seed)`**, pooled 3 seed + per-seed delta báo kèm. **Không bao giờ** t-test trên 3 seed mean (df=2).
32B: primary vs zero_shot 67,96% (McNemar cùng kiểu), SC@3 báo cạnh; kết quả âm = model-conditional finding.
Mọi so sánh khác (rung delta, precision, subset) = secondary, hiệu chỉnh **Holm**.

**GO — được phép claim trong paper** ("equal-or-lower cost than SC@3 với disagreement pipeline đo được, calibrated; +1–3pp trên 8B") khi TẤT CẢ:
1. Pre-pilot gates (a)–(d) pass, hoặc fallback pre-mapped đã kích hoạt và disclose (evidence-only / rank-flip off);
2. 8B full run: Δ = acc(`deda_med`) − acc(SC@3) > 0, McNemar p < 0,05 (pooled), per-seed delta cùng dấu cả 3 seed (descriptive consistency);
3. Chi phí đo được: mean `api_calls` ≤ 3,0 VÀ token budget đo được ≤ SC@3;
4. Trên full run: `elimination_precision` ≥ 85% (cluster-bootstrap CI), `gold_survival` ≥ 90%, telemetry verdict-marginal sạch (không chạm ngưỡng degeneracy);
5. `validity.py` parser-skew: không SUSPECT vs zero_shot cùng model.

**NO-GO / downgrade map (quyết định trước, không thương lượng sau khi thấy số):**
- Điều kiện 2 fail (không significant hoặc Δ ≤ 0) → **không claim accuracy**; báo negative/neutral result + ablation analysis (rung nào chết); 32B fail riêng → framing model-conditional (đã pre-registered ở H4).
- Điều kiện 4 fail nhưng 2 pass → claim hạ cấp: "gating + audit-as-evidence", không claim elimination capability.
- Gate P1/degeneracy trip ở bất kỳ phase nào → **abort trước full run** (không đốt 95h GPU cho một contrarian mới).
- Chi phí đo được > SC@3 → claim "equal-or-lower cost" bị rút, chỉ còn so accuracy thuần (yếu hơn hẳn — cân nhắc dừng).
- Claim **không bao giờ** được phát biểu bằng 74,66% (SC@5-unanimous là subset; số đúng cho tập gate 2-of-2 là 58,84%/75,27% đã đo).

### 2.11 Hardware & Commands

- **Hardware:** GPU 48GB; `ollama` shared :11434 (llama3.1-8b), `ollama2` :11435 (qwen3-32b-ctx8k, instance riêng của hoangnv → 2 model chạy song song); `NUM_WORKERS=4` khớp `OLLAMA_NUM_PARALLEL=4`. Framework phải thread-safe (đã đảm bảo: stateless + per-question RNG).
- **Commands** (runner tự chạy đủ 3 seed 42/43/44; KHÔNG dùng `--all`):
```bash
# Phase 0 — 0 API call
pytest tests/test_deda_med.py           # parse_audit fixtures + render_ballot gap-ballots + decide_programmatic + api_calls-path assert

# Phase 1 — pilot_100 (ghi vào results_test/)
python runner/run_experiments.py --framework deda_med        --model ollama-llama3.1-8b  --dataset data/pilot_100.jsonl
python runner/run_experiments.py --framework deda_med_nogate --model ollama-llama3.1-8b  --dataset data/pilot_100.jsonl
python runner/run_experiments.py --framework deda_med        --model ollama2-qwen3-32b   --dataset data/pilot_100.jsonl
python runner/run_experiments.py --framework deda_med_nogate --model ollama2-qwen3-32b   --dataset data/pilot_100.jsonl
python runner/run_experiments.py --framework cot_sc          --model ollama-llama3.1-8b  --dataset data/pilot_100.jsonl   # nếu chưa có, cho Gate P3
python analysis/deda_pilot_gates.py     # in PASS/FAIL cho Gate P1/P2/P3

# Phase 2 — pre-pilot mined ~1k (stem prefix "pilot" -> results_test/)
python scripts/prepilot_deda.py --model ollama-llama3.1-8b --n 1000
python scripts/prepilot_deda.py --model ollama2-qwen3-32b  --n 1000

# Phase 3 — full run (headline + superset)
python runner/run_experiments.py --framework deda_med        --model ollama-llama3.1-8b --dataset data/vm14k_clean.jsonl
python runner/run_experiments.py --framework deda_med_nogate --model ollama-llama3.1-8b --dataset data/vm14k_clean.jsonl
python runner/run_experiments.py --framework deda_med        --model ollama2-qwen3-32b  --dataset data/vm14k_clean.jsonl
# nogate 32B: chỉ seed 42 — chạy rồi Ctrl+C sau seed 42, hoặc thêm cờ seed nếu owner duyệt sửa runner (mặc định: dừng thủ công, disclose)

# Ablation (offline, 0 call)
python analysis/deda_ablation.py        # R1..R5 từ superset nogate; import parse/decide từ frameworks/tier3/deda_med.py
```
- Lưu ý: `GROQ_API_KEY` đang comment (`config.py:11-16`) → mọi row groq sẽ NameError; exp này **ollama-only**.

### 2.12 Δ vs previous

So với `cot_sc` (SC@3): giữ nguyên 2 sample đầu (byte-identical prompt, cùng T=0.5) — **biến đổi duy nhất là thay sample thứ 3
bằng cặp {rule-out audit + blinded arbiter} chỉ chạy trên disagreement**. So với `self_refine`: cùng họ "critic" nhưng đảo cả 3 biến gây hại
(critic có target per-option thay vì find-flaws; được phép abstain; không bao giờ chạm vào answer không có đối thủ).
Trong nội bộ exp: mỗi rung ablation R1→R5 đổi đúng 1 biến (trừ ghi chú gate-inert-ở-R1, đã disclose ở 2.4).

---

## 3. What I did (log)

> Khung checklist dưới đây là **plan**; theo template, khi chạy sẽ ghi đè bằng log chronological những gì THẬT SỰ xảy ra (surprises, mid-run fixes).

- [ ] **P0.1** Viết `frameworks/tier3/deda_med.py` + registration bundle (2.2) trong 1 commit.
- [ ] **P0.2** Viết unit tests parser với fixture adversarial BẮT BUỘC trước call API đầu tiên:
      `"cannot be ruled out"`, `"cannot be eliminated"`, `"not supported"`, token trần `"INCORRECT"`,
      `"B: RULED_OUT — the correct answer is A"`, `"X: UNCERTAIN | NONE | it cannot be fully ruled out"`,
      dòng tiếng Việt trộn (`"đáp án B: không thể loại trừ"`), dòng bullet/số thứ tự/prefix Option-Đáp án-Lựa chọn,
      gap ballots {A,C,D}/{B,D}/{C}, assert api_calls-khớp-path, assert không survivor nào nằm dưới ELIMINATED.
- [ ] **P0.3** Cập nhật `PROMPTS.md` (2 P-number mới + MAX_TOKENS_AUDIT) — verbatim.
- [ ] **P1.1** Chạy 4 run pilot_100 + `cot_sc` pilot nếu thiếu (lệnh ở 2.11). Ghi lại wall-clock/call thực tế để hiệu chỉnh cost model 2.9.
- [ ] **P1.2** Chạy `analysis/deda_pilot_gates.py` → điền bảng Gate P1/P2/P3 vào mục 4. FAIL gate nào → quay lại redesign, log lại tại đây.
- [ ] **P2.1** Mine ~1.000 disagreement id từ logs `cot_sc` vm14k (offline); disclose overlap với pilot/headline.
- [ ] **P2.2** Chạy pre-pilot 2 model; tính discrimination + CI, precision, gold_survival, flip damage/gain, degeneracy telemetry.
- [ ] **P2.3** Chốt 3 công tắc pre-registered theo outcome map 2.8: pruning on/off, rank-flip on/off, prompt rework hay không. **FREEZE parser**, ghi `parser_version`.
- [ ] **P2.4** Xin owner xác nhận bằng văn bản: (i) ngân sách GPU 2.9, (ii) nogate 32B 1 seed hay 3 seed, (iii) row gated vật lý là headline duy nhất — R5 chỉ là ablation-internal (không bao giờ có 2 số "full system" trong paper).
- [ ] **P3.1** Full run theo bảng 2.9 (resume mặc định; theo dõi >10-errors abort).
- [ ] **P3.2** Sau ≥500 record: chạy `validity.py` parser-skew check giữa chừng — bắt sớm skew thay vì sau 40h.
- [ ] **P3.3** Chạy `metrics/deda.py` + `analysis/deda_ablation.py`; điền mục 4/5/6; đối chiếu go/no-go 2.10.

---

## 4. Results

*(chưa chạy — điền sau; khung bảng chuẩn bị sẵn)*

**(a) Pilot gates (Phase 1):**

| Gate | Ngưỡng | llama3.1-8b | qwen3-32b | PASS? |
|---|---|---|---|---|
| P1 verdict-entropy | không verdict nào >90%; UNCERTAIN ≤90%; ≥3-RULED_OUT ≤50%; parse_ok ≥95%; trunc <5% | | | |
| P2 elimination | precision ≥85% (điểm ước lượng, CI rộng — disclose); gold_survival ≥90% | | | |
| P3 vs SC@3 | Δacc ≥ −3pp; mean api_calls <3,0; gate_rate trong band kỳ vọng | | | |

**(b) Pre-pilot (Phase 2):** discrimination + 95% CI / precision / gold_survival / flip damage-gain / verdict marginals — per model.

**(c) Full run (Phase 3):** bảng chính accuracy mean±std 3 seed + McNemar p; per_branch_accuracy; gate_rate; elimination metrics; anchor_override_rate; measured token budget vs SC@3; ablation R1→R5 (paired, từ superset).

| Metric | deda_med | SC@3 (baseline) | zero_shot (32B) | Δ | p (McNemar) |
|---|---:|---:|---:|---:|---:|
| Accuracy 8B | | 49,14% | — | | |
| Accuracy 32B | | | 67,96% | | |
| avg api_calls | | 3,0 | 1,0 | | |

## 5. Error analysis

*(chưa chạy — điền sau)* Danh sách sẽ soi: các câu gate-pass nhưng sai (confident-wrong — trong bảng MAS-failure, `flawed_consensus`
của deda_med đọc là "confident-wrong gate rate", footnote rõ "agents" là 2 temperature sample, không phải persona);
audit ruled-out gold (đọc raw text để phân loại lỗi kiến thức vs lỗi parse); arbiter position-bias; các nhánh fallback_min_survivor
(đường cùng — tần suất phải rất nhỏ); phân bố lỗi theo difficulty (io.py normalize Easy/Moderate/Hard).

## 6. Conclusion

*(chưa chạy — sẽ trả lời đúng các câu pre-registered ở 2.10: GO hay NO-GO cho paper claim, công tắc nào đã kích hoạt, và claim cuối cùng được phép phát biểu ở mức nào.)*

## 7. Next step

- Nếu GO: viết section paper (framing "equal-or-lower cost, calibrated disagreement pipeline"), bảng ablation R1–R5, appendix prompts verbatim.
- Nếu NO-GO ở gate elimination: exp kế tiếp thử audit-as-evidence-only làm chính (pruning off) hoặc audit 2-call (per-pair) — chỉ sau khi có số pre-pilot.
- Deferred (chờ owner): bilingual MedQA-en arm — CHỈ khi có bản 4-option + preflight assert `len(options)==4, gold in A-D`
  (MedQA chuẩn 5-option sẽ bị chấm sai âm thầm vì `parse_answer`/`io.OPTIONS` hardcode A-D); guard cho GROQ_API_KEY (ngoài sanctioned points, cần sign-off).

## 8. Artifacts & links

- Code (sẽ tạo): `frameworks/tier3/deda_med.py`, `metrics/deda.py`, `scripts/prepilot_deda.py`, `analysis/deda_ablation.py`, `analysis/deda_pilot_gates.py`, `tests/test_deda_med.py`
- Registration: [`infrastructure/config.py`](../../infrastructure/config.py), [`frameworks/__init__.py`](../../frameworks/__init__.py), [`runner/run_experiments.py`](../../runner/run_experiments.py), [`metrics/io.py`](../../metrics/io.py), [`PROMPTS.md`](../../PROMPTS.md)
- Results: `results_test/tier3/deda_med*/` (pilot + pre-pilot), `results/tier3/deda_med*/` (full), baseline `results/tier1/cot_sc/`, `results/tier1/zero_shot/`
- Số liệu nền: logs `self_refine` (pathology 99,8%), `cot_sc` (hằng số gate), exp `005-medagents-vi-review-findings` (bug pad role — lý do không import medagents_vi)
- Literature: Huang et al. ICLR'24; Tyen et al.; Smit et al. ICML'24; POE EMNLP'23; Eliminate-Then-Select 2025; ColMAD; MAD failure analysis 2025; judge-anchoring
- Related: exp `003-tier3-llama8b-pilot`, `004-ollama2-instance-setup`, `005-medagents-vi-review-findings`; `metrics/INFORMATION_LOSS_TODO.md`
