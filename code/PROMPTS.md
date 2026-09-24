# Prompt Inventory — ViMedAgent

Complete verbatim inventory of every prompt used in the experiments, for the
paper appendix. **All text below is copied from source, not paraphrased.**
Regenerate this file if any prompt changes; a paraphrased appendix is a
reproducibility defect.

Notation used throughout:

- `{question}` — the VM14K question stem, verbatim, in Vietnamese.
- `{options}` — the answer options rendered by `format_options()`
  (`frameworks/utils.py:53`) as `"A. <opt>\nB. <opt>\nC. <opt>\nD. <opt>"`.
- `{role}`, `{report}`, … — runtime-substituted values, described per prompt.
- Each **P-number is one prompt** = its system message plus its user message.
  A prompt with no system message is marked *(user only)*.

**Language note.** Every prompt is written in **English** while `{question}`
and `{options}` are inserted in **Vietnamese**. This cross-lingual asymmetry is
deliberate and matches the MedAgentAudit reference setup used for the CLD
metric; it is a property of the experiment, not an oversight.

---

## 1. Prompt taxonomy

**18 numbered prompts, 17 distinct texts** (P6 is byte-identical to P1). They
fall into six functional classes. This classification is the recommended
organising principle for the appendix, because it makes the Tier-1 → Tier-3
progression visible: higher tiers do not use *better* prompts, they use *more
classes* of them.

| Class | Function | Prompts | Used by |
|---|---|---|---|
| **A. Direct answer** | elicit a bare label, no reasoning | P1, P6, P14 | Zero-shot, Voting Ensemble, MDAgents(-VI) |
| **B. Elicited reasoning** | reason, then answer — single agent | P2, P3, P13, P17 | CoT, CoT-SC, Self-Refine, MDAgents(-VI), MAD |
| **C. Self-evaluation** | critique / revise own output | P4, P5 | Self-Refine |
| **D. Role conditioning** | bind an agent to a specialist persona | P8, P15 | MedAgents(-VI), MDAgents(-VI) |
| **E. Inter-agent exchange** | condition on *other agents'* output | P10, P18 | MedAgents(-VI), MAD Debate |
| **F. Aggregation & control** | collapse many positions, or route | P7, P9, P11, P12, P16 | MedAgents(-VI), MDAgents(-VI) |

Two further classes are **not prompts** and must be reported separately, since
they are the mechanism behind the `-VI` variants' call savings:

| Class | Function | Mechanism | Used by |
|---|---|---|---|
| **G. Dataset shortcut** | replace an LLM call with a lookup | `TOPIC_TO_ROLES`, `map_difficulty()` | MedAgents-VI, MDAgents-VI |
| **H. Programmatic** | replace an LLM call with an algorithm | `majority_vote()`, option shuffling | CoT-SC, Voting Ensemble, MAD Debate |

### Per-framework summary

| Framework | Tier | Prompts used | Count | Classes | LLM calls/question |
|---|---|---|---|---|---|
| Zero-shot | 1 | P1 | 1 | A | 1 |
| CoT | 1 | P2 | 1 | B | 1 |
| CoT-SC | 1 | P2 *(reused)* | 1 | B + H | 5 |
| Self-Refine | 1 | P3, P4, P5 | 3 | B, C | 2–5 |
| Voting Ensemble | 2 | P6 *(≡ P1)* | 1 | A + H | 3 |
| MedAgents | 3 | P7, P8, P9, P10, P11 | 5 | D, E, F | 9 |
| MedAgents-VI | 3 | P8, P9, P10, P11 | 4 | D, E, F + G | 8 |
| MDAgents | 3 | P12, P13, P14, P15, P16 | 5 | A, B, D, F | 2 / 4 / 7 |
| MDAgents-VI | 3 | P13, P14, P15, P16 | 4 | A, B, D, F + G | 1 / 3 / 6 |
| MAD Debate | 3 | P17, P18 | 2 | B, E + H | ≤ 9 |

Prompts reused across frameworks are counted once and cross-referenced, never
duplicated.

---

## 2. Shared inference configuration

Applies to every prompt unless overridden in its own subsection.
Source: `infrastructure/config.py`.

| Parameter | Value | Note |
|---|---|---|
| Temperature (default) | `0.3` | uniform across all frameworks |
| Temperature (CoT-SC only) | `0.5` | sampling diversity |
| Temperature (critique only) | `0.0` | deterministic verdict |
| Seeds | `42, 43, 44` | 3 runs; mean ± std reported |
| Max retries | `3` | exponential backoff |
| `think` | `False` | **see below** |

**Token budgets** (`max_tokens`), assigned per prompt class:

| Constant | Value | Applied to |
|---|---|---|
| `MAX_TOKENS_COMPLEX` | 8 | complexity classification (P12) |
| `MAX_TOKENS_SHORT` | 16 | single-character decisions (P11, P16) |
| `MAX_TOKENS_MEDIUM` | 300 | report, consultation, critique, PCC (P4, P9, P10, P13) |
| `MAX_TOKENS_DEBATE` | 400 | debate rounds (P17, P18) |
| `MAX_TOKENS_ROLES` | 512 | role listing (P7) |
| `MAX_TOKENS_LONG` | 512 | CoT reasoning, refinement (P2, P3, P5) |
| `MAX_TOKENS_AGENT` | 600 | full per-expert analysis (P8, P15) |

**`think=False` is a required disclosure.** Every call passes `think=False`,
which for Ollama hybrid-reasoning models (Qwen3, DeepSeek-R1) routes through
Ollama's native `/api/chat` to suppress the hidden `<think>` pass
(`infrastructure/llm_client.py:53`). Without it, Qwen3-32B emits reasoning
blocks that (a) consume the token budget before the answer is produced and
(b) break `parse_answer()`. Any replication on a hybrid-reasoning model that
omits this flag will not reproduce our numbers. Experiments
`001-tier1-disable-thinking`, `002-tier2-disable-thinking` and
`003-tier3-disable-thinking` document the effect.

---

## 3. Tier 1 — Single-LLM prompting

### P1 · Zero-shot *(user only)*
`frameworks/tier1/zero_shot.py:14` · class A · 1 call · `max_tokens` default

Exact replication of the VM14K Figure 6 prompt. No system message —
reproduced exactly as published, including the ungrammatical
`"Only response with 1 character"`, which is **not corrected** because this
prompt is the reproducibility anchor against VM14K Table 2.

```
{question}
Choose the correct option from these answers:
{options}
Only response with 1 character. Example: A
```

### P2 · Chain-of-Thought
`frameworks/tier1/cot.py:14` · class B · 1 call · `MAX_TOKENS_LONG`

System:
```
You are a medical expert. Think step by step before giving your final answer.
```

User:
```
{question}
Choose the correct option from these answers:
{options}

Think step by step. After your reasoning, end with: 'Final answer: X' where X is A, B, C, or D.
```

> **CoT-SC uses P2 unchanged.** `frameworks/tier1/cot_sc.py:16` is byte-identical
> to `cot.py:14`. CoT-SC introduces no new prompt: the k = 5 reasoning paths
> differ only through stochastic sampling at **T = 0.5**, and aggregation is
> programmatic (`majority_vote()`, ties broken by the first valid path). In the
> appendix, cite it as "P2, k = 5, T = 0.5" rather than reprinting the text —
> reprinting would misrepresent the method as prompt-based diversity.

### P3 · Self-Refine — initial answer
`frameworks/tier1/self_refine.py:23` · class B · `MAX_TOKENS_LONG` · T = 0.3

Differs from P2 in both messages: the system message says
`carefully / well-reasoned` rather than `step by step`, and the user message
omits `"After your reasoning,"`. The divergence is inherited from Madaan et al.
and is preserved rather than harmonised.

System:
```
You are a medical expert. Think carefully and provide a well-reasoned answer.
```

User:
```
{question}
Choose the correct option from these answers:
{options}

Think step by step. End with: 'Final answer: X' where X is A, B, C, or D.
```

### P4 · Self-Refine — critique
`frameworks/tier1/self_refine.py:45` · class C · `MAX_TOKENS_MEDIUM` · **T = 0.0**

`{prev_reasoning}` is the previous round's output truncated by
`extract_reasoning()` (≤ 500 chars, cut at the last sentence boundary).

The verdict is parsed by string match on `CORRECT. NO CHANGES` /
`NO CHANGES NEEDED` / `INCORRECT`. **Anything unparseable is treated as
"incorrect" and refinement continues** (`_is_correct_verdict()`, line 101) —
a conservative default that inflates call counts rather than stopping early.

System:
```
You are a critical medical reviewer. Evaluate the reasoning for errors.
```

User:
```
Medical question:
{question}

Options:
{options}

Previous reasoning:
{prev_reasoning}
Previous answer: {prev_answer}

Is this reasoning medically correct?
If correct: reply exactly 'CORRECT. No changes needed.'
If incorrect: reply 'INCORRECT.' and explain the error.
```

### P5 · Self-Refine — refine
`frameworks/tier1/self_refine.py:74` · class C · `MAX_TOKENS_LONG` · T = 0.3

System:
```
You are a medical expert refining your answer based on critique.
```

User:
```
Medical question:
{question}

Options:
{options}

Your previous reasoning:
{prev_reasoning}
Your previous answer: {prev_answer}

Critique:
{critique}

Revise your answer based on the critique. End with: 'Final answer: X' where X is A, B, C, or D.
```

---

## 4. Tier 2 — Ensemble

### P6 · Voting Ensemble *(user only)*
`frameworks/tier2/voting_ensemble.py:18` · class A + H · 3 calls · `max_tokens` default

**Text is byte-identical to P1.** The difference is not in the prompt:
`{options}` receives a **permuted** option order, reshuffled per run with the
run seed, and answers are mapped back to original-space letters via
`reverse_map` before voting so `individual_votes` stay comparable across runs.

Diversity here is **positional**, not stochastic — that is precisely what
separates Tier 2 from CoT-SC, and the reason both can share a tier-1 prompt
while measuring different things.

```
{question}
Choose the correct option from these answers:
{options}          ← permuted order, seed-controlled
Only response with 1 character. Example: A
```

---

## 5. Tier 3 — Multi-agent collaboration

### 5.1 MedAgents (Tang et al., ACL 2024)

Five steps, five prompts, 9 calls with N = 3 experts.

#### P7 · Step 1 — Expert gathering *(user only)*
`frameworks/tier3/medagents.py:29` · class F · `MAX_TOKENS_ROLES`

```
You are coordinating a medical consultation.

Medical question:
{question}

Options:
{options}

Which medical specialists would be most relevant to answer this question? List exactly 3 specialist roles, comma-separated. Example: Cardiologist, Nephrologist, General Practitioner
Only output the comma-separated list, nothing else.
```

Post-processing (`parse_roles()`, line 48): `<think>…</think>` blocks are
stripped before parsing; the list is split on commas, padded with
`"General Practitioner"` if fewer than 3 are returned and truncated if more;
empty output falls back to
`["General Practitioner", "Internal Medicine Physician", "Clinical Pharmacologist"]`.

#### P8 · Step 2 — Analysis proposition
`frameworks/tier3/medagents.py:72` · class D · `MAX_TOKENS_AGENT` · T = 0.3
Issued once per role (3 calls).

System:
```
You are an experienced {role}. Analyze medical questions carefully using your specialist expertise.
```

User:
```
Medical question:
{question}

Options:
{options}

As a {role}, provide:
1. Your analysis of the question
2. Your evaluation of each option (A, B, C, D)
3. Your answer (A/B/C/D)

Be concise and systematic.
```

#### P9 · Step 3 — Report summarization
`frameworks/tier3/medagents.py:102` · class F · `MAX_TOKENS_MEDIUM`

`{analyses}` is `"[{role}] (Answer: {answer}):\n{analysis}"` joined by blank
lines, where `{analysis}` is `extract_reasoning()`-truncated (≤ 500 chars).

System:
```
You are a medical coordinator summarizing specialist consultations.
```

User:
```
Medical question:
{question}

Options:
{options}

Specialist analyses:
{analyses}

Summarize the key points of agreement and disagreement into a concise consultation report (3-5 sentences).
```

#### P10 · Step 4 — Collaborative consultation
`frameworks/tier3/medagents.py:132` · class E · `MAX_TOKENS_MEDIUM` · T = 0.3
Issued once per role (3 calls).

**Single pass, not iterated.** Tang et al. iterate consultation to consensus;
this implementation runs exactly one round per expert, which is why the call
count is a fixed 9 rather than a range. Any figure or prose description must
not show a loop here.

System:
```
You are an experienced {role} participating in a multi-specialist consultation.
```

User:
```
Medical question:
{question}

Options:
{options}

Consultation report from all specialists:
{report}

Your previous analysis:
{own_analysis}

Given the collective consultation, do you maintain or update your answer? State your final answer (A/B/C/D) and brief reasoning.
```

#### P11 · Step 5 — Decision making
`frameworks/tier3/medagents.py:162` · class F · `MAX_TOKENS_SHORT`

`{opinions}` is `"- {role} → {answer}: {reasoning}"` joined by newlines.

System:
```
You are a senior physician making the final clinical decision.
```

User:
```
Medical question:
{question}

Options:
{options}

Final specialist positions:
{opinions}

Based on the collective consultation, what is the final answer? Respond with only 1 character: A, B, C, or D.
```

### 5.2 MedAgents-VI (this paper)

**Reuses P8–P11 verbatim** via direct import
(`frameworks/tier3/medagents_vi.py:19`). The only change is that **P7 is
deleted**, not rewritten: expert gathering becomes a dictionary lookup on
VM14K's `medical_topic` field.

| | MedAgents | MedAgents-VI |
|---|---|---|
| Step 1 | P7, 1 LLM call | `TOPIC_TO_ROLES` lookup, **0 calls** |
| Steps 2–5 | P8–P11 | P8–P11, unchanged |
| Total | 9 calls | 8 calls |

`TOPIC_TO_ROLES` (`medagents_vi.py:28`) maps **67 VM14K medical topics** onto
**48 distinct specialist roles**, 1–2 per topic
(e.g. `"Urology" → ["Urologist", "Nephrologist"]`).
`get_roles_from_topic()` accepts the multi-label topic list, deduplicates
preserving insertion order, caps at 3 roles, and falls back to
`GENERIC_ROLES = ["General Practitioner", "Internal Medicine Physician",
"Clinical Pharmacologist"]` when no topic matches. The full 67-row mapping
belongs in the appendix as its own table, not inline in the prompt list.

### 5.3 MDAgents (Kim et al., NeurIPS 2024)

Adaptive routing: complexity determines team size. 2 / 4 / 7 calls.

#### P12 · Step 1 — Complexity check
`frameworks/tier3/mdagents.py:44` · class F · **`MAX_TOKENS_COMPLEX` = 8**

System:
```
You are a General Practitioner assessing the complexity of medical queries. Classify based on clinical decision-making requirements.
```

User:
```
Medical question:
{question}

Options:
{options}

Classify this question's complexity:
- Low: straightforward factual/definitional question, single correct answer
- Moderate: requires mechanistic reasoning or pharmacological knowledge
- High: complex multi-symptom case requiring integrative clinical reasoning

Respond with exactly one word: Low, Moderate, or High.
```

Parsing (`parse_complexity()`, line 72) is worth documenting: first-word match,
then a word-boundary regex with a **negation guard** — `(?<!not\s)\bhigh\b` —
so `"not high"` does not classify as High and `"highly"` does not match at all.
Unparseable output defaults to `Moderate`.

#### P13 · Low branch — Primary Care Clinician
`frameworks/tier3/mdagents.py:106` · class B · `MAX_TOKENS_MEDIUM`

System:
```
You are a Primary Care Clinician. Think step by step.
```

User:
```
{question}

Options:
{options}

Think step by step, then answer with a single character: A, B, C, or D.
```

#### P14 · Moderate branch — MDT member
`frameworks/tier3/mdagents.py:126` · class A · `max_tokens` default

Issued 3× with `{agent_id}` ∈ {1, 2, 3}. The three calls are **independent** —
no agent sees another's output — and are aggregated by `majority_vote()`.
The persona is a bare index, not a specialty; classed as direct-answer rather
than role-conditioned for that reason.

System:
```
You are Medical Expert #{agent_id} in a multi-disciplinary team.
```

User:
```
Medical question:
{question}

Options:
{options}

Analyze carefully. Respond with only 1 character: A, B, C, or D.
```

#### P15 · High branch — ICT specialist
`frameworks/tier3/mdagents.py:147` · class D · `MAX_TOKENS_AGENT`

Issued once per role over the **fixed** roster `ICT_ROLES` (line 33): Internal
Medicine Physician, Clinical Pharmacologist, Pathologist, Radiologist, General
Practitioner. Unlike MedAgents, these roles are hard-coded rather than
LLM-selected — a design difference worth naming in the paper, since it means
MDAgents never pays for role selection but also never adapts roles to topic.

System:
```
You are a specialist {role} in an Integrated Care Team.
```

User:
```
Complex medical question requiring expert analysis:
{question}

Options:
{options}

Provide your expert reasoning and final answer (A/B/C/D).
```

#### P16 · High branch — ICT synthesizer
`frameworks/tier3/mdagents.py:166` · class F · `MAX_TOKENS_SHORT`

`{opinions}` is `"- {role} (Answer: {answer}): {reasoning}"` joined by newlines.

System:
```
You are a senior physician synthesizing an Integrated Care Team consultation.
```

User:
```
Medical question:
{question}

Options:
{options}

ICT specialist opinions:
{opinions}

Based on the collective expert reasoning, what is the final answer? Respond with only 1 character: A, B, C, or D.
```

### 5.4 MDAgents-VI (this paper)

**Reuses P13–P16 verbatim** via direct import
(`frameworks/tier3/mdagents_vi.py:22`). **P12 is deleted**: complexity comes
from VM14K's `difficulty_level` field through `map_difficulty()` —
`Easy → Low`, `Medium`/unknown `→ Moderate`, `Challenging`/`Hard → High`.

| Branch | MDAgents | MDAgents-VI |
|---|---|---|
| Low | 1 (P12) + 1 = 2 | 0 + 1 = **1** |
| Moderate | 1 (P12) + 3 = 4 | 0 + 3 = **3** |
| High | 1 (P12) + 5 + 1 = 7 | 0 + 5 + 1 = **6** |

Required for the limitations section: `difficulty_level` is a dataset label
read at inference time. This is an **oracle** condition — it bounds what
perfect complexity routing could achieve; it is not a deployable system. The
caveat already appears in the Bảng II footnote and must be repeated wherever
MDAgents-VI results are reported.

### 5.5 MAD Debate (Du et al., ICML 2024)

Role-free: agents are anonymous and numbered, never given a specialty. Two
prompts, ≤ 9 calls (3 agents × ≤ 3 rounds, early stop on consensus).

#### P17 · Round 1 — initial position
`frameworks/tier3/mad_debate.py:24` · class B · `MAX_TOKENS_DEBATE`

System:
```
You are Medical Agent #{agent_id}. Your goal is to reason carefully and argue for the correct answer.
```

User:
```
Medical question:
{question}

Options:
{options}

State your reasoning and answer (A/B/C/D).
```

#### P18 · Rounds 2+ — debate
`frameworks/tier3/mad_debate.py:46` · class E · `MAX_TOKENS_DEBATE`

`{others}` is `"Agent #{i}: {response}"` joined by blank lines, carrying the
**full raw responses** of the other agents — not `extract_reasoning()`
summaries. This follows Du et al. and is the one place in the codebase where
untruncated agent output is passed between agents; it is also the dominant
contributor to MAD Debate's prompt length.

System:
```
You are Medical Agent #{agent_id}, Round {round_num}. Review other agents' arguments and refine your answer.
```

User:
```
Medical question:
{question}

Options:
{options}

Other agents' arguments:
{others}

Do you maintain your position or update it? Argue concisely. End with your answer (A/B/C/D).
```

---

## 6. Cross-cutting observations for the paper

These are the points a reviewer will look for; each is defensible from the
inventory above.

**1. Prompt count does not track performance.** MedAgents-VI (4 prompts,
8 calls) outperforms MedAgents (5 prompts, 9 calls) on Qwen3-32B
(69.67% vs 68.33%). Deleting a prompt improved accuracy — evidence that the
LLM's own role selection was adding noise, not signal.

**2. Both `-VI` contributions are deletions, not rewrites.** Neither variant
introduces a new prompt: MedAgents-VI removes P7, MDAgents-VI removes P12.
This makes the ablation unusually clean — the manipulated variable is the
presence or absence of exactly one LLM call, with every other prompt held
byte-identical by *direct import* rather than by copy-paste. Worth stating
explicitly, because it forecloses the obvious reviewer objection that the
variants might differ in unlogged prompt wording.

**3. Output-format instructions are not uniform, by inheritance.** Three
distinct formats coexist: `"Only response with 1 character"` (P1, P6 — VM14K
verbatim), `"Final answer: X"` (P2, P3, P5 — CoT convention), and
`"Respond with only 1 character"` (P11, P14, P16 — MedAgents/MDAgents
convention). This follows from replicating three source papers faithfully and
should be disclosed rather than silently harmonised — harmonising it would
break comparability with the published baselines.

**4. Aggregation is split between LLM and algorithm, and that split *is* the
tier boundary.** Tier 1–2 aggregate programmatically (`majority_vote()`);
Tier 3 aggregates through an LLM (P11, P16) except MAD Debate, which reverts
to majority vote. This is the operational definition of "inter-agent
communication" used to assign tiers, and it also explains why Voting Ensemble
is excluded from the failure-mode table: a hard-majority aggregation step
admits no minority-suppression event by construction.

**5. Truncation is a real experimental variable.** Every inter-agent handoff
except P18's passes `extract_reasoning()` output — the first ≤ 500 characters,
cut at a sentence boundary. Information an agent produced but which falls past
that cut is invisible downstream. This directly bounds the Information Loss
metric (see `metrics/INFORMATION_LOSS_TODO.md`) and should be stated as a
limitation rather than left implicit.

---

## 7. Appendix packaging

Recommended structure for the paper, in order:

1. **Table A1** — the taxonomy table from §1 (classes A–H × frameworks). One
   page; gives the reader the map before the territory.
2. **Table A2** — shared inference configuration from §2, including the
   `think=False` disclosure.
3. **§A.3–A.5** — prompts P1–P18 verbatim, in the tier order used here, in a
   monospace verbatim environment. Do **not** reflow or re-wrap the text.
4. **Table A6** — the 67-row `TOPIC_TO_ROLES` mapping.
5. Cross-reference reused prompts (CoT-SC → P2, P6 ≡ P1, MedAgents-VI → P8–P11,
   MDAgents-VI → P13–P16) instead of reprinting them.

---

*Generated 2026-08-26 from `frameworks/` and `infrastructure/config.py`.
If a prompt changes, regenerate rather than hand-editing — the value of this
file is that it is verbatim.*
