"""
MedAgents consultation ablations on vm14k_sub2000: one flow, two knobs.

  run_medagents_ablation(question, options, model, panel, consult_input)
    panel          model answering expert i (analysis AND consultation of that expert)
    consult_input  what each consulting expert reads besides its own analysis:
                     "report"              Step 3 report, as in MedAgents          (A)
                     "analyses_full"       no report; all three analyses verbatim  (B, chosen: B2)
                     "analyses_truncated"  no report; all three analyses as the report
                                           writer sees them (extract_reasoning, ~500 chars) (B1)

  A  medagents_hetero    panel = (Qwen3-32B, Llama-3.3-70B, Llama-3.1-8B), consult_input="report"
  B  medagents_noreport  panel = (model,) * 3,                           consult_input="analyses_full"

Prompts and flow are those of frameworks/tier3/medagents.py: the prompt builders and the role
parser are imported, not copied. With panel = (model,)*3 and consult_input="report" the calls are
identical, word for word, to run_medagents (tests/test_ablation.py checks this).
Expert gathering, report and decision always use `model` (the run's --model).

For B the consultation prompt is prompt_consult() with its report block
"Consultation report from all specialists:\n<report>" replaced by
"Analyses from all specialists:\n<the three analyses>"; every other word is unchanged.
Each analysis is shown as "[role] (Answer: X):\n<text>", the format prompt_report already uses.
The decision step is unchanged (it never read the report).

Each audit_trail step also carries "model" and "call" (tokens_in, tokens_out, finish_reason,
temperature, max_tokens) from llm_client.get_last_call(); finish_reason "length" = cap hit.
API calls per question: 9 with "report", 8 without (1 + 3 + [1] + 3 + 1).
"""

from infrastructure.llm_client import call_llm, parse_answer, get_last_call
from infrastructure.config import (
    MAX_TOKENS_SHORT, MAX_TOKENS_ROLES, MAX_TOKENS_MEDIUM, MAX_TOKENS_AGENT,
)
from frameworks.utils import extract_reasoning
from frameworks.tier3.medagents import (
    prompt_gather_experts, parse_roles, prompt_analysis,
    prompt_report, prompt_consult, prompt_decision,
)

ORCHESTRATOR = "sglang-qwen3-32b"
HETERO_PANEL = ("sglang-qwen3-32b", "sglang-llama3.3-70b", "sglang-llama3.1-8b-awq")
NOREPORT_CONSULT_INPUT = "analyses_full"          # B2, chosen 2026-11-11
CONSULT_INPUTS = ("report", "analyses_full", "analyses_truncated")

_REPORT_HEADING = "Consultation report from all specialists:\n"
_ANALYSES_HEADING = "Analyses from all specialists:\n"
_PLACEHOLDER = "\x00REPORT_BLOCK\x00"
_CALL_KEYS = ("tokens_in", "tokens_out", "finish_reason", "temperature", "max_tokens")


def _call(model: str, msgs: list[dict], **kwargs) -> tuple[str, dict]:
    raw = call_llm(model, msgs, **kwargs)
    info = get_last_call()
    return raw, {k: info.get(k) for k in _CALL_KEYS}


def analyses_block(analyses: list[dict], full: bool) -> str:
    """The three analyses in prompt_report's format; full=True uses the raw text verbatim."""
    return "\n\n".join(
        f"[{a['role']}] (Answer: {a['answer']}):\n{a['raw'] if full else a['analysis']}"
        for a in analyses
    )


def prompt_consult_analyses(question: str, options: list[str], role: str,
                            block: str, own_analysis: str) -> list[dict]:
    """prompt_consult with its report block swapped for the analyses block, nothing else."""
    msgs = prompt_consult(question, options, role, _PLACEHOLDER, own_analysis)
    user = msgs[-1]["content"]
    target = _REPORT_HEADING + _PLACEHOLDER
    if user.count(target) != 1:
        raise RuntimeError("prompt_consult no longer has the expected report block; "
                           "update medagents_ablation before running B")
    return [*msgs[:-1], {**msgs[-1], "content": user.replace(target, _ANALYSES_HEADING + block, 1)}]


def run_medagents_ablation(question: str, options: list[str], model: str,
                           panel: tuple[str, ...], consult_input: str) -> dict:
    if consult_input not in CONSULT_INPUTS:
        raise ValueError(f"consult_input must be one of {CONSULT_INPUTS}, got {consult_input!r}")
    n_experts = len(panel)
    api_calls = 0
    audit_trail = []

    # ── Step 1: Expert Gathering (model) ──────────────────
    msgs = prompt_gather_experts(question, options)
    raw_roles, call = _call(model, msgs, max_tokens=MAX_TOKENS_ROLES, think=False)
    roles = parse_roles(raw_roles, fallback_n=n_experts)
    api_calls += 1
    audit_trail.append({"step": "expert_gathering", "raw": raw_roles, "roles": roles,
                        "model": model, "call": call})

    # ── Step 2: Analysis Proposition (expert i → panel[i]) ─
    analyses = []
    for role, expert_model in zip(roles, panel):
        msgs = prompt_analysis(question, options, role)
        raw, call = _call(expert_model, msgs, temperature=0.3, max_tokens=MAX_TOKENS_AGENT, think=False)
        ans = parse_answer(raw, n_options=len(options))
        api_calls += 1
        analyses.append({
            "role":     role,
            "analysis": extract_reasoning(raw),
            "answer":   ans,
            "raw":      raw,
            "model":    expert_model,
        })
        audit_trail.append({"step": "analysis", "role": role, "answer": ans,
                            "reasoning": analyses[-1]["analysis"], "raw": raw,
                            "model": expert_model, "call": call})

    # ── Step 3: Report Summarization (model) — only when consultants read the report ─
    report = None
    if consult_input == "report":
        msgs = prompt_report(question, options, analyses)
        report, call = _call(model, msgs, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        api_calls += 1
        audit_trail.append({"step": "report", "raw": report, "model": model, "call": call})
    else:
        block = analyses_block(analyses, full=(consult_input == "analyses_full"))

    # ── Step 4: Collaborative Consultation (same model as the expert's analysis) ─
    final_opinions = []
    for a in analyses:
        if consult_input == "report":
            msgs = prompt_consult(question, options, a["role"], report, a["analysis"])
        else:
            msgs = prompt_consult_analyses(question, options, a["role"], block, a["analysis"])
        raw, call = _call(a["model"], msgs, temperature=0.3, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        ans = parse_answer(raw, n_options=len(options))
        api_calls += 1
        final_opinions.append({
            "role":      a["role"],
            "answer":    ans,
            "reasoning": extract_reasoning(raw),
            "raw":       raw,
        })
        audit_trail.append({"step": "consultation", "role": a["role"], "answer": ans,
                            "reasoning": final_opinions[-1]["reasoning"], "raw": raw,
                            "model": a["model"], "call": call})

    # ── Step 5: Decision Making (model) ───────────────────
    msgs = prompt_decision(question, options, final_opinions)
    raw_dec, call = _call(model, msgs, max_tokens=MAX_TOKENS_SHORT, think=False)
    ans_dec = parse_answer(raw_dec, n_options=len(options))
    api_calls += 1
    audit_trail.append({"step": "decision", "answer": ans_dec, "raw": raw_dec,
                        "model": model, "call": call})

    return {
        "answer":          ans_dec,
        "roles":           roles,
        "round1_answers":  [a["answer"] for a in analyses],
        "round2_answers":  [op["answer"] for op in final_opinions],
        "individual_votes":[op["answer"] for op in final_opinions],
        "audit_trail":     audit_trail,
        "api_calls":       api_calls,
    }


def run_medagents_hetero(question: str, options: list[str], model: str = ORCHESTRATOR) -> dict:
    """Ablation A: heterogeneous panel, report kept."""
    return run_medagents_ablation(question, options, model, HETERO_PANEL, "report")


def run_medagents_noreport(question: str, options: list[str], model: str = ORCHESTRATOR) -> dict:
    """Ablation B: one-model panel, no report; consultants read the analyses (B2: verbatim)."""
    return run_medagents_ablation(question, options, model, (model,) * 3, NOREPORT_CONSULT_INPUT)
