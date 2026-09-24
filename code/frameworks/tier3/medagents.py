"""
ViMedAgent — MedAgents (original, Tang et al. ACL 2024)
https://arxiv.org/abs/2311.10537

5-step Multi-disciplinary Collaboration (MC) framework:
  Step 1 — Expert Gathering:   LLM suggests relevant specialist roles
  Step 2 — Analysis Proposition: each expert analyzes question + each option
  Step 3 — Report Summarization: aggregate all analyses into a report
  Step 4 — Collaborative Consultation: experts discuss report, iterate to consensus
  Step 5 — Decision Making:    final answer from consensus

This is the ORIGINAL variant where the LLM autonomously selects roles.
For the VM14K-adapted variant (using medical_topic), see medagents_vi.py.

API calls per question: 1 (gather) + N (analysis) + 1 (report) + N (discuss) + 1 (decision)
  With N=3 experts: 1+3+1+3+1 = 9 calls
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import (
    MAX_TOKENS_SHORT, MAX_TOKENS_ROLES, MAX_TOKENS_MEDIUM,
    MAX_TOKENS_AGENT,
)
from frameworks.utils import extract_reasoning, format_options


# ── Step 1: Expert Gathering ──────────────────────────────

def prompt_gather_experts(question: str, options: list[str]) -> list[dict]:
    """
    Ask the LLM which medical specialists are relevant for this question.
    Returns a comma-separated list of roles.
    """
    return [{
        "role": "user",
        "content": (
            f"You are coordinating a medical consultation.\n\n"
            f"Medical question:\n{question}\n\n"
            f"Options:\n{format_options(options)}\n\n"
            f"Which medical specialists would be most relevant to answer this question? "
            f"List exactly 3 specialist roles, comma-separated. "
            f"Example: Cardiologist, Nephrologist, General Practitioner\n"
            f"Only output the comma-separated list, nothing else."
        ),
    }]


def parse_roles(raw: str, fallback_n: int = 3) -> list[str]:
    """Parse comma-separated roles from LLM output.
    Strips <think>...</think> blocks before parsing (Qwen3, DeepSeek-R1).
    """
    import re
    FALLBACK = ["General Practitioner", "Internal Medicine Physician",
                "Clinical Pharmacologist"]
    if not raw:
        return FALLBACK[:fallback_n]
    # Strip extended thinking blocks before parsing
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if not cleaned:
        return FALLBACK[:fallback_n]
    roles = [r.strip() for r in cleaned.split(",") if r.strip()]
    if not roles:
        return FALLBACK[:fallback_n]
    # Cap at fallback_n, pad if fewer
    while len(roles) < fallback_n:
        roles.append("General Practitioner")
    return roles[:fallback_n]


# ── Step 2: Analysis Proposition ─────────────────────────

def prompt_analysis(question: str, options: list[str], role: str) -> list[dict]:
    """
    Each expert analyzes: (a) the question, (b) each option A/B/C/D.
    Per MedAgents paper Section 2.2.
    """
    return [
        {
            "role": "system",
            "content": (
                f"You are an experienced {role}. "
                f"Analyze medical questions carefully using your specialist expertise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"As a {role}, provide:\n"
                f"1. Your analysis of the question\n"
                f"2. Your evaluation of each option (A, B, C, D)\n"
                f"3. Your answer (A/B/C/D)\n\n"
                f"Be concise and systematic."
            ),
        },
    ]


# ── Step 3: Report Summarization ─────────────────────────

def prompt_report(question: str, options: list[str],
                  analyses: list[dict]) -> list[dict]:
    """
    Aggregate all expert analyses into a unified report.
    Per MedAgents paper Section 2.3.
    """
    analyses_str = "\n\n".join(
        f"[{a['role']}] (Answer: {a['answer']}):\n{a['analysis']}"
        for a in analyses
    )
    return [
        {
            "role": "system",
            "content": "You are a medical coordinator summarizing specialist consultations.",
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Specialist analyses:\n{analyses_str}\n\n"
                f"Summarize the key points of agreement and disagreement "
                f"into a concise consultation report (3-5 sentences)."
            ),
        },
    ]


# ── Step 4: Collaborative Consultation ───────────────────

def prompt_consult(question: str, options: list[str],
                   role: str, report: str,
                   own_analysis: str) -> list[dict]:
    """
    Each expert reads the report and updates their position.
    Per MedAgents paper Section 2.4.
    """
    return [
        {
            "role": "system",
            "content": (
                f"You are an experienced {role} participating in a multi-specialist consultation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Consultation report from all specialists:\n{report}\n\n"
                f"Your previous analysis:\n{own_analysis}\n\n"
                f"Given the collective consultation, do you maintain or update your answer? "
                f"State your final answer (A/B/C/D) and brief reasoning."
            ),
        },
    ]


# ── Step 5: Decision Making ───────────────────────────────

def prompt_decision(question: str, options: list[str],
                    final_opinions: list[dict]) -> list[dict]:
    """
    Final decision from all expert positions after consultation.
    Per MedAgents paper Section 2.5.
    """
    opinions_str = "\n".join(
        f"- {op['role']} → {op['answer']}: {op['reasoning']}"
        for op in final_opinions
    )
    return [
        {
            "role": "system",
            "content": "You are a senior physician making the final clinical decision.",
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Final specialist positions:\n{opinions_str}\n\n"
                f"Based on the collective consultation, what is the final answer? "
                f"Respond with only 1 character: A, B, C, or D."
            ),
        },
    ]


# ── Runner ────────────────────────────────────────────────

def run_medagents(question: str, options: list[str], model: str,
                  n_experts: int = 3) -> dict:
    """
    MedAgents original (Tang et al. ACL 2024).
    LLM autonomously selects specialist roles in Step 1.
    API calls: 1 + n_experts + 1 + n_experts + 1 = 2*n_experts + 3
      (default n=3 → 9 calls)
    """
    api_calls   = 0
    audit_trail = []

    # ── Step 1: Expert Gathering ──────────────────────────
    msgs      = prompt_gather_experts(question, options)
    raw_roles = call_llm(model, msgs, max_tokens=MAX_TOKENS_ROLES, think=False)
    roles     = parse_roles(raw_roles, fallback_n=n_experts)
    api_calls += 1
    audit_trail.append({"step": "expert_gathering", "raw": raw_roles, "roles": roles})

    # ── Step 2: Analysis Proposition ─────────────────────
    analyses = []
    for role in roles:
        msgs = prompt_analysis(question, options, role)
        raw  = call_llm(model, msgs, temperature=0.3, max_tokens=MAX_TOKENS_AGENT, think=False)
        ans  = parse_answer(raw)
        api_calls += 1
        analyses.append({
            "role":     role,
            "analysis": extract_reasoning(raw),
            "answer":   ans,
            "raw":      raw,
        })
        audit_trail.append({"step": "analysis", "role": role, "answer": ans,
                            "reasoning": analyses[-1]["analysis"], "raw": raw})

    # ── Step 3: Report Summarization ─────────────────────
    msgs   = prompt_report(question, options, analyses)
    report = call_llm(model, msgs, max_tokens=MAX_TOKENS_MEDIUM, think=False)
    api_calls += 1
    audit_trail.append({"step": "report", "raw": report})

    # ── Step 4: Collaborative Consultation ───────────────
    final_opinions = []
    for a in analyses:
        msgs = prompt_consult(question, options, a["role"], report, a["analysis"])
        raw  = call_llm(model, msgs, temperature=0.3, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        ans  = parse_answer(raw)
        api_calls += 1
        final_opinions.append({
            "role":      a["role"],
            "answer":    ans,
            "reasoning": extract_reasoning(raw),
            "raw":       raw,
        })
        audit_trail.append({"step": "consultation", "role": a["role"], "answer": ans,
                            "reasoning": final_opinions[-1]["reasoning"], "raw": raw})

    # ── Step 5: Decision Making ───────────────────────────
    msgs    = prompt_decision(question, options, final_opinions)
    raw_dec = call_llm(model, msgs, max_tokens=MAX_TOKENS_SHORT, think=False)
    ans_dec = parse_answer(raw_dec)
    api_calls += 1
    audit_trail.append({"step": "decision", "answer": ans_dec, "raw": raw_dec})

    return {
        "answer":          ans_dec,
        "roles":           roles,
        "round1_answers":  [a["answer"] for a in analyses],
        "round2_answers":  [op["answer"] for op in final_opinions],
        "individual_votes":[op["answer"] for op in final_opinions],
        "audit_trail":     audit_trail,
        "api_calls":       api_calls,
    }