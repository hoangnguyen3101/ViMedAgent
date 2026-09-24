"""
ViMedAgent — MDAgents (original, Kim et al. NeurIPS 2024)
https://arxiv.org/abs/2404.15155

4-step adaptive framework:
  Step 1 — Complexity Check:  moderator LLM classifies question as Low/Moderate/High
  Step 2 — Expert Recruitment: recruit team based on complexity
             Low      → 1 Primary Care Clinician (PCC), CoT
             Moderate → Multi-disciplinary Team (MDT), 3 agents vote
             High     → Integrated Care Team (ICT), 5 agents + synthesizer
  Step 3 — Analysis & Synthesis
  Step 4 — Decision Making

This is the ORIGINAL variant where LLM autonomously classifies complexity.
For the VM14K-adapted variant (using difficulty_level), see mdagents_vi.py.

API calls:
  Low:      1 (check) + 1 (PCC) = 2
  Moderate: 1 (check) + 3 (MDT) = 4
  High:     1 (check) + 5 (ICT) + 1 (synth) = 7
"""

import re

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import (
    MAX_TOKENS_SHORT, MAX_TOKENS_COMPLEX, MAX_TOKENS_MEDIUM, MAX_TOKENS_AGENT,
)
from frameworks.utils import majority_vote, extract_reasoning, format_options


# ── Expert roles for High complexity ICT ─────────────────
ICT_ROLES = [
    "Internal Medicine Physician",
    "Clinical Pharmacologist",
    "Pathologist",
    "Radiologist",
    "General Practitioner",
]


# ── Step 1: Complexity Check ──────────────────────────────

def prompt_complexity_check(question: str, options: list[str]) -> list[dict]:
    """
    Moderator LLM (as General Practitioner) classifies query complexity.
    Per MDAgents paper Section 3.1.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a General Practitioner assessing the complexity of medical queries. "
                "Classify based on clinical decision-making requirements."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Classify this question's complexity:\n"
                f"- Low: straightforward factual/definitional question, single correct answer\n"
                f"- Moderate: requires mechanistic reasoning or pharmacological knowledge\n"
                f"- High: complex multi-symptom case requiring integrative clinical reasoning\n\n"
                f"Respond with exactly one word: Low, Moderate, or High."
            ),
        },
    ]


def parse_complexity(raw: str) -> str:
    """
    Parse complexity level from LLM output. Returns 'Low'/'Moderate'/'High'.
    Uses word-boundary matching to avoid false positives
    (e.g. 'not high' or 'highly' should not match 'High').
    Falls back to first exact word found, then substring.
    """
    if not raw:
        return "Moderate"
    text = raw.strip()

    # Try exact match first (LLM was asked for single word)
    word = text.split()[0].lower().rstrip(".,;:") if text.split() else ""
    if word == "high":
        return "High"
    if word == "low":
        return "Low"
    if word in ("moderate", "medium"):
        return "Moderate"

    # Fallback: word-boundary search in full response
    # Exclude negated forms like "not high"
    if re.search(r'(?<!not\s)\bhigh\b', text, re.IGNORECASE):
        return "High"
    if re.search(r'(?<!not\s)\blow\b', text, re.IGNORECASE):
        return "Low"
    if re.search(r'\b(moderate|medium)\b', text, re.IGNORECASE):
        return "Moderate"

    return "Moderate"


# ── Low: Single PCC with CoT ──────────────────────────────

def prompt_pcc(question: str, options: list[str]) -> list[dict]:
    """Primary Care Clinician with Chain-of-Thought."""
    return [
        {
            "role": "system",
            "content": "You are a Primary Care Clinician. Think step by step.",
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Think step by step, then answer with a single character: A, B, C, or D."
            ),
        },
    ]


# ── Moderate: MDT — 3 independent agents vote ─────────────

def prompt_mdt_agent(question: str, options: list[str],
                     agent_id: int) -> list[dict]:
    """One MDT member votes independently."""
    return [
        {
            "role": "system",
            "content": f"You are Medical Expert #{agent_id + 1} in a multi-disciplinary team.",
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Analyze carefully. Respond with only 1 character: A, B, C, or D."
            ),
        },
    ]


# ── High: ICT — 5 experts + synthesizer ──────────────────

def prompt_ict_expert(question: str, options: list[str],
                      role: str) -> list[dict]:
    """One ICT specialist provides expert analysis."""
    return [
        {
            "role": "system",
            "content": f"You are a specialist {role} in an Integrated Care Team.",
        },
        {
            "role": "user",
            "content": (
                f"Complex medical question requiring expert analysis:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Provide your expert reasoning and final answer (A/B/C/D)."
            ),
        },
    ]


def prompt_ict_synthesizer(question: str, options: list[str],
                            opinions: list[dict]) -> list[dict]:
    """ICT synthesizer integrates all expert opinions."""
    opinions_str = "\n".join(
        f"- {op['role']} (Answer: {op['answer']}): {op['reasoning']}"
        for op in opinions
    )
    return [
        {
            "role": "system",
            "content": "You are a senior physician synthesizing an Integrated Care Team consultation.",
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"ICT specialist opinions:\n{opinions_str}\n\n"
                f"Based on the collective expert reasoning, what is the final answer? "
                f"Respond with only 1 character: A, B, C, or D."
            ),
        },
    ]


# ── Runner ────────────────────────────────────────────────

def run_mdagents(question: str, options: list[str], model: str) -> dict:
    """
    MDAgents original (Kim et al. NeurIPS 2024).
    LLM autonomously classifies question complexity in Step 1.
    """
    api_calls   = 0
    audit_trail = []

    # ── Step 1: Complexity Check ──────────────────────────
    msgs       = prompt_complexity_check(question, options)
    raw_check  = call_llm(model, msgs, max_tokens=MAX_TOKENS_COMPLEX, think=False)
    complexity = parse_complexity(raw_check)
    api_calls += 1
    audit_trail.append({"step": "complexity_check", "raw": raw_check,
                        "complexity": complexity})

    # ── Step 2-4: Route by complexity ────────────────────

    if complexity == "Low":
        msgs = prompt_pcc(question, options)
        raw  = call_llm(model, msgs, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        ans  = parse_answer(raw)
        api_calls += 1
        audit_trail.append({"step": "pcc", "answer": ans,
                            "reasoning": extract_reasoning(raw), "raw": raw})
        return {
            "answer":     ans,
            "complexity": complexity,
            "mode":       "pcc",
            "audit_trail": audit_trail,
            "api_calls":  api_calls,
        }

    elif complexity == "Moderate":
        votes = []
        for i in range(3):
            msgs = prompt_mdt_agent(question, options, i)
            raw  = call_llm(model, msgs, think=False)
            ans  = parse_answer(raw)
            votes.append(ans)
            api_calls += 1
            audit_trail.append({"step": "mdt", "agent": i + 1, "answer": ans,
                                "reasoning": extract_reasoning(raw), "raw": raw})
        return {
            "answer":           majority_vote(votes),
            "complexity":       complexity,
            "mode":             "mdt",
            "individual_votes": votes,
            "audit_trail":      audit_trail,
            "api_calls":        api_calls,
        }

    else:  # High
        opinions = []
        for role in ICT_ROLES:
            msgs = prompt_ict_expert(question, options, role)
            raw  = call_llm(model, msgs, max_tokens=MAX_TOKENS_AGENT, think=False)
            ans  = parse_answer(raw)
            api_calls += 1
            opinions.append({
                "role":      role,
                "answer":    ans,
                "reasoning": extract_reasoning(raw),
            })
            audit_trail.append({"step": "ict_expert", "role": role, "answer": ans,
                                "reasoning": opinions[-1]["reasoning"], "raw": raw})

        msgs    = prompt_ict_synthesizer(question, options, opinions)
        raw_syn = call_llm(model, msgs, max_tokens=MAX_TOKENS_SHORT, think=False)
        ans_syn = parse_answer(raw_syn)
        api_calls += 1
        audit_trail.append({"step": "ict_synthesizer", "answer": ans_syn, "raw": raw_syn})

        return {
            "answer":           ans_syn,
            "complexity":       complexity,
            "mode":             "ict",
            "individual_votes": [op["answer"] for op in opinions],
            "audit_trail":      audit_trail,
            "api_calls":        api_calls,
        }
