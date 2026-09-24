"""
ViMedAgent — Tier 1: Self-Refine
Madaan et al. NeurIPS 2023 — "Self-Refine: Iterative Refinement with Self-Feedback"

Protocol:
  Round 0: initial CoT answer
  Round r: critique previous answer (T=0) → if CORRECT → stop
                                           → else refine (T=0.3)
  Final: last round's answer

API calls: 1 + r*(1+1) where r = actual rounds run (1..max_rounds)
  Best case (correct on round 1 critique): 1 + 1 = 2 calls
  Worst case (max_rounds=2, no early stop): 1 + 2*2 = 5 calls
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import MAX_TOKENS_LONG, MAX_TOKENS_MEDIUM
from frameworks.utils import extract_reasoning, format_options


# ── Prompts ───────────────────────────────────────────────

def _prompt_initial(question: str, options: list[str]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a medical expert. "
                "Think carefully and provide a well-reasoned answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{question}\n"
                f"Choose the correct option from these answers:\n"
                f"{format_options(options)}\n\n"
                f"Think step by step. "
                f"End with: 'Final answer: X' where X is A, B, C, or D."
            ),
        },
    ]


def _prompt_critique(
    question: str,
    options: list[str],
    prev_reasoning: str,
    prev_answer: str,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a critical medical reviewer. "
                "Evaluate the reasoning for errors."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Previous reasoning:\n{prev_reasoning}\n"
                f"Previous answer: {prev_answer}\n\n"
                f"Is this reasoning medically correct?\n"
                f"If correct: reply exactly 'CORRECT. No changes needed.'\n"
                f"If incorrect: reply 'INCORRECT.' and explain the error."
            ),
        },
    ]


def _prompt_refine(
    question: str,
    options: list[str],
    prev_reasoning: str,
    prev_answer: str,
    critique: str,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a medical expert refining your answer based on critique.",
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Your previous reasoning:\n{prev_reasoning}\n"
                f"Your previous answer: {prev_answer}\n\n"
                f"Critique:\n{critique}\n\n"
                f"Revise your answer based on the critique. "
                f"End with: 'Final answer: X' where X is A, B, C, or D."
            ),
        },
    ]


def _is_correct_verdict(critique_raw: str) -> bool:
    """Return True if the critique says the previous answer is already correct."""
    t = critique_raw.upper()
    if "CORRECT. NO CHANGES" in t or "NO CHANGES NEEDED" in t:
        return True
    if "INCORRECT" in t:
        return False
    return False  # conservative default: keep refining


# ── Runner ────────────────────────────────────────────────

def run_self_refine(
    question: str,
    options: list[str],
    model: str,
    max_rounds: int = 2,
) -> dict:
    """
    Self-Refine: iterative self-critique and refinement.
    audit_trail records each step for failure analysis.
    API calls: 2 (early stop) to 5 (max_rounds=2, no stop).
    """
    api_calls   = 0
    audit_trail = []

    # Round 0: initial answer
    raw = call_llm(model, _prompt_initial(question, options),
                   temperature=0.3, max_tokens=MAX_TOKENS_LONG, think=False)
    ans = parse_answer(raw)
    api_calls += 1

    current_reasoning = extract_reasoning(raw)
    current_answer    = ans
    audit_trail.append({
        "round": 0, "step": "initial",
        "answer": ans, "reasoning": current_reasoning, "raw": raw,
    })

    # Refinement rounds
    for rnd in range(1, max_rounds + 1):

        # Critique (deterministic — T=0)
        raw_crit = call_llm(
            model,
            _prompt_critique(question, options, current_reasoning,
                             current_answer or "?"),
            temperature=0.0, max_tokens=MAX_TOKENS_MEDIUM, think=False,
        )
        api_calls += 1
        verdict = "correct" if _is_correct_verdict(raw_crit) else "incorrect"
        audit_trail.append({
            "round": rnd, "step": "critique",
            "verdict": verdict, "raw": raw_crit,
        })

        if verdict == "correct":
            audit_trail.append({"round": rnd, "event": "early_stop"})
            break

        # Refine
        raw_ref = call_llm(
            model,
            _prompt_refine(question, options, current_reasoning,
                           current_answer or "?",
                           extract_reasoning(raw_crit)),
            temperature=0.3, max_tokens=MAX_TOKENS_LONG, think=False,
        )
        ans_ref = parse_answer(raw_ref)
        api_calls += 1

        current_reasoning = extract_reasoning(raw_ref)
        current_answer    = ans_ref
        audit_trail.append({
            "round": rnd, "step": "refine",
            "answer": ans_ref, "reasoning": current_reasoning, "raw": raw_ref,
        })

    return {
        "answer":      current_answer,
        "audit_trail": audit_trail,
        "api_calls":   api_calls,
    }
