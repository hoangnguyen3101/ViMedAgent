"""
ViMedAgent — Tier 1: CoT Self-Consistency (CoT-SC)
Wang et al. NeurIPS 2023 — "Self-Consistency Improves CoT Reasoning in LLMs"

k=5 independent CoT paths → majority vote.
Diversity from temperature>0 sampling (not shuffling).
Tie-breaking: first path's answer (deterministic fallback).
API calls: k (default 5)
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import COT_SC_TEMPERATURE, MAX_TOKENS_LONG
from frameworks.utils import majority_vote, format_options


def build_prompt(question: str, options: list[str]) -> list[dict]:
    """Same CoT prompt — diversity comes from stochastic sampling."""
    return [
        {
            "role": "system",
            "content": (
                "You are a medical expert. "
                "Think step by step before giving your final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{question}\n"
                f"Choose the correct option from these answers:\n"
                f"{format_options(options)}\n\n"
                f"Think step by step. "
                f"After your reasoning, end with: "
                f"'Final answer: X' where X is A, B, C, or D."
            ),
        },
    ]


def run_cot_sc(
    question: str,
    options: list[str],
    model: str,
    k: int = 5,
    temperature: float = COT_SC_TEMPERATURE,
) -> dict:
    """
    CoT Self-Consistency: k independent CoT paths → majority vote.
    Tie-breaking: first path's answer (fallback).

    individual_votes: all k answers — used for Pass@k computation.
    Pass@k here is Pass@5; for fair comparison with voting_ensemble
    Pass@3, use individual_votes[:3].

    API calls: k (default 5)
    """
    votes: list[str | None] = []
    raws:  list[str]        = []

    for _ in range(k):
        msgs = build_prompt(question, options)
        raw  = call_llm(model, msgs, temperature=temperature, max_tokens=MAX_TOKENS_LONG, think=False)
        ans  = parse_answer(raw)
        votes.append(ans)
        raws.append(raw)

    first_valid = next((v for v in votes if v in ("A", "B", "C", "D")), None)
    final       = majority_vote(votes, fallback=first_valid)

    return {
        "answer":           final,
        "individual_votes": votes,
        "raw":              raws,
        "api_calls":        k,
    }
