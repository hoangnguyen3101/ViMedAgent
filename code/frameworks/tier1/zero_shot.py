"""
ViMedAgent — Tier 1: Zero-shot baseline
Exact VM14K Figure 6 prompt replication.

API calls: 1
Temperature: DEFAULT (0.3) — but diversity comes from multi-run seeds,
             not within a single call.
"""

from infrastructure.llm_client import call_llm, parse_answer
from frameworks.utils import format_options


def build_prompt(question: str, options: list[str]) -> list[dict]:
    """Exact VM14K Figure 6 prompt."""
    return [{
        "role": "user",
        "content": (
            f"{question}\n"
            f"Choose the correct option from these answers:\n"
            f"{format_options(options)}\n"
            f"Only response with 1 character. Example: A"
        ),
    }]


def run_zero_shot(question: str, options: list[str], model: str) -> dict:
    """
    Single zero-shot call. Primary reproducibility anchor:
    should match VM14K Table 2 published values.
    API calls: 1
    """
    msgs = build_prompt(question, options)
    raw  = call_llm(model, msgs, think=False)
    return {
        "answer":    parse_answer(raw),
        "raw":       raw,
        "api_calls": 1,
    }
