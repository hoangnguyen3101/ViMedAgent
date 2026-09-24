"""
ViMedAgent — Tier 1: Chain-of-Thought (CoT)
Wei et al. NeurIPS 2022 — "Chain-of-Thought Prompting Elicits Reasoning in LLMs"

Standard baseline in MedAgentBoard, MedAgentsBench, MDAgents.
API calls: 1
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import MAX_TOKENS_LONG
from frameworks.utils import format_options


def build_prompt(question: str, options: list[str]) -> list[dict]:
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


def run_cot(question: str, options: list[str], model: str) -> dict:
    """
    Chain-of-Thought. Strong single-LLM baseline.
    API calls: 1
    """
    msgs = build_prompt(question, options)
    raw  = call_llm(model, msgs, max_tokens=MAX_TOKENS_LONG, think=False)
    return {
        "answer":    parse_answer(raw),
        "raw":       raw,
        "api_calls": 1,
    }
