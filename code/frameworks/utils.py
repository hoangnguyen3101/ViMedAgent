"""
ViMedAgent — Shared helpers for all frameworks
"""

from collections import Counter

from infrastructure.choices import choice_letters, choice_names


def majority_vote(answers: list, fallback: str | None = None,
                  n_options: int = 4) -> str | None:
    """
    Majority vote over the labels available for this question.
    Tie-breaking: fall back to `fallback` if provided (e.g. first run's answer),
    otherwise alphabetical (A > B > C > D) for determinism.
    Returns None if no valid answer.
    """
    letters = tuple(choice_letters(n_options))
    valid = [a for a in answers if a in letters]
    if not valid:
        return None

    counts    = Counter(valid)
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Tie: use fallback if it's one of the tied candidates
    if fallback in candidates:
        return fallback

    # Otherwise alphabetical
    return sorted(candidates)[0]


def extract_reasoning(raw: str, max_chars: int = 500) -> str:
    """
    Truncate raw LLM output to last complete sentence within max_chars.
    Used to pass reasoning summaries between agents without token bloat.
    """
    if not raw:
        return ""
    summary = raw[:max_chars]
    last_period = max(
        summary.rfind("。"),
        summary.rfind("."),
        summary.rfind("!"),
        summary.rfind("?"),
    )
    if last_period > 200:
        summary = summary[:last_period + 1]
    return summary


def format_options(options: list[str]) -> str:
    """Format list of options as 'A. opt\\nB. opt\\n...'"""
    return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
