"""Answer labels shared by four- and five-option MCQ pipelines."""

SUPPORTED_LETTERS = "ABCDE"


def choice_letters(n_options: int = 4) -> str:
    if n_options not in (4, 5):
        raise ValueError(f"Expected 4 or 5 options, got {n_options}")
    return SUPPORTED_LETTERS[:n_options]


def choice_names(n_options: int = 4) -> str:
    """Keep the original four-option prompt wording exactly."""
    letters = choice_letters(n_options)
    return ", ".join(letters[:-1]) + ", or " + letters[-1]
