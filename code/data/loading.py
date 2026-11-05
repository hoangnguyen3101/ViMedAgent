"""Read VM14K or MedQA JSONL without modifying the source dataset."""

import json
from pathlib import Path

from infrastructure.choices import choice_letters


def normalize_sample(record: dict, namespace: str, line_number: int) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Each record must be a JSON object")
    sample = dict(record)
    question = sample.get("question")
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    # VM14K already contains two empty question strings. Preserve its sample
    # population and expose their count in --validate-only rather than filter.

    options = sample.get("options")
    if not isinstance(options, (dict, list)):
        raise ValueError("options must be a list or a dictionary keyed A-D/A-E")
    letters = choice_letters(len(options))
    if isinstance(options, dict):
        if set(options) != set(letters):
            raise ValueError(f"Option keys must be exactly {letters}")
        options = [options[letter] for letter in letters]
    if any(not isinstance(opt, str) or not opt.strip() for opt in options):
        raise ValueError("Every option must be a non-empty string")
    sample["options"] = options

    if "answer_idx" in sample:
        answer = sample["answer_idx"]
        if answer not in tuple(letters):
            raise ValueError(f"answer_idx must be one of {letters}")
        text = options[letters.index(answer)]
        if "answer" in sample and sample["answer"] not in (answer, text):
            raise ValueError("answer text does not match answer_idx")
    else:
        answer = sample.get("answer")
        if answer not in tuple(letters):
            raise ValueError(f"answer must be one of {letters}")
    sample["answer"] = answer

    # IDs depend on source position, never on seed, worker order or randomness.
    # Prefixing MedQA avoids collisions with VM14K question IDs.
    prefix = "medqa_us_" if "answer_idx" in sample else ""
    sample.setdefault("id", f"{prefix}{namespace}_{line_number:06d}")
    sample_id = sample["id"]
    if (isinstance(sample_id, bool) or not isinstance(sample_id, (str, int))
            or (isinstance(sample_id, str) and not sample_id.strip())):
        raise ValueError("id must be a non-empty string or an integer")
    return sample


def load_dataset(path: str) -> list[dict]:
    dataset = Path(path)
    samples = []
    ids = set()
    with dataset.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                sample = normalize_sample(json.loads(line), dataset.stem, line_number)
                if sample["id"] in ids:
                    raise ValueError(f"Duplicate id: {sample['id']!r}")
            except ValueError as exc:
                raise ValueError(f"{dataset}:{line_number}: {exc}") from exc
            ids.add(sample["id"])
            samples.append(sample)
    if not samples:
        raise ValueError(f"Dataset is empty: {dataset}")
    return samples
