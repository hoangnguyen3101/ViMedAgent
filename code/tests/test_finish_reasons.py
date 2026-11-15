"""finish_reason of every call_llm is counted per question, so every framework reports truncation."""

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import infrastructure.llm_client as llm
import infrastructure.config as cfg

MODEL = next(m for m, c in cfg.MODELS.items() if c["provider"] == "sglang")


def _client(reasons):
    it = iter(reasons)

    def create(**kw):
        reason = next(it)
        usage = None if reason == "no-usage" else SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
            finish_reason=None if reason == "no-usage" else reason,
            message=SimpleNamespace(content="B"))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_finish_reasons_follow_call_order_and_match_call_count(monkeypatch):
    client = _client(["stop", "length", "stop"])
    monkeypatch.setattr(llm, "_get_client", lambda api_key, base_url: client)
    llm.reset_usage()
    for _ in range(3):
        llm.call_llm(MODEL, [{"role": "user", "content": "q"}], max_tokens=1024)
    u = llm.get_usage()
    assert u["finish_reasons"] == ["stop", "length", "stop"]
    assert len(u["finish_reasons"]) == u["n_llm_calls"] == 3


def test_missing_finish_reason_is_recorded_as_none(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda api_key, base_url: _client(["no-usage"]))
    llm.reset_usage()
    llm.call_llm(MODEL, [{"role": "user", "content": "q"}])
    u = llm.get_usage()
    assert u["finish_reasons"] == [None] and u["n_usage_missing"] == 1


def test_reset_clears_previous_question(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda api_key, base_url: _client(["length"]))
    llm.reset_usage()
    llm.call_llm(MODEL, [{"role": "user", "content": "q"}])
    llm.reset_usage()
    assert llm.get_usage()["finish_reasons"] == []
