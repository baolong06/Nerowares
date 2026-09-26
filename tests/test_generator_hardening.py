"""Provider-boundary tests for an explicit, sized generation contract."""
import pytest

from src.prompt.generator import GenerationResult, generate_text_result


def _prompt():
    return {"subject": {"label": "window"}, "attributes": []}


def test_local_generation_is_explicit_and_bounded(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.prompt.generator.settings.max_output_chars", 4000)
    result = generate_text_result(_prompt(), [{"surface": "window"}], use_llm=False, tenant_id="default")
    assert isinstance(result, GenerationResult)
    assert result.status == "local_template"
    assert result.provider == "local"
    assert len(result.text) <= 4000


def test_provider_failure_is_explicit(monkeypatch):
    monkeypatch.setattr("src.prompt.generator._call_anthropic", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("provider down")))
    monkeypatch.setattr("src.prompt.generator.settings.llm_enabled", True)
    result = generate_text_result(_prompt(), [{"surface": "window"}], use_llm=True, tenant_id="default")
    assert result.status == "provider_failed"
    assert result.provider == "anthropic"
    assert result.text == ""


def test_provider_output_is_capped(monkeypatch):
    monkeypatch.setattr("src.prompt.generator._call_anthropic", lambda *_a, **_k: "x" * 5000)
    monkeypatch.setattr("src.prompt.generator.settings.max_output_chars", 100)
    monkeypatch.setattr("src.prompt.generator.settings.llm_enabled", True)
    result = generate_text_result(_prompt(), [{"surface": "window"}], use_llm=True, tenant_id="default")
    assert result.status == "provider_failed"
    assert result.text == ""
