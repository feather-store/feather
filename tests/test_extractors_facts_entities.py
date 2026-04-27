"""Tests for FactExtractor + EntityResolver using mocked LLM provider.

Live tests against real providers are skipped unless the appropriate
API key env var is set (kept out of CI).
"""
from __future__ import annotations
import os
import pytest

from feather_db.extractors import FactExtractor, EntityResolver, Fact, Entity


class MockProvider:
    """LLMProvider stub that returns canned responses by call index."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, max_tokens=512, temperature=0.0) -> str:
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if not self._responses:
            raise RuntimeError("MockProvider exhausted")
        return self._responses.pop(0)


# ── FactExtractor ──────────────────────────────────────────────────────────

def test_facts_clean_json():
    provider = MockProvider(['''[
      {"subject":"Acme","predicate":"launched","object":"Summer Sale","confidence":0.95,"valid_at":"2024-03-15T00:00:00Z"},
      {"subject":"Summer Sale first week","predicate":"had_average_CTR","object":"4.5%","confidence":0.9,"valid_at":null}
    ]'''])
    fx = FactExtractor(provider=provider)
    facts = fx.extract(
        "Acme launched the Summer Sale on March 15, 2024. CTR averaged 4.5% week 1.",
        context={"source_id": "memo-001", "namespace": "acme"},
    )
    assert len(facts) == 2
    assert all(isinstance(f, Fact) for f in facts)
    assert facts[0].subject == "Acme"
    assert facts[0].predicate == "launched"
    assert facts[0].provenance == "memo-001"
    assert facts[0].confidence == 0.95
    assert facts[0].valid_at is not None  # ISO parsed
    assert facts[1].valid_at is None


def test_facts_drops_low_confidence():
    provider = MockProvider(['''[
      {"subject":"A","predicate":"is","object":"B","confidence":0.95},
      {"subject":"X","predicate":"is","object":"Y","confidence":0.2}
    ]'''])
    fx = FactExtractor(provider=provider, min_confidence=0.5)
    facts = fx.extract("doesn't matter")
    assert len(facts) == 1
    assert facts[0].subject == "A"


def test_facts_caps_max_facts():
    items = ",".join([
        f'{{"subject":"s{i}","predicate":"p","object":"o{i}","confidence":1}}'
        for i in range(40)
    ])
    provider = MockProvider([f"[{items}]"])
    fx = FactExtractor(provider=provider, max_facts_per_call=10)
    facts = fx.extract("text")
    assert len(facts) == 10


def test_facts_handles_markdown_fences():
    provider = MockProvider(['''Here is the JSON requested:

```json
[{"subject":"A","predicate":"is","object":"B","confidence":1.0}]
```

Let me know if you need more.'''])
    fx = FactExtractor(provider=provider)
    facts = fx.extract("text")
    assert len(facts) == 1
    assert facts[0].subject == "A"


def test_facts_empty_input():
    provider = MockProvider([])
    fx = FactExtractor(provider=provider)
    assert fx.extract("") == []
    assert fx.extract("   \n\t   ") == []
    assert provider.calls == []  # never called


def test_facts_empty_response():
    provider = MockProvider(["[]"])
    fx = FactExtractor(provider=provider)
    assert fx.extract("nothing extractable") == []


def test_facts_malformed_json_returns_empty():
    errors = []
    provider = MockProvider(["this is not json at all"])
    fx = FactExtractor(provider=provider, max_retries=0)
    facts = fx.extract("text", context={"on_error": errors.append})
    assert facts == []
    assert len(errors) == 1
    assert "JSON parse failed" in errors[0]


def test_facts_drops_incomplete_triples():
    provider = MockProvider(['''[
      {"subject":"A","predicate":"is","object":"B","confidence":1},
      {"subject":"","predicate":"is","object":"X","confidence":1},
      {"subject":"C","predicate":"","object":"D","confidence":1},
      {"subject":"E","predicate":"is","confidence":1}
    ]'''])
    fx = FactExtractor(provider=provider)
    facts = fx.extract("x")
    assert len(facts) == 1
    assert facts[0].subject == "A"


def test_facts_provider_failure_returns_empty_with_callback():
    class FailingProvider:
        def complete(self, *a, **k): raise RuntimeError("boom")
    errors = []
    fx = FactExtractor(provider=FailingProvider(), max_retries=0)
    facts = fx.extract("text", context={"on_error": errors.append})
    assert facts == []
    assert any("LLM call failed" in e for e in errors)


# ── EntityResolver ─────────────────────────────────────────────────────────

def test_entities_clean_resolution():
    provider = MockProvider(['''[
      {"surface_form":"Acme Corp","canonical_id":"brand::acme","kind":"Brand","confidence":0.95,"aliases":["ACME","Acme"]},
      {"surface_form":"Summer Sale","canonical_id":"campaign::summer_2024","kind":"Campaign","confidence":0.9,"aliases":[]}
    ]'''])
    res = EntityResolver(provider=provider)
    out = res.resolve(["Acme Corp", "Summer Sale"], context={"namespace": "acme"})
    assert len(out) == 2
    assert out[0].canonical_id == "brand::acme"
    assert out[0].aliases == ["ACME", "Acme"]
    assert out[1].kind == "Campaign"


def test_entities_fallback_on_missing_resolution():
    provider = MockProvider(['''[
      {"surface_form":"Acme Corp","canonical_id":"brand::acme","kind":"Brand","confidence":0.95}
    ]'''])
    res = EntityResolver(provider=provider)
    # Resolver dropped one of the inputs; we should still get 2 outputs
    out = res.resolve(["Acme Corp", "Some Mystery"])
    assert len(out) == 2
    assert out[0].canonical_id == "brand::acme"
    assert out[1].canonical_id.startswith("unknown::")
    assert out[1].confidence == 0.2


def test_entities_known_seed_in_user_message():
    provider = MockProvider(['[]'])
    known = [Entity(
        surface_form="Acme Corp",
        canonical_id="brand::acme",
        kind="Brand",
        aliases=["ACME"],
    )]
    res = EntityResolver(provider=provider, known_entities=known)
    res.resolve(["Acme"])
    user_msg = provider.calls[0]["messages"][1]["content"]
    assert "brand::acme" in user_msg
    assert "ACME" in user_msg


def test_entities_provider_failure_returns_fallbacks():
    class FailingProvider:
        def complete(self, *a, **k): raise RuntimeError("rate limit")
    res = EntityResolver(provider=FailingProvider(), max_retries=0)
    out = res.resolve(["A", "B"])
    assert len(out) == 2
    assert all(e.canonical_id.startswith("unknown::") for e in out)
    assert all(e.confidence == 0.2 for e in out)


def test_entities_empty_input():
    provider = MockProvider([])
    assert EntityResolver(provider=provider).resolve([]) == []


# ── Live tests (gated on API keys) ─────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY",
)
def test_facts_live_anthropic():
    from feather_db.providers import ClaudeProvider
    fx = FactExtractor(provider=ClaudeProvider())
    facts = fx.extract(
        "Acme launched the Summer Sale campaign on March 15, 2024. "
        "Click-through rate averaged 4.5% in the first week."
    )
    assert len(facts) >= 1
    # at least one fact should mention Acme
    assert any("Acme" in f.subject or "Acme" in f.object for f in facts)
