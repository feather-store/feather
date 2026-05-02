"""Tests for ContradictionResolver — rule path + LLM-scored path."""
from __future__ import annotations
import os
import pytest

from feather_db.extractors import (
    ContradictionResolver, ContradictionFinding, Fact,
)


class MockProvider:
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


# ── Rule path (no LLM) ────────────────────────────────────────────────

def test_no_findings_when_candidates_empty():
    res = ContradictionResolver()
    assert res.detect(Fact("A", "is", "B"), []) == []


def test_no_finding_for_unrelated_facts():
    res = ContradictionResolver()
    new = Fact("Acme", "had_CTR", "4.5%")
    candidates = [
        Fact("Acme", "launched", "Summer Sale"),  # different predicate
        Fact("Nike", "had_CTR", "5.2%"),          # different subject
    ]
    assert res.detect(new, candidates) == []


def test_no_finding_when_object_matches():
    res = ContradictionResolver()
    new = Fact("Acme", "had_CTR", "4.5%")
    candidates = [Fact("Acme", "had_CTR", "4.5%")]
    assert res.detect(new, candidates) == []


def test_finding_same_sp_different_object_non_numeric():
    res = ContradictionResolver()
    new = Fact("Acme", "ceo_is", "Alice")
    candidates = [Fact("Acme", "ceo_is", "Bob")]
    findings = res.detect(new, candidates)
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, ContradictionFinding)
    assert f.severity == "definite"
    assert f.suggested_resolution == "review"
    assert f.rationale and "non-numeric" in f.rationale


def test_finding_numeric_outside_tolerance():
    res = ContradictionResolver(numeric_tolerance=0.02)  # 2%
    new = Fact("Acme Summer Sale", "had_CTR", "4.5%")
    candidates = [Fact("Acme Summer Sale", "had_CTR", "4.2%")]
    findings = res.detect(new, candidates)
    assert len(findings) == 1
    assert findings[0].severity == "probable"
    assert findings[0].suggested_resolution == "review"


def test_no_finding_numeric_within_tolerance_default():
    """4.50% vs 4.51% → within default 2% tolerance, treated as merge."""
    res = ContradictionResolver(numeric_tolerance=0.02)
    new = Fact("Acme", "had_CTR", "4.50%")
    candidates = [Fact("Acme", "had_CTR", "4.51%")]
    findings = res.detect(new, candidates)
    # Within tolerance → emitted as 'merge' suggestion (not silent drop —
    # we still surface the pair so audit trail shows both observations).
    assert len(findings) == 1
    assert findings[0].suggested_resolution == "merge"
    assert findings[0].severity == "possible"


def test_supersedes_when_new_has_later_valid_at():
    res = ContradictionResolver()
    new = Fact("Acme", "had_CTR", "4.5%", valid_at=2_000_000_000)
    candidates = [Fact("Acme", "had_CTR", "4.2%", valid_at=1_700_000_000)]
    findings = res.detect(new, candidates)
    assert len(findings) == 1
    assert findings[0].suggested_resolution == "supersedes"
    assert findings[0].severity == "possible"
    assert "later valid_at" in (findings[0].rationale or "")


def test_subject_predicate_normalized_case_and_whitespace():
    res = ContradictionResolver()
    new = Fact("  Acme  ", "Had_CTR", "4.5%")
    candidates = [Fact("acme", "had_CTR", "4.2%")]
    assert len(res.detect(new, candidates)) == 1


def test_self_skipped():
    res = ContradictionResolver()
    f = Fact("A", "is", "B")
    assert res.detect(f, [f]) == []


def test_ignore_supersedure_drops_those_findings():
    res = ContradictionResolver(ignore_supersedure=True)
    new = Fact("Acme", "had_CTR", "4.5%", valid_at=2_000_000_000)
    candidates = [Fact("Acme", "had_CTR", "4.2%", valid_at=1_700_000_000)]
    assert res.detect(new, candidates) == []


def test_multiple_candidates_all_returned():
    res = ContradictionResolver()
    new = Fact("Acme", "ceo_is", "Alice")
    candidates = [
        Fact("Acme", "ceo_is", "Bob"),
        Fact("Acme", "ceo_is", "Carol"),
        Fact("Acme", "ceo_is", "Dave"),
    ]
    findings = res.detect(new, candidates)
    assert len(findings) == 3
    assert all(f.severity == "definite" for f in findings)


def test_dollar_and_comma_numeric_parsed():
    """$1,250 vs $1,200 → numeric, within 5% tolerance."""
    res = ContradictionResolver(numeric_tolerance=0.05)
    new = Fact("Q3 spend", "totalled", "$1,250")
    candidates = [Fact("Q3 spend", "totalled", "$1,200")]
    findings = res.detect(new, candidates)
    assert len(findings) == 1
    assert findings[0].suggested_resolution == "merge"


def test_new_earlier_than_candidate_is_review_not_supersedes():
    res = ContradictionResolver()
    new = Fact("Acme", "had_CTR", "4.5%", valid_at=1_700_000_000)
    candidates = [Fact("Acme", "had_CTR", "4.2%", valid_at=2_000_000_000)]
    findings = res.detect(new, candidates)
    assert len(findings) == 1
    assert findings[0].suggested_resolution == "review"
    assert "EARLIER valid_at" in (findings[0].rationale or "")


# ── LLM-scored path ───────────────────────────────────────────────────

def test_llm_severity_overrides_rule_severity():
    provider = MockProvider(['''[
      {"severity":"definite","suggested_resolution":"review",
       "rationale":"A/B test attribution mismatch detected by LLM."}
    ]'''])
    res = ContradictionResolver(provider=provider)
    new = Fact("Acme", "had_CTR", "4.5%")
    candidates = [Fact("Acme", "had_CTR", "4.2%")]
    findings = res.detect(new, candidates)
    assert len(findings) == 1
    # Rule said 'probable', LLM bumped to 'definite'
    assert findings[0].severity == "definite"
    assert "A/B test" in (findings[0].rationale or "")
    assert provider.calls  # LLM was called


def test_llm_invalid_severity_falls_back_to_rule_severity():
    provider = MockProvider(['''[
      {"severity":"BOGUS","suggested_resolution":"review","rationale":"x"}
    ]'''])
    res = ContradictionResolver(provider=provider)
    new = Fact("Acme", "had_CTR", "4.5%")
    candidates = [Fact("Acme", "had_CTR", "4.2%")]
    findings = res.detect(new, candidates)
    assert findings[0].severity == "probable"  # rule's value preserved


def test_llm_failure_returns_rule_findings():
    class FailingProvider:
        def complete(self, *a, **k): raise RuntimeError("rate limit")
    errors = []
    res = ContradictionResolver(provider=FailingProvider(), max_retries=0)
    new = Fact("Acme", "ceo_is", "Alice")
    candidates = [Fact("Acme", "ceo_is", "Bob")]
    findings = res.detect(new, candidates,
                          context={"on_error": errors.append})
    assert len(findings) == 1
    assert findings[0].severity == "definite"  # rule path preserved
    assert any("LLM call failed" in e for e in errors)


def test_llm_malformed_json_returns_rule_findings():
    provider = MockProvider(["not json at all"])
    errors = []
    res = ContradictionResolver(provider=provider, max_retries=0)
    new = Fact("Acme", "ceo_is", "Alice")
    candidates = [Fact("Acme", "ceo_is", "Bob")]
    findings = res.detect(new, candidates,
                          context={"on_error": errors.append})
    assert len(findings) == 1
    assert findings[0].severity == "definite"
    assert any("JSON parse failed" in e for e in errors)


def test_llm_short_response_passes_through_remainder():
    """LLM only scored the first finding — rest pass through."""
    provider = MockProvider(['''[
      {"severity":"probable","suggested_resolution":"review",
       "rationale":"only first scored"}
    ]'''])
    res = ContradictionResolver(provider=provider)
    new = Fact("Acme", "ceo_is", "Alice")
    candidates = [
        Fact("Acme", "ceo_is", "Bob"),
        Fact("Acme", "ceo_is", "Carol"),
    ]
    findings = res.detect(new, candidates)
    assert len(findings) == 2
    assert findings[0].severity == "probable"  # LLM-scored
    assert findings[1].severity == "definite"  # rule pass-through


# ── Live tests (gated on API keys) ────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY",
)
def test_detect_live_anthropic():
    from feather_db.providers import ClaudeProvider  # type: ignore
    res = ContradictionResolver(provider=ClaudeProvider())
    new = Fact("Acme Summer Sale", "had_CTR", "4.5%")
    candidates = [Fact("Acme Summer Sale", "had_CTR", "4.2%",
                       valid_at=1_700_000_000)]
    findings = res.detect(new, candidates)
    assert findings  # at least one finding
    assert all(f.severity in {"definite", "probable", "possible"}
               for f in findings)
