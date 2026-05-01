"""Tests for OntologyLinker using mocked LLM provider.

Live tests against real providers are skipped unless the appropriate
API key env var is set (kept out of CI).
"""
from __future__ import annotations
import os
import pytest

from feather_db.extractors import (
    OntologyLinker, OntologyEdge, Fact, Entity,
)


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


def _facts_two_ctr_values():
    """Two facts about the same subject/predicate with different values
    — natural fodder for `contradicts` / `supersedes`."""
    return [
        Fact(subject="Acme Summer Sale", predicate="had_CTR",
             object="4.5%", confidence=0.9),
        Fact(subject="Acme Summer Sale", predicate="had_CTR",
             object="4.2%", confidence=0.95, valid_at=1712000000),
    ]


# ── Basic edge inference ──────────────────────────────────────────────────

def test_link_clean_json():
    provider = MockProvider(['''[
      {"source_id":"f_1","target_id":"f_0","rel_type":"supersedes",
       "weight":0.85,"confidence":0.9,
       "rationale":"f_1 has a later valid_at and higher confidence."}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1
    e = edges[0]
    assert isinstance(e, OntologyEdge)
    assert e.source_id == "f_1"
    assert e.target_id == "f_0"
    assert e.rel_type == "supersedes"
    assert e.weight == 0.85
    assert e.confidence == 0.9
    assert e.rationale and "later valid_at" in e.rationale


def test_link_supports_relation_no_rationale_required():
    provider = MockProvider(['''[
      {"source_id":"f_0","target_id":"f_1","rel_type":"supports",
       "weight":0.7,"confidence":0.7}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1
    assert edges[0].rel_type == "supports"
    assert edges[0].rationale is None


# ── Validation / filtering ────────────────────────────────────────────────

def test_link_filters_unknown_relations():
    provider = MockProvider(['''[
      {"source_id":"f_0","target_id":"f_1","rel_type":"BOGUS_REL",
       "weight":0.9},
      {"source_id":"f_1","target_id":"f_0","rel_type":"supports",
       "weight":0.7}
    ]'''])
    linker = OntologyLinker(provider=provider,
                            allowed_relations=["supports", "caused_by"])
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1
    assert edges[0].rel_type == "supports"


def test_link_filters_invalid_ids():
    """LLM-hallucinated ids (not in the rendered list) are dropped."""
    provider = MockProvider(['''[
      {"source_id":"f_99","target_id":"f_0","rel_type":"supports",
       "weight":0.9},
      {"source_id":"f_1","target_id":"f_0","rel_type":"supports",
       "weight":0.9}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1
    assert edges[0].source_id == "f_1"


def test_link_self_edges_dropped():
    provider = MockProvider(['''[
      {"source_id":"f_0","target_id":"f_0","rel_type":"supports",
       "weight":0.9},
      {"source_id":"f_0","target_id":"f_1","rel_type":"supports",
       "weight":0.9}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1
    assert edges[0].source_id != edges[0].target_id


def test_link_contradicts_requires_rationale():
    provider = MockProvider(['''[
      {"source_id":"f_0","target_id":"f_1","rel_type":"contradicts",
       "weight":0.9},
      {"source_id":"f_1","target_id":"f_0","rel_type":"contradicts",
       "weight":0.9,"rationale":"different CTR values for the same campaign"}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1
    assert edges[0].rationale and "different CTR" in edges[0].rationale


def test_link_supersedes_requires_rationale():
    provider = MockProvider(['''[
      {"source_id":"f_1","target_id":"f_0","rel_type":"supersedes",
       "weight":0.9}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert edges == []


def test_link_weight_clamped_and_defaulted():
    provider = MockProvider(['''[
      {"source_id":"f_0","target_id":"f_1","rel_type":"supports",
       "weight":2.5,"confidence":-0.3},
      {"source_id":"f_1","target_id":"f_0","rel_type":"supports",
       "weight":"not a number"}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 2
    assert edges[0].weight == 1.0
    assert edges[0].confidence == 0.0
    # invalid weight string falls back to 0.5
    assert edges[1].weight == 0.5


# ── Inputs / boundaries ───────────────────────────────────────────────────

def test_link_empty_items_returns_empty():
    provider = MockProvider([])
    linker = OntologyLinker(provider=provider)
    assert linker.link([]) == []
    assert provider.calls == []  # never called


def test_link_single_item_returns_empty():
    provider = MockProvider([])
    linker = OntologyLinker(provider=provider)
    assert linker.link([_facts_two_ctr_values()[0]]) == []
    assert provider.calls == []


def test_link_unrecognized_item_types_ignored():
    """Items that aren't Fact or Entity are silently skipped — if that
    leaves <2 items, no LLM call is made."""
    provider = MockProvider([])
    linker = OntologyLinker(provider=provider)
    edges = linker.link([_facts_two_ctr_values()[0], "raw text", 42])
    assert edges == []
    assert provider.calls == []  # only 1 valid item → no call


def test_link_caps_max_items():
    items = [Fact(subject=f"s{i}", predicate="p", object=f"o{i}")
             for i in range(20)]
    # Capture the rendered prompt to confirm only first N items are sent.
    provider = MockProvider(["[]"])
    linker = OntologyLinker(provider=provider, max_pairs_per_call=5)
    linker.link(items)
    user_msg = provider.calls[0]["messages"][1]["content"]
    assert "f_4" in user_msg
    assert "f_5" not in user_msg


# ── Mixed input types ─────────────────────────────────────────────────────

def test_link_handles_mixed_facts_and_entities():
    items = [
        Fact(subject="Summer Sale", predicate="targets", object="runners"),
        Entity(surface_form="runners",
               canonical_id="audience::runners", kind="Audience"),
    ]
    provider = MockProvider(['''[
      {"source_id":"f_0","target_id":"e_0","rel_type":"targets_segment",
       "weight":0.95}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(items)
    assert len(edges) == 1
    assert edges[0].source_id == "f_0"
    assert edges[0].target_id == "e_0"
    assert edges[0].rel_type == "targets_segment"


def test_link_renders_facts_and_entities_with_stable_ids():
    """Stable id scheme: f_0, f_1 for Facts; e_0, e_1 for Entities."""
    provider = MockProvider(["[]"])
    linker = OntologyLinker(provider=provider)
    items = [
        Fact(subject="A", predicate="p", object="B"),
        Entity(surface_form="B", canonical_id="x::b", kind="X"),
        Fact(subject="C", predicate="p", object="D"),
    ]
    linker.link(items)
    user_msg = provider.calls[0]["messages"][1]["content"]
    assert "f_0:" in user_msg
    assert "f_1:" in user_msg
    assert "e_0:" in user_msg


# ── Failure paths ─────────────────────────────────────────────────────────

def test_link_provider_failure_returns_empty_with_callback():
    class FailingProvider:
        def complete(self, *a, **k): raise RuntimeError("boom")
    errors = []
    linker = OntologyLinker(provider=FailingProvider(), max_retries=0)
    edges = linker.link(_facts_two_ctr_values(),
                        context={"on_error": errors.append})
    assert edges == []
    assert any("LLM call failed" in e for e in errors)


def test_link_malformed_json_returns_empty_with_callback():
    errors = []
    provider = MockProvider(["this is not json at all"])
    linker = OntologyLinker(provider=provider, max_retries=0)
    edges = linker.link(_facts_two_ctr_values(),
                        context={"on_error": errors.append})
    assert edges == []
    assert any("JSON parse failed" in e for e in errors)


def test_link_response_not_a_list_returns_empty():
    provider = MockProvider(['{"source_id":"f_0","target_id":"f_1",'
                             '"rel_type":"supports"}'])
    linker = OntologyLinker(provider=provider, max_retries=0)
    edges = linker.link(_facts_two_ctr_values())
    assert edges == []


def test_link_handles_markdown_fences():
    provider = MockProvider(['''Here are the edges:

```json
[{"source_id":"f_0","target_id":"f_1","rel_type":"supports","weight":0.7}]
```

End.'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1
    assert edges[0].rel_type == "supports"


def test_link_drops_incomplete_edges():
    provider = MockProvider(['''[
      {"source_id":"f_0","target_id":"f_1","rel_type":"supports",
       "weight":0.9},
      {"source_id":"","target_id":"f_1","rel_type":"supports"},
      {"source_id":"f_0","rel_type":"supports"},
      {"source_id":"f_0","target_id":"f_1","rel_type":""}
    ]'''])
    linker = OntologyLinker(provider=provider)
    edges = linker.link(_facts_two_ctr_values())
    assert len(edges) == 1


# ── Live tests (gated on API keys) ────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY",
)
def test_link_live_anthropic():
    from feather_db.providers import ClaudeProvider  # type: ignore
    linker = OntologyLinker(provider=ClaudeProvider())
    edges = linker.link(_facts_two_ctr_values())
    # We don't assert exact rel_type — but a contradicts/supersedes
    # edge with rationale is the natural inference here.
    assert all(isinstance(e, OntologyEdge) for e in edges)
