"""Tests for Phase 9 MCP tools: feather_ingest + feather_recall.

Uses mock providers so no API keys needed.
"""
from __future__ import annotations
import json
import tempfile
import time

import numpy as np
import pytest

import feather_db
from feather_db.providers import LLMProvider
from feather_db.integrations.base import FeatherTools, TOOL_SPECS


class MockSystemProvider(LLMProvider):
    def complete(self, messages, max_tokens=512, temperature=0.0):
        last = messages[-1]["content"]
        # EntityResolver prompt contains "entities"; FactExtractor contains "triples"
        if "entities" in last.lower() or "canonical" in last.lower():
            return json.dumps([
                {"canonical": "User", "aliases": ["user"], "kind": "person",
                 "confidence": 0.9, "attributes": {}},
            ])
        return json.dumps([
            {"subject": "User", "predicate": "prefers", "object": "dark mode",
             "confidence": 0.9, "valid_at": None},
        ])


class MockEmbed:
    def __init__(self, dim=64):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return rng.random(self.dim).astype(np.float32)


@pytest.fixture
def tools_raw():
    path = tempfile.mktemp(suffix="_mcp_raw.feather")
    return FeatherTools(db_path=path, dim=64, embedder=MockEmbed(), auto_save=False)


@pytest.fixture
def tools_phase9():
    path = tempfile.mktemp(suffix="_mcp_p9.feather")
    return FeatherTools(
        db_path=path, dim=64, embedder=MockEmbed(),
        system_provider=MockSystemProvider(),
        namespace="hawky",
        auto_save=False,
    )


class TestToolSpecs:
    def test_feather_ingest_in_specs(self):
        names = [s["name"] for s in TOOL_SPECS]
        assert "feather_ingest" in names

    def test_feather_recall_in_specs(self):
        names = [s["name"] for s in TOOL_SPECS]
        assert "feather_recall" in names

    def test_feather_ingest_spec_required(self):
        spec = next(s for s in TOOL_SPECS if s["name"] == "feather_ingest")
        assert "content" in spec["required"]

    def test_feather_recall_spec_required(self):
        spec = next(s for s in TOOL_SPECS if s["name"] == "feather_recall")
        assert "query" in spec["required"]

    def test_total_tool_count(self):
        assert len(TOOL_SPECS) == 16  # 14 original + 2 Phase 9


class TestFeatherIngestRaw:
    def test_raw_mode_returns_ok(self, tools_raw):
        result = json.loads(tools_raw.feather_ingest(
            content="Alice prefers dark mode.",
            source_id="s1::t1",
            timestamp=int(time.time()),
        ))
        assert result["status"] == "ok"
        assert result["mode"] == "raw"
        assert "id" in result

    def test_raw_mode_note_present(self, tools_raw):
        result = json.loads(tools_raw.feather_ingest(content="hello"))
        assert "note" in result

    def test_raw_stored_in_db(self, tools_raw):
        tools_raw.feather_ingest(content="Test content for retrieval.")
        ids = tools_raw.db.get_all_ids(modality="text")
        assert len(ids) > 0

    def test_via_handle_dispatch(self, tools_raw):
        result = json.loads(tools_raw.handle(
            "feather_ingest", {"content": "Dispatch test"}
        ))
        assert result["status"] == "ok"

    def test_empty_content_no_crash(self, tools_raw):
        result = json.loads(tools_raw.feather_ingest(content=""))
        # Should not crash — either ok or error with message
        assert "status" in result or "error" in result


class TestFeatherIngestPhase9:
    def test_phase9_mode_label(self, tools_phase9):
        result = json.loads(tools_phase9.feather_ingest(
            content="Alice said she loves dark mode."
        ))
        assert result["status"] == "ok"
        assert result["mode"] == "phase9"

    def test_phase9_stats_fields(self, tools_phase9):
        result = json.loads(tools_phase9.feather_ingest(
            content="Alice said she loves dark mode."
        ))
        assert "facts_extracted" in result
        assert "entities_resolved" in result
        assert "records_ingested" in result

    def test_phase9_namespace_override(self, tools_phase9):
        result = json.loads(tools_phase9.feather_ingest(
            content="Content for custom namespace.",
            namespace="custom_ns",
        ))
        assert result["status"] == "ok"

    def test_phase9_pipeline_init(self, tools_phase9):
        assert tools_phase9._pipeline is not None


class TestFeatherRecall:
    def _seed(self, tools, n=3):
        for i in range(n):
            tools.feather_ingest(
                content=f"Memory fact {i}: user prefers dark mode setting {i}",
                source_id=f"seed::{i}",
                timestamp=int(time.time()) - i * 100,
            )

    def test_recall_returns_results(self, tools_raw):
        self._seed(tools_raw)
        result = json.loads(tools_raw.feather_recall(query="dark mode preference"))
        assert "results" in result
        assert "count" in result

    def test_recall_decay_fields(self, tools_raw):
        self._seed(tools_raw)
        result = json.loads(tools_raw.feather_recall(
            query="dark mode", half_life_days=14.0, time_weight=0.5
        ))
        assert result["decay"]["half_life"] == 14.0
        assert result["decay"]["time_weight"] == 0.5

    def test_recall_k_limit(self, tools_raw):
        self._seed(tools_raw, n=5)
        result = json.loads(tools_raw.feather_recall(query="fact", k=2))
        assert result["count"] <= 2

    def test_recall_empty_db(self, tools_raw):
        result = json.loads(tools_raw.feather_recall(query="anything"))
        assert result["count"] == 0

    def test_recall_via_handle(self, tools_raw):
        self._seed(tools_raw)
        result = json.loads(tools_raw.handle(
            "feather_recall", {"query": "dark mode"}
        ))
        assert "results" in result

    def test_recall_namespace_filter(self, tools_phase9):
        tools_phase9.feather_ingest(content="Some memory", namespace="ns_a")
        result = json.loads(tools_phase9.feather_recall(
            query="memory", namespace="ns_b"  # no match
        ))
        assert result["count"] == 0
