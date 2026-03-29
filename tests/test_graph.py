"""
Context Graph tests: link, get_edges, get_incoming, auto_link, context_chain.
"""
import pytest
import feather_db
from feather_db import DB, RelType
from .conftest import EMBED, make_embedder


class TestEdges:
    def test_link_creates_edge(self, populated_db):
        populated_db.link(1, 2, rel_type=RelType.RELATED_TO, weight=0.9)
        edges = populated_db.get_edges(1)
        assert any(e.target_id == 2 for e in edges)

    def test_link_weight_stored(self, populated_db):
        populated_db.link(1, 3, rel_type=RelType.CAUSED_BY, weight=0.7)
        edges = populated_db.get_edges(1)
        match = next(e for e in edges if e.target_id == 3)
        assert match.weight == pytest.approx(0.7, abs=0.01)

    def test_link_rel_type_stored(self, populated_db):
        populated_db.link(2, 4, rel_type=RelType.SUPPORTS, weight=0.5)
        edges = populated_db.get_edges(2)
        match = next(e for e in edges if e.target_id == 4)
        assert match.rel_type == RelType.SUPPORTS

    def test_get_incoming(self, populated_db):
        populated_db.link(1, 5, rel_type=RelType.RELATED_TO, weight=0.6)
        populated_db.link(3, 5, rel_type=RelType.CAUSED_BY, weight=0.8)
        incoming = populated_db.get_incoming(5)
        source_ids = [e.source_id for e in incoming]
        assert 1 in source_ids
        assert 3 in source_ids

    def test_link_deduplication(self, populated_db):
        populated_db.link(1, 2, rel_type=RelType.RELATED_TO, weight=0.5)
        populated_db.link(1, 2, rel_type=RelType.RELATED_TO, weight=0.9)
        edges = [e for e in populated_db.get_edges(1) if e.target_id == 2]
        assert len(edges) == 1  # deduplicated, latest weight wins

    def test_custom_rel_type_string(self, populated_db):
        populated_db.link(2, 3, rel_type="blocks_progress", weight=0.4)
        edges = populated_db.get_edges(2)
        match = next((e for e in edges if e.target_id == 3), None)
        assert match is not None
        assert match.rel_type == "blocks_progress"


class TestAutoLink:
    def test_auto_link_creates_edges(self, populated_db):
        populated_db.auto_link(modality="text", threshold=0.0, rel_type=RelType.RELATED_TO)
        # With threshold=0 every pair gets linked; just verify some edges exist
        edges = populated_db.get_edges(1)
        assert len(edges) > 0

    def test_auto_link_threshold_filters(self, populated_db):
        populated_db.auto_link(modality="text", threshold=0.99, rel_type=RelType.RELATED_TO)
        # With threshold=0.99 very few (possibly zero) edges should appear
        total_edges = sum(len(populated_db.get_edges(i)) for i in range(1, 7))
        assert total_edges < 6  # substantially fewer than full graph


class TestContextChain:
    def test_context_chain_returns_nodes(self, populated_db):
        populated_db.link(1, 2, rel_type=RelType.RELATED_TO, weight=0.9)
        result = populated_db.context_chain(EMBED("onboarding"), k=3, hops=2)
        assert len(result.nodes) >= 1

    def test_context_chain_hop_distances(self, populated_db):
        populated_db.link(1, 2, rel_type=RelType.RELATED_TO, weight=0.9)
        populated_db.link(2, 3, rel_type=RelType.CAUSED_BY,  weight=0.8)
        result = populated_db.context_chain(EMBED("onboarding"), k=1, hops=2)
        hop_distances = {n.id: n.hop for n in result.nodes}
        # Node found by vector search is hop 0
        assert 0 in hop_distances.values()

    def test_context_chain_edges_returned(self, populated_db):
        populated_db.link(1, 2, rel_type=RelType.RELATED_TO, weight=0.9)
        result = populated_db.context_chain(EMBED("onboarding"), k=3, hops=2)
        assert hasattr(result, "edges")

    def test_context_chain_modality(self, populated_db):
        result = populated_db.context_chain(
            make_embedder(64)("latency"), k=3, hops=1, modality="visual"
        )
        assert len(result.nodes) >= 1


class TestGraphJson:
    def test_export_graph_json_valid(self, populated_db):
        import json
        populated_db.link(1, 2, rel_type=RelType.RELATED_TO, weight=0.8)
        raw = populated_db.export_graph_json()
        data = json.loads(raw)
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 6

    def test_export_graph_json_namespace_filter(self, populated_db):
        import json
        raw = populated_db.export_graph_json(namespace_filter="infra", entity_filter="")
        data = json.loads(raw)
        assert all(n["namespace_id"] == "infra" for n in data["nodes"])
        assert len(data["nodes"]) == 1  # only node 5 is "infra"

    def test_visualize_produces_html(self, populated_db, tmp_path):
        from feather_db.graph import visualize
        out = str(tmp_path / "graph.html")
        visualize(populated_db, output_path=out)
        content = open(out).read()
        assert "<html" in content.lower()
        assert "d3" in content.lower()
