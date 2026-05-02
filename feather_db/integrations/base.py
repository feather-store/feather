"""
FeatherTools — Base Agent Tool Layer
=====================================
Wraps a Feather DB instance as a callable tool-set for any LLM agent.
All tool logic lives here. Each LLM connector (Claude, OpenAI, Gemini)
inherits from this class and overrides only the schema-formatting methods.

Tools exposed:
  feather_search         Semantic search by text query → top-k nodes
  feather_context_chain  Vector search + BFS graph expansion (n hops)
  feather_get_node       Fetch full metadata for a node by ID
  feather_get_related    Get graph neighbours (edges in/out) of a node
  feather_add_intel      Ingest a new intelligence node into the DB
  feather_link_nodes     Create a typed weighted edge between two nodes
  feather_timeline       Chronological node list for a product / entity
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Optional

import numpy as np

try:
    import feather_db as _fdb
except ImportError:
    _fdb = None  # type: ignore


# ── Tool definitions ──────────────────────────────────────────────────────────
# Single source of truth — connectors read from here.

TOOL_SPECS: list[dict] = [
    {
        "name": "feather_search",
        "description": (
            "Semantic vector search over the Feather DB knowledge graph. "
            "Converts the text query to an embedding and returns the k most "
            "similar nodes with their scores, content, and metadata. "
            "Use this to find relevant creatives, competitor events, strategy "
            "briefs, or performance intel by meaning — not keyword."
        ),
        "parameters": {
            "query":     {"type": "string",  "description": "Natural language search query"},
            "k":         {"type": "integer", "description": "Number of results to return (default 5)", "default": 5},
            "namespace": {"type": "string",  "description": "Filter by namespace (optional)"},
            "entity":    {"type": "string",  "description": "Filter by entity type, e.g. 'competitor_intel', 'strategy_intel', 'ad_creative' (optional)"},
            "product":   {"type": "string",  "description": "Filter by product attribute, e.g. 'FD', 'CC', 'Bond', 'MF' (optional)"},
        },
        "required": ["query"],
    },
    {
        "name": "feather_context_chain",
        "description": (
            "Runs a two-phase retrieval: first finds the k most similar nodes "
            "by vector search (hop=0), then expands outward over typed graph "
            "edges for `hops` levels (BFS). Returns all reached nodes with "
            "their hop distance and score. "
            "Use this to trace root causes — e.g. start from a CTR drop and "
            "surface the competitor event or strategy brief that explains it."
        ),
        "parameters": {
            "query": {"type": "string",  "description": "Seed query — describes the starting context"},
            "k":     {"type": "integer", "description": "Number of vector-search seeds (default 3)", "default": 3},
            "hops":  {"type": "integer", "description": "Number of graph hops to expand (default 2)", "default": 2},
        },
        "required": ["query"],
    },
    {
        "name": "feather_get_node",
        "description": (
            "Fetch the full metadata record for a single node by its integer ID. "
            "Returns content, entity type, product, all attributes, and outgoing "
            "graph edges. Use after feather_search or feather_context_chain to "
            "inspect a specific node in detail."
        ),
        "parameters": {
            "id": {"type": "integer", "description": "Feather DB node ID"},
        },
        "required": ["id"],
    },
    {
        "name": "feather_get_related",
        "description": (
            "Get the direct graph neighbours of a node — nodes connected by "
            "typed edges. Returns edge type, weight, and the connected node's "
            "content and metadata. "
            "Use to explore the knowledge graph from a known node."
        ),
        "parameters": {
            "id":        {"type": "integer", "description": "Source node ID"},
            "rel_type":  {"type": "string",  "description": "Filter by edge type, e.g. 'caused_by', 'contradicts', 'supports', 'same_ad' (optional)"},
            "direction": {
                "type": "string",
                "description": "Edge direction: 'outgoing' (default), 'incoming', or 'both'",
                "enum": ["outgoing", "incoming", "both"],
                "default": "outgoing",
            },
        },
        "required": ["id"],
    },
    {
        "name": "feather_add_intel",
        "description": (
            "Ingest a new intelligence node into the Feather DB. "
            "Use when the agent discovers new information that should persist "
            "as memory — competitor activity, market signals, or creative insights."
        ),
        "parameters": {
            "content":     {"type": "string",  "description": "Full text content of the intelligence note"},
            "entity_type": {"type": "string",  "description": "Type label, e.g. 'competitor_intel', 'social_trend', 'strategy_intel'"},
            "product":     {"type": "string",  "description": "Product this intel relates to, e.g. 'FD', 'CC', 'Bond', 'MF' (optional)"},
            "importance":  {"type": "number",  "description": "Relevance weight 0.0–1.0 (default 0.8)", "default": 0.8},
        },
        "required": ["content", "entity_type"],
    },
    {
        "name": "feather_link_nodes",
        "description": (
            "Create a typed weighted edge between two existing nodes. "
            "Edge types: caused_by, supports, contradicts, derived_from, "
            "references, same_ad, related_to. "
            "Use when the agent identifies a causal or semantic relationship "
            "that should be preserved in the knowledge graph."
        ),
        "parameters": {
            "from_id":  {"type": "integer", "description": "Source node ID"},
            "to_id":    {"type": "integer", "description": "Target node ID"},
            "rel_type": {"type": "string",  "description": "Edge type (caused_by | supports | contradicts | derived_from | references | same_ad | related_to)"},
            "weight":   {"type": "number",  "description": "Edge weight 0.0–1.0 (default 0.8)", "default": 0.8},
        },
        "required": ["from_id", "to_id", "rel_type"],
    },
    {
        "name": "feather_timeline",
        "description": (
            "Return nodes sorted chronologically (newest first) for a given "
            "product or entity type. Useful for understanding how context has "
            "evolved over time — e.g. all competitor moves for FD in order."
        ),
        "parameters": {
            "product":     {"type": "string",  "description": "Filter by product: 'FD', 'CC', 'Bond', 'MF' (optional)"},
            "entity_type": {"type": "string",  "description": "Filter by entity type (optional)"},
            "limit":       {"type": "integer", "description": "Max nodes to return (default 10)", "default": 10},
        },
        "required": [],
    },
    # ── Phase 6 tools ─────────────────────────────────────────────────────────
    {
        "name": "feather_forget",
        "description": (
            "Soft-delete a node from the knowledge graph. "
            "Removes it from search results but preserves the graph shell so "
            "existing edges remain traversable. "
            "Use when intel is outdated, incorrect, or must be removed for privacy."
        ),
        "parameters": {
            "id": {"type": "integer", "description": "Node ID to forget"},
        },
        "required": ["id"],
    },
    {
        "name": "feather_health",
        "description": (
            "Return a health summary of the Feather DB knowledge graph. "
            "Includes: total node count, hot/warm/cold tier distribution, "
            "orphan nodes (no edges), expired TTL nodes, recall histogram, "
            "and average importance/confidence scores. "
            "Use to understand the current state of the knowledge graph "
            "and decide if consolidation or cleanup is needed."
        ),
        "parameters": {
            "modality": {"type": "string", "description": "Modality to inspect (default 'text')"},
        },
        "required": [],
    },
    {
        "name": "feather_why",
        "description": (
            "Explain why a specific node would be retrieved for a given query. "
            "Returns a score breakdown: similarity, stickiness (recall bonus), "
            "effective age (decay), recency, importance, confidence, and "
            "the final computed score with the formula. "
            "Use to understand retrieval decisions and debug unexpected results."
        ),
        "parameters": {
            "id":    {"type": "integer", "description": "Node ID to explain"},
            "query": {"type": "string",  "description": "Query text to score against"},
        },
        "required": ["id", "query"],
    },
    {
        "name": "feather_mmr_search",
        "description": (
            "Semantic search with Maximal Marginal Relevance (MMR) diversity. "
            "Like feather_search but post-processes results to maximize both "
            "relevance to the query AND diversity among results — preventing "
            "5 near-identical nodes from dominating the result set. "
            "Use when you need a broad view of the topic rather than the single "
            "most relevant cluster."
        ),
        "parameters": {
            "query":     {"type": "string",  "description": "Natural language search query"},
            "k":         {"type": "integer", "description": "Number of results (default 5)"},
            "diversity": {"type": "number",  "description": "Diversity weight 0.0–1.0 (default 0.5). Higher = more diverse."},
        },
        "required": ["query"],
    },
    {
        "name": "feather_consolidate",
        "description": (
            "Cluster similar nodes in a namespace and merge each cluster into "
            "a single summary node. Original nodes are preserved but linked "
            "with 'consolidated_into' edges and their importance is lowered. "
            "Use to compact the knowledge graph when similar intel has accumulated, "
            "reducing noise and improving retrieval quality. "
            "Returns IDs of newly created consolidated nodes."
        ),
        "parameters": {
            "namespace":   {"type": "string",  "description": "Namespace to consolidate"},
            "since_hours": {"type": "number",  "description": "Only consolidate nodes added within this many hours (default 24)"},
            "threshold":   {"type": "number",  "description": "Cosine similarity threshold to cluster (default 0.85)"},
        },
        "required": ["namespace"],
    },
    {
        "name": "feather_episode_get",
        "description": (
            "Retrieve all nodes belonging to a named episode, ordered by timestamp. "
            "An episode groups related events into a narrative sequence — e.g. "
            "all nodes added during a campaign launch, a competitor incident, "
            "or a strategy review. "
            "Use to replay what happened in a given context window."
        ),
        "parameters": {
            "episode_id": {"type": "string", "description": "Episode identifier string"},
        },
        "required": ["episode_id"],
    },
    {
        "name": "feather_expire",
        "description": (
            "Scan all nodes and soft-delete any that have exceeded their TTL "
            "(time-to-live). TTL nodes are typically short-lived working memory "
            "like session state, ephemeral signals, or real-time feed items. "
            "Returns count of nodes expired. "
            "Call periodically to keep the graph clean."
        ),
        "parameters": {},
        "required": [],
    },
    # ── Phase 9 tools ──────────────────────────────────────────────────────────
    {
        "name": "feather_ingest",
        "description": (
            "Phase 9 structured ingestion: runs FactExtractor + EntityResolver "
            "on a raw text turn and stores the atomic facts (subject–predicate–object "
            "triples) alongside the source text. When an LLM system provider is "
            "configured, extracted facts are independently searchable and graph-linked. "
            "Falls back to raw-text storage when no provider is configured. "
            "Use this to feed conversation turns, documents, or any raw text into "
            "the knowledge graph."
        ),
        "parameters": {
            "content":   {"type": "string", "description": "Raw text to ingest (a single turn, doc, or note)"},
            "source_id": {"type": "string", "description": "Unique source identifier (e.g. 'session_42::turn_7')"},
            "timestamp": {"type": "integer", "description": "Unix timestamp of the turn (0 = now)"},
            "namespace": {"type": "string",  "description": "Namespace to store under (defaults to server namespace)"},
        },
        "required": ["content"],
    },
    {
        "name": "feather_recall",
        "description": (
            "Hybrid retrieval with adaptive decay scoring. Combines vector similarity "
            "(BM25 + HNSW) with recency weighting so frequently-accessed facts resist "
            "decay. Use for questions that need temporally-aware context — e.g. "
            "'what did the user say recently about X?' — where pure similarity search "
            "would surface stale results."
        ),
        "parameters": {
            "query":            {"type": "string",  "description": "Natural language recall query"},
            "k":                {"type": "integer", "description": "Number of results to return (default 8)"},
            "namespace":        {"type": "string",  "description": "Filter by namespace (optional)"},
            "half_life_days":   {"type": "number",  "description": "Decay half-life in days (default 30)"},
            "time_weight":      {"type": "number",  "description": "Blend weight for recency [0-1] (default 0.3)"},
        },
        "required": ["query"],
    },
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


class FeatherTools:
    """
    LLM-agnostic Feather DB tool layer.

    Parameters
    ----------
    db_path : str
        Path to the .feather file.
    dim : int
        Vector dimension (must match the DB).
    embedder : callable, optional
        Function (text: str) -> np.ndarray.  If not provided, falls back to
        a lightweight mock embedder so the class works without an API key.
    auto_save : bool
        Call db.save() after every write operation (add_intel, link_nodes).
    system_provider : LLMProvider, optional
        Phase 9 system LLM (FactExtractor + EntityResolver).  When provided,
        ``feather_ingest`` runs structured extraction; otherwise raw storage.
    namespace : str
        Default namespace for Phase 9 ingestion (default "default").
    """

    def __init__(
        self,
        db_path: str,
        dim: int = 3072,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        auto_save: bool = True,
        system_provider=None,
        namespace: str = "default",
    ):
        if _fdb is None:
            raise ImportError("feather_db is not installed or not built.")
        self.db        = _fdb.DB.open(db_path, dim=dim)
        self.dim       = dim
        self.auto_save = auto_save
        self._next_id  = 90001   # auto-assigned IDs for add_intel
        self._namespace = namespace
        self._pipeline  = None   # built lazily when system_provider is set

        if embedder is not None:
            self._embed = embedder
        else:
            # Built-in lightweight mock (works offline, no API key needed)
            self._embed = self._mock_embed

        if system_provider is not None:
            self._init_pipeline(system_provider)

    def _init_pipeline(self, system_provider) -> None:
        try:
            from feather_db.extractors import FactExtractor, EntityResolver
            from feather_db.pipelines import IngestPipeline

            class _EmbedAdapter:
                def __init__(self, fn, dim):
                    self.dim = dim
                    self._fn = fn
                def embed(self, text):
                    return self._fn(text)

            self._pipeline = IngestPipeline(
                db=self.db,
                embedder=_EmbedAdapter(self._embed, self.dim),
                fact_extractor=FactExtractor(provider=system_provider),
                entity_resolver=EntityResolver(provider=system_provider),
                namespace=self._namespace,
            )
        except Exception:
            self._pipeline = None

    # ── Embed ─────────────────────────────────────────────────────────────────

    def _mock_embed(self, text: str) -> np.ndarray:
        import hashlib
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().replace(",", " ").replace(".", " ").split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            for j in range(6):
                vec[(h >> (j * 5)) % self.dim] += 1.0 / (j + 1)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        return self._embed(text)

    # ── Tool implementations ──────────────────────────────────────────────────

    def feather_search(
        self,
        query: str,
        k: int = 5,
        namespace: Optional[str] = None,
        entity: Optional[str] = None,
        product: Optional[str] = None,
    ) -> str:
        vec     = self.embed(query)
        results = self.db.search(vec, k=k * 3)  # over-fetch, then filter

        output = []
        for r in results:
            m = r.metadata
            if namespace and m.namespace_id != namespace:
                continue
            if entity and m.get_attribute("entity_type") != entity:
                continue
            if product and m.get_attribute("product") != product:
                continue
            output.append({
                "id":          r.id,
                "score":       round(r.score, 4),
                "content":     m.content[:200],
                "entity_type": m.get_attribute("entity_type"),
                "product":     m.get_attribute("product"),
                "modality":    m.get_attribute("modality"),
                "importance":  round(m.importance, 3),
                "recall_count": m.recall_count,
            })
            if len(output) >= k:
                break

        return json.dumps({"results": output, "count": len(output)}, indent=2)

    def feather_context_chain(
        self,
        query: str,
        k: int = 3,
        hops: int = 2,
    ) -> str:
        vec   = self.embed(query)
        chain = self.db.context_chain(vec, k=k, hops=hops, modality="text")

        nodes = []
        for node in sorted(chain.nodes, key=lambda n: (n.hop, -n.score)):
            m = node.metadata
            nodes.append({
                "id":          node.id,
                "hop":         node.hop,
                "score":       round(node.score, 4),
                "content":     m.content[:200],
                "entity_type": m.get_attribute("entity_type"),
                "product":     m.get_attribute("product"),
                "modality":    m.get_attribute("modality"),
            })

        edges = []
        for e in chain.edges:
            edges.append({
                "source":   e.source,
                "target":   e.target,
                "rel_type": e.rel_type,
                "weight":   round(e.weight, 3),
            })

        return json.dumps({
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count":  len(edges),
        }, indent=2)

    def feather_get_node(self, id: int) -> str:
        try:
            m = self.db.get_metadata(id)
        except Exception as e:
            return json.dumps({"error": str(e)})

        edges = [
            {"target_id": e.target_id, "rel_type": e.rel_type, "weight": round(e.weight, 3)}
            for e in m.edges
        ]
        incoming = [
            {"source_id": e.source_id, "rel_type": e.rel_type, "weight": round(e.weight, 3)}
            for e in self.db.get_incoming(id)
        ]

        return json.dumps({
            "id":           id,
            "content":      m.content,
            "entity_type":  m.get_attribute("entity_type"),
            "product":      m.get_attribute("product"),
            "modality":     m.get_attribute("modality"),
            "importance":   round(m.importance, 3),
            "recall_count": m.recall_count,
            "timestamp":    m.timestamp,
            "namespace":    m.namespace_id,
            "edges_out":    edges,
            "edges_in":     incoming,
        }, indent=2)

    def feather_get_related(
        self,
        id: int,
        rel_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> str:
        results = []

        if direction in ("outgoing", "both"):
            meta = self.db.get_metadata(id)
            for e in meta.edges:
                if rel_type and e.rel_type != rel_type:
                    continue
                try:
                    tm = self.db.get_metadata(e.target_id)
                    results.append({
                        "direction": "outgoing",
                        "edge":      {"target_id": e.target_id, "rel_type": e.rel_type, "weight": round(e.weight, 3)},
                        "node":      {"content": tm.content[:150], "entity_type": tm.get_attribute("entity_type"), "product": tm.get_attribute("product")},
                    })
                except Exception:
                    pass

        if direction in ("incoming", "both"):
            for e in self.db.get_incoming(id):
                if rel_type and e.rel_type != rel_type:
                    continue
                try:
                    sm = self.db.get_metadata(e.source_id)
                    results.append({
                        "direction": "incoming",
                        "edge":      {"source_id": e.source_id, "rel_type": e.rel_type, "weight": round(e.weight, 3)},
                        "node":      {"content": sm.content[:150], "entity_type": sm.get_attribute("entity_type"), "product": sm.get_attribute("product")},
                    })
                except Exception:
                    pass

        return json.dumps({"related": results, "count": len(results)}, indent=2)

    def feather_add_intel(
        self,
        content: str,
        entity_type: str,
        product: Optional[str] = None,
        importance: float = 0.8,
    ) -> str:
        vec  = self.embed(content)
        nid  = self._next_id
        self._next_id += 1

        meta = _fdb.Metadata()
        meta.timestamp    = int(time.time())
        meta.importance   = max(0.0, min(1.0, importance))
        meta.type         = _fdb.ContextType.EVENT
        meta.source       = "agent_ingested"
        meta.content      = content
        meta.namespace_id = "agent"
        meta.entity_id    = entity_type
        meta.set_attribute("entity_type", entity_type)
        if product:
            meta.set_attribute("product", product)
        meta.set_attribute("modality", "text")

        self.db.add(id=nid, vec=vec, meta=meta)
        if self.auto_save:
            self.db.save()

        return json.dumps({"id": nid, "status": "added", "entity_type": entity_type, "product": product or ""})

    def feather_link_nodes(
        self,
        from_id: int,
        to_id: int,
        rel_type: str,
        weight: float = 0.8,
    ) -> str:
        try:
            self.db.link(from_id=from_id, to_id=to_id, rel_type=rel_type, weight=weight)
            if self.auto_save:
                self.db.save()
            return json.dumps({"status": "linked", "from_id": from_id, "to_id": to_id, "rel_type": rel_type, "weight": weight})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def feather_timeline(
        self,
        product: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        all_ids = self.db.get_all_ids(modality="text")
        nodes   = []
        for nid in all_ids:
            try:
                m = self.db.get_metadata(nid)
                if product and m.get_attribute("product") != product:
                    continue
                if entity_type and m.get_attribute("entity_type") != entity_type:
                    continue
                nodes.append({
                    "id":          nid,
                    "timestamp":   m.timestamp,
                    "content":     m.content[:150],
                    "entity_type": m.get_attribute("entity_type"),
                    "product":     m.get_attribute("product"),
                    "importance":  round(m.importance, 3),
                })
            except Exception:
                continue

        nodes.sort(key=lambda n: n["timestamp"], reverse=True)
        return json.dumps({"timeline": nodes[:limit], "total_matching": len(nodes)}, indent=2)

    # ── Phase 6 tools ─────────────────────────────────────────────────────────

    def feather_forget(self, id: int) -> str:
        try:
            self.db.forget(id)
            if self.auto_save:
                self.db.save()
            return json.dumps({"status": "forgotten", "id": id})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def feather_health(self, modality: str = "text") -> str:
        from feather_db.memory import MemoryManager
        report = MemoryManager.health_report(self.db, modality=modality)
        return json.dumps(report, indent=2)

    def feather_why(self, id: int, query: str) -> str:
        from feather_db.memory import MemoryManager
        vec = self.embed(query)
        result = MemoryManager.why_retrieved(self.db, node_id=id, query_vec=vec)
        return json.dumps(result, indent=2)

    def feather_mmr_search(
        self,
        query: str,
        k: int = 5,
        diversity: float = 0.5,
    ) -> str:
        from feather_db.memory import MemoryManager
        vec     = self.embed(query)
        results = MemoryManager.search_mmr(self.db, query_vec=vec, k=k, diversity=diversity)
        output  = []
        for r in results:
            m = r.metadata
            output.append({
                "id":          r.id,
                "score":       round(r.score, 4),
                "content":     m.content[:200],
                "entity_type": m.get_attribute("entity_type"),
                "product":     m.get_attribute("product"),
                "importance":  round(m.importance, 3),
                "confidence":  round(m.confidence, 3),
            })
        return json.dumps({"results": output, "count": len(output),
                           "diversity": diversity}, indent=2)

    def feather_consolidate(
        self,
        namespace: str,
        since_hours: float = 24.0,
        threshold: float = 0.85,
    ) -> str:
        from feather_db.memory import MemoryManager
        new_ids = MemoryManager.consolidate(
            self.db,
            namespace=namespace,
            since_hours=since_hours,
            similarity_threshold=threshold,
        )
        return json.dumps({
            "status":    "consolidated",
            "namespace": namespace,
            "new_nodes": new_ids,
            "count":     len(new_ids),
        }, indent=2)

    def feather_episode_get(self, episode_id: str) -> str:
        from feather_db.episodes import EpisodeManager
        em      = EpisodeManager()
        members = em.get_episode(self.db, episode_id)
        output  = []
        for m in members:
            output.append({
                "id":          m["id"],
                "timestamp":   m["timestamp"],
                "content":     m["content"][:200],
                "entity_type": m["entity_type"],
                "importance":  m["importance"],
            })
        return json.dumps({
            "episode_id": episode_id,
            "nodes":      output,
            "count":      len(output),
        }, indent=2)

    def feather_expire(self) -> str:
        try:
            count = self.db.forget_expired()
            if self.auto_save and count > 0:
                self.db.save()
            return json.dumps({"status": "expired", "count": count})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Phase 9 tools ─────────────────────────────────────────────────────────

    def feather_ingest(
        self,
        content: str,
        source_id: str = "",
        timestamp: int = 0,
        namespace: Optional[str] = None,
    ) -> str:
        import time as _time
        ts = timestamp or int(_time.time())
        ns = namespace or self._namespace

        if self._pipeline is not None:
            try:
                from feather_db.pipelines import IngestRecord
                rec = IngestRecord(
                    content=content,
                    source_id=source_id or f"mcp::{ts}",
                    timestamp=ts,
                    metadata={},
                )
                # Temporarily override namespace if different
                orig_ns = self._pipeline._namespace
                self._pipeline._namespace = ns
                stats = self._pipeline.ingest([rec])
                self._pipeline._namespace = orig_ns
                if self.auto_save:
                    self.db.save()
                return json.dumps({
                    "status": "ok",
                    "mode": "phase9",
                    "facts_extracted": stats.facts_extracted,
                    "entities_resolved": stats.entities_resolved,
                    "records_ingested": stats.records_ingested,
                    "extraction_failures": stats.extraction_failures,
                }, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e), "mode": "phase9"})
        else:
            # Raw storage fallback (no LLM provider configured)
            try:
                meta = _fdb.Metadata()
                meta.content = content
                meta.timestamp = ts
                meta.namespace_id = ns
                meta.source = source_id or "mcp"
                meta.type = _fdb.ContextType.CONVERSATION
                vec = self._embed(content)
                nid = self._next_id
                self._next_id += 1
                self.db.add(id=nid, vec=vec, meta=meta)
                if self.auto_save:
                    self.db.save()
                return json.dumps({
                    "status": "ok",
                    "mode": "raw",
                    "id": nid,
                    "note": "No system_provider configured — raw storage only. "
                            "Start server with --system-provider to enable Phase 9.",
                }, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e), "mode": "raw"})

    def feather_recall(
        self,
        query: str,
        k: int = 8,
        namespace: Optional[str] = None,
        half_life_days: float = 30.0,
        time_weight: float = 0.3,
    ) -> str:
        vec = self.embed(query)
        try:
            scoring = _fdb.ScoringConfig(
                half_life=half_life_days,
                weight=time_weight,
                min=0.0,
            )
            try:
                results = self.db.hybrid_search(
                    vec, query, k=k * 2, modality="text", scoring=scoring
                )
            except Exception:
                results = self.db.search(vec, k=k * 2, modality="text", scoring=scoring)

            output = []
            for r in results:
                m = r.metadata
                if namespace and m.namespace_id != namespace:
                    continue
                output.append({
                    "id":       r.id,
                    "score":    round(r.score, 4),
                    "content":  m.content[:300],
                    "namespace": m.namespace_id,
                    "source":   m.source,
                    "recall_count": m.recall_count,
                })
                if len(output) >= k:
                    break

            return json.dumps({"results": output, "count": len(output),
                               "decay": {"half_life": half_life_days,
                                         "time_weight": time_weight}}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def handle(self, tool_name: str, tool_input: dict) -> str:
        """Route a tool call by name to the correct method."""
        dispatch = {
            "feather_search":        self.feather_search,
            "feather_context_chain": self.feather_context_chain,
            "feather_get_node":      self.feather_get_node,
            "feather_get_related":   self.feather_get_related,
            "feather_add_intel":     self.feather_add_intel,
            "feather_link_nodes":    self.feather_link_nodes,
            "feather_timeline":      self.feather_timeline,
            # Phase 6
            "feather_forget":        self.feather_forget,
            "feather_health":        self.feather_health,
            "feather_why":           self.feather_why,
            "feather_mmr_search":    self.feather_mmr_search,
            "feather_consolidate":   self.feather_consolidate,
            "feather_episode_get":   self.feather_episode_get,
            "feather_expire":        self.feather_expire,
            # Phase 9
            "feather_ingest":        self.feather_ingest,
            "feather_recall":        self.feather_recall,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return fn(**tool_input)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name, "input": tool_input})
