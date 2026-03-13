"""
LlamaIndex Integration — Feather DB v0.6.0
============================================
Drop-in adapters for LlamaIndex:

  FeatherVectorStore      — VectorStore implementation (query + add)
  FeatherReader           — BaseReader that loads a .feather file as Documents

Install:
    pip install llama-index llama-index-core

Quick start:
    from feather_db.integrations.llamaindex_compat import (
        FeatherVectorStore, FeatherReader,
    )

    # Load existing .feather as LlamaIndex documents
    reader = FeatherReader()
    docs   = reader.load_data(db_path="my.feather", dim=3072)

    # Use as a vector store
    store  = FeatherVectorStore(db_path="my.feather", dim=3072, embed_fn=embed)
    result = store.query(VectorStoreQuery(query_embedding=vec, similarity_top_k=5))
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

# ── Optional llama-index imports ─────────────────────────────────────────────
try:
    from llama_index.core.schema import TextNode, NodeWithScore, Document as LIDocument
    from llama_index.core.vector_stores.types import (
        VectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.readers.base import BaseReader
    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False

    class VectorStore:   pass   # type: ignore
    class VectorStoreQuery: pass  # type: ignore
    class VectorStoreQueryResult: pass  # type: ignore
    class BaseReader: pass   # type: ignore
    class TextNode:          # type: ignore
        def __init__(self, text="", metadata=None):
            self.text = text; self.metadata = metadata or {}
    class NodeWithScore:     # type: ignore
        def __init__(self, node=None, score=0.0):
            self.node = node; self.score = score
    class LIDocument:        # type: ignore
        def __init__(self, text="", metadata=None):
            self.text = text; self.metadata = metadata or {}


def _require_llamaindex():
    if not _LLAMAINDEX_AVAILABLE:
        raise ImportError(
            "pip install llama-index llama-index-core  # required for LlamaIndex adapters"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FeatherVectorStore
# ──────────────────────────────────────────────────────────────────────────────

class FeatherVectorStore(VectorStore):
    """
    LlamaIndex VectorStore backed by Feather DB.

    Implements: add(), query(), delete().
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        db_path: str,
        dim: int = 3072,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        namespace: str = "llamaindex",
        modality: str = "text",
    ):
        _require_llamaindex()
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be installed")

        self._db      = _fdb.DB.open(db_path, dim=dim)
        self._fdb     = _fdb
        self._embed   = embed_fn
        self._ns      = namespace
        self._mod     = modality
        self._dim     = dim
        self._next_id = int(time.time() * 1000) % (2 ** 40)

    @property
    def client(self):
        return self._db

    def add(
        self,
        nodes: list,  # list[BaseNode]
        **kwargs: Any,
    ) -> list[str]:
        """Add LlamaIndex nodes to Feather DB."""
        ids: list[str] = []
        for node in nodes:
            text = getattr(node, "get_content", lambda: getattr(node, "text", ""))()

            if hasattr(node, "embedding") and node.embedding:
                vec = np.array(node.embedding, dtype=np.float32)
            elif self._embed is not None:
                vec = np.array(self._embed(text), dtype=np.float32)
            else:
                vec = np.zeros(self._dim, dtype=np.float32)

            nid = self._next_id; self._next_id += 1

            meta = self._fdb.Metadata()
            meta.timestamp    = int(time.time())
            meta.importance   = float(getattr(node, "score", None) or 0.9)
            meta.type         = self._fdb.ContextType.FACT
            meta.source       = "llamaindex"
            meta.content      = text
            meta.namespace_id = self._ns

            node_meta = getattr(node, "metadata", {}) or {}
            for k, v in node_meta.items():
                meta.set_attribute(str(k), str(v))

            # Store LlamaIndex node_id for round-trip retrieval
            node_id_str = getattr(node, "node_id", None) or str(nid)
            meta.set_attribute("li_node_id", node_id_str)

            self._db.add(id=nid, vec=vec, meta=meta, modality=self._mod)
            ids.append(node_id_str)

        self._db.save()
        return ids

    def query(
        self,
        query: Any,  # VectorStoreQuery
        **kwargs: Any,
    ) -> Any:  # VectorStoreQueryResult
        """Query Feather DB and return VectorStoreQueryResult."""
        k   = getattr(query, "similarity_top_k", 5) or 5
        vec = getattr(query, "query_embedding", None)

        if vec is None:
            query_str = getattr(query, "query_str", None) or ""
            if self._embed is not None and query_str:
                vec = self._embed(query_str)
            else:
                return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])

        q_arr   = np.array(vec, dtype=np.float32)
        results = self._db.search(q_arr, k=k, modality=self._mod)

        nodes:  list = []
        ids:    list[str] = []
        scores: list[float] = []

        for r in results:
            m = r.metadata
            node = TextNode(
                text=m.content,
                metadata={
                    "source":       m.source,
                    "importance":   round(m.importance, 3),
                    "namespace_id": m.namespace_id,
                    "entity_id":    m.entity_id,
                    "recall_count": m.recall_count,
                    **{k: v for k, v in m.attributes.items()},
                },
            )
            node_id = m.get_attribute("li_node_id") or str(r.id)
            if hasattr(node, "node_id"):
                node.node_id = node_id

            nodes.append(NodeWithScore(node=node, score=r.score))
            ids.append(node_id)
            scores.append(round(r.score, 4))

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Soft-delete nodes matching ref_doc_id attribute."""
        ids = self._db.get_all_ids(self._mod)
        for nid in ids:
            meta = self._db.get_metadata(nid)
            if meta is None:
                continue
            if meta.get_attribute("li_node_id") == ref_doc_id:
                self._db.forget(nid)
        self._db.save()


# ──────────────────────────────────────────────────────────────────────────────
# FeatherReader
# ──────────────────────────────────────────────────────────────────────────────

class FeatherReader(BaseReader):
    """
    LlamaIndex BaseReader that loads a .feather file as LlamaIndex Documents.

    Usage:
        reader = FeatherReader()
        docs   = reader.load_data(db_path="my.feather", dim=3072)
        index  = VectorStoreIndex.from_documents(docs)
    """

    def load_data(
        self,
        db_path: str,
        dim: int = 3072,
        modality: str = "text",
        namespace_filter: str = "",
        min_importance: float = 0.0,
    ) -> list:
        """
        Load all nodes from a .feather file as LlamaIndex Documents.

        Returns list of llama_index.core.schema.Document.
        """
        _require_llamaindex()
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be installed")

        db  = _fdb.DB.open(db_path, dim=dim)
        ids = db.get_all_ids(modality)
        docs: list = []

        for nid in ids:
            meta = db.get_metadata(nid)
            if meta is None:
                continue
            if namespace_filter and meta.namespace_id != namespace_filter:
                continue
            if meta.importance < min_importance:
                continue

            doc = LIDocument(
                text=meta.content,
                metadata={
                    "feather_id":   nid,
                    "source":       meta.source,
                    "namespace_id": meta.namespace_id,
                    "entity_id":    meta.entity_id,
                    "importance":   round(meta.importance, 3),
                    "confidence":   round(meta.confidence, 3),
                    "recall_count": meta.recall_count,
                    "timestamp":    meta.timestamp,
                    **{k: v for k, v in meta.attributes.items()},
                },
            )
            if hasattr(doc, "doc_id"):
                doc.doc_id = str(nid)
            docs.append(doc)

        return docs
