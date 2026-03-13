"""
LangChain Integration — Feather DB v0.6.0
==========================================
Drop-in adapters for LangChain:

  FeatherVectorStore   — LangChain VectorStore implementation
  FeatherMemory        — LangChain BaseMemory with adaptive decay
  FeatherRetriever     — LangChain BaseRetriever wrapping context_chain()

Install:
    pip install langchain langchain-core

Quick start:
    from feather_db.integrations.langchain_compat import (
        FeatherVectorStore, FeatherMemory, FeatherRetriever,
    )
    from langchain.chat_models import ChatOpenAI

    store  = FeatherVectorStore(db_path="my.feather", dim=3072, embed_fn=embed)
    store.add_texts(["Fixed deposit at 8.5%", "Competitor launched 8.75%"])
    docs = store.similarity_search("FD rate", k=3)

    mem    = FeatherMemory(db_path="my.feather", dim=3072, embed_fn=embed)
    ret    = FeatherRetriever(db_path="my.feather", dim=3072, embed_fn=embed)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.memory import BaseMemory
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # Stub base classes so the module can be imported without langchain
    class VectorStore:   pass   # type: ignore
    class BaseRetriever: pass   # type: ignore
    class BaseMemory:    pass   # type: ignore
    class Document:              # type: ignore
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


def _require_langchain():
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "pip install langchain langchain-core  # required for LangChain adapters"
        )


def _result_to_doc(r) -> "Document":
    m = r.metadata
    return Document(
        page_content=m.content,
        metadata={
            "id":           r.id,
            "score":        round(r.score, 4),
            "source":       m.source,
            "importance":   round(m.importance, 3),
            "confidence":   round(m.confidence, 3),
            "namespace_id": m.namespace_id,
            "entity_id":    m.entity_id,
            "recall_count": m.recall_count,
            "timestamp":    m.timestamp,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# FeatherVectorStore
# ──────────────────────────────────────────────────────────────────────────────

class FeatherVectorStore(VectorStore):
    """
    LangChain VectorStore backed by Feather DB.

    Implements: add_texts, similarity_search, from_texts.
    """

    def __init__(
        self,
        db_path: str,
        dim: int = 3072,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        namespace: str = "langchain",
        modality: str = "text",
    ):
        _require_langchain()
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be installed")

        self._db       = _fdb.DB.open(db_path, dim=dim)
        self._embed    = embed_fn or self._default_embed
        self._ns       = namespace
        self._mod      = modality
        self._dim      = dim
        self._next_id  = int(time.time() * 1000) % (2 ** 40)
        self._fdb      = _fdb

    def _default_embed(self, text: str) -> np.ndarray:
        import hashlib
        vec = np.zeros(self._dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        n = np.linalg.norm(vec)
        return (vec / n) if n > 0 else vec

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add to Feather DB. Returns list of string IDs."""
        ids: list[str] = []
        for i, text in enumerate(texts):
            vec = np.array(self._embed(text), dtype=np.float32)
            nid = self._next_id; self._next_id += 1

            meta = self._fdb.Metadata()
            meta.timestamp    = int(time.time())
            meta.importance   = 0.9
            meta.type         = self._fdb.ContextType.FACT
            meta.source       = "langchain"
            meta.content      = text
            meta.namespace_id = self._ns

            if metadatas and i < len(metadatas):
                for k, v in metadatas[i].items():
                    meta.set_attribute(str(k), str(v))

            self._db.add(id=nid, vec=vec, meta=meta, modality=self._mod)
            ids.append(str(nid))

        self._db.save()
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list["Document"]:
        """Search by query text and return LangChain Documents."""
        vec     = np.array(self._embed(query), dtype=np.float32)
        results = self._db.search(vec, k=k, modality=self._mod)
        return [_result_to_doc(r) for r in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple["Document", float]]:
        vec     = np.array(self._embed(query), dtype=np.float32)
        results = self._db.search(vec, k=k, modality=self._mod)
        return [(_result_to_doc(r), r.score) for r in results]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: Optional[list[dict]] = None,
        db_path: str = "langchain.feather",
        dim: int = 3072,
        **kwargs: Any,
    ) -> "FeatherVectorStore":
        """Class method: create store and populate with texts."""
        embed_fn = getattr(embedding, "embed_query", None) or getattr(embedding, "embed", None)
        store    = cls(db_path=db_path, dim=dim, embed_fn=embed_fn, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list["Document"],
        embedding: Any,
        db_path: str = "langchain.feather",
        dim: int = 3072,
        **kwargs: Any,
    ) -> "FeatherVectorStore":
        texts     = [d.page_content for d in documents]
        metas     = [d.metadata for d in documents]
        embed_fn  = getattr(embedding, "embed_query", None) or getattr(embedding, "embed", None)
        store     = cls(db_path=db_path, dim=dim, embed_fn=embed_fn, **kwargs)
        store.add_texts(texts, metadatas=metas)
        return store

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, self._embed(t))) for t in texts]

    def _embed_query(self, text: str) -> list[float]:
        return list(map(float, self._embed(text)))


# ──────────────────────────────────────────────────────────────────────────────
# FeatherMemory
# ──────────────────────────────────────────────────────────────────────────────

class FeatherMemory(BaseMemory):
    """
    LangChain conversation memory backed by Feather DB with adaptive decay.

    On load_memory_variables: retrieves contextually relevant history via
    semantic search (not just last-N turns) with stickiness-modulated scoring.

    On save_context: embeds the assistant's output and stores it.
    """

    def __init__(
        self,
        db_path: str,
        dim: int = 3072,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        k: int = 5,
        namespace: str = "memory",
        input_key: str = "input",
        output_key: str = "output",
        memory_key: str = "history",
    ):
        _require_langchain()
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be installed")

        self._db         = _fdb.DB.open(db_path, dim=dim)
        self._fdb        = _fdb
        self._embed      = embed_fn or (lambda t: np.zeros(dim))
        self._k          = k
        self._ns         = namespace
        self._input_key  = input_key
        self._output_key = output_key
        self._memory_key = memory_key
        self._next_id    = int(time.time() * 1000) % (2 ** 40)

    @property
    def memory_variables(self) -> list[str]:
        return [self._memory_key]

    def load_memory_variables(self, inputs: dict) -> dict:
        query = inputs.get(self._input_key, "")
        if not query:
            return {self._memory_key: ""}

        vec     = np.array(self._embed(query), dtype=np.float32)
        scoring = self._fdb.ScoringConfig(half_life=7.0, weight=0.4, min=0.0)
        results = self._db.search(vec, k=self._k, scoring=scoring)

        lines = []
        for r in results:
            ts   = time.strftime("%b %d %H:%M", time.localtime(r.metadata.timestamp))
            role = r.metadata.get_attribute("role") or "assistant"
            lines.append(f"[{ts}] {role}: {r.metadata.content}")

        return {self._memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        output = outputs.get(self._output_key, "")
        if not output:
            return

        vec = np.array(self._embed(output), dtype=np.float32)
        nid = self._next_id; self._next_id += 1

        meta = self._fdb.Metadata()
        meta.timestamp    = int(time.time())
        meta.importance   = 0.8
        meta.type         = self._fdb.ContextType.CONVERSATION
        meta.source       = "conversation"
        meta.content      = output
        meta.namespace_id = self._ns
        meta.set_attribute("role",       "assistant")
        meta.set_attribute("input",      str(inputs.get(self._input_key, ""))[:200])

        self._db.add(id=nid, vec=vec, meta=meta)
        self._db.save()

    def clear(self) -> None:
        self._db.purge(self._ns)


# ──────────────────────────────────────────────────────────────────────────────
# FeatherRetriever
# ──────────────────────────────────────────────────────────────────────────────

class FeatherRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapping Feather DB's context_chain().

    Returns Documents including graph-expanded nodes, not just top-k similarity.
    """

    def __init__(
        self,
        db_path: str,
        dim: int = 3072,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        k: int = 5,
        hops: int = 2,
        modality: str = "text",
    ):
        _require_langchain()
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be installed")

        self._db     = _fdb.DB.open(db_path, dim=dim)
        self._embed  = embed_fn or (lambda t: np.zeros(dim))
        self._k      = k
        self._hops   = hops
        self._mod    = modality

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> list["Document"]:
        vec    = np.array(self._embed(query), dtype=np.float32)
        result = self._db.context_chain(vec, k=self._k, hops=self._hops, modality=self._mod)

        docs: list[Document] = []
        for node in sorted(result.nodes, key=lambda n: (n.hop, -n.score)):
            m = node.metadata
            docs.append(Document(
                page_content=m.content,
                metadata={
                    "id":          node.id,
                    "score":       round(node.score, 4),
                    "hop":         node.hop,
                    "similarity":  round(node.similarity, 4),
                    "entity_type": m.get_attribute("entity_type"),
                    "namespace":   m.namespace_id,
                    "importance":  round(m.importance, 3),
                },
            ))
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> list["Document"]:
        # Sync wrapper — feather_db is synchronous
        return self._get_relevant_documents(query, run_manager=run_manager)
