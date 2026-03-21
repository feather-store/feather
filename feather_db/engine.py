"""
feather_db.engine — Self-Aligned Context Engine (Phase 1)
==========================================================
ContextEngine wraps a Feather DB instance and an LLM provider to create
a self-organizing knowledge base: raw text goes in, the engine decides
how to structure, classify, link, and store it autonomously.

Works with ANY LLM backend — closed-source (Claude, OpenAI, Gemini) or
self-hosted open-source (Ollama, vLLM, LM Studio, Groq, …).

Architecture
------------
  Raw text
      │
      ▼
  Embed (your embedder fn)
      │
      ▼
  Sample similar existing nodes → build LLM prompt
      │
      ▼
  LLM classifies: entity_type, importance, confidence, ttl,
                  namespace, episode_id, suggested_links
      │
      ▼
  db.add() → db.link() → episodes → triggers → contradiction check
      │
      ▼
  node_id returned

Quick start
-----------
  from feather_db.engine    import ContextEngine
  from feather_db.providers import ClaudeProvider, OllamaProvider
  import numpy as np

  # Closed-source
  engine = ContextEngine(
      db_path  = "my.feather",
      dim      = 3072,
      provider = ClaudeProvider(model="claude-haiku-4-5-20251001"),
      embedder = my_embed_fn,
  )

  # Open-source local
  engine = ContextEngine(
      db_path  = "my.feather",
      dim      = 768,
      provider = OllamaProvider(model="llama3.1:8b"),
      embedder = my_embed_fn,
  )

  node_id = engine.ingest("Competitor launched a new product targeting our core segment.")
  node_id = engine.ingest("User prefers short bullet-point responses.",
                           hint={"namespace": "user_prefs", "ttl": 86400})

  ids = engine.ingest_batch(["fact one", "fact two", "fact three"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Callable, Optional

import numpy as np

from .providers import LLMProvider

log = logging.getLogger("feather_db.engine")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(val)
        return max(lo, min(hi, v)) if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return max(0, int(val))
    except (TypeError, ValueError):
        return default


def _norm_entity_type(s: str) -> str:
    """'Competitor Intel' → 'competitor_intel'"""
    return re.sub(r"[^a-z0-9_]", "_", str(s).lower().strip())[:64]


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a knowledge graph classifier. Given a text input and \
optional context from an existing knowledge graph, output a single JSON object \
that classifies the text for storage.

RESPOND WITH ONLY THIS JSON OBJECT — no explanation, no markdown, no code fences:
{
  "entity_type":     string,  // what kind of thing this is, snake_case, e.g. "fact", "event", "preference", "observation", "question", "competitor_intel", "strategy_brief"
  "importance":      float,   // 0.0 to 1.0 — how significant is this information
  "confidence":      float,   // 0.0 to 1.0 — certainty of your classification
  "ttl":             integer, // seconds until expiry; 0 = never expires. Use 0 for facts, 3600 for session state, 86400 for daily signals
  "namespace":       string,  // partition key (use the caller's default namespace if unsure)
  "episode_id":      string,  // episode this belongs to, or "" if none
  "suggested_links": [        // up to 3 links to EXISTING nodes in the graph context; empty [] if none
    {"target_id": integer, "rel_type": string, "weight": float}
  ]
}

VALID rel_type values: "related_to", "caused_by", "supports", "contradicts", "derived_from", "references"

EXAMPLE:
Input text: "User always prefers responses in bullet points, not paragraphs."
Context: []
Output: {"entity_type":"preference","importance":0.8,"confidence":0.95,"ttl":0,"namespace":"default","episode_id":"","suggested_links":[]}"""

_USER_TEMPLATE = """EXISTING GRAPH CONTEXT (most similar nodes already stored):
{context_block}

TEXT TO CLASSIFY:
{text}

Respond with ONLY the JSON object. No explanation, no markdown, no code fences."""

_FALLBACK_ENTITY_TYPE = "fact"


# ── ContextEngine ──────────────────────────────────────────────────────────────

class ContextEngine:
    """
    Self-organizing context engine — wraps Feather DB + an LLM provider.

    The engine autonomously decides how to structure, link, and store any
    raw text. It works with closed-source models (Claude, OpenAI, Gemini)
    and self-hosted open-source models (Ollama, vLLM, Groq, Mistral, …).

    Parameters
    ----------
    db_path      : path to the .feather file (created if absent)
    dim          : vector dimension — must match your embedder
    provider     : any LLMProvider instance (ClaudeProvider, OllamaProvider, …)
    embedder     : callable (text: str) -> np.ndarray
    namespace    : default namespace for ingested nodes
    modality     : HNSW modality to search/store in (default "text")
    sample_k     : max existing nodes sampled for LLM context
    auto_save    : call db.save() after each ingest
    watch_manager         : optional WatchManager for semantic triggers
    episode_manager       : optional EpisodeManager for episode membership
    contradiction_detector: optional ContradictionDetector
    source_tag   : metadata.source value written to all ingested nodes
    provider_fallback     : if True, fall back to heuristic classify when
                            LLM call fails (default True)
    """

    def __init__(
        self,
        db_path:   str,
        dim:       int,
        provider:  Optional[LLMProvider],
        embedder:  Callable[[str], np.ndarray],
        namespace: str = "default",
        modality:  str = "text",
        sample_k:  int = 10,
        auto_save: bool = True,
        watch_manager=None,
        episode_manager=None,
        contradiction_detector=None,
        source_tag: str = "context_engine",
        provider_fallback: bool = True,
    ):
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be built. Run: python setup.py build_ext --inplace")

        self._fdb        = _fdb
        self._db         = _fdb.DB.open(db_path, dim=dim)
        self._dim        = dim
        self._provider   = provider
        self._embedder   = embedder
        self._ns         = namespace
        self._mod        = modality
        self._sample_k   = sample_k
        self._auto_save  = auto_save
        self._source     = source_tag
        self._fallback   = provider_fallback

        self._watch_mgr  = watch_manager
        self._ep_mgr     = episode_manager
        self._contra_det = contradiction_detector

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, text: str, hint: Optional[dict] = None) -> int:
        """
        Full pipeline: embed → sample → LLM classify → store → link → triggers.
        Returns the node ID.

        Parameters
        ----------
        text : raw text to ingest
        hint : optional caller overrides, any of:
                 namespace, entity_type, importance, confidence,
                 ttl, episode_id, source
        """
        if not text.strip():
            raise ValueError("text cannot be empty")

        # 1. Embed
        vec = np.array(self._embedder(text), dtype=np.float32)

        # 2. Sample similar existing nodes for LLM context
        context_nodes = self._sample_context(vec)

        # 3. Classify via LLM (or heuristic fallback)
        parsed = self._classify(text, context_nodes)

        # 4. Apply caller hints (always win over LLM)
        if hint:
            for k in ("namespace", "entity_type", "importance", "confidence",
                      "ttl", "episode_id"):
                if k in hint:
                    parsed[k] = hint[k]

        # 5. Store
        node_id = self._apply(text, vec, parsed, hint)

        # 6. Create suggested graph links (best-effort)
        for link in (parsed.get("suggested_links") or [])[:5]:
            try:
                self._db.link(
                    from_id  = node_id,
                    to_id    = int(link["target_id"]),
                    rel_type = str(link.get("rel_type", "related_to")),
                    weight   = _clamp(link.get("weight"), 0.0, 1.0, 0.5),
                )
            except Exception:
                pass

        # 7. Episode membership (best-effort)
        ep_id = str(parsed.get("episode_id") or "").strip()
        if ep_id and self._ep_mgr:
            try:
                self._ep_mgr.add_to_episode(self._db, node_id, ep_id)
            except Exception:
                pass

        # 8. Semantic triggers (pass vec to avoid re-embedding)
        if self._watch_mgr:
            try:
                self._watch_mgr.check_triggers(
                    self._db, new_node_id=node_id, new_vec=vec,
                    embed_fn=self._embedder,
                )
            except Exception:
                pass

        # 9. Contradiction detection (best-effort, can be expensive)
        if self._contra_det:
            try:
                self._contra_det.check(
                    self._db, new_node_id=node_id, auto_link=True,
                )
            except Exception:
                pass

        # 10. Save
        if self._auto_save:
            self._db.save()

        log.debug(
            "ingest id=%d entity_type=%s importance=%.2f provider=%s",
            node_id,
            parsed.get("entity_type", "?"),
            _clamp(parsed.get("importance"), 0.0, 1.0, 0.7),
            getattr(self._provider, "name", lambda: "none")(),
        )
        return node_id

    def ingest_batch(
        self,
        texts: list[str],
        hints: Optional[list[Optional[dict]]] = None,
    ) -> list[int]:
        """
        Ingest multiple texts sequentially.
        Returns list of node IDs in the same order.
        """
        if hints is None:
            hints = [None] * len(texts)
        elif len(hints) != len(texts):
            raise ValueError("hints must be same length as texts")

        return [self.ingest(t, h) for t, h in zip(texts, hints)]

    @property
    def db(self):
        """Direct access to the underlying feather_db.DB instance."""
        return self._db

    # ── Internal: classification ──────────────────────────────────────────────

    def _classify(self, text: str, context_nodes: list[dict]) -> dict:
        """Call LLM → parse JSON. Falls back to heuristic on any error."""
        if self._provider is None:
            return self._heuristic_classify(text)

        try:
            messages = self._build_messages(text, context_nodes)
            raw      = self._provider.complete(messages, max_tokens=512, temperature=0.0)
            parsed   = self._extract_json(raw)
            return parsed
        except Exception as exc:
            log.warning("LLM classify failed (%s): %s — using heuristic", type(exc).__name__, exc)
            if self._fallback:
                return self._heuristic_classify(text)
            raise

    def _heuristic_classify(self, text: str) -> dict:
        """
        Zero-cost offline fallback — no LLM needed.
        Uses simple keyword rules to assign entity_type and importance.
        """
        t = text.lower()
        if any(w in t for w in ("prefer", "always", "never", "like", "dislike", "want")):
            etype, imp = "preference", 0.8
        elif any(w in t for w in ("competitor", "rival", "launched", "released", "announced")):
            etype, imp = "competitor_intel", 0.9
        elif any(w in t for w in ("strategy", "plan", "goal", "objective", "target")):
            etype, imp = "strategy_brief", 0.85
        elif any(w in t for w in ("error", "bug", "fail", "crash", "issue", "problem")):
            etype, imp = "issue", 0.9
        elif any(w in t for w in ("user", "customer", "feedback", "complaint", "request")):
            etype, imp = "user_feedback", 0.8
        elif any(w in t for w in ("market", "trend", "report", "survey", "data")):
            etype, imp = "market_signal", 0.75
        else:
            etype, imp = "fact", 0.7

        return {
            "entity_type":     etype,
            "importance":      imp,
            "confidence":      0.5,    # low confidence signals heuristic was used
            "ttl":             0,
            "namespace":       self._ns,
            "episode_id":      "",
            "suggested_links": [],
        }

    def _build_messages(self, text: str, context_nodes: list[dict]) -> list[dict]:
        if context_nodes:
            lines = []
            for n in context_nodes:
                snippet = n.get("content", "")[:120]
                lines.append(
                    f"[id={n['id']}] (entity_type={n.get('entity_type','?')}, "
                    f"importance={n.get('importance',0):.2f}) \"{snippet}\""
                )
            context_block = "\n".join(lines)
        else:
            context_block = "(none — this is the first node)"

        user_msg = _USER_TEMPLATE.format(
            context_block=context_block,
            text=text[:1500],   # cap to avoid blowing context on small models
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

    # ── Internal: JSON extraction (robust for small models) ───────────────────

    def _extract_json(self, raw: str) -> dict:
        """
        5-stage JSON extraction — handles small-model quirks:
          1. Direct parse
          2. Strip markdown code fences
          3. Balanced-brace extraction (first match)
          4. Balanced-brace extraction (last match)
          5. Repair trailing commas then retry
          → fallback default on all failures
        """
        s = raw.strip()

        # Stage 1: direct
        try:
            return self._validate(json.loads(s))
        except Exception:
            pass

        # Stage 2: strip code fences  (```json...``` or ```...```)
        s2 = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s2 = re.sub(r"\s*```$", "", s2)
        try:
            return self._validate(json.loads(s2.strip()))
        except Exception:
            pass

        # Stages 3 & 4: balanced brace extraction
        for blob in self._find_json_blobs(s):
            # Stage 5: repair trailing commas
            for candidate in (blob, self._repair_json(blob)):
                try:
                    return self._validate(json.loads(candidate))
                except Exception:
                    pass

        log.warning("JSON extraction failed for raw output: %s", raw[:200])
        return self._default_parsed()

    def _find_json_blobs(self, s: str) -> list[str]:
        """
        Walk the string and yield balanced {...} substrings, first and last.
        """
        blobs: list[str] = []
        for start in range(len(s)):
            if s[start] != "{":
                continue
            depth = 0
            for end in range(start, len(s)):
                if s[end] == "{":
                    depth += 1
                elif s[end] == "}":
                    depth -= 1
                if depth == 0:
                    blobs.append(s[start:end + 1])
                    break
        # Return first and last distinct blobs
        seen: list[str] = []
        for b in blobs:
            if b not in seen:
                seen.append(b)
        return seen[:1] + (seen[-1:] if len(seen) > 1 else [])

    @staticmethod
    def _repair_json(s: str) -> str:
        """Remove trailing commas before } or ]."""
        return re.sub(r",\s*([}\]])", r"\1", s)

    def _validate(self, d: dict) -> dict:
        """Clamp and normalise all fields after parsing."""
        d["entity_type"]  = _norm_entity_type(d.get("entity_type") or _FALLBACK_ENTITY_TYPE)
        d["importance"]   = _clamp(d.get("importance"), 0.0, 1.0, 0.7)
        d["confidence"]   = _clamp(d.get("confidence"), 0.0, 1.0, 0.5)
        d["ttl"]          = _safe_int(d.get("ttl"), 0)
        d["namespace"]    = str(d.get("namespace") or self._ns).strip() or self._ns
        d["episode_id"]   = str(d.get("episode_id") or "").strip()

        # Normalise suggested_links
        raw_links = d.get("suggested_links")
        if not isinstance(raw_links, list):
            raw_links = []
        links = []
        for lnk in raw_links:
            if not isinstance(lnk, dict):
                continue
            try:
                links.append({
                    "target_id": int(lnk["target_id"]),
                    "rel_type":  str(lnk.get("rel_type", "related_to")),
                    "weight":    _clamp(lnk.get("weight"), 0.0, 1.0, 0.5),
                })
            except (KeyError, TypeError, ValueError):
                pass
        d["suggested_links"] = links
        return d

    def _default_parsed(self) -> dict:
        return {
            "entity_type":     _FALLBACK_ENTITY_TYPE,
            "importance":      0.7,
            "confidence":      0.3,    # very low — signals extraction failed
            "ttl":             0,
            "namespace":       self._ns,
            "episode_id":      "",
            "suggested_links": [],
        }

    # ── Internal: DB write ────────────────────────────────────────────────────

    def _apply(
        self,
        text:   str,
        vec:    np.ndarray,
        parsed: dict,
        hint:   Optional[dict],
    ) -> int:
        """Construct Metadata, call db.add(), return node_id."""
        node_id = self._make_id(text)

        meta = self._fdb.Metadata()
        meta.timestamp    = int(time.time())
        meta.importance   = parsed["importance"]
        meta.confidence   = parsed["confidence"]
        meta.type         = self._fdb.ContextType.FACT
        meta.source       = (hint or {}).get("source", self._source)
        meta.content      = text
        meta.namespace_id = parsed["namespace"]
        meta.entity_id    = parsed["entity_type"]
        if parsed["ttl"] > 0:
            meta.ttl = parsed["ttl"]

        meta.set_attribute("entity_type", parsed["entity_type"])
        meta.set_attribute("classified_by",
                           getattr(self._provider, "name",
                                   lambda: "heuristic")() if self._provider else "heuristic")

        self._db.add(id=node_id, vec=vec, meta=meta, modality=self._mod)
        return node_id

    def _sample_context(self, vec: np.ndarray) -> list[dict]:
        """
        Return up to sample_k nodes similar to vec — gives the LLM graph context
        for making link suggestions and understanding existing entity types.
        """
        try:
            results = self._db.search(vec, k=self._sample_k, modality=self._mod)
            nodes = []
            for r in results:
                m = r.metadata
                nodes.append({
                    "id":          r.id,
                    "score":       round(r.score, 3),
                    "entity_type": m.get_attribute("entity_type") or m.entity_id or "fact",
                    "importance":  round(m.importance, 2),
                    "content":     m.content,
                })
            return nodes
        except Exception:
            return []

    @staticmethod
    def _make_id(text: str) -> int:
        """
        Collision-resistant node ID from content + timestamp + PID.
        Mirrors the pattern used in EpisodeManager._episode_header_id.
        """
        src = f"{text[:200]}:{time.time()}:{os.getpid()}"
        return int.from_bytes(
            hashlib.sha256(src.encode()).digest()[:7], "little"
        ) % (2 ** 50)
