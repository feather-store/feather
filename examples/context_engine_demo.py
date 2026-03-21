"""
Feather DB — Self-Aligned Context Engine Demo (Phase 1)
========================================================
Shows engine.ingest() with four backends:
  - Claude   (ANTHROPIC_API_KEY)
  - OpenAI   (OPENAI_API_KEY)
  - Gemini   (GOOGLE_API_KEY)
  - Ollama   (local, no key — run: ollama pull llama3.1:8b)

Without any API key: falls back to the built-in heuristic classifier.
All backends produce identical output schema — the engine is provider-agnostic.

Run:
  python3 examples/context_engine_demo.py
  ANTHROPIC_API_KEY=sk-ant-... python3 examples/context_engine_demo.py
  OLLAMA_MODEL=mistral:7b     python3 examples/context_engine_demo.py
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import feather_db
from feather_db.engine    import ContextEngine
from feather_db.providers import (
    ClaudeProvider, OpenAIProvider, GeminiProvider, OllamaProvider,
)


# ── Simple offline embedder (no API key needed) ───────────────────────────────
import hashlib

def embed(text: str, dim: int = 768) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for j in range(8):
            vec[(h >> (j * 5)) % dim] += 1.0 / (j + 1)
    n = np.linalg.norm(vec)
    return (vec / n) if n > 0 else vec


# ── Pick a provider based on available env vars ───────────────────────────────
def pick_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  Using Claude (claude-haiku-4-5)")
        return ClaudeProvider(model="claude-haiku-4-5-20251001")

    if os.environ.get("OPENAI_API_KEY"):
        print("  Using OpenAI (gpt-4o-mini)")
        return OpenAIProvider(model="gpt-4o-mini")

    if os.environ.get("GOOGLE_API_KEY"):
        print("  Using Gemini (gemini-2.0-flash)")
        return GeminiProvider(model="gemini-2.0-flash")

    # Try Ollama
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
            print(f"  Using Ollama ({model})")
            return OllamaProvider(model=model)
    except Exception:
        pass

    print("  No API key found — using built-in heuristic classifier (offline mode)")
    return None   # ContextEngine falls back to heuristic


# ── Texts to ingest ───────────────────────────────────────────────────────────
TEXTS = [
    "Competitor Y just launched a developer SDK with native streaming support and MIT license. "
    "10k GitHub stars in 24 hours. Directly targets our core open-source audience.",

    "User always prefers responses in bullet points rather than long paragraphs. "
    "Keep answers concise and scannable.",

    "VS Code extension weekly active users: 42,000. Up 18% month-over-month. "
    "Primary driver is autocomplete adoption in the 25-35 developer cohort.",

    "Strategy: prioritise time-to-first-value under 90 seconds for all onboarding flows. "
    "Frictionless OAuth is the single highest-leverage improvement.",

    "Community Discord request: offline mode (no internet required). "
    "47 upvotes. Three enterprise pilots are blocked specifically by this gap. "
    "Security and data-residency requirements.",

    "The retention analysis shows power users (5+ sessions/week) have 8.4x 90-day "
    "retention vs casual users. Habit formation in week 1 is the key lever.",
]


def main():
    db_path  = tempfile.mktemp(suffix=".feather")
    provider = pick_provider()

    print(f"\n{'='*60}")
    print("Feather DB — Self-Aligned Context Engine (Phase 1)")
    print(f"{'='*60}\n")

    engine = ContextEngine(
        db_path   = db_path,
        dim       = 768,
        provider  = provider,
        embedder  = embed,
        namespace = "devtools",
    )

    print(f"Ingesting {len(TEXTS)} nodes...\n")

    ids = []
    for i, text in enumerate(TEXTS, 1):
        nid = engine.ingest(text)
        ids.append(nid)
        m = engine.db.get_metadata(nid)
        print(f"  [{i}] id={nid}")
        print(f"       entity_type  = {m.get_attribute('entity_type')}")
        print(f"       importance   = {m.importance:.2f}")
        print(f"       confidence   = {m.confidence:.2f}")
        print(f"       classified_by= {m.get_attribute('classified_by')}")
        print()

    print(f"{'─'*60}")
    print("Semantic search: 'what competitor moves should I watch?'\n")

    q   = embed("what competitor moves should I watch?")
    res = engine.db.search(q, k=3)
    for r in res:
        m = r.metadata
        print(f"  score={r.score:.4f}  [{m.get_attribute('entity_type')}]")
        print(f"  {m.content[:120]}...")
        print()

    print(f"{'─'*60}")
    print("Context chain from 'slow onboarding drop-off' (2 hops)\n")

    q2    = embed("slow onboarding drop-off user friction")
    chain = engine.db.context_chain(q2, k=3, hops=2, modality="text")
    for node in sorted(chain.nodes, key=lambda n: (n.hop, -n.score)):
        prefix = "  " * node.hop + "└─ " if node.hop > 0 else "   "
        m = node.metadata
        print(f"  hop={node.hop} {prefix}[{m.get_attribute('entity_type')}] {m.content[:90]}…")

    if chain.edges:
        print(f"\n  Graph edges traversed:")
        for e in chain.edges:
            print(f"    {e.source} ──{e.rel_type}──▶ {e.target}  (w={e.weight:.2f})")

    print(f"\n{'='*60}")
    print(f"Done. DB: {db_path}")
    print(f"Nodes stored: {len(engine.db.get_all_ids('text'))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
