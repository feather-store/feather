"""
Feather DB — Interactive Demo
HuggingFace Space · feather-db v0.6.0

Demonstrates:
  - Semantic search over a pre-loaded knowledge graph
  - Context chain (vector search + graph BFS expansion)
  - Graph health report
  - feather_why — retrieval score breakdown
  - Add new intel nodes live
"""

import hashlib
import json
import time

import gradio as gr
import numpy as np

try:
    import feather_db
    _FEATHER_OK = True
except ImportError:
    _FEATHER_OK = False


# ── Offline embedder (no API key needed) ──────────────────────────────────────
def _embed(text: str, dim: int = 768) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().replace(",", " ").replace(".", " ").split()
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for j in range(8):
            vec[(h >> (j * 5)) % dim] += 1.0 / (j + 1)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


# ── Seed knowledge graph — AI developer tools / product intelligence ──────────
#
# Domain: a team building an AI-powered developer tool (editor, CLI, SDK, cloud)
# tracks feature performance, competitor moves, community signals, and strategy.
# All data points are realistic and meaningful for this domain.
#
SEED_NODES = [
    (
        1,
        "AI autocomplete in the editor: 68% daily active usage, avg 12 completions accepted per session. "
        "Highest adoption of any feature shipped this quarter. Strongest signal in power-user cohort.",
        "feature_performance", "Editor", 0.92,
    ),
    (
        2,
        "Competitor launched inline AI debugging with natural-language error explanations. "
        "3,400 GitHub stars in 48 hours. Announcement dominated dev Twitter for two days. "
        "Directly targets our core editor user base.",
        "competitor_intel", "Editor", 0.95,
    ),
    (
        3,
        "StackOverflow Developer Survey 2026: 71% of developers now use AI coding assistants daily, "
        "up from 44% last year. Willingness to pay for productivity tools at an all-time high. "
        "Enterprise segment growing fastest.",
        "market_signal", "SDK", 0.90,
    ),
    (
        4,
        "CLI onboarding funnel: 34% of new users drop off at step 3 (API key setup). "
        "Median time-to-first-output is 4.2 minutes — well above our 90-second target. "
        "Friction is authentication, not comprehension.",
        "user_feedback", "CLI", 0.87,
    ),
    (
        5,
        "SDK v2 launched with streaming and tool-use support. Download velocity 2.1x faster than SDK v1 "
        "in the first week. Community PRs opened within 6 hours of release. "
        "Streaming is the most-requested missing feature now resolved.",
        "feature_performance", "SDK", 0.89,
    ),
    (
        6,
        "Strategy brief: reduce time-to-first-value under 90 seconds for all entry points. "
        "Frictionless auth (OAuth + token auto-detect) identified as the highest-leverage lever. "
        "Target: onboarding completion rate from 66% to 85% in Q2.",
        "strategy_brief", "CLI", 0.93,
    ),
    (
        7,
        "Community Discord: offline / air-gapped mode has 47 upvotes and is the top feature request. "
        "Users cite enterprise security policy and data-residency requirements. "
        "Three Fortune 500 pilots blocked specifically by this gap.",
        "community_signal", "Cloud", 0.88,
    ),
    (
        8,
        "VS Code extension outperforms JetBrains plugin 3.1x in weekly active users and 4.8x in session length. "
        "Recommend 70/30 investment split. JetBrains users skew toward Java/Kotlin — "
        "worth a targeted language-server improvement sprint.",
        "channel_insight", "Editor", 0.86,
    ),
    (
        9,
        "Retention analysis: power users (5+ sessions/week) show 8.4x 90-day retention vs casual users. "
        "Habit formation — not feature breadth — is the primary retention driver. "
        "Users who complete 3 sessions in week 1 have 72% chance of being active at day 90.",
        "user_feedback", "SDK", 0.91,
    ),
    (
        10,
        "Open-source alternative launched under MIT license: 12k GitHub stars in first month. "
        "No cloud sync, no team features, local-only. Actively targeting our free-tier users "
        "with 'no vendor lock-in' messaging. Poses risk to top-of-funnel acquisition.",
        "competitor_intel", "Cloud", 0.93,
    ),
]

SEED_EDGES = [
    (2,  1,  "contradicts",   0.90),   # competitor launch threatens editor feature lead
    (3,  5,  "supports",      0.85),   # market survey supports SDK investment
    (4,  6,  "references",    0.92),   # onboarding drop-off directly informs strategy brief
    (6,  4,  "derived_from",  0.88),   # strategy brief derived from CLI feedback
    (8,  1,  "supports",      0.78),   # VS Code dominance supports editor focus
    (9,  6,  "supports",      0.87),   # retention data supports onboarding strategy
    (10, 7,  "supports",      0.80),   # OSS competitor validates offline mode demand
    (3,  1,  "supports",      0.75),   # rising AI adoption supports editor feature investment
]

DIM      = 768
_DB_PATH = "/tmp/feather_demo.feather"
_db      = None


def _get_db():
    global _db
    if _db is not None:
        return _db
    if not _FEATHER_OK:
        return None

    db = feather_db.DB.open(_DB_PATH, dim=DIM)
    t0 = int(time.time()) - 7 * 86400  # seed nodes spread across last 7 days

    for nid, content, etype, product, imp in SEED_NODES:
        vec  = _embed(content, DIM)
        meta = feather_db.Metadata()
        meta.timestamp    = t0 + nid * 14400   # 4-hour intervals
        meta.importance   = imp
        meta.confidence   = 0.9
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "demo_seed"
        meta.content      = content
        meta.namespace_id = "devtools"
        meta.entity_id    = etype
        meta.set_attribute("entity_type", etype)
        meta.set_attribute("product",     product)
        db.add(id=nid, vec=vec, meta=meta)

    for src, tgt, rel, w in SEED_EDGES:
        db.link(src, tgt, rel, w)

    db.save()
    _db = db
    return db


# ── Tool implementations ───────────────────────────────────────────────────────

def do_search(query: str, k: int, product_filter: str) -> str:
    db = _get_db()
    if db is None:
        return "⚠️ feather_db not installed. Run: pip install feather-db"
    if not query.strip():
        return "Enter a query above."

    vec     = _embed(query, DIM)
    results = db.search(vec, k=k * 3)

    rows = []
    for r in results:
        m = r.metadata
        p = m.get_attribute("product")
        if product_filter and product_filter != "All" and p != product_filter:
            continue
        rows.append({
            "id":           r.id,
            "score":        round(r.score, 4),
            "entity_type":  m.get_attribute("entity_type"),
            "product":      p,
            "content":      m.content,
            "recall_count": m.recall_count,
            "importance":   round(m.importance, 3),
        })
        if len(rows) >= k:
            break

    if not rows:
        return "No results found."
    return json.dumps(rows, indent=2)


def do_context_chain(query: str, k: int, hops: int) -> str:
    db = _get_db()
    if db is None:
        return "⚠️ feather_db not installed."
    if not query.strip():
        return "Enter a query above."

    vec   = _embed(query, DIM)
    chain = db.context_chain(vec, k=k, hops=hops, modality="text")

    nodes = []
    for node in sorted(chain.nodes, key=lambda n: (n.hop, -n.score)):
        m = node.metadata
        nodes.append({
            "id":          node.id,
            "hop":         node.hop,
            "score":       round(node.score, 4),
            "entity_type": m.get_attribute("entity_type"),
            "product":     m.get_attribute("product"),
            "content":     m.content[:140] + ("…" if len(m.content) > 140 else ""),
        })

    edges = [
        {"source": e.source, "target": e.target,
         "rel_type": e.rel_type, "weight": round(e.weight, 3)}
        for e in chain.edges
    ]

    return json.dumps({
        "summary": f"{len(nodes)} nodes reached across {hops} graph hop(s)",
        "nodes":   nodes,
        "edges":   edges,
    }, indent=2)


def do_why(node_id: int, query: str) -> str:
    db = _get_db()
    if db is None:
        return "⚠️ feather_db not installed."
    if not query.strip():
        return "Enter a query above."

    from feather_db.memory import MemoryManager
    vec    = _embed(query, DIM)
    result = MemoryManager.why_retrieved(db, node_id=int(node_id), query_vec=vec)
    return json.dumps(result, indent=2)


def do_health() -> str:
    db = _get_db()
    if db is None:
        return "⚠️ feather_db not installed."

    from feather_db.memory import MemoryManager
    report = MemoryManager.health_report(db, modality="text")
    return json.dumps(report, indent=2)


def do_add(content: str, entity_type: str, product: str, importance: float) -> str:
    db = _get_db()
    if db is None:
        return "⚠️ feather_db not installed."
    if not content.strip():
        return "Content cannot be empty."

    nid  = int(time.time() * 1000) % (2 ** 32)
    vec  = _embed(content, DIM)
    meta = feather_db.Metadata()
    meta.timestamp    = int(time.time())
    meta.importance   = float(importance)
    meta.confidence   = 0.85
    meta.type         = feather_db.ContextType.EVENT
    meta.source       = "gradio_user"
    meta.content      = content
    meta.namespace_id = "devtools"
    meta.entity_id    = entity_type
    meta.set_attribute("entity_type", entity_type)
    meta.set_attribute("product",     product)
    db.add(id=nid, vec=vec, meta=meta)
    db.save()
    return json.dumps({
        "status":      "added",
        "id":          nid,
        "entity_type": entity_type,
        "product":     product,
        "tip":         "Node is now live — try searching for it in the Search tab.",
    })


# ── Preload on startup ────────────────────────────────────────────────────────
_get_db()

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Feather DB — Living Context Engine",
    theme=gr.themes.Soft(),
    css=".tool-output { font-family: monospace; font-size: 0.84rem; }",
) as demo:

    gr.HTML("""
    <div style="border-left:4px solid #6366f1;padding-left:1rem;margin-bottom:1rem">
      <h1 style="margin:0">🪶 Feather DB — Living Context Engine</h1>
      <p style="margin:0.4rem 0 0 0">
        Embedded vector DB · HNSW search · typed context graph · adaptive decay · MCP server<br/>
        <a href="https://pypi.org/project/feather-db/" target="_blank">PyPI</a> ·
        <a href="https://github.com/feather-store/feather" target="_blank">GitHub</a> ·
        <code>pip install feather-db</code>
      </p>
    </div>
    """)

    gr.Markdown("""
**Demo graph:** 10 nodes representing product intelligence for an AI developer tools team —
feature performance, competitor moves, community signals, strategy briefs, and user research.
8 typed causal edges connect them (`contradicts`, `supports`, `derived_from`, `references`).
""")

    with gr.Tabs():

        # ── Search ────────────────────────────────────────────────────────────
        with gr.TabItem("🔍 Semantic Search"):
            gr.Markdown("Find nodes by **meaning**, not keywords. Filtered by product or entity type.")
            with gr.Row():
                with gr.Column(scale=2):
                    s_query   = gr.Textbox(label="Query",
                                           placeholder="Why is user onboarding failing?")
                    s_k       = gr.Slider(1, 10, value=5, step=1, label="Top-k")
                    s_product = gr.Dropdown(["All","Editor","CLI","SDK","Cloud"],
                                            value="All", label="Product filter")
                    s_btn     = gr.Button("Search", variant="primary")
                with gr.Column(scale=3):
                    s_out = gr.Code(label="Results", language="json",
                                    elem_classes=["tool-output"])

            gr.Examples(
                examples=[
                    ["Why is user onboarding failing?",         5, "All"],
                    ["What competitor moves should we watch?",  5, "All"],
                    ["Which features drive retention?",         5, "SDK"],
                    ["What does the community want most?",      5, "Cloud"],
                    ["Where should we invest in the editor?",   5, "Editor"],
                ],
                inputs=[s_query, s_k, s_product],
            )
            s_btn.click(do_search, [s_query, s_k, s_product], s_out)

        # ── Context Chain ─────────────────────────────────────────────────────
        with gr.TabItem("🕸️ Context Chain"):
            gr.Markdown(
                "**Two-phase retrieval** — vector search finds seed nodes (hop 0), "
                "then BFS expands outward over typed graph edges.\n\n"
                "Use this to trace root causes: *start from a symptom, surface the events that explain it.*"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    c_query = gr.Textbox(label="Seed query",
                                         placeholder="CLI adoption is slow")
                    c_k     = gr.Slider(1, 5, value=3, step=1, label="Seed nodes (k)")
                    c_hops  = gr.Slider(1, 3, value=2, step=1, label="Graph hops")
                    c_btn   = gr.Button("Run Context Chain", variant="primary")
                with gr.Column(scale=3):
                    c_out = gr.Code(label="Chain result", language="json",
                                    elem_classes=["tool-output"])

            gr.Examples(
                examples=[
                    ["CLI adoption is slow",                    3, 2],
                    ["Why is the competitor threat serious?",   3, 2],
                    ["What drives long-term user retention?",   3, 2],
                    ["Why do enterprise deals stall?",          3, 1],
                ],
                inputs=[c_query, c_k, c_hops],
            )
            c_btn.click(do_context_chain, [c_query, c_k, c_hops], c_out)

        # ── Why Retrieved ─────────────────────────────────────────────────────
        with gr.TabItem("🔬 Why Retrieved?"):
            gr.Markdown(
                "Score breakdown for any node — **similarity**, **stickiness** (recall bonus), "
                "**recency** (adaptive decay), **importance**, **confidence**, and the full formula.\n\n"
                "Use to understand and debug retrieval decisions."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    w_id    = gr.Number(label="Node ID (1–10)", value=4, precision=0)
                    w_query = gr.Textbox(label="Query",
                                         placeholder="onboarding drop-off")
                    w_btn   = gr.Button("Explain", variant="primary")
                with gr.Column(scale=3):
                    w_out = gr.Code(label="Score breakdown", language="json",
                                    elem_classes=["tool-output"])

            gr.Examples(
                examples=[
                    [4,  "onboarding drop-off time to value"],
                    [2,  "competitor launch editor feature"],
                    [9,  "retention power users habit"],
                    [7,  "offline mode enterprise security"],
                    [6,  "strategy brief Q2 auth friction"],
                ],
                inputs=[w_id, w_query],
            )
            w_btn.click(do_why, [w_id, w_query], w_out)

        # ── Health ────────────────────────────────────────────────────────────
        with gr.TabItem("🩺 Graph Health"):
            gr.Markdown(
                "Snapshot of the knowledge graph: **hot / warm / cold** tier distribution, "
                "orphan nodes, expired TTL count, recall histogram, avg importance and confidence."
            )
            h_btn = gr.Button("Run Health Check", variant="primary")
            h_out = gr.Code(label="Health report", language="json",
                            elem_classes=["tool-output"])
            h_btn.click(do_health, [], h_out)

        # ── Add Intel ─────────────────────────────────────────────────────────
        with gr.TabItem("➕ Add Intel"):
            gr.Markdown(
                "Ingest a new intelligence node into the live graph. "
                "It becomes **immediately searchable** — try adding something then switching to Search."
            )
            with gr.Row():
                with gr.Column():
                    a_content = gr.Textbox(
                        label="Content", lines=3,
                        placeholder="Competitor Y just open-sourced their SDK. "
                                    "10k stars overnight. Targets our developer acquisition funnel.",
                    )
                    a_etype = gr.Dropdown(
                        ["competitor_intel", "feature_performance", "user_feedback",
                         "strategy_brief", "market_signal", "community_signal", "channel_insight"],
                        value="competitor_intel", label="Entity Type",
                    )
                    a_product    = gr.Dropdown(["Editor","CLI","SDK","Cloud"],
                                               value="SDK", label="Product")
                    a_importance = gr.Slider(0.0, 1.0, value=0.85, step=0.05,
                                             label="Importance")
                    a_btn = gr.Button("Add to Graph", variant="primary")
                with gr.Column():
                    a_out = gr.Code(label="Result", language="json",
                                    elem_classes=["tool-output"])

            a_btn.click(do_add, [a_content, a_etype, a_product, a_importance], a_out)

    gr.Markdown("""
---
**Connect Feather DB to any LLM in 5 lines:**
```python
pip install feather-db

from feather_db.integrations import ClaudeConnector
conn   = ClaudeConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
result = conn.run_loop(client,
    messages=[{"role": "user", "content": "Why is onboarding drop-off so high?"}],
    model="claude-opus-4-6")
```
Works with **Claude · OpenAI · Gemini · Groq · Mistral · Ollama · MCP (Claude Desktop, Cursor)**

[PyPI](https://pypi.org/project/feather-db/) · [GitHub](https://github.com/feather-store/feather) · [Integrations Guide](https://github.com/feather-store/feather/blob/main/docs/integrations.md) · [Hawky.ai](https://hawky.ai)
""")

if __name__ == "__main__":
    demo.launch()
