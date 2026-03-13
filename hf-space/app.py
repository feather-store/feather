"""
Feather DB — Interactive Demo
HuggingFace Space · feather-db v0.6.0

Demonstrates:
  - Semantic search over a pre-loaded knowledge graph
  - Context chain (vector search + graph BFS expansion)
  - Graph health report
  - feather_why — retrieval score breakdown
  - D3 graph visualization
"""

import hashlib
import json
import time

import gradio as gr
import numpy as np

# ── Try to import feather_db (built from source or pip) ───────────────────────
try:
    import feather_db
    _FEATHER_OK = True
except ImportError:
    _FEATHER_OK = False

# ── Deterministic offline embedder (no API key needed) ────────────────────────
def _embed(text: str, dim: int = 768) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().replace(",", " ").replace(".", " ").split()
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for j in range(8):
            vec[(h >> (j * 5)) % dim] += 1.0 / (j + 1)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


# ── Seed knowledge graph ──────────────────────────────────────────────────────
SEED_NODES = [
    (1,  "FD video ad — senior couple hook. CTR 3.2%, ROAS 4.1x. Best format this quarter.",
         "ad_performance", "FD",  0.92),
    (2,  "Competitor bank launched 8.75% APY fixed deposit on Budget Day Feb 1. Directly undercuts our 8.5% rate.",
         "competitor_intel", "FD", 0.95),
    (3,  "Budget Day 2026: RBI held repo rate at 6.5%. FD search queries up 220%. High intent window.",
         "market_signal",  "FD",  0.90),
    (4,  "FD static banner creative showing fatigue — frequency 8.2, CTR dropped 12% WoW. Recommend creative rotation.",
         "creative_insight","FD",  0.85),
    (5,  "CC cashback reel — unlimited 5% cashback hook. CTR 3.8%. Best performing CC format this quarter.",
         "ad_performance", "CC",  0.88),
    (6,  "Strategy: lead with rate in first 2 seconds for FD video. 35-55 segment shows 3x ROAS vs general audience.",
         "strategy_intel", "FD",  0.93),
    (7,  "Mutual Fund SIP campaign — market volatility angle. CTR 2.1%, down from 2.8% last month.",
         "ad_performance", "MF",  0.80),
    (8,  "Instagram Reels outperforming Stories 2.4x for FD creatives. Recommend 70/30 budget split.",
         "channel_insight", "FD", 0.87),
    (9,  "Senior demographic (55+) shows highest FD conversion rate at 6.8%. Safety and guaranteed return messaging resonates.",
         "audience_insight","FD",  0.89),
    (10, "Bond fund competitor launched zero-commission offer. Targeting our MF audience on Google Search.",
         "competitor_intel","MF",  0.91),
]

SEED_EDGES = [
    (2, 1, "contradicts",  0.9),
    (3, 1, "supports",     0.85),
    (4, 6, "references",   0.9),
    (6, 1, "supports",     0.8),
    (8, 1, "supports",     0.75),
    (9, 6, "supports",     0.88),
    (10, 7, "contradicts", 0.85),
]

DIM = 768
_DB_PATH = "/tmp/feather_demo.feather"
_db = None


def _get_db():
    global _db
    if _db is not None:
        return _db
    if not _FEATHER_OK:
        return None

    db = feather_db.DB.open(_DB_PATH, dim=DIM)
    t0 = int(time.time()) - 86400

    for nid, content, etype, product, imp in SEED_NODES:
        vec = _embed(content, DIM)
        meta = feather_db.Metadata()
        meta.timestamp    = t0 + nid * 3600
        meta.importance   = imp
        meta.confidence   = 0.9
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "demo_seed"
        meta.content      = content
        meta.namespace_id = "demo"
        meta.entity_id    = etype
        meta.set_attribute("entity_type", etype)
        meta.set_attribute("product",     product)
        db.add(id=nid, vec=vec, meta=meta)

    for src, tgt, rel, w in SEED_EDGES:
        db.link(src, tgt, rel, w)

    db.save()
    _db = db
    return db


# ── Tool: semantic search ──────────────────────────────────────────────────────
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
            "ID":          r.id,
            "Score":       round(r.score, 4),
            "Entity Type": m.get_attribute("entity_type"),
            "Product":     p,
            "Content":     m.content,
            "Recall Count": m.recall_count,
            "Importance":  round(m.importance, 3),
        })
        if len(rows) >= k:
            break

    if not rows:
        return "No results found."
    return json.dumps(rows, indent=2)


# ── Tool: context chain ────────────────────────────────────────────────────────
def do_context_chain(query: str, k: int, hops: int) -> str:
    db = _get_db()
    if db is None:
        return "⚠️ feather_db not installed."
    if not query.strip():
        return "Enter a query above."

    vec    = _embed(query, DIM)
    chain  = db.context_chain(vec, k=k, hops=hops, modality="text")

    nodes = []
    for node in sorted(chain.nodes, key=lambda n: (n.hop, -n.score)):
        m = node.metadata
        nodes.append({
            "ID":          node.id,
            "Hop":         node.hop,
            "Score":       round(node.score, 4),
            "Entity Type": m.get_attribute("entity_type"),
            "Product":     m.get_attribute("product"),
            "Content":     m.content[:120] + ("…" if len(m.content) > 120 else ""),
        })

    edges = [
        {"Source": e.source, "Target": e.target,
         "Rel Type": e.rel_type, "Weight": round(e.weight, 3)}
        for e in chain.edges
    ]

    return json.dumps({
        "nodes": nodes,
        "edges": edges,
        "summary": f"{len(nodes)} nodes reached across {hops} graph hop(s)",
    }, indent=2)


# ── Tool: feather_why ─────────────────────────────────────────────────────────
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


# ── Tool: health report ────────────────────────────────────────────────────────
def do_health() -> str:
    db = _get_db()
    if db is None:
        return "⚠️ feather_db not installed."

    from feather_db.memory import MemoryManager
    report = MemoryManager.health_report(db, modality="text")
    return json.dumps(report, indent=2)


# ── Tool: add a node ──────────────────────────────────────────────────────────
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
    meta.namespace_id = "demo"
    meta.entity_id    = entity_type
    meta.set_attribute("entity_type", entity_type)
    meta.set_attribute("product",     product)
    db.add(id=nid, vec=vec, meta=meta)
    db.save()
    return json.dumps({"status": "added", "id": nid, "entity_type": entity_type, "product": product})


# ── Preload DB on startup ─────────────────────────────────────────────────────
_get_db()


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Feather DB — Living Context Engine Demo",
    theme=gr.themes.Soft(),
    css="""
    .tool-output { font-family: monospace; font-size: 0.85rem; }
    .header-block { border-left: 4px solid #6366f1; padding-left: 1rem; }
    """,
) as demo:

    gr.HTML("""
    <div class="header-block">
      <h1>🪶 Feather DB — Living Context Engine</h1>
      <p>Embedded vector database with HNSW search, typed context graph, and adaptive decay.
         <a href="https://pypi.org/project/feather-db/" target="_blank">PyPI</a> ·
         <a href="https://github.com/feather-store/feather" target="_blank">GitHub</a> ·
         <code>pip install feather-db</code>
      </p>
    </div>
    """)

    gr.Markdown("""
    **10 seed nodes** are pre-loaded: ad performance, competitor intel, market signals, strategy briefs — across FD, CC, and MF products.
    Try the tabs below to explore the living context engine.
    """)

    with gr.Tabs():

        # ── Tab 1: Semantic Search ────────────────────────────────────────────
        with gr.TabItem("🔍 Semantic Search"):
            gr.Markdown("Vector similarity search. Finds nodes by meaning, not keywords.")
            with gr.Row():
                with gr.Column(scale=2):
                    search_query   = gr.Textbox(label="Query", placeholder="Why is our FD CTR dropping?")
                    search_k       = gr.Slider(1, 10, value=5, step=1, label="Top-k results")
                    search_product = gr.Dropdown(["All","FD","CC","MF"], value="All", label="Product filter")
                    search_btn     = gr.Button("Search", variant="primary")
                with gr.Column(scale=3):
                    search_out = gr.Code(label="Results (JSON)", language="json", elem_classes=["tool-output"])

            gr.Examples(
                examples=[
                    ["Why is our FD CTR dropping?",   5, "FD"],
                    ["Competitor threats this month",  5, "All"],
                    ["Best performing ad creative",    5, "All"],
                    ["What audience converts best?",   5, "FD"],
                ],
                inputs=[search_query, search_k, search_product],
            )
            search_btn.click(do_search, [search_query, search_k, search_product], search_out)

        # ── Tab 2: Context Chain ──────────────────────────────────────────────
        with gr.TabItem("🕸️ Context Chain"):
            gr.Markdown(
                "Two-phase retrieval: find seed nodes by vector similarity (hop=0), "
                "then expand outward over typed graph edges (BFS). "
                "Great for tracing **root causes** — e.g. start from a CTR drop and surface the competitor event that caused it."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    chain_query = gr.Textbox(label="Seed query", placeholder="FD CTR underperforming")
                    chain_k     = gr.Slider(1, 5, value=3, step=1, label="Seed nodes (k)")
                    chain_hops  = gr.Slider(1, 3, value=2, step=1, label="Graph hops")
                    chain_btn   = gr.Button("Run Context Chain", variant="primary")
                with gr.Column(scale=3):
                    chain_out = gr.Code(label="Chain result (JSON)", language="json", elem_classes=["tool-output"])

            gr.Examples(
                examples=[
                    ["FD CTR underperforming",     3, 2],
                    ["Budget Day competitor moves", 3, 2],
                    ["Which strategy drives ROAS?", 3, 1],
                ],
                inputs=[chain_query, chain_k, chain_hops],
            )
            chain_btn.click(do_context_chain, [chain_query, chain_k, chain_hops], chain_out)

        # ── Tab 3: Why Retrieved ──────────────────────────────────────────────
        with gr.TabItem("🔬 Why Retrieved?"):
            gr.Markdown(
                "Score breakdown for a specific node — shows exactly why it would (or wouldn't) "
                "surface for your query: **similarity**, **stickiness** (recall bonus), **recency**, **importance**, **confidence**, and the final formula."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    why_id    = gr.Number(label="Node ID (1–10)", value=4, precision=0)
                    why_query = gr.Textbox(label="Query", placeholder="FD creative performance")
                    why_btn   = gr.Button("Explain", variant="primary")
                with gr.Column(scale=3):
                    why_out = gr.Code(label="Score breakdown (JSON)", language="json", elem_classes=["tool-output"])

            gr.Examples(
                examples=[
                    [4,  "creative fatigue CTR drop"],
                    [2,  "competitor FD rate"],
                    [6,  "FD video strategy ROAS"],
                    [9,  "best audience segment"],
                ],
                inputs=[why_id, why_query],
            )
            why_btn.click(do_why, [why_id, why_query], why_out)

        # ── Tab 4: Health Report ──────────────────────────────────────────────
        with gr.TabItem("🩺 Graph Health"):
            gr.Markdown(
                "Snapshot of the knowledge graph: tier distribution (hot/warm/cold), "
                "orphan nodes (no edges), expired TTL count, recall histogram, avg importance/confidence."
            )
            health_btn = gr.Button("Run Health Check", variant="primary")
            health_out = gr.Code(label="Health report (JSON)", language="json", elem_classes=["tool-output"])
            health_btn.click(do_health, [], health_out)

        # ── Tab 5: Add Intel ──────────────────────────────────────────────────
        with gr.TabItem("➕ Add Intel"):
            gr.Markdown("Ingest a new intelligence node into the live graph. It becomes immediately searchable.")
            with gr.Row():
                with gr.Column():
                    add_content     = gr.Textbox(label="Content", lines=3,
                                                  placeholder="Competitor X launched a 9% FD rate targeting senior segment.")
                    add_entity_type = gr.Dropdown(
                        ["competitor_intel","ad_performance","market_signal",
                         "strategy_intel","creative_insight","channel_insight","audience_insight"],
                        value="competitor_intel", label="Entity Type")
                    add_product     = gr.Dropdown(["FD","CC","MF","Bond"], value="FD", label="Product")
                    add_importance  = gr.Slider(0.0, 1.0, value=0.85, step=0.05, label="Importance")
                    add_btn         = gr.Button("Add to Graph", variant="primary")
                with gr.Column():
                    add_out = gr.Code(label="Result (JSON)", language="json", elem_classes=["tool-output"])

            add_btn.click(do_add, [add_content, add_entity_type, add_product, add_importance], add_out)

    gr.Markdown("""
    ---
    **Feather DB v0.6.0** — embedded vector database + living context engine · MIT License

    ```python
    pip install feather-db

    from feather_db.integrations import ClaudeConnector
    conn = ClaudeConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
    result = conn.run_loop(client, messages=[{"role":"user","content":"Why is CTR dropping?"}])
    ```
    """)


if __name__ == "__main__":
    demo.launch()
