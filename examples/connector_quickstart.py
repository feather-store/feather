"""
Feather DB Connector — Quickstart Test
========================================
Shows exactly how to connect Feather DB to Claude and Gemini.

Run with real API keys:
    ANTHROPIC_API_KEY=sk-ant-... python3 examples/connector_quickstart.py
    GOOGLE_API_KEY=AIza...      python3 examples/connector_quickstart.py

Without keys: falls back to mock mode automatically.
"""

import os, sys, json, time, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feather_db
from feather_db.integrations import ClaudeConnector, GeminiConnector, GeminiEmbedder


# ── Simple deterministic embedder (works offline, no API key) ──────────────
def local_embed(text: str, dim: int = 3072) -> np.ndarray:
    import hashlib
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for j in range(4):
            vec[(h >> (j * 8)) % dim] += 1.0 / (j + 1)
    n = np.linalg.norm(vec)
    return (vec / n) if n > 0 else vec


# ── Build a small knowledge graph ─────────────────────────────────────────
def build_db(path: str, dim: int) -> None:
    db = feather_db.DB.open(path, dim=dim)

    nodes = [
        (1, "Our FD product offers 8.5% interest rate. Video creative with senior couple hook has CTR 3.2% and ROAS 4.1x.",
         "ad_performance", "FD"),
        (2, "Competitor bank launched 8.75% FD rate on Budget Day Feb 1. Directly undercuts our rate. Heavy Instagram spend.",
         "competitor_intel", "FD"),
        (3, "Budget Day 2026: RBI held repo rate at 6.5%. Fixed deposits now more attractive vs equity. FD search queries +220%.",
         "market_signal", "FD"),
        (4, "FD static banner creative showing fatigue — frequency 8.2, CTR dropped 12% week-over-week. Recommend rotation.",
         "creative_insight", "FD"),
        (5, "CC cashback reel — unlimited 5% cashback hook. CTR 3.8%. Best performing CC format this quarter.",
         "ad_performance", "CC"),
        (6, "Strategy: lead with rate in first 2 seconds for FD video. 35-55 age segment shows 3x ROAS vs general audience.",
         "strategy", "FD"),
    ]

    t0 = int(time.time()) - 3600
    for nid, content, etype, product in nodes:
        vec  = local_embed(content, dim)
        meta = feather_db.Metadata()
        meta.timestamp    = t0 + nid * 300
        meta.importance   = 0.85 + nid * 0.02
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "demo_seed"
        meta.content      = content
        meta.namespace_id = "demo"
        meta.entity_id    = etype
        meta.confidence   = 0.9
        meta.set_attribute("entity_type", etype)
        meta.set_attribute("product",     product)
        db.add(id=nid, vec=vec, meta=meta)

    # Add causal edges
    db.link(2, 1, "contradicts",  0.9)   # competitor rate contradicts our rate
    db.link(3, 1, "supports",     0.85)  # budget day supports FD demand
    db.link(4, 6, "references",   0.9)   # fatigue insight references strategy
    db.link(6, 1, "supports",     0.8)   # strategy supports video ad performance
    db.save()
    print(f"  DB ready: {db.size()} nodes, path={path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS — step by step
# ══════════════════════════════════════════════════════════════════════════════

def explain_architecture():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         HOW FEATHER DB CONNECTS TO AN LLM                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Your question                                                   ║
║       │                                                          ║
║       ▼                                                          ║
║  LLM (Claude / Gemini / GPT)                                     ║
║       │  decides it needs context → calls a tool                 ║
║       ▼                                                          ║
║  Feather DB Connector  ◄── you create this once                  ║
║   • translates the tool call to db.search() / context_chain()   ║
║   • returns JSON results back to the LLM                         ║
║       │                                                          ║
║       ▼                                                          ║
║  Feather DB  (.feather file on disk)                             ║
║   • 14 tools: search, context_chain, get_node, add_intel …      ║
║   • living context: recall counts, decay, typed graph edges      ║
║                                                                  ║
║  The LLM keeps calling tools until it has enough context,        ║
║  then writes its final answer. run_loop() handles all of this.   ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE connector demo
# ══════════════════════════════════════════════════════════════════════════════

def demo_claude(db_path: str, dim: int):
    print("─" * 64)
    print("  CLAUDE CONNECTOR")
    print("─" * 64)

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Step 1: Create the connector
    print("""
STEP 1 — Create the connector
  conn = ClaudeConnector(
      db_path  = "my.feather",   # your DB file
      dim      = 3072,           # vector dimension
      embedder = embed_fn,       # text → np.ndarray
  )
""")
    conn = ClaudeConnector(db_path=db_path, dim=dim, embedder=lambda t: local_embed(t, dim))

    # Step 2: Show what tools look like to Claude
    print("STEP 2 — Tools passed to Claude (input_schema format):")
    tools = conn.tools()
    for t in tools[:3]:  # show first 3
        print(f"  • {t['name']}: {t['description'][:70]}...")
    print(f"  … {len(tools)} tools total\n")

    # Step 3: Run the agent loop
    question = "Why is our FD CTR underperforming? What should we do?"
    print(f"STEP 3 — Ask Claude: \"{question}\"")
    print()

    if not api_key:
        print("  [No ANTHROPIC_API_KEY — showing mock tool calls]\n")
        # Manual demo of what the loop does
        print("  Claude would call tools in sequence:")
        r1 = conn.handle("feather_search", {"query": question, "k": 4})
        d1 = json.loads(r1)
        print(f"  1. feather_search → {d1['count']} results")
        for r in d1['results']:
            print(f"       [{r['entity_type']}] {r['content'][:70]}…")

        r2 = conn.handle("feather_context_chain", {"query": question, "k": 3, "hops": 2})
        d2 = json.loads(r2)
        print(f"\n  2. feather_context_chain → {d2['node_count']} nodes, {d2['edge_count']} edges")
        for n in d2['nodes'][:4]:
            hop = "  " * n['hop'] + ("└─ " if n['hop'] > 0 else "   ")
            print(f"       hop={n['hop']} {hop}[{n['entity_type']}] {n['content'][:65]}…")

        r3 = conn.handle("feather_why", {"id": 4, "query": question})
        d3 = json.loads(r3)
        print(f"\n  3. feather_why(id=4) →")
        print(f"       similarity={d3['similarity']}, stickiness={d3['stickiness']}")
        print(f"       formula: {d3['formula']}")

        print(f"\n  [Claude would synthesise these results into a final answer]\n")
        return

    # Real Claude call
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    print("  Calling claude-opus-4-6…\n")
    result = conn.run_loop(
        client,
        messages=[{"role": "user", "content": question}],
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "You are a performance marketing analyst. "
            "Use the Feather DB tools to retrieve relevant context before answering. "
            "Cite node IDs you found. Be concise."
        ),
        verbose=True,
    )
    print(f"\n  Claude's answer:\n")
    for line in result.strip().split("\n"):
        print(f"  {line}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI connector demo
# ══════════════════════════════════════════════════════════════════════════════

def demo_gemini(db_path: str, dim: int):
    print("─" * 64)
    print("  GEMINI CONNECTOR")
    print("─" * 64)

    api_key = os.environ.get("GOOGLE_API_KEY")

    print("""
STEP 1 — Create the connector
  # Option A: use Gemini's own embedder (3072-dim, real API)
  emb  = GeminiEmbedder(api_key="AIza...")
  conn = GeminiConnector(db_path="my.feather", dim=emb.dim, embedder=emb.embed_text)

  # Option B: use any embedder (mock, sentence-transformers, etc.)
  conn = GeminiConnector(db_path="my.feather", dim=3072, embedder=my_embed_fn)
""")

    if api_key:
        emb      = GeminiEmbedder(api_key=api_key)
        conn     = GeminiConnector(db_path=db_path, dim=emb.dim, embedder=emb.embed_text)
        used_dim = emb.dim
        print(f"  Using real Gemini Embedding 2 (dim={used_dim})\n")
    else:
        conn     = GeminiConnector(db_path=db_path, dim=dim, embedder=lambda t: local_embed(t, dim))
        used_dim = dim
        print("  [No GOOGLE_API_KEY — using local embedder]\n")

    print("STEP 2 — Tools formatted as Gemini FunctionDeclarations:")
    tools = conn.tools()
    # tools() returns a list with one types.Tool containing all FunctionDeclarations
    fn_decls = tools[0].function_declarations if hasattr(tools[0], 'function_declarations') else []
    for fd in fn_decls[:3]:
        print(f"  • {fd.name}: {fd.description[:70]}...")
    print(f"  … {len(fn_decls)} tools total\n")

    question = "Which competitor moves should I be most worried about for FD?"
    print(f"STEP 3 — Ask Gemini: \"{question}\"")
    print()

    if not api_key:
        print("  [No GOOGLE_API_KEY — showing mock tool calls]\n")
        r1 = conn.handle("feather_search", {"query": question, "k": 3, "entity": "competitor_intel"})
        d1 = json.loads(r1)
        print(f"  1. feather_search(entity='competitor_intel') → {d1['count']} results")
        for r in d1['results']:
            print(f"       [{r['entity_type']}] {r['content'][:70]}…")

        r2 = conn.handle("feather_mmr_search", {"query": question, "k": 4, "diversity": 0.6})
        d2 = json.loads(r2)
        print(f"\n  2. feather_mmr_search(diversity=0.6) → {d2['count']} diverse results")
        for r in d2['results']:
            print(f"       score={r['score']} [{r['entity_type']}] {r['content'][:65]}…")

        r3 = conn.handle("feather_health", {})
        d3 = json.loads(r3)
        print(f"\n  3. feather_health →")
        print(f"       total={d3['total']} nodes, hot={d3['hot_count']}, cold={d3['cold_count']}, orphans={d3['orphan_count']}")

        print(f"\n  [Gemini would synthesise these results into a final answer]\n")
        return

    # Real Gemini call
    from google import genai
    client = genai.Client(api_key=api_key)

    print("  Calling gemini-2.0-flash…\n")
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=conn.chat_config(
            system=(
                "You are a performance marketing analyst. "
                "Use the Feather DB tools to find relevant context before answering. "
                "Cite node IDs. Be concise."
            )
        ),
    )
    result = conn.run_loop(chat, question, max_rounds=6, verbose=True)
    print(f"\n  Gemini's answer:\n")
    for line in result.strip().split("\n"):
        print(f"  {line}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MCP CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def show_mcp_config(db_path: str):
    print("─" * 64)
    print("  MCP SERVER — Claude Desktop / Cursor")
    print("─" * 64)
    print(f"""
Install the MCP SDK:
  pip install mcp feather-db

Start the server:
  feather-serve --db {db_path} --dim 3072

Claude Desktop config  (~/.claude/claude_desktop_config.json):
  {{
    "mcpServers": {{
      "feather": {{
        "command": "feather-serve",
        "args": ["--db", "{db_path}", "--dim", "3072"]
      }}
    }}
  }}

Cursor config  (.cursor/mcp.json):
  {{
    "mcpServers": {{
      "feather": {{
        "command": "feather-serve",
        "args": ["--db", "{db_path}", "--dim", "3072"]
      }}
    }}
  }}

Once connected, Claude Desktop / Cursor can call all 14 Feather tools
directly from the chat UI — no code required.
""")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    DIM = 3072
    db_path = tempfile.mktemp(suffix=".feather")

    print("\nFeather DB — Connector Quickstart")
    print("=" * 64)

    explain_architecture()

    print("Building demo knowledge graph…")
    build_db(db_path, DIM)

    demo_claude(db_path, DIM)
    demo_gemini(db_path, DIM)
    show_mcp_config(db_path)

    print("=" * 64)
    print("Done. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY for live LLM calls.")
    print()


if __name__ == "__main__":
    main()
