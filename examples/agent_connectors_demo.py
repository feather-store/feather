"""
Feather DB — LLM Agent Connectors Demo
========================================
Demonstrates all three connectors running a simulated multi-turn agent loop
against a pre-seeded Feather DB knowledge graph.

Runs fully offline in mock mode (no API keys needed).
Enable real LLM calls by setting environment variables:

    ANTHROPIC_API_KEY   → activates Claude connector demo
    OPENAI_API_KEY      → activates OpenAI connector demo
    GOOGLE_API_KEY      → activates Gemini connector demo
    GROQ_API_KEY        → activates Groq (OpenAI-compat) connector demo

Usage:
    # Mock / offline (no keys needed)
    python3 examples/agent_connectors_demo.py

    # Real APIs (uses whichever keys are set)
    ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... python3 examples/agent_connectors_demo.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import textwrap
import hashlib
from typing import Callable

import numpy as np

# ── Add repo root to path when running from examples/ ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feather_db
from feather_db.integrations import (
    ClaudeConnector, OpenAIConnector, GeminiConnector, GeminiEmbedder,
    TOOL_SPECS,
)

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ──────────────────────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA= "\033[35m"
    BLUE   = "\033[34m"
    RED    = "\033[31m"

def hdr(title: str, color: str = C.CYAN) -> None:
    bar = "─" * 68
    print(f"\n{color}{C.BOLD}{bar}{C.RESET}")
    print(f"{color}{C.BOLD}  {title}{C.RESET}")
    print(f"{color}{C.BOLD}{bar}{C.RESET}")

def section(label: str) -> None:
    print(f"\n{C.YELLOW}{C.BOLD}▶ {label}{C.RESET}")

def ok(msg: str) -> None:
    print(f"  {C.GREEN}✓{C.RESET}  {msg}")

def info(msg: str) -> None:
    print(f"  {C.DIM}{msg}{C.RESET}")

def tool_call(name: str, args: dict) -> None:
    print(f"  {C.MAGENTA}[tool_call]{C.RESET} {C.BOLD}{name}{C.RESET}({json.dumps(args, separators=(',',':'))[:100]})")

def tool_result(snippet: str) -> None:
    print(f"  {C.DIM}[tool_result]{C.RESET} {snippet[:120]}…")

def agent_text(text: str) -> None:
    wrapped = textwrap.fill(text.strip(), width=70, initial_indent="  ", subsequent_indent="  ")
    print(f"\n{C.CYAN}  Agent:{C.RESET}\n{wrapped}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Mock embedder  (deterministic, no API key)
# ──────────────────────────────────────────────────────────────────────────────

DIM = 3072

_VOCAB = [
    "fd","fixeddeposit","bond","creditcard","mutualfund","roas","ctr","cpm",
    "invest","return","rate","interest","savings","growth","yield","tax",
    "campaign","ad","creative","static","video","hook","cta","click",
    "retargeting","acquisition","retention","attribution","spend",
    "competitor","budget","rbi","india","viral","sentiment","instagram",
    "context","memory","recall","decay","sticky","graph","edge","chain",
]

def mock_embed(text: str, salt: str = "text") -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    tokens = text.lower().replace(",", " ").replace(".", " ").split()
    for tok in tokens:
        for i, kw in enumerate(_VOCAB):
            if kw in tok or tok in kw:
                idx = (i * 11 + len(tok) * 7) % DIM
                vec[idx] += 1.0
                vec[(idx + 37) % DIM] += 0.3
    if salt:
        h = int(hashlib.md5(salt.encode()).hexdigest(), 16)
        vec = np.roll(vec, h % 12)
        vec += np.random.default_rng(h % 10000).random(DIM).astype(np.float32) * 0.05
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


# ──────────────────────────────────────────────────────────────────────────────
# Seed a demo Feather DB with performance marketing intel
# ──────────────────────────────────────────────────────────────────────────────

def seed_demo_db(db_path: str) -> None:
    """Populate a .feather file with synthetic but realistic intel nodes."""
    db = feather_db.DB.open(db_path, dim=DIM)

    nodes = [
        # ── FD creative performance ──────────────────────────────────────────
        (1001, "FD video ad — senior couple, 8.5% guaranteed return hook. CTR 3.2%, ROAS 4.1. Best performer Q1.",
         "ad_creative", "FD", 1.0),
        (1002, "FD static banner — gold coins, trust badge. CTR 1.8%, ROAS 2.7. Underperforming vs video.",
         "ad_creative", "FD", 0.8),
        (1003, "FD carousel ad — split-test: rate-first vs trust-first. Rate-first: CTR 2.9%. Trust-first: CTR 1.4%.",
         "ad_creative", "FD", 0.9),
        (1004, "FD reels creative — Budget Day 2026, Finance Minister announcement hook. CTR spike +140% at 11 AM.",
         "ad_creative", "FD", 0.95),

        # ── CC creative performance ──────────────────────────────────────────
        (2001, "CC lounge access ad — airport visual, premium lifestyle. CTR 2.1%, ROAS 1.8. Young professional segment.",
         "ad_creative", "CC", 0.75),
        (2002, "CC cashback reel — animated coins, 5% unlimited cashback hook. CTR 3.8%. Best CC format.",
         "ad_creative", "CC", 0.92),

        # ── Competitor intel ─────────────────────────────────────────────────
        (9001, "Competitor A (incumbent bank) launched FD at 8.75% APY on Feb 1. Directly undercuts our 8.5% offer. Budget Day timing deliberate.",
         "competitor_intel", "FD", 0.95),
        (9002, "Competitor B (fintech) social campaign: 'Beat FD with MF SIP'. Targeting our FD user cohort. Spend up 3x this week.",
         "competitor_intel", "MF", 0.85),
        (9003, "Competitor C (neobank) CC approval rate claim: 'Approved in 60 seconds'. Direct attack on our CC funnel friction narrative.",
         "competitor_intel", "CC", 0.80),
        (9004, "Competitor A MF push paused — compliance review flagged misleading returns claim. Window for us to capture MF intent traffic.",
         "competitor_intel", "MF", 0.88),

        # ── Social / macro intel ─────────────────────────────────────────────
        (9101, "Budget Day 2026 (Feb 1): RBI held repo rate at 6.5%. FD appeal increased — fixed rates now more attractive than floating.",
         "social_trend", "FD", 0.90),
        (9102, "Twitter/X trend #InflationHedge peaked at 12 PM Feb 1. FD and Bond queries +220%. Retargeting window: next 48 hours.",
         "social_trend", "FD", 0.87),
        (9103, "Instagram Reels CTR benchmark Feb 2026: Finance vertical avg 2.4%. Video with voice hook outperforms static 2.3x.",
         "social_trend", None, 0.75),
        (9104, "Post-budget MF inflows hit 12-month high. Equity SIP narrative gaining. Potential headwind to FD conversion if MF yields trend up.",
         "social_trend", "MF", 0.82),

        # ── Strategy intel ───────────────────────────────────────────────────
        (9201, "Q1 strategy: Lead with FD rate hook in first 2 seconds of video. Senior + 35-55 age segment shows 3x ROAS vs general audience.",
         "strategy_intel", "FD", 0.93),
        (9202, "Creative fatigue detected on FD static banner (id 1002) — frequency 8.2, CTR declining 12% week-over-week. Rotate creative.",
         "strategy_intel", "FD", 0.91),
        (9203, "Attribution insight: 62% of FD conversions come via 3+ touch retargeting sequence. Single-impression ROAS misleading.",
         "strategy_intel", "FD", 0.88),
        (9204, "MF SIP creative opportunity: post-budget intent window open. Recommend launching MF carousel with tax-saving angle within 72h.",
         "strategy_intel", "MF", 0.86),
    ]

    t_base = int(time.time()) - 86400  # 24h ago
    for offset, (nid, content, entity_type, product, importance) in enumerate(nodes):
        vec  = mock_embed(content)
        meta = feather_db.Metadata()
        meta.timestamp  = t_base + offset * 900   # 15-min intervals
        meta.importance = importance
        meta.type       = feather_db.ContextType.EVENT
        meta.source     = "demo_seed"
        meta.content    = content
        meta.namespace_id = "demo"
        meta.entity_id    = entity_type
        meta.set_attribute("entity_type", entity_type)
        meta.set_attribute("modality", "text")
        if product:
            meta.set_attribute("product", product)
        db.add(id=nid, vec=vec, meta=meta)

    # ── Graph edges ──────────────────────────────────────────────────────────
    db.link(from_id=9001, to_id=1001, rel_type="contradicts",   weight=0.9)   # competitor rate undercuts our video
    db.link(from_id=9001, to_id=9101, rel_type="supports",      weight=0.85)  # budget day → rate attractiveness
    db.link(from_id=9101, to_id=1004, rel_type="caused_by",     weight=0.95)  # budget day caused CTR spike
    db.link(from_id=9102, to_id=1004, rel_type="supports",      weight=0.80)  # social trend supports reels spike
    db.link(from_id=9201, to_id=1001, rel_type="supports",      weight=0.90)  # strategy supports video
    db.link(from_id=9202, to_id=1002, rel_type="references",    weight=0.95)  # fatigue references static banner
    db.link(from_id=9001, to_id=9202, rel_type="caused_by",     weight=0.70)  # competitor launch → fatigue pressure
    db.link(from_id=9203, to_id=9201, rel_type="supports",      weight=0.85)  # attribution insight → strategy
    db.link(from_id=9104, to_id=9204, rel_type="supports",      weight=0.88)  # MF trend → MF strategy

    db.save()
    ok(f"Seeded {len(nodes)} nodes + 9 typed edges → {db_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Mock agent loop  (simulates tool-use without a real LLM)
# ──────────────────────────────────────────────────────────────────────────────

def mock_agent_loop(conn, question: str) -> None:
    """
    Simulates a 3-round agent loop:
      Round 1 — feather_search for direct hits
      Round 2 — feather_context_chain to follow causal graph
      Round 3 — feather_get_node for deep-dive on most relevant hit
    Then synthesises a final text response.
    """
    print(f"\n  {C.BOLD}Question:{C.RESET} {question}\n")

    # Round 1: search
    args1 = {"query": question, "k": 4}
    tool_call("feather_search", args1)
    r1 = conn.handle("feather_search", args1)
    data1 = json.loads(r1)
    tool_result(r1)

    # Round 2: context_chain from same question
    args2 = {"query": question, "k": 3, "hops": 2}
    tool_call("feather_context_chain", args2)
    r2 = conn.handle("feather_context_chain", args2)
    data2 = json.loads(r2)
    tool_result(r2)

    # Round 3: get_node on top search result (if available)
    top_id = data1["results"][0]["id"] if data1["results"] else None
    if top_id:
        args3 = {"id": top_id}
        tool_call("feather_get_node", args3)
        r3 = conn.handle("feather_get_node", args3)
        data3 = json.loads(r3)
        tool_result(r3)
    else:
        data3 = {}

    # Synthesise final response from retrieved data
    lines = []
    if data1["results"]:
        top = data1["results"][0]
        lines.append(f"Top signal: {top['content'][:120]}")
    chain_nodes = [n for n in data2.get("nodes", []) if n["hop"] > 0]
    if chain_nodes:
        lines.append(f"Causal graph expanded {len(chain_nodes)} node(s) beyond seed.")
        for n in chain_nodes[:2]:
            lines.append(f"  ↳ [{n['entity_type']}] {n['content'][:100]}")
    if data3.get("edges_out"):
        rels = [f"{e['rel_type']}→{e['target_id']}" for e in data3["edges_out"][:3]]
        lines.append(f"Node {top_id} connects to: {', '.join(rels)}")

    summary = " | ".join(lines) if lines else "No relevant context found."
    agent_text(f"Based on Feather DB context graph:\n\n" + "\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Real Claude agent loop
# ──────────────────────────────────────────────────────────────────────────────

def run_claude(db_path: str, question: str) -> None:
    hdr("Claude Connector  (claude-opus-4-6)", C.MAGENTA)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        info("ANTHROPIC_API_KEY not set — running mock loop")
        conn = ClaudeConnector(db_path=db_path, dim=DIM, embedder=mock_embed)
        mock_agent_loop(conn, question)
        return

    try:
        import anthropic
    except ImportError:
        info("pip install anthropic — skipping real Claude demo")
        return

    conn   = ClaudeConnector(db_path=db_path, dim=DIM, embedder=mock_embed)
    client = anthropic.Anthropic(api_key=api_key)

    section("Running real Claude agent loop")
    print(f"  {C.BOLD}Question:{C.RESET} {question}\n")

    result = conn.run_loop(
        client,
        messages=[{"role": "user", "content": question}],
        model="claude-opus-4-6",
        system=(
            "You are a performance marketing analyst with access to Feather DB, "
            "a living context engine that stores campaign intelligence. "
            "Use the available tools to retrieve relevant context before answering. "
            "Always cite the node IDs you found."
        ),
        max_rounds=6,
    )
    agent_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Real OpenAI agent loop
# ──────────────────────────────────────────────────────────────────────────────

def run_openai(db_path: str, question: str) -> None:
    hdr("OpenAI Connector  (gpt-4o)", C.BLUE)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        info("OPENAI_API_KEY not set — running mock loop")
        conn = OpenAIConnector(db_path=db_path, dim=DIM, embedder=mock_embed)
        mock_agent_loop(conn, question)
        return

    try:
        from openai import OpenAI
    except ImportError:
        info("pip install openai — skipping real OpenAI demo")
        return

    conn   = OpenAIConnector(db_path=db_path, dim=DIM, embedder=mock_embed)
    client = OpenAI(api_key=api_key)

    section("Running real OpenAI agent loop")
    print(f"  {C.BOLD}Question:{C.RESET} {question}\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a performance marketing analyst. "
                "Use the Feather DB tools to retrieve context before answering. "
                "Cite node IDs in your final answer."
            ),
        },
        {"role": "user", "content": question},
    ]
    result = conn.run_loop(client, messages, model="gpt-4o", max_rounds=6)
    agent_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Groq (OpenAI-compatible) agent loop
# ──────────────────────────────────────────────────────────────────────────────

def run_groq(db_path: str, question: str) -> None:
    hdr("Groq Connector  (llama-3.3-70b-versatile)", C.GREEN)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        info("GROQ_API_KEY not set — running mock loop")
        conn = OpenAIConnector(db_path=db_path, dim=DIM, embedder=mock_embed)
        mock_agent_loop(conn, question)
        return

    try:
        from openai import OpenAI
    except ImportError:
        info("pip install openai — skipping real Groq demo")
        return

    conn   = OpenAIConnector(db_path=db_path, dim=DIM, embedder=mock_embed)
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    section("Running real Groq agent loop")
    print(f"  {C.BOLD}Question:{C.RESET} {question}\n")

    messages = [{"role": "user", "content": question}]
    result   = conn.run_loop(
        client, messages,
        model="llama-3.3-70b-versatile",
        system="You are a marketing analyst. Use Feather DB tools to retrieve context before answering.",
        max_rounds=6,
    )
    agent_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Real Gemini agent loop
# ──────────────────────────────────────────────────────────────────────────────

def run_gemini(db_path: str, question: str) -> None:
    hdr("Gemini Connector  (gemini-2.0-flash)", C.YELLOW)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        info("GOOGLE_API_KEY not set — running mock loop")
        conn = GeminiConnector(db_path=db_path, dim=DIM, embedder=mock_embed)
        mock_agent_loop(conn, question)
        return

    try:
        from google import genai
    except ImportError:
        info("pip install google-genai — skipping real Gemini demo")
        return

    emb    = GeminiEmbedder(api_key=api_key)
    conn   = GeminiConnector(db_path=db_path, dim=emb.dim, embedder=emb.embed_text)
    client = genai.Client(api_key=api_key)

    section("Running real Gemini agent loop")
    print(f"  {C.BOLD}Question:{C.RESET} {question}\n")

    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=conn.chat_config(
            system=(
                "You are a performance marketing analyst. "
                "Use Feather DB tools to retrieve context before answering. "
                "Cite node IDs in your final answer."
            )
        ),
    )
    result = conn.run_loop(chat, question, max_rounds=6)
    agent_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Tool schema inspection
# ──────────────────────────────────────────────────────────────────────────────

def show_tools_summary() -> None:
    hdr("Tool Schema Overview  (provider-agnostic)", C.CYAN)
    print(f"\n  {len(TOOL_SPECS)} tools registered:\n")
    for spec in TOOL_SPECS:
        params = ", ".join(f"{p}" + (" *" if p in spec.get("required", []) else "") for p in spec["parameters"])
        print(f"  {C.BOLD}{spec['name']:<30}{C.RESET}  params: {params}")
    print(f"\n  {C.DIM}( * = required ){C.RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Additional tool demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_write_tools(conn, db_path: str) -> None:
    hdr("Write Tools Demo  (feather_add_intel + feather_link_nodes)", C.GREEN)

    section("feather_add_intel — agent ingests a new competitor signal")
    args = {
        "content": "Competitor D launched 'Instant FD' with 8.9% rate. Zero-paperwork claim. Heavy Instagram spend detected — estimated ₹2Cr/day.",
        "entity_type": "competitor_intel",
        "product": "FD",
        "importance": 0.95,
    }
    tool_call("feather_add_intel", args)
    result = conn.handle("feather_add_intel", args)
    data   = json.loads(result)
    tool_result(result)
    new_id = data["id"]
    ok(f"New intel node assigned ID: {new_id}")

    section("feather_link_nodes — connect new intel to existing strategy node")
    link_args = {
        "from_id":  new_id,
        "to_id":    9201,
        "rel_type": "contradicts",
        "weight":   0.88,
    }
    tool_call("feather_link_nodes", link_args)
    r2 = conn.handle("feather_link_nodes", link_args)
    tool_result(r2)
    ok("Edge created: new intel ──contradicts──▶ strategy node 9201")


def demo_timeline(conn) -> None:
    hdr("Timeline Demo  (feather_timeline)", C.BLUE)

    section("Chronological FD intel (limit 6)")
    args = {"product": "FD", "limit": 6}
    tool_call("feather_timeline", args)
    r = conn.handle("feather_timeline", args)
    data = json.loads(r)
    tool_result(r)

    print(f"\n  {C.BOLD}Timeline — {data['total_matching']} FD nodes total:{C.RESET}")
    for n in data["timeline"]:
        ts = time.strftime("%b %d %H:%M", time.localtime(n["timestamp"]))
        etype = f"[{n['entity_type']}]"
        print(f"  {C.DIM}{ts}{C.RESET}  {etype:<22}  {n['content'][:70]}")


def demo_context_chain(conn) -> None:
    hdr("Context Chain Demo  (vector search + BFS graph expansion)", C.MAGENTA)

    question = "Why did our FD CTR spike on Budget Day?"
    section(f"Query: '{question}'")
    args = {"query": question, "k": 3, "hops": 2}
    tool_call("feather_context_chain", args)
    r = conn.handle("feather_context_chain", args)
    data = json.loads(r)
    tool_result(r)

    print(f"\n  {C.BOLD}Chain result — {data['node_count']} nodes, {data['edge_count']} edges:{C.RESET}")
    for n in data["nodes"]:
        hop_label = f"hop={n['hop']}"
        etype = f"[{n['entity_type'] or '?'}]"
        bar   = "  " * n["hop"] + "└─ " if n["hop"] > 0 else "   "
        print(f"  {C.DIM}{hop_label}{C.RESET}  {bar}{etype:<22}  {n['content'][:75]}")

    if data["edges"]:
        print(f"\n  {C.BOLD}Edges traversed:{C.RESET}")
        for e in data["edges"]:
            print(f"    {e['source']} ──{e['rel_type']}──▶ {e['target']}  (w={e['weight']})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{C.BOLD}Feather DB — LLM Agent Connectors Demo{C.RESET}")
    print(f"{C.DIM}v0.6.0  |  Claude · OpenAI · Groq · Gemini{C.RESET}")

    # ── Build temp DB ──────────────────────────────────────────────────────
    db_path = os.path.join(tempfile.gettempdir(), "feather_agent_demo.feather")
    hdr("Seeding Demo Knowledge Graph", C.CYAN)
    seed_demo_db(db_path)
    ok(f"DB path: {db_path}")

    # ── Show tool schema ───────────────────────────────────────────────────
    show_tools_summary()

    # ── Create a shared connector (mock embedder) for offline demos ────────
    conn = ClaudeConnector(db_path=db_path, dim=DIM, embedder=mock_embed)

    # ── Offline tool demos ─────────────────────────────────────────────────
    demo_context_chain(conn)
    demo_timeline(conn)
    demo_write_tools(conn, db_path)

    # ── Agent loop demos (mock or real depending on env vars) ──────────────
    questions = [
        "Why is our FD CTR underperforming vs competitors on Budget Day?",
        "Which ad creatives should we rotate next week?",
    ]

    q1, q2 = questions[0], questions[1]

    run_claude(db_path,  q1)
    run_openai(db_path,  q2)
    run_groq(db_path,    q1)
    run_gemini(db_path,  q2)

    # ── Summary ────────────────────────────────────────────────────────────
    hdr("Summary", C.GREEN)
    has_real = any([
        os.environ.get("ANTHROPIC_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
    ])
    if has_real:
        ok("Real LLM agent loop(s) completed.")
    else:
        info("All connectors ran in mock mode (no API keys set).")
        info("Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY for live demos.")

    print(f"""
  {C.BOLD}Connector quick-start:{C.RESET}

    from feather_db.integrations import ClaudeConnector, OpenAIConnector
    from feather_db.integrations import GeminiConnector, GeminiEmbedder

    # Claude
    import anthropic
    conn   = ClaudeConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
    client = anthropic.Anthropic()
    result = conn.run_loop(client, [{{"role":"user","content":"your question"}}])

    # OpenAI / Groq / Mistral (OpenAI-compatible)
    from openai import OpenAI
    conn   = OpenAIConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
    client = OpenAI()            # or OpenAI(base_url="https://api.groq.com/openai/v1")
    result = conn.run_loop(client, [{{"role":"user","content":"your question"}}])

    # Gemini
    from google import genai
    emb    = GeminiEmbedder(api_key="AIza...")
    conn   = GeminiConnector(db_path="my.feather", dim=emb.dim, embedder=emb.embed_text)
    client = genai.Client(api_key="AIza...")
    chat   = client.chats.create(model="gemini-2.0-flash", config=conn.chat_config())
    result = conn.run_loop(chat, "your question")
""")


if __name__ == "__main__":
    main()
