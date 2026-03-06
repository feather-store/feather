"""
Feather Agent Tools
===================
Wraps Feather DB as a tool-set for LLM agents (Claude, OpenAI, LangChain, etc.)

Provides:
  - FeatherTools class — methods agents call directly
  - Claude API tool definitions (tool_use format)
  - OpenAI-compatible function definitions
  - Simulated agent demo showing how an agent reasons over the Stable Money graph

Run after stable_money_intel_demo.py:
    python3 examples/feather_agent_tools.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feather_db
import numpy as np
import json
from feather_db import FilterBuilder, ScoringConfig

DB_PATH = "/tmp/stable_money.feather"
DIM = 128

# ─── Vocabulary (must match stable_money_intel_demo.py) ──────────────────────

VOCAB = [
    "fixed deposit","fd","interest rate","maturity","minimum deposit",
    "1000 inr","500 inr","rupee","bond","yield","credit card","lounge",
    "airport","premium","reward","cashback","annual fee","approval","cagr",
    "return","investment","savings","lock-in","tenure","renewal","nri",
    "ctr","roas","impressions","clicks","spend","conversion","cpc","cpm",
    "reach","frequency","engagement","instagram","google","youtube","meta",
    "performance","creative","campaign","audience","targeting","awareness",
    "consideration","retargeting","video","image","carousel",
    "acquisition","funnel","entry product","cross-sell","upsell",
    "positioning","competitor","differentiation","value proposition",
    "hook","intent","strategy","counter offer","pricing","threshold",
    "response","launch","reduce","increase","feature","benefit",
    "stable money","finflex","q1","q2","millennial","salaried",
    "business owner","india","drop","decline","growth","spike","anomaly",
]
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}


def embed_text(text, dim=DIM):
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for w in words:
        if w in VOCAB_INDEX:
            vec[VOCAB_INDEX[w] % dim] += 1.0
        for vocab_w, vi in VOCAB_INDEX.items():
            if vocab_w in text.lower():
                vec[vi % dim] += 0.5
    for i, ch in enumerate(text[:40]):
        vec[48 + (ord(ch) * 7 + i * 13) % 80] += 0.1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ─── FeatherTools ─────────────────────────────────────────────────────────────

class FeatherTools:
    """
    Agent-facing query interface for Feather DB.

    Each method returns clean Python dicts — ready for LLM consumption.
    Pass as tools to Claude API, OpenAI, LangChain, or any agent framework.
    """

    def __init__(self, db_path: str, dim: int = 128):
        self.db = feather_db.DB.open(db_path, dim=dim)
        self.dim = dim

    # ── Core tools ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        namespace: str = None,
        entity: str = None,
        product: str = None,
        time_decay: bool = False,
        half_life_days: float = 30.0,
    ) -> list[dict]:
        """
        Semantic search over Feather DB using natural language.
        Returns the k most relevant records, optionally filtered and decay-scored.

        Args:
            query:          Natural language search query
            k:              Number of results to return
            namespace:      Filter by namespace (e.g. "stable_money")
            entity:         Filter by entity type ("perf_snapshot", "strategy",
                            "competitor_signal", "creative", "campaign", "product")
            product:        Filter by product attribute ("fd", "credit_card", "bond")
            time_decay:     Apply adaptive time-decay scoring (recent = higher score)
            half_life_days: Half-life for decay (default 30 days)
        """
        builder = FilterBuilder()
        if namespace: builder = builder.namespace(namespace)
        if entity:    builder = builder.entity(entity)
        if product:   builder = builder.attribute("product", product)
        f = builder.build() if (namespace or entity or product) else None

        scoring = ScoringConfig(half_life=half_life_days, weight=0.4, min=0.0) if time_decay else None
        vec = embed_text(query, self.dim)
        results = self.db.search(vec, k=k, filter=f, scoring=scoring)

        return [self._format_result(r) for r in results]

    def context_chain(
        self,
        query: str,
        k: int = 5,
        hops: int = 2,
        modality: str = "text",
    ) -> dict:
        """
        Search + graph expansion. Finds seed nodes via vector search, then
        traverses the knowledge graph via BFS for `hops` steps.

        Use this to answer WHY questions — e.g. "why did FD performance drop?"
        will find the performance nodes (hop 0) then traverse to competitor events,
        strategic decisions, and counter-strategies.

        Args:
            query:    Natural language query to seed the graph traversal
            k:        Number of seed nodes from vector search
            hops:     BFS depth (2 = seed → direct neighbors → their neighbors)
            modality: Which vector index to search ("text" or "visual")
        """
        vec = embed_text(query, self.dim)
        chain = self.db.context_chain(vec, k=k, hops=hops, modality=modality)

        nodes = []
        for node in sorted(chain.nodes, key=lambda n: n.hop):
            m = node.metadata
            if m:
                nodes.append({
                    "id": node.id,
                    "hop": node.hop,
                    "score": round(node.score, 4),
                    "entity": m.entity_id,
                    "content": m.content,
                    "product": m.get_attribute("product"),
                    "severity": m.get_attribute("severity"),
                    "day": m.get_attribute("day"),
                    "ctr": m.get_attribute("ctr"),
                    "roas": m.get_attribute("roas"),
                })

        edges = [
            {"source": e.source, "target": e.target,
             "rel_type": e.rel_type, "weight": round(e.weight, 3)}
            for e in chain.edges
        ]

        return {"nodes": nodes, "edges": edges,
                "summary": f"{len(nodes)} nodes across {hops} hops, {len(edges)} edges traversed"}

    def get_node(self, node_id: int) -> dict:
        """
        Retrieve full details for a specific node by ID.
        Use after search() or context_chain() to inspect a specific record.
        """
        m = self.db.get_metadata(node_id)
        if not m:
            return {"error": f"Node {node_id} not found"}

        edges_out = self.db.get_edges(node_id)
        edges_in  = self.db.get_incoming(node_id)

        return {
            "id": node_id,
            "content": m.content,
            "entity": m.entity_id,
            "namespace": m.namespace_id,
            "importance": m.importance,
            "recall_count": m.recall_count,
            "product": m.get_attribute("product"),
            "source": m.source,
            "outgoing_edges": [
                {"target": e.target_id, "rel_type": e.rel_type, "weight": e.weight}
                for e in edges_out
            ],
            "incoming_edges": [
                {"source": e.source_id, "rel_type": e.rel_type, "weight": e.weight}
                for e in edges_in
            ],
        }

    def get_related(
        self,
        node_id: int,
        rel_type: str = None,
        direction: str = "outgoing",
    ) -> list[dict]:
        """
        Get all nodes connected to a given node via graph edges.

        Args:
            node_id:   The node to start from
            rel_type:  Optional filter by relationship type
                       (e.g. "caused_by", "contradicts", "supports", "part_of")
            direction: "outgoing" (what this node points to) or
                       "incoming" (what points TO this node)
        """
        if direction == "outgoing":
            edges = self.db.get_edges(node_id)
            related = [
                {"id": e.target_id, "rel_type": e.rel_type, "weight": e.weight,
                 **self._node_summary(e.target_id)}
                for e in edges if (rel_type is None or e.rel_type == rel_type)
            ]
        else:
            edges = self.db.get_incoming(node_id)
            related = [
                {"id": e.source_id, "rel_type": e.rel_type, "weight": e.weight,
                 **self._node_summary(e.source_id)}
                for e in edges if (rel_type is None or e.rel_type == rel_type)
            ]
        return related

    def timeline(
        self,
        product: str,
        days_back: int = 30,
        namespace: str = "stable_money",
    ) -> list[dict]:
        """
        Retrieve time-ordered performance snapshots for a product.
        Returns daily metrics sorted by day (most recent first).

        Args:
            product:   "fd", "credit_card", or "bond"
            days_back: How many days of history to return
            namespace: Namespace filter (default "stable_money")
        """
        f = (FilterBuilder()
             .namespace(namespace)
             .entity("perf_snapshot")
             .attribute("product", product)
             .build())

        cfg = ScoringConfig(half_life=float(days_back) / 2, weight=0.8, min=0.0)
        vec = embed_text(f"{product} performance ctr roas daily snapshot")
        results = self.db.search(vec, k=days_back * 3, filter=f, scoring=cfg)

        seen_days = {}
        for r in results:
            m = r.metadata
            day = m.get_attribute("day")
            if day and day not in seen_days:
                seen_days[day] = {
                    "day": int(day),
                    "ctr": float(m.get_attribute("ctr") or 0),
                    "roas": float(m.get_attribute("roas") or 0),
                    "shock": m.get_attribute("shock") == "true",
                    "score": round(r.score, 4),
                }

        return sorted(seen_days.values(), key=lambda x: x["day"], reverse=True)[:days_back]

    def explain(self, node_id: int) -> str:
        """
        Generate a human-readable explanation of a node and its graph connections.
        Useful for agents to synthesize a narrative from raw graph data.
        """
        node = self.get_node(node_id)
        if "error" in node:
            return node["error"]

        lines = [
            f"NODE {node_id} [{node['entity'].upper()}]",
            f"Content: {node['content'][:200]}",
            f"Importance: {node['importance']:.2f} | Accessed: {node['recall_count']} times",
        ]
        if node["outgoing_edges"]:
            lines.append("This node CAUSES / SUPPORTS / REFERENCES:")
            for e in node["outgoing_edges"][:5]:
                target = self.db.get_metadata(e["target"])
                if target:
                    lines.append(f"  --{e['rel_type']}--> {target.content[:80]}...")
        if node["incoming_edges"]:
            lines.append("Things attributed TO this node:")
            for e in node["incoming_edges"][:5]:
                source = self.db.get_metadata(e["source"])
                if source:
                    lines.append(f"  <--{e['rel_type']}-- {source.content[:80]}...")
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _format_result(self, r) -> dict:
        m = r.metadata
        return {
            "id": r.id,
            "score": round(r.score, 4),
            "entity": m.entity_id,
            "content": m.content,
            "product": m.get_attribute("product"),
            "day": m.get_attribute("day"),
            "ctr": m.get_attribute("ctr"),
            "roas": m.get_attribute("roas"),
            "severity": m.get_attribute("severity"),
            "competitor": m.get_attribute("competitor"),
            "importance": m.importance,
        }

    def _node_summary(self, node_id: int) -> dict:
        m = self.db.get_metadata(node_id)
        if not m:
            return {}
        return {
            "entity": m.entity_id,
            "content_preview": m.content[:100],
            "product": m.get_attribute("product"),
        }

    # ── Tool definitions for LLM agents ──────────────────────────────────────

    def as_claude_tools(self) -> list[dict]:
        """
        Returns tool definitions in Anthropic Claude API format (tool_use).
        Pass directly to the `tools` parameter of client.messages.create().

        Usage:
            import anthropic
            client = anthropic.Anthropic()
            tools = FeatherTools(DB_PATH).as_claude_tools()
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                tools=tools,
                messages=[{"role": "user", "content": "Why is our FD performance dropping?"}]
            )
        """
        return [
            {
                "name": "feather_search",
                "description": (
                    "Semantic search over the Stable Money knowledge graph. "
                    "Returns the most relevant records matching your query. "
                    "Use for: finding performance data, strategies, competitor events, creatives. "
                    "Supports filtering by entity type and product."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query, e.g. 'FD CTR drop competitor impact'"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results (default 5, max 20)",
                            "default": 5
                        },
                        "entity": {
                            "type": "string",
                            "enum": ["perf_snapshot", "strategy", "competitor_signal",
                                     "creative", "campaign", "product", "intelligence"],
                            "description": "Filter by record type"
                        },
                        "product": {
                            "type": "string",
                            "enum": ["fd", "credit_card", "bond"],
                            "description": "Filter by product"
                        },
                        "time_decay": {
                            "type": "boolean",
                            "description": "Weight results by recency (recent = higher score)",
                            "default": False
                        },
                        "half_life_days": {
                            "type": "number",
                            "description": "Half-life for decay in days (default 30)",
                            "default": 30.0
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "feather_context_chain",
                "description": (
                    "Search the knowledge graph AND traverse connected nodes via BFS. "
                    "Use this for WHY questions — starts from semantic matches, "
                    "then follows edges to find causes, strategies, competitor events. "
                    "Example: 'why is FD CTR dropping?' will find the drop → competitor event "
                    "→ contradicted strategy → counter-strategy."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to seed the graph traversal"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of seed nodes from vector search (default 5)",
                            "default": 5
                        },
                        "hops": {
                            "type": "integer",
                            "description": "Graph traversal depth (1=direct neighbors, 2=2 hops, default 2)",
                            "default": 2
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "feather_get_related",
                "description": (
                    "Get all nodes connected to a specific node via graph edges. "
                    "Use after search or context_chain to explore a specific node's connections. "
                    "Can filter by relationship type and direction."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "integer",
                            "description": "The node ID to get connections for"
                        },
                        "rel_type": {
                            "type": "string",
                            "enum": ["caused_by", "contradicts", "supports", "part_of",
                                     "derived_from", "references", "precedes", "related_to"],
                            "description": "Filter by relationship type (optional)"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["outgoing", "incoming"],
                            "description": "'outgoing' = what this node points to, 'incoming' = what points TO this node",
                            "default": "outgoing"
                        }
                    },
                    "required": ["node_id"]
                }
            },
            {
                "name": "feather_timeline",
                "description": (
                    "Get time-ordered daily performance metrics for a product. "
                    "Returns CTR, ROAS, and anomaly flags sorted by day. "
                    "Use to understand performance trends over time."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "string",
                            "enum": ["fd", "credit_card", "bond"],
                            "description": "Product to get performance timeline for"
                        },
                        "days_back": {
                            "type": "integer",
                            "description": "Number of days of history (default 30)",
                            "default": 30
                        }
                    },
                    "required": ["product"]
                }
            },
            {
                "name": "feather_explain",
                "description": (
                    "Get a human-readable explanation of a node and all its graph connections. "
                    "Use to synthesize a narrative about why something is important or connected."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "integer",
                            "description": "Node ID to explain"
                        }
                    },
                    "required": ["node_id"]
                }
            }
        ]

    def as_openai_tools(self) -> list[dict]:
        """Returns tool definitions in OpenAI function-calling format."""
        claude_tools = self.as_claude_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
            }
            for t in claude_tools
        ]

    def handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """
        Dispatch a tool call from an agent and return JSON-serializable result.
        Pass this to the tool_result content block in Claude API.
        """
        method = {
            "feather_search":        self.search,
            "feather_context_chain": self.context_chain,
            "feather_get_related":   self.get_related,
            "feather_timeline":      self.timeline,
            "feather_explain":       self.explain,
        }.get(tool_name)

        if not method:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        result = method(**tool_input)
        return json.dumps(result, indent=2, default=str)


# ─── Simulated Agent Demo ─────────────────────────────────────────────────────

def run_agent_demo():
    print("=" * 60)
    print("  Feather Agent Tools — Simulated Agent Demo")
    print("=" * 60)
    print()
    print("  This demo shows how a Claude agent would use Feather tools")
    print("  to answer: 'Our FD CTR has dropped. What happened and what")
    print("  should we do?'")
    print()

    tools = FeatherTools(DB_PATH, dim=DIM)

    # ── Agent turn 1: search for the anomaly ──────────────────────────────────
    print("─" * 60)
    print("AGENT → Tool call: feather_search")
    print('  query="FD performance drop CTR decline anomaly"')
    print('  entity="perf_snapshot", product="fd", time_decay=True')
    print("─" * 60)

    results = tools.search(
        query="FD performance drop CTR decline anomaly",
        k=5, entity="perf_snapshot", product="fd",
        time_decay=True, half_life_days=14
    )
    print("FEATHER → Results:")
    for r in results[:3]:
        print(f"  node={r['id']} day={r['day']} CTR={r['ctr']} ROAS={r['roas']} score={r['score']}")

    # ── Agent turn 2: context chain from the drop ─────────────────────────────
    print()
    print("─" * 60)
    print("AGENT → Tool call: feather_context_chain")
    print('  query="FD CTR drop competitor impact caused by", hops=2')
    print("─" * 60)

    chain = tools.context_chain(
        query="FD CTR drop competitor impact caused by external event",
        k=4, hops=2
    )
    print(f"FEATHER → {chain['summary']}")
    print("  Graph path:")
    for node in chain["nodes"]:
        indent = "  " + "  " * node["hop"]
        entity = node["entity"].upper()
        preview = node["content"][:90].replace("\n", " ")
        print(f"{indent}[hop {node['hop']}] [{entity}] {preview}...")

    print()
    print("  Edge types traversed:")
    from collections import Counter
    edge_types = Counter(e["rel_type"] for e in chain["edges"])
    for rel, count in edge_types.most_common():
        print(f"    {rel}: {count}")

    # ── Agent turn 3: get competitor node details ─────────────────────────────
    FINFLEX_NODE = 300
    print()
    print("─" * 60)
    print(f"AGENT → Tool call: feather_get_related")
    print(f"  node_id={FINFLEX_NODE} (FinFlex ₹500 event), direction='incoming'")
    print("─" * 60)

    attributed = tools.get_related(FINFLEX_NODE, direction="incoming")
    print(f"FEATHER → {len(attributed)} nodes attributed to this competitor event:")
    rel_counts = Counter(n["rel_type"] for n in attributed)
    for rel, count in rel_counts.most_common():
        print(f"  {rel}: {count} nodes")

    # ── Agent turn 4: find the counter-strategy ───────────────────────────────
    print()
    print("─" * 60)
    print("AGENT → Tool call: feather_search")
    print('  query="counter strategy response finflex 500 inr combo offer"')
    print('  entity="strategy"')
    print("─" * 60)

    strategy = tools.search(
        query="counter strategy response finflex 500 inr fd credit card combo",
        k=3, entity="strategy"
    )
    print("FEATHER → Recommended strategies:")
    for r in strategy:
        print(f"  [score={r['score']}] {r['content'][:120]}...")

    # ── Agent turn 5: performance timeline ───────────────────────────────────
    print()
    print("─" * 60)
    print("AGENT → Tool call: feather_timeline")
    print('  product="fd", days_back=10')
    print("─" * 60)

    timeline = tools.timeline(product="fd", days_back=10)
    print("FEATHER → FD performance (last 10 days, most recent first):")
    print(f"  {'Day':>4}  {'CTR':>7}  {'ROAS':>6}  {'Flag'}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*15}")
    for t in timeline[:10]:
        flag = "⚠ COMPETITOR IMPACT" if t["shock"] else ""
        print(f"  {t['day']:>4}  {t['ctr']:>7.4f}  {t['roas']:>6.2f}  {flag}")

    # ── Agent synthesis ───────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("AGENT SYNTHESIS (what Claude would say after these tool calls):")
    print("─" * 60)
    print("""
  Based on the Feather knowledge graph, here is what happened:

  ROOT CAUSE:
  On day 21, FinFlex Bank launched a ₹500 minimum FD (cutting from ₹2000).
  This directly contradicts Stable Money's ₹1000 premium positioning strategy.
  FD CTR dropped ~20% across all creatives on days 21-26 (caused_by this event).

  CASCADING IMPACT:
  - FD Lounge Video: CTR 0.038 → 0.025 (-34%)
  - FD 1000 INR Banner: hardest hit (directly undermined by FinFlex messaging)
  - FinFlex also launched 'seShuru' campaign on same Instagram audience
  - HDFC lounge ads running simultaneously — audience fatigue on lounge hook

  WHAT THE GRAPH SAYS TO DO:
  The counter-strategy node (importance=0.98) is already in the intelligence layer:
  → Run FD + CC combo: "Open ₹1000 FD → instant CC approval → 3 months free lounge"
  → Do NOT reduce minimum deposit — maintain premium positioning
  → Pause pure interest-rate creatives (low CTR even before shock)
  → Double down on video format (2.1x CTR vs static per intelligence node)

  RECOVERY SIGNAL:
  Day 27-30 shows FD performance recovering (+4%/day) as counter-strategy
  messaging takes effect. Bond and CC unaffected throughout.
""")

    # ── Show Claude API usage ─────────────────────────────────────────────────
    print("─" * 60)
    print("HOW TO CONNECT TO A REAL CLAUDE AGENT:")
    print("─" * 60)
    print("""
  import anthropic
  from examples.feather_agent_tools import FeatherTools

  client = anthropic.Anthropic()
  ft = FeatherTools("/tmp/stable_money.feather")

  messages = [{"role": "user", "content": "Why is our FD CTR dropping?"}]

  while True:
      response = client.messages.create(
          model="claude-opus-4-6",
          max_tokens=4096,
          tools=ft.as_claude_tools(),
          messages=messages,
      )

      if response.stop_reason == "end_turn":
          print(response.content[0].text)
          break

      # Process tool calls
      tool_results = []
      for block in response.content:
          if block.type == "tool_use":
              result = ft.handle_tool_call(block.name, block.input)
              tool_results.append({
                  "type": "tool_result",
                  "tool_use_id": block.id,
                  "content": result,
              })

      messages.append({"role": "assistant", "content": response.content})
      messages.append({"role": "user", "content": tool_results})
""")


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found.")
        print("Run stable_money_intel_demo.py first.")
        sys.exit(1)

    run_agent_demo()
    print("=" * 60)
    print("  Tool definitions available as:")
    print("    ft = FeatherTools(DB_PATH)")
    print("    ft.as_claude_tools()   # Anthropic format")
    print("    ft.as_openai_tools()   # OpenAI format")
    print("    ft.handle_tool_call(name, input)  # dispatch")
    print("=" * 60)
