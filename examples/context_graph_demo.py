"""
context_graph_demo.py — Feather DB v0.5.0 Living Context Engine

Demonstrates the full context graph stack:
  1.  Manual typed edges (link with rel_type + weight)
  2.  Auto-link by vector similarity
  3.  get_edges() — outgoing edges with type + weight
  4.  get_incoming() — reverse index (who points to me?)
  5.  context_chain() — vector search + n-hop graph expansion
  6.  export_graph_json() — D3/Cytoscape JSON
  7.  visualize() — self-contained interactive HTML graph
  8.  Backward compat: meta.links still works
  9.  Multi-embedding models (text + visual in separate modalities)
  10. Full persistence round-trip
"""

import sys, time, json
import numpy as np
sys.path.insert(0, '.')

import feather_db
from feather_db import (DB, Metadata, ContextType, FilterBuilder,
                         MarketingProfile, RelType, visualize, export_graph)

print("=" * 65)
print(f"  Feather DB v{feather_db.__version__} — Context Graph Demo")
print("=" * 65)

DB_PATH = "/tmp/context_graph_demo.feather"
DIM     = 64   # small dim for fast demo
rng     = np.random.default_rng(42)

def vec(seed=None):
    if seed is not None:
        return np.random.default_rng(seed).random(DIM).astype(np.float32)
    return rng.random(DIM).astype(np.float32)

db = DB.open(DB_PATH, dim=DIM)


# ─────────────────────────────────────────────────────────────
# 1. Build a knowledge graph — Nike marketing context
#    Nodes: Campaign Brief → Creative Concept → Ad Assets
#           → Performance Report → Follow-up Campaign
# ─────────────────────────────────────────────────────────────
print("\n[1] Inserting knowledge graph nodes (Nike marketing)...")

records = [
    # id, content, type, source, importance
    (1,  "Nike Q2 Brand Brief: summer performance narrative",   ContextType.FACT,         "strategy",   1.0),
    (2,  "Creative concept: Just Move — athlete stories",       ContextType.FACT,         "creative",   0.95),
    (3,  "Instagram Reel: 30s athlete sprint video",            ContextType.EVENT,        "production", 0.9),
    (4,  "TikTok Ad: GenZ lifestyle collab #JustMove",          ContextType.EVENT,        "production", 0.88),
    (5,  "Email campaign: loyalty segment re-engagement",       ContextType.EVENT,        "crm",        0.75),
    (6,  "Q2 Performance Report: ROAS=4.2, CTR=0.067",         ContextType.FACT,         "analytics",  0.85),
    (7,  "User preference: prefers video ads over static",      ContextType.PREFERENCE,   "insight",    0.8),
    (8,  "A/B test: short-form vs long-form — short wins",      ContextType.FACT,         "analytics",  0.7),
    (9,  "Follow-up brief: Q3 extend JustMove campaign",        ContextType.FACT,         "strategy",   0.95),
    (10, "Competitor signal: Adidas uses similar narrative",    ContextType.FACT,         "intel",      0.6),
]

for rid, content, ctype, source, imp in records:
    mp = MarketingProfile().set_brand("nike").set_user("team_creative")
    m  = mp.to_metadata()
    m.timestamp  = int(time.time()) - (len(records) - rid) * 86400
    m.type       = ctype
    m.source     = source
    m.content    = content
    m.importance = imp
    # Make similar records have similar vectors (add noise to base)
    base = vec(rid % 3 + 1)        # 3 cluster seeds
    noise = rng.random(DIM).astype(np.float32) * 0.3
    db.add(id=rid, vec=(base + noise) / np.linalg.norm(base + noise), meta=m)

print(f"   {len(records)} nodes inserted")


# ─────────────────────────────────────────────────────────────
# 2. Manual typed edges — building the knowledge graph
# ─────────────────────────────────────────────────────────────
print("\n[2] Creating typed, weighted edges...")

# Campaign Brief → Creative Concept (derived_from)
db.link(2, 1, RelType.DERIVED_FROM, weight=0.95)

# Creative Concept → Ad Assets (derived_from)
db.link(3, 2, RelType.DERIVED_FROM, weight=0.9)
db.link(4, 2, RelType.DERIVED_FROM, weight=0.85)
db.link(5, 2, RelType.DERIVED_FROM, weight=0.7)

# Performance Report caused by campaign assets
db.link(6, 3, RelType.CAUSED_BY, weight=0.8)
db.link(6, 4, RelType.CAUSED_BY, weight=0.75)

# A/B test supports performance findings
db.link(8, 6, RelType.SUPPORTS, weight=0.9)

# User preference supported by A/B test
db.link(7, 8, RelType.SUPPORTS, weight=0.85)

# Follow-up brief derived from original brief + performance report
db.link(9, 1, RelType.DERIVED_FROM, weight=0.8)
db.link(9, 6, RelType.CAUSED_BY,    weight=0.9)

# Competitor signal contradicts the uniqueness of the narrative
db.link(10, 2, RelType.CONTRADICTS, weight=0.6)

print("   10 typed edges created")


# ─────────────────────────────────────────────────────────────
# 3. Auto-link by vector similarity
# ─────────────────────────────────────────────────────────────
print("\n[3] Auto-linking by vector similarity (threshold=0.75)...")
n_auto = db.auto_link(modality="text", threshold=0.75, rel_type=RelType.RELATED_TO, candidates=5)
print(f"   {n_auto} similarity-based edges auto-created")


# ─────────────────────────────────────────────────────────────
# 4. Query outgoing + incoming edges
# ─────────────────────────────────────────────────────────────
print("\n[4] Inspecting edges for node 6 (Performance Report)...")

out = db.get_edges(6)
print(f"   Outgoing edges ({len(out)}):")
for e in out:
    print(f"     → id:{e.target_id:<3}  [{e.rel_type:<14}]  w={e.weight:.2f}")

inc = db.get_incoming(6)
print(f"   Incoming edges ({len(inc)}):")
for ie in inc:
    print(f"     ← id:{ie.source_id:<3}  [{ie.rel_type:<14}]  w={ie.weight:.2f}")


# ─────────────────────────────────────────────────────────────
# 5. backward compat: meta.links still returns target IDs
# ─────────────────────────────────────────────────────────────
print("\n[5] Backward compat: meta.links (read-only target IDs)...")
m6 = db.get_metadata(6)
print(f"   meta.links for node 6 = {m6.links}  (derived from edges)")
assert all(isinstance(x, int) for x in m6.links), "links must be list[int]"


# ─────────────────────────────────────────────────────────────
# 6. context_chain — vector search + graph expansion
# ─────────────────────────────────────────────────────────────
print("\n[6] context_chain() — search from Q2 campaign angle, 2 hops...")
query_vec = vec(seed=1)   # close to cluster-1 seeds
chain = db.context_chain(query_vec, k=3, hops=2, modality="text")

print(f"   Subgraph: {len(chain.nodes)} nodes, {len(chain.edges)} edges")
print("   Top nodes by score:")
for n in chain.nodes[:6]:
    hop_tag = f"(seed)" if n.hop == 0 else f"(hop {n.hop})"
    sim_tag = f" sim={n.similarity:.3f}" if n.similarity > 0 else ""
    label = n.metadata.content[:45]
    print(f"     id:{n.id:<3}  score={n.score:.3f}{sim_tag}  {hop_tag}  '{label}'")

print("   Edges in chain:")
for e in chain.edges[:8]:
    print(f"     id:{e.source} --[{e.rel_type}]--> id:{e.target}  w={e.weight:.2f}")


# ─────────────────────────────────────────────────────────────
# 7. Multi-modal embeddings — add visual pocket
# ─────────────────────────────────────────────────────────────
print("\n[7] Adding visual embeddings (CLIP-style, same IDs)...")
VIS_DIM = 32
for rid in [3, 4]:   # ad assets get both text + visual vectors
    vis_vec = rng.random(VIS_DIM).astype(np.float32)
    db.add(id=rid, vec=vis_vec, modality="visual")
    # Link visual to text modality representation
    db.link(rid, rid, RelType.MULTIMODAL_OF, weight=1.0)
print(f"   Text dim: {db.dim('text')}   Visual dim: {db.dim('visual')}")


# ─────────────────────────────────────────────────────────────
# 8. Export graph JSON
# ─────────────────────────────────────────────────────────────
print("\n[8] export_graph_json() — raw JSON for D3/Cytoscape...")
graph_dict = export_graph(db, namespace_filter="nike")
print(f"   Nodes: {len(graph_dict['nodes'])}  Edges: {len(graph_dict['edges'])}")
sample_node = graph_dict['nodes'][0]
print(f"   Sample node keys: {list(sample_node.keys())}")
sample_edge = graph_dict['edges'][0]
print(f"   Sample edge: {sample_edge}")


# ─────────────────────────────────────────────────────────────
# 9. Generate HTML visualizer
# ─────────────────────────────────────────────────────────────
print("\n[9] Generating interactive HTML graph visualizer...")
html_path = visualize(db, output_path="/tmp/feather_context_graph.html",
                      namespace_filter="nike")
print(f"   → Open in browser: {html_path}")


# ─────────────────────────────────────────────────────────────
# 10. Persistence round-trip
# ─────────────────────────────────────────────────────────────
print("\n[10] Persistence round-trip...")
db.save()
db2 = DB.open(DB_PATH, dim=DIM)

m9 = db2.get_metadata(9)
assert m9 is not None, "id=9 not found after reload"
edges9 = db2.get_edges(9)
assert len(edges9) >= 2, f"Expected >=2 edges for id=9, got {len(edges9)}"
rels9 = {e.rel_type for e in edges9}
assert RelType.DERIVED_FROM in rels9 or RelType.RELATED_TO in rels9

chain2 = db2.context_chain(query_vec, k=3, hops=1)
assert len(chain2.nodes) > 0, "context_chain empty after reload"

inc6 = db2.get_incoming(6)
assert len(inc6) >= 2, f"Expected >=2 incoming for id=6, got {len(inc6)}"

print(f"   id=9 edges after reload: {[e.rel_type for e in edges9]}")
print(f"   context_chain nodes: {len(chain2.nodes)}, edges: {len(chain2.edges)}")
print(f"   id=6 incoming: {[ie.rel_type for ie in inc6]}")
print("   ✓ All persistence assertions passed")


print()
print("=" * 65)
print("  DONE — Context Graph fully operational")
print(f"  Visualizer: file://{html_path}")
print("=" * 65)
