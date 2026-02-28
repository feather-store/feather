"""
feather_inspector.py — Feather DB v0.5.0 Interactive Inspector

Local web UI with four features:
  1. Force Graph       — existing context graph
  2. Embedding Scatter — PCA 2D projection of all vectors
  3. Similarity Search — click any node → top-K similar highlighted
  4. Edit / Delete     — update metadata or deactivate records

Usage:
    python3 examples/feather_inspector.py
    → opens http://localhost:7777 in your browser
"""

import sys, json, math, time, hashlib, re, os, threading, webbrowser
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, '.')
from feather_db import DB, Metadata, ContextType, RelType
from feather_db.graph import export_graph, _HTML_TEMPLATE

DB_PATH = '/tmp/real_meta_ads.feather'
DIM     = 128
PORT    = 7777

print("=" * 60)
print("  Feather DB Inspector — loading DB...")
print("=" * 60)

db = DB.open(DB_PATH, dim=DIM)
print(f"  Loaded: {db.size()} nodes")

# ─── Load D3 ─────────────────────────────────────────────────────
_d3_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../feather_db/d3.min.js'),
    '/tmp/d3.min.js',
]
D3_SRC = ""
for p in _d3_paths:
    if os.path.exists(p):
        with open(p) as f: D3_SRC = f.read()
        break


# ─── PCA (numpy only, no sklearn) ────────────────────────────────
def pca_2d(matrix):
    """Project N×D matrix to N×2 using top-2 principal components."""
    X = matrix - matrix.mean(axis=0)
    # Use SVD on the covariance for stability
    cov = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order, take last 2 (largest)
    top2 = eigenvectors[:, -2:][:, ::-1]
    return X @ top2


# ─── Collect all vectors + build graph data ──────────────────────
def build_inspector_data(namespace_filter="hawky_meta"):
    print("  Computing embeddings + similarity index...")

    all_ids    = db.get_all_ids("text")
    graph_data = json.loads(db.export_graph_json(namespace_filter))
    node_id_set = {n['id'] for n in graph_data['nodes']}

    # Filter to nodes in graph
    ids_in_graph = [i for i in all_ids if i in node_id_set]

    # Fetch vectors
    vecs = []
    valid_ids = []
    for nid in ids_in_graph:
        v = db.get_vector(nid, "text")
        if len(v) == DIM:
            vecs.append(v)
            valid_ids.append(nid)

    print(f"  Got {len(valid_ids)} vectors for PCA")

    # PCA 2D
    mat = np.array(vecs, dtype=np.float32)
    coords_2d = pca_2d(mat)

    # Normalize to [-1, 1]
    for col in range(2):
        mn, mx = coords_2d[:, col].min(), coords_2d[:, col].max()
        rng = mx - mn or 1.0
        coords_2d[:, col] = 2 * (coords_2d[:, col] - mn) / rng - 1

    pca_map = {nid: [float(coords_2d[i, 0]), float(coords_2d[i, 1])]
               for i, nid in enumerate(valid_ids)}

    # Pre-compute top-8 similar per node (cosine similarity)
    print("  Pre-computing top-8 similar per node...")
    # Normalize rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_norm = mat / norms

    similar_map = {}
    for i, nid in enumerate(valid_ids):
        sims = mat_norm @ mat_norm[i]
        top_idx = np.argsort(sims)[::-1][1:9]   # skip self (index 0)
        similar_map[nid] = [
            {"id": int(valid_ids[j]), "score": float(sims[j])}
            for j in top_idx
        ]

    # Attach pca coords and similar to graph nodes
    pca_map_js = {str(k): v for k, v in pca_map.items()}
    similar_js  = {str(k): v for k, v in similar_map.items()}

    print(f"  Done. {len(pca_map)} PCA points  {len(similar_map)} similarity entries")
    return graph_data, pca_map_js, similar_js


graph_data, pca_map, similar_map = build_inspector_data()

# ─── Build the HTML ───────────────────────────────────────────────
HTML_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feather DB Inspector</title>
<script>{D3_SRC}</script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

#header {{ padding: 10px 16px; background: #1a1d27; border-bottom: 1px solid #2d3748; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
#header h1 {{ font-size: 14px; font-weight: 700; color: #a78bfa; }}
.badge {{ background: #2d3748; border-radius: 4px; padding: 2px 8px; font-size: 11px; color: #94a3b8; }}
.tab-btn {{ background: none; border: 1px solid #4a5568; border-radius: 6px; color: #94a3b8; padding: 4px 14px; font-size: 12px; cursor: pointer; transition: all 0.15s; }}
.tab-btn.active {{ background: #a78bfa; border-color: #a78bfa; color: #fff; font-weight: 600; }}
.tab-btn:hover:not(.active) {{ background: #2d3748; color: #e2e8f0; }}
#search-wrap {{ display: flex; gap: 6px; margin-left: auto; }}
#search-wrap input {{ background: #2d3748; border: 1px solid #4a5568; border-radius: 6px; color: #e2e8f0; padding: 4px 10px; font-size: 12px; width: 200px; }}
#search-wrap input:focus {{ outline: none; border-color: #a78bfa; }}
#btn-reset {{ background: #2d3748; border: 1px solid #4a5568; border-radius: 6px; color: #e2e8f0; padding: 4px 12px; font-size: 12px; cursor: pointer; }}
#btn-reset:hover {{ background: #4a5568; }}

#main {{ display: flex; flex: 1; overflow: hidden; }}
#canvas-wrap {{ flex: 1; position: relative; overflow: hidden; }}
#graph-svg, #scatter-svg {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; }}
#scatter-svg {{ display: none; }}

#sidebar {{ width: 300px; background: #1a1d27; border-left: 1px solid #2d3748; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }}
#sidebar-title {{ padding: 10px 14px; font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid #2d3748; }}
#node-detail {{ padding: 12px; overflow-y: auto; flex: 1; font-size: 12px; }}
.placeholder {{ color: #4a5568; font-size: 12px; text-align: center; margin-top: 40px; }}

.detail-label {{ font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin: 10px 0 3px; }}
.detail-value {{ color: #e2e8f0; word-break: break-word; }}
.detail-value.highlight {{ color: #a78bfa; font-weight: 700; }}
.kv-row {{ display: flex; gap: 6px; margin-bottom: 3px; align-items: flex-start; }}
.kv-key {{ background: #2d3748; border-radius: 3px; padding: 1px 5px; font-size: 10px; color: #94a3b8; white-space: nowrap; flex-shrink: 0; }}
.kv-val {{ font-size: 11px; color: #cbd5e1; word-break: break-all; }}

.sim-item {{ background: #1e293b; border-radius: 5px; padding: 6px 8px; margin-bottom: 4px; cursor: pointer; border: 1px solid transparent; transition: border-color 0.15s; }}
.sim-item:hover {{ border-color: #a78bfa; }}
.sim-score {{ color: #22d3ee; font-size: 10px; font-weight: 700; }}
.sim-label {{ color: #cbd5e1; font-size: 11px; }}

.action-row {{ display: flex; gap: 6px; margin-top: 12px; }}
.btn-edit {{ flex: 1; background: #1e3a5f; border: 1px solid #3b82f6; border-radius: 5px; color: #60a5fa; padding: 6px; font-size: 11px; cursor: pointer; font-weight: 600; }}
.btn-edit:hover {{ background: #2563eb; color: #fff; }}
.btn-delete {{ flex: 1; background: #3b0f0f; border: 1px solid #ef4444; border-radius: 5px; color: #f87171; padding: 6px; font-size: 11px; cursor: pointer; font-weight: 600; }}
.btn-delete:hover {{ background: #dc2626; color: #fff; }}

/* Edit modal */
#edit-modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 1000; align-items: center; justify-content: center; }}
#edit-modal.show {{ display: flex; }}
#edit-box {{ background: #1a1d27; border: 1px solid #4a5568; border-radius: 10px; padding: 20px; width: 420px; max-height: 80vh; overflow-y: auto; }}
#edit-box h2 {{ color: #a78bfa; font-size: 14px; margin-bottom: 14px; }}
.edit-field {{ margin-bottom: 12px; }}
.edit-field label {{ display: block; font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
.edit-field input, .edit-field textarea {{ width: 100%; background: #2d3748; border: 1px solid #4a5568; border-radius: 5px; color: #e2e8f0; padding: 6px 10px; font-size: 12px; font-family: inherit; }}
.edit-field input:focus, .edit-field textarea:focus {{ outline: none; border-color: #a78bfa; }}
.edit-field textarea {{ resize: vertical; min-height: 60px; }}
#attrs-edit {{ font-size: 11px; }}
.attr-row {{ display: flex; gap: 4px; margin-bottom: 4px; }}
.attr-row input {{ flex: 1; }}
.btn-save {{ background: #a78bfa; border: none; border-radius: 5px; color: #fff; padding: 8px 20px; font-size: 12px; font-weight: 700; cursor: pointer; width: 100%; margin-top: 10px; }}
.btn-save:hover {{ background: #8b5cf6; }}
.btn-cancel {{ background: #2d3748; border: none; border-radius: 5px; color: #94a3b8; padding: 8px 20px; font-size: 12px; cursor: pointer; width: 100%; margin-top: 6px; }}
.toast {{ position: fixed; bottom: 20px; right: 20px; background: #22d3ee; color: #0f1117; border-radius: 6px; padding: 8px 16px; font-size: 12px; font-weight: 700; z-index: 2000; display: none; }}
.toast.error {{ background: #ef4444; color: #fff; }}

/* graph */
.node circle {{ cursor: pointer; stroke-width: 1.5; }}
.node circle:hover {{ filter: brightness(1.4); }}
.node.selected circle {{ stroke: #fff !important; stroke-width: 3; }}
.node.highlighted circle {{ stroke: #22d3ee; stroke-width: 2.5; }}
.node text {{ pointer-events: none; font-size: 9px; fill: #94a3b8; text-anchor: middle; }}
.link {{ fill: none; stroke-opacity: 0.3; }}

/* scatter */
.dot {{ cursor: pointer; }}
.dot:hover {{ opacity: 1 !important; }}
.dot.selected {{ stroke: #fff; stroke-width: 2; }}
.dot.highlighted {{ stroke: #22d3ee; stroke-width: 2; opacity: 1 !important; }}
#scatter-tooltip {{ position: absolute; background: #1a1d27; border: 1px solid #4a5568; border-radius: 6px; padding: 8px 10px; font-size: 11px; pointer-events: none; z-index: 50; display: none; max-width: 240px; }}

#stats {{ position: absolute; bottom: 10px; left: 10px; background: rgba(26,29,39,0.9); border: 1px solid #2d3748; border-radius: 6px; padding: 5px 10px; font-size: 10px; color: #64748b; pointer-events: none; }}
</style>
</head>
<body>
<div id="header">
  <h1>⚡ Feather DB Inspector</h1>
  <span class="badge" id="stat-badge">— nodes</span>
  <button class="tab-btn active" id="tab-graph" onclick="switchTab('graph')">Force Graph</button>
  <button class="tab-btn" id="tab-scatter" onclick="switchTab('scatter')">Embedding Space (PCA)</button>
  <div id="search-wrap">
    <input id="search-input" type="text" placeholder="Search nodes…" oninput="onSearch(this.value)">
    <button id="btn-reset" onclick="resetView()">Reset</button>
  </div>
</div>

<div id="main">
  <div id="canvas-wrap">
    <svg id="graph-svg"></svg>
    <svg id="scatter-svg"></svg>
    <div id="scatter-tooltip"></div>
    <div id="stats">scroll to zoom · drag to pan · click to inspect</div>
  </div>
  <div id="sidebar">
    <div id="sidebar-title">Node Inspector</div>
    <div id="node-detail"><div class="placeholder">Click a node to inspect</div></div>
  </div>
</div>

<div id="edit-modal">
  <div id="edit-box">
    <h2>Edit Record</h2>
    <div class="edit-field"><label>Content / Label</label><textarea id="edit-content" rows="3"></textarea></div>
    <div class="edit-field"><label>Importance (0.0 – 1.0)</label><input id="edit-importance" type="number" min="0" max="1" step="0.01"></div>
    <div class="edit-field"><label>Source</label><input id="edit-source" type="text"></div>
    <div class="edit-field">
      <label>Attributes</label>
      <div id="attrs-edit"></div>
    </div>
    <button class="btn-save" onclick="saveEdit()">Save Changes</button>
    <button class="btn-cancel" onclick="closeEdit()">Cancel</button>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const GRAPH_DATA  = {json.dumps(graph_data, separators=(',', ':'))};
const PCA_MAP     = {json.dumps(pca_map,    separators=(',', ':'))};
const SIMILAR_MAP = {json.dumps(similar_map, separators=(',', ':'))};

const NODE_COLORS = {{ 0:"#60a5fa", 1:"#fbbf24", 2:"#34d399", 3:"#f472b6" }};
const REL_COLORS  = {{
  related_to:"#475569", derived_from:"#a78bfa", caused_by:"#f97316",
  contradicts:"#f43f5e", supports:"#22d3ee", precedes:"#facc15",
  part_of:"#86efac", references:"#c4b5fd", multimodal_of:"#fb923c"
}};
function relColor(r) {{ return REL_COLORS[r] || "#6b7280"; }}
function nodeColor(n) {{ return NODE_COLORS[n.type] || "#60a5fa"; }}
function nodeRadius(n) {{
  const s = 1 + Math.log(1 + (n.recall_count || 0));
  return Math.max(5, Math.min(18, 4 + n.importance * 7 * s * 0.4));
}}

let currentTab = 'graph';
let selectedId = null;

document.getElementById('stat-badge').textContent =
  `${{GRAPH_DATA.nodes.length}} nodes · ${{GRAPH_DATA.edges.length}} edges`;

// ── Tab switching ────────────────────────────────────────────────
function switchTab(tab) {{
  currentTab = tab;
  document.getElementById('tab-graph').classList.toggle('active', tab === 'graph');
  document.getElementById('tab-scatter').classList.toggle('active', tab === 'scatter');
  document.getElementById('graph-svg').style.display   = tab === 'graph'   ? '' : 'none';
  document.getElementById('scatter-svg').style.display = tab === 'scatter' ? '' : 'none';
  if (tab === 'scatter') initScatter();
}}

// ═══════════════════════════════════════════════════════════════
// FORCE GRAPH
// ═══════════════════════════════════════════════════════════════
window.addEventListener('load', function() {{
  const svg   = d3.select('#graph-svg');
  const W = window.innerWidth - 300, H = window.innerHeight - 50;
  const g = svg.append('g');

  const nodeIdSet = new Set(GRAPH_DATA.nodes.map(n => n.id));
  const nodes = GRAPH_DATA.nodes.map(n => ({{...n}}));
  const edges = GRAPH_DATA.edges
    .filter(e => nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
    .map(e => ({{...e}}));

  // Spread initial positions
  const spread = Math.sqrt(nodes.length) * 55;
  nodes.forEach((n, i) => {{
    const a = (i / nodes.length) * 2 * Math.PI;
    n.x = W/2 + spread * Math.cos(a) * (0.5 + Math.random()*0.5);
    n.y = H/2 + spread * Math.sin(a) * (0.5 + Math.random()*0.5);
  }});

  const defs = svg.append('defs');
  [...new Set(edges.map(e => e.rel_type))].forEach(rel => {{
    defs.append('marker')
      .attr('id', 'arr-' + rel.replace(/[^a-z0-9]/g,'_'))
      .attr('viewBox','0 -4 8 8').attr('refX',16).attr('refY',0)
      .attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
      .append('path').attr('d','M0,-4L8,0L0,4')
      .attr('fill', relColor(rel)).attr('fill-opacity',0.7);
  }});

  const sim = d3.forceSimulation(nodes)
    .alphaDecay(0.04)
    .force('link',   d3.forceLink(edges).id(d=>d.id).distance(65).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-120).distanceMax(300))
    .force('center', d3.forceCenter(W/2, H/2).strength(0.05))
    .force('collide',d3.forceCollide().radius(d=>nodeRadius(d)+4));

  const link = g.append('g').selectAll('line').data(edges).join('line')
    .attr('class','link')
    .attr('stroke', d=>relColor(d.rel_type))
    .attr('stroke-width', d=>Math.max(0.5, 0.5+d.weight*2))
    .attr('marker-end', d=>`url(#arr-${{d.rel_type.replace(/[^a-z0-9]/g,'_')}})`);

  const nodeG = g.append('g').selectAll('g').data(nodes).join('g')
    .attr('class','node')
    .call(d3.drag()
      .on('start',(ev,d)=>{{ if(!ev.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
      .on('drag', (ev,d)=>{{ d.fx=ev.x; d.fy=ev.y; }})
      .on('end',  (ev,d)=>{{ if(!ev.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}));

  nodeG.append('circle')
    .attr('r', nodeRadius)
    .attr('fill', nodeColor)
    .attr('stroke', d=>d3.color(nodeColor(d)).darker(0.8));

  nodeG.append('text')
    .attr('dy', d=>nodeRadius(d)+11)
    .text(d=>(d.label||String(d.id)).substring(0,18));

  nodeG.on('click', (ev, d) => {{
    nodeG.classed('selected', false);
    d3.select(ev.currentTarget).classed('selected', true);
    selectedId = d.id;
    highlightSimilar(d.id, nodeG);
    showDetail(d);
    ev.stopPropagation();
  }});
  svg.on('click', ()=>{{ nodeG.classed('selected highlighted', false); selectedId=null; }});

  const zoom = d3.zoom().scaleExtent([0.03,4])
    .on('zoom', ev=>g.attr('transform',ev.transform));
  svg.call(zoom);

  sim.on('tick', ()=>{{
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
        .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    nodeG.attr('transform', d=>`translate(${{d.x}},${{d.y}})`);
  }});

  window._graphSim   = sim;
  window._graphNodeG = nodeG;
  window._graphZoom  = zoom;
  window._graphSvg   = svg;

  document.getElementById('btn-reset').onclick = () => {{
    if (currentTab === 'graph') {{
      svg.transition().duration(400)
        .call(zoom.transform, d3.zoomIdentity.translate(W/2,H/2).scale(0.7));
    }}
  }};
}});

// ── Highlight similar nodes on selection ───────────────────────
function highlightSimilar(id, nodeG) {{
  const similar = SIMILAR_MAP[String(id)] || [];
  const simIds  = new Set(similar.map(s=>s.id));
  if (!nodeG) nodeG = window._graphNodeG;
  if (nodeG) {{
    nodeG.classed('highlighted', false);
    nodeG.filter(d=>simIds.has(d.id)).classed('highlighted', true);
  }}
  // Also highlight in scatter
  if (window._scatterDots) {{
    window._scatterDots.classed('highlighted', false);
    window._scatterDots.filter(d=>simIds.has(d.id) || d.id===id).classed('highlighted', true);
  }}
}}

// ═══════════════════════════════════════════════════════════════
// PCA SCATTER PLOT
// ═══════════════════════════════════════════════════════════════
let scatterInited = false;
function initScatter() {{
  if (scatterInited) return;
  scatterInited = true;

  const svg = d3.select('#scatter-svg');
  const W = window.innerWidth - 300, H = window.innerHeight - 50;
  const PAD = 40;

  const nodesWithPCA = GRAPH_DATA.nodes.filter(n => PCA_MAP[String(n.id)]);
  const xArr = nodesWithPCA.map(n=>PCA_MAP[String(n.id)][0]);
  const yArr = nodesWithPCA.map(n=>PCA_MAP[String(n.id)][1]);

  const xScale = d3.scaleLinear().domain([d3.min(xArr), d3.max(xArr)]).range([PAD, W-PAD]);
  const yScale = d3.scaleLinear().domain([d3.min(yArr), d3.max(yArr)]).range([H-PAD, PAD]);

  const g = svg.append('g');

  // Axes (subtle)
  g.append('line').attr('x1',PAD).attr('x2',W-PAD).attr('y1',H/2).attr('y2',H/2)
    .attr('stroke','#2d3748').attr('stroke-width',0.5);
  g.append('line').attr('x1',W/2).attr('x2',W/2).attr('y1',PAD).attr('y2',H-PAD)
    .attr('stroke','#2d3748').attr('stroke-width',0.5);
  g.append('text').attr('x',W-PAD+4).attr('y',H/2+4).attr('fill','#4a5568').attr('font-size',10).text('PC1');
  g.append('text').attr('x',W/2+4).attr('y',PAD-6).attr('fill','#4a5568').attr('font-size',10).text('PC2');

  const dots = g.append('g').selectAll('circle').data(nodesWithPCA).join('circle')
    .attr('class','dot')
    .attr('cx', d=>xScale(PCA_MAP[String(d.id)][0]))
    .attr('cy', d=>yScale(PCA_MAP[String(d.id)][1]))
    .attr('r',  d=>nodeRadius(d))
    .attr('fill', nodeColor)
    .attr('fill-opacity', 0.75)
    .attr('stroke', 'none');

  const ttip = document.getElementById('scatter-tooltip');

  dots.on('mouseover', (ev, d) => {{
    ttip.style.display = 'block';
    ttip.innerHTML = `<b>${{(d.label||'id:'+d.id).substring(0,50)}}</b><br>
      imp=${{d.importance.toFixed(2)}} recalls=${{d.recall_count}}<br>
      ${{d.attributes?.record_type || ''}} ${{d.attributes?.roas ? 'roas='+d.attributes.roas : ''}}`;
    ttip.style.left = (ev.clientX+12)+'px';
    ttip.style.top  = (ev.clientY-10)+'px';
  }}).on('mousemove', ev=>{{
    ttip.style.left=(ev.clientX+12)+'px'; ttip.style.top=(ev.clientY-10)+'px';
  }}).on('mouseout', ()=>{{ ttip.style.display='none'; }});

  dots.on('click', (ev, d) => {{
    dots.classed('selected', false);
    d3.select(ev.currentTarget).classed('selected', true);
    selectedId = d.id;
    highlightSimilar(d.id, window._graphNodeG);
    showDetail(d);
    ev.stopPropagation();
  }});

  svg.on('click', ()=>{{ dots.classed('selected highlighted', false); selectedId=null; }});

  const zoom = d3.zoom().scaleExtent([0.1,10])
    .on('zoom', ev=>g.attr('transform',ev.transform));
  svg.call(zoom);

  window._scatterDots = dots;
}}

// ═══════════════════════════════════════════════════════════════
// SIDEBAR DETAIL
// ═══════════════════════════════════════════════════════════════
function showDetail(d) {{
  const attrs = Object.entries(d.attributes||{{}})
    .map(([k,v])=>`<div class="kv-row"><span class="kv-key">${{k}}</span><span class="kv-val">${{v}}</span></div>`)
    .join('');

  const similar = (SIMILAR_MAP[String(d.id)]||[]).slice(0,8);
  const simHtml = similar.map(s=>{{
    const nd = GRAPH_DATA.nodes.find(n=>n.id===s.id);
    const lbl = nd ? (nd.label||'id:'+s.id).substring(0,40) : 'id:'+s.id;
    return `<div class="sim-item" onclick="jumpToNode(${{s.id}})">
      <span class="sim-score">${{(s.score*100).toFixed(1)}}%</span>
      <span class="sim-label"> ${{lbl}}</span>
    </div>`;
  }}).join('');

  document.getElementById('node-detail').innerHTML = `
    <div class="detail-label">ID</div>
    <div class="detail-value highlight">${{d.id}}</div>

    <div class="detail-label">Label</div>
    <div class="detail-value">${{(d.label||'—').substring(0,120)}}</div>

    <div class="detail-label">Namespace / Entity</div>
    <div class="detail-value">${{d.namespace_id||'—'}} · ${{d.entity_id||'—'}}</div>

    <div class="detail-label">Importance · Recalls</div>
    <div class="detail-value">${{d.importance.toFixed(3)}} · ${{d.recall_count}}</div>

    <div class="detail-label">Attributes</div>
    <div class="detail-value">${{attrs||'—'}}</div>

    ${{simHtml.length ? `<div class="detail-label">Top Similar Nodes</div>${{simHtml}}` : ''}}

    <div class="action-row">
      <button class="btn-edit" onclick="openEdit(${{d.id}})">✏️ Edit</button>
      <button class="btn-delete" onclick="deleteNode(${{d.id}})">🗑 Delete</button>
    </div>
  `;
}}

function jumpToNode(id) {{
  const nd = GRAPH_DATA.nodes.find(n=>n.id===id);
  if (!nd) return;
  selectedId = id;
  showDetail(nd);
  highlightSimilar(id);
  if (window._graphNodeG) {{
    window._graphNodeG.classed('selected', false);
    window._graphNodeG.filter(d=>d.id===id).classed('selected', true);
  }}
  if (window._scatterDots) {{
    window._scatterDots.classed('selected', false);
    window._scatterDots.filter(d=>d.id===id).classed('selected', true);
  }}
}}

// ═══════════════════════════════════════════════════════════════
// SEARCH
// ═══════════════════════════════════════════════════════════════
function onSearch(q) {{
  q = q.toLowerCase().trim();
  if (window._graphNodeG) {{
    window._graphNodeG.selectAll('circle').attr('opacity', d=>{{
      if (!q) return 1;
      return (String(d.id).includes(q) ||
              (d.label||'').toLowerCase().includes(q) ||
              Object.values(d.attributes||{{}}).join(' ').toLowerCase().includes(q)) ? 1 : 0.08;
    }});
  }}
  if (window._scatterDots) {{
    window._scatterDots.attr('fill-opacity', d=>{{
      if (!q) return 0.75;
      return (String(d.id).includes(q) ||
              (d.label||'').toLowerCase().includes(q) ||
              Object.values(d.attributes||{{}}).join(' ').toLowerCase().includes(q)) ? 1 : 0.06;
    }});
  }}
}}

function resetView() {{
  document.getElementById('search-input').value='';
  onSearch('');
  if (window._graphNodeG) window._graphNodeG.classed('selected highlighted', false);
  if (window._scatterDots) window._scatterDots.classed('selected highlighted', false);
  selectedId = null;
}}

// ═══════════════════════════════════════════════════════════════
// EDIT
// ═══════════════════════════════════════════════════════════════
let _editId = null;
function openEdit(id) {{
  _editId = id;
  const nd = GRAPH_DATA.nodes.find(n=>n.id===id);
  if (!nd) return;
  document.getElementById('edit-content').value    = nd.label || '';
  document.getElementById('edit-importance').value = nd.importance.toFixed(3);
  document.getElementById('edit-source').value     = nd.source || '';
  const attrsDiv = document.getElementById('attrs-edit');
  attrsDiv.innerHTML = Object.entries(nd.attributes||{{}}).map(([k,v])=>`
    <div class="attr-row">
      <input type="text" value="${{k}}" placeholder="key" class="attr-key" style="max-width:110px">
      <input type="text" value="${{v}}" placeholder="value" class="attr-val">
    </div>`).join('');
  document.getElementById('edit-modal').classList.add('show');
}}

function closeEdit() {{ document.getElementById('edit-modal').classList.remove('show'); }}

async function saveEdit() {{
  const content    = document.getElementById('edit-content').value;
  const importance = parseFloat(document.getElementById('edit-importance').value);
  const source     = document.getElementById('edit-source').value;
  const attrRows   = document.querySelectorAll('#attrs-edit .attr-row');
  const attributes = {{}};
  attrRows.forEach(row=>{{
    const k = row.querySelector('.attr-key').value.trim();
    const v = row.querySelector('.attr-val').value.trim();
    if (k) attributes[k]=v;
  }});
  try {{
    const resp = await fetch('/api/edit', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{ id:_editId, content, importance, source, attributes }})
    }});
    const result = await resp.json();
    if (result.ok) {{
      // Update local data
      const nd = GRAPH_DATA.nodes.find(n=>n.id===_editId);
      if (nd) {{ nd.label=content; nd.importance=importance; nd.source=source; nd.attributes=attributes; }}
      showToast('Saved ✓');
      closeEdit();
      if (selectedId===_editId) showDetail(nd);
    }} else {{ showToast('Error: '+result.error, true); }}
  }} catch(e) {{ showToast('Server error: '+e.message, true); }}
}}

// ═══════════════════════════════════════════════════════════════
// DELETE
// ═══════════════════════════════════════════════════════════════
async function deleteNode(id) {{
  if (!confirm(`Delete record id=${{id}}? This will set importance=0 and mark it inactive.`)) return;
  try {{
    const resp = await fetch('/api/delete', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{id}})
    }});
    const result = await resp.json();
    if (result.ok) {{
      // Visually remove from graph
      if (window._graphNodeG) window._graphNodeG.filter(d=>d.id===id).remove();
      if (window._scatterDots) window._scatterDots.filter(d=>d.id===id).remove();
      // Remove from GRAPH_DATA
      const idx = GRAPH_DATA.nodes.findIndex(n=>n.id===id);
      if (idx>=0) GRAPH_DATA.nodes.splice(idx,1);
      document.getElementById('node-detail').innerHTML='<div class="placeholder">Node deleted</div>';
      showToast('Deleted ✓');
    }} else {{ showToast('Error: '+result.error, true); }}
  }} catch(e) {{ showToast('Server error: '+e.message, true); }}
}}

// ═══════════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════════
function showToast(msg, isError=false) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError?' error':'');
  t.style.display = 'block';
  setTimeout(()=>{{ t.style.display='none'; }}, 2500);
}}
</script>
</body>
</html>"""


# ─── HTTP Server ──────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass   # suppress access logs

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            body = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path    = urlparse(self.path).path
        length  = int(self.headers.get('Content-Length', 0))
        body    = json.loads(self.rfile.read(length))

        if path == '/api/edit':
            try:
                nid  = int(body['id'])
                meta = db.get_metadata(nid)
                if meta is None:
                    self.send_json({'ok': False, 'error': 'id not found'}); return

                meta.content    = body.get('content', meta.content)
                meta.importance = float(body.get('importance', meta.importance))
                meta.source     = body.get('source', meta.source)
                for k, v in body.get('attributes', {}).items():
                    meta.set_attribute(k, v)

                db.update_metadata(nid, meta)
                db.save()
                self.send_json({'ok': True})
            except Exception as e:
                self.send_json({'ok': False, 'error': str(e)})

        elif path == '/api/delete':
            try:
                nid = int(body['id'])
                # Mark as deleted: importance=0, attribute deleted=true
                db.update_importance(nid, 0.0)
                meta = db.get_metadata(nid)
                if meta:
                    meta.set_attribute('deleted', 'true')
                    db.update_metadata(nid, meta)
                db.save()
                self.send_json({'ok': True})
            except Exception as e:
                self.send_json({'ok': False, 'error': str(e)})

        else:
            self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# ─── Launch ───────────────────────────────────────────────────────
server = HTTPServer(('localhost', PORT), Handler)
url = f'http://localhost:{PORT}'
print(f"\n  Server running at {url}")
print("  Features:")
print("    • Force Graph — full context graph")
print("    • Embedding Space (PCA) — 2D projection of all vectors")
print("    • Click any node → see top-8 similar nodes highlighted")
print("    • Edit button → update content, importance, attributes")
print("    • Delete button → deactivate record")
print("\n  Press Ctrl+C to stop.\n")

threading.Timer(0.8, lambda: webbrowser.open(url)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n  Server stopped.")
