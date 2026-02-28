"""
feather_db.graph — Context graph export and visualization.

Usage:
    from feather_db.graph import visualize, export_graph, RelType

    # Export graph data as Python dict (D3 / Cytoscape-compatible)
    graph = export_graph(db, namespace_filter="nike")

    # Generate self-contained HTML with interactive D3 force graph
    path = visualize(db, output_path="context_graph.html", namespace_filter="nike")
    # → open context_graph.html in any browser, no server needed
"""

import json
import os
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Relationship type constants
# ─────────────────────────────────────────────────────────────
class RelType:
    RELATED_TO   = "related_to"
    DERIVED_FROM = "derived_from"
    CAUSED_BY    = "caused_by"
    CONTRADICTS  = "contradicts"
    SUPPORTS     = "supports"
    PRECEDES     = "precedes"
    PART_OF      = "part_of"
    REFERENCES   = "references"
    MULTIMODAL_OF = "multimodal_of"


# ─────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────
def export_graph(db, namespace_filter: str = "", entity_filter: str = "") -> dict:
    """
    Export the context graph as a Python dict with 'nodes' and 'edges' lists.
    Compatible with D3.js, Cytoscape.js, vis.js, and NetworkX.
    """
    raw = db.export_graph_json(namespace_filter, entity_filter)
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────
# HTML Visualizer — self-contained, no server needed
# ─────────────────────────────────────────────────────────────
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feather DB — Context Graph</title>
<script>__D3_INLINE__</script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }

  #header { padding: 12px 20px; background: #1a1d27; border-bottom: 1px solid #2d3748; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }
  #header h1 { font-size: 15px; font-weight: 600; color: #a78bfa; letter-spacing: 0.5px; }
  #header .badge { background: #2d3748; border-radius: 4px; padding: 2px 8px; font-size: 11px; color: #94a3b8; }
  #controls { display: flex; gap: 8px; margin-left: auto; align-items: center; }
  #controls input { background: #2d3748; border: 1px solid #4a5568; border-radius: 6px; color: #e2e8f0; padding: 4px 10px; font-size: 12px; width: 180px; }
  #controls input:focus { outline: none; border-color: #a78bfa; }
  #controls button { background: #2d3748; border: 1px solid #4a5568; border-radius: 6px; color: #e2e8f0; padding: 4px 12px; font-size: 12px; cursor: pointer; }
  #controls button:hover { background: #4a5568; }

  #main { display: flex; flex: 1; overflow: hidden; }
  #canvas-wrap { flex: 1; position: relative; overflow: hidden; }
  svg { width: 100%; height: 100%; }

  #sidebar { width: 280px; background: #1a1d27; border-left: 1px solid #2d3748; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
  #sidebar-title { padding: 12px 14px; font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid #2d3748; }
  #node-detail { padding: 14px; overflow-y: auto; flex: 1; }
  #node-detail .placeholder { color: #4a5568; font-size: 12px; text-align: center; margin-top: 40px; }
  .detail-row { margin-bottom: 10px; }
  .detail-label { font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 3px; }
  .detail-value { font-size: 12px; color: #e2e8f0; word-break: break-word; }
  .detail-value.highlight { color: #a78bfa; font-weight: 600; }
  .kv-pair { display: flex; gap: 6px; margin-bottom: 4px; }
  .kv-key { background: #2d3748; border-radius: 3px; padding: 1px 6px; font-size: 10px; color: #94a3b8; }
  .kv-val { font-size: 11px; color: #e2e8f0; }
  .edge-item { background: #2d3748; border-radius: 4px; padding: 6px 8px; margin-bottom: 5px; font-size: 11px; }
  .edge-rel { color: #f59e0b; font-weight: 600; font-size: 10px; }
  .edge-target { color: #a78bfa; }
  .badge-score { display: inline-block; background: #1e3a5f; color: #60a5fa; border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600; }

  #legend { padding: 12px 14px; border-top: 1px solid #2d3748; font-size: 11px; }
  #legend-title { color: #64748b; text-transform: uppercase; letter-spacing: 0.6px; font-size: 10px; font-weight: 600; margin-bottom: 8px; }
  .legend-row { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .legend-line { width: 20px; height: 3px; border-radius: 2px; flex-shrink: 0; }
  .legend-text { color: #94a3b8; font-size: 11px; }

  /* Graph elements */
  .node circle { cursor: pointer; stroke-width: 2; transition: filter 0.15s; }
  .node circle:hover { filter: brightness(1.3); }
  .node.selected circle { stroke: #fff !important; stroke-width: 3; }
  .node text { pointer-events: none; font-size: 9px; fill: #cbd5e1; text-anchor: middle; }

  .link { stroke-opacity: 0.6; fill: none; }
  .link-label { font-size: 8px; fill: #64748b; pointer-events: none; }

  marker { overflow: visible; }

  .tooltip { position: absolute; background: #1a1d27; border: 1px solid #4a5568; border-radius: 6px; padding: 8px 10px; font-size: 11px; pointer-events: none; max-width: 220px; z-index: 100; display: none; }
  .tooltip .tt-label { font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
  .tooltip .tt-row { color: #94a3b8; }

  #stats { position: absolute; bottom: 10px; left: 10px; background: rgba(26,29,39,0.85); border: 1px solid #2d3748; border-radius: 6px; padding: 6px 10px; font-size: 10px; color: #64748b; pointer-events: none; }
</style>
</head>
<body>

<div id="header">
  <h1>⚡ Feather DB — Context Graph</h1>
  <span class="badge" id="stat-badge">0 nodes · 0 edges</span>
  <div id="controls">
    <input id="search-input" type="text" placeholder="Search nodes…">
    <button id="btn-reset">Reset zoom</button>
    <button id="btn-auto">Auto-layout</button>
  </div>
</div>

<div id="main">
  <div id="canvas-wrap">
    <svg id="graph-svg"></svg>
    <div class="tooltip" id="tooltip"></div>
    <div id="stats">scroll to zoom · drag to pan · click node to inspect</div>
  </div>

  <div id="sidebar">
    <div id="sidebar-title">Node Inspector</div>
    <div id="node-detail">
      <div class="placeholder">Click a node to inspect</div>
    </div>
    <div id="legend">
      <div id="legend-title">Legend</div>
      <div class="legend-row"><div class="legend-dot" style="background:#60a5fa"></div><div class="legend-text">FACT</div></div>
      <div class="legend-row"><div class="legend-dot" style="background:#34d399"></div><div class="legend-text">EVENT</div></div>
      <div class="legend-row"><div class="legend-dot" style="background:#fbbf24"></div><div class="legend-text">PREFERENCE</div></div>
      <div class="legend-row"><div class="legend-dot" style="background:#f472b6"></div><div class="legend-text">CONVERSATION</div></div>
      <div style="margin: 8px 0 6px; border-top: 1px solid #2d3748;"></div>
      <div id="edge-legend"></div>
    </div>
  </div>
</div>

<script>
const GRAPH_DATA = __GRAPH_DATA__;

window.addEventListener("load", function() {

// ── Colour maps ───────────────────────────────────────────────────
const NODE_COLORS = { 0: "#60a5fa", 1: "#fbbf24", 2: "#34d399", 3: "#f472b6" };
const NODE_TYPE_LABELS = { 0: "FACT", 1: "PREFERENCE", 2: "EVENT", 3: "CONVERSATION" };
const REL_COLORS = {
  "related_to":    "#94a3b8",
  "derived_from":  "#a78bfa",
  "caused_by":     "#f97316",
  "contradicts":   "#f43f5e",
  "supports":      "#22d3ee",
  "precedes":      "#facc15",
  "part_of":       "#86efac",
  "references":    "#c4b5fd",
  "multimodal_of": "#fb923c",
};
function relColor(rel) { return REL_COLORS[rel] || "#6b7280"; }

// ── Edge legend ───────────────────────────────────────────────────
const usedRels = [...new Set(GRAPH_DATA.edges.map(e => e.rel_type))];
const legendEl = document.getElementById("edge-legend");
usedRels.forEach(rel => {
  const row = document.createElement("div"); row.className = "legend-row";
  row.innerHTML = `<div class="legend-line" style="background:${relColor(rel)}"></div>
                   <div class="legend-text">${rel}</div>`;
  legendEl.appendChild(row);
});

// ── Stat badge ────────────────────────────────────────────────────
document.getElementById("stat-badge").textContent =
  `${GRAPH_DATA.nodes.length} nodes · ${GRAPH_DATA.edges.length} edges`;

// ── Build D3 force graph ──────────────────────────────────────────
const svg = d3.select("#graph-svg");
// Use window dimensions — clientWidth can be 0 if layout not yet painted
const width  = window.innerWidth  - 280;  // subtract sidebar width
const height = window.innerHeight - 50;   // subtract header height

const g = svg.append("g");

// Arrow markers per rel_type
const defs = svg.append("defs");
usedRels.forEach(rel => {
  defs.append("marker")
    .attr("id", "arrow-" + rel.replace(/[^a-z0-9]/g, "_"))
    .attr("viewBox", "0 -4 8 8")
    .attr("refX", 18).attr("refY", 0)
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-4L8,0L0,4")
    .attr("fill", relColor(rel))
    .attr("fill-opacity", 0.7);
});

// Deep-copy nodes for D3 (it mutates them)
const nodes = GRAPH_DATA.nodes.map(n => ({ ...n }));
// Filter edges: drop any where source or target node is missing
const nodeIdSet = new Set(GRAPH_DATA.nodes.map(n => n.id));
const edges = GRAPH_DATA.edges
  .filter(e => nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
  .map(e => ({ ...e }));

// Spread nodes initially so force sim converges faster
const isLarge = nodes.length > 100;
const spread  = isLarge ? Math.sqrt(nodes.length) * 60 : 200;
nodes.forEach((n, i) => {
  const angle = (i / nodes.length) * 2 * Math.PI;
  n.x = width / 2  + spread * Math.cos(angle) * (0.5 + Math.random() * 0.5);
  n.y = height / 2 + spread * Math.sin(angle) * (0.5 + Math.random() * 0.5);
});

// Node size: importance × stickiness factor
function nodeRadius(n) {
  const sticky = 1 + Math.log(1 + (n.recall_count || 0));
  return Math.max(5, Math.min(20, 4 + n.importance * 7 * sticky * 0.4));
}

const linkDist  = isLarge ? 60  : 90;
const charge    = isLarge ? -120 : -280;
const alphaDecay = isLarge ? 0.04 : 0.0228;

const simulation = d3.forceSimulation(nodes)
  .alphaDecay(alphaDecay)
  .force("link",   d3.forceLink(edges).id(d => d.id).distance(linkDist).strength(0.3))
  .force("charge", d3.forceManyBody().strength(charge).distanceMax(300))
  .force("center", d3.forceCenter(width / 2, height / 2).strength(0.05))
  .force("collide", d3.forceCollide().radius(d => nodeRadius(d) + 4));

// Links
const link = g.append("g").selectAll("line")
  .data(edges).join("line")
  .attr("class", "link")
  .attr("stroke", d => relColor(d.rel_type))
  .attr("stroke-width", d => Math.max(0.5, 0.5 + d.weight * 2))
  .attr("stroke-opacity", isLarge ? 0.35 : 0.6)
  .attr("marker-end", d => `url(#arrow-${d.rel_type.replace(/[^a-z0-9]/g, "_")})`);

// Link labels — only shown when graph is small or on hover
const linkLabel = g.append("g").selectAll("text")
  .data(isLarge ? [] : edges).join("text")
  .attr("class", "link-label")
  .text(d => d.rel_type.replace(/_/g, " "));

// Node groups
const nodeG = g.append("g").selectAll("g")
  .data(nodes).join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on("end",   (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

nodeG.append("circle")
  .attr("r", nodeRadius)
  .attr("fill", d => NODE_COLORS[d.type] || "#60a5fa")
  .attr("stroke", d => d3.color(NODE_COLORS[d.type] || "#60a5fa").darker(0.8));

nodeG.append("text")
  .attr("dy", d => nodeRadius(d) + 11)
  .text(d => (d.label || String(d.id)).substring(0, 20));

// ── Tooltip ───────────────────────────────────────────────────────
const tooltip = document.getElementById("tooltip");
nodeG.on("mouseover", (event, d) => {
  tooltip.style.display = "block";
  tooltip.innerHTML = `<div class="tt-label">${(d.label || "id:" + d.id).substring(0, 50)}</div>
    <div class="tt-row">ns: ${d.namespace_id || "—"} · entity: ${d.entity_id || "—"}</div>
    <div class="tt-row">importance: ${d.importance.toFixed(2)} · recalls: ${d.recall_count}</div>`;
  tooltip.style.left = (event.offsetX + 14) + "px";
  tooltip.style.top  = (event.offsetY - 10) + "px";
}).on("mousemove", (event) => {
  tooltip.style.left = (event.offsetX + 14) + "px";
  tooltip.style.top  = (event.offsetY - 10) + "px";
}).on("mouseout", () => { tooltip.style.display = "none"; });

// ── Node click → sidebar inspector ───────────────────────────────
let selectedNode = null;
nodeG.on("click", (event, d) => {
  nodeG.classed("selected", false);
  d3.select(event.currentTarget).classed("selected", true);
  selectedNode = d.id;
  renderDetail(d);
  event.stopPropagation();
});
svg.on("click", () => { nodeG.classed("selected", false); selectedNode = null; });

function renderDetail(d) {
  const outEdges = edges.filter(e => (e.source.id || e.source) === d.id);
  const inEdges  = edges.filter(e => (e.target.id || e.target) === d.id);

  const attrs = Object.entries(d.attributes || {})
    .map(([k, v]) => `<div class="kv-pair"><span class="kv-key">${k}</span><span class="kv-val">${v}</span></div>`)
    .join("") || "<span style='color:#4a5568;font-size:11px'>none</span>";

  const outHtml = outEdges.map(e => {
    const tid = e.target.id || e.target;
    return `<div class="edge-item"><span class="edge-rel">${e.rel_type}</span> → <span class="edge-target">id:${tid}</span> <span class="badge-score">${e.weight.toFixed(2)}</span></div>`;
  }).join("") || "<span style='color:#4a5568;font-size:11px'>none</span>";

  const inHtml = inEdges.map(e => {
    const sid = e.source.id || e.source;
    return `<div class="edge-item"><span class="edge-target">id:${sid}</span> → <span class="edge-rel">${e.rel_type}</span> <span class="badge-score">${e.weight.toFixed(2)}</span></div>`;
  }).join("") || "<span style='color:#4a5568;font-size:11px'>none</span>";

  document.getElementById("node-detail").innerHTML = `
    <div class="detail-row"><div class="detail-label">ID</div><div class="detail-value highlight">${d.id}</div></div>
    <div class="detail-row"><div class="detail-label">Label</div><div class="detail-value">${d.label || "—"}</div></div>
    <div class="detail-row"><div class="detail-label">Type</div><div class="detail-value">${NODE_TYPE_LABELS[d.type] || d.type}</div></div>
    <div class="detail-row"><div class="detail-label">Namespace</div><div class="detail-value">${d.namespace_id || "—"}</div></div>
    <div class="detail-row"><div class="detail-label">Entity</div><div class="detail-value">${d.entity_id || "—"}</div></div>
    <div class="detail-row"><div class="detail-label">Source</div><div class="detail-value">${d.source || "—"}</div></div>
    <div class="detail-row"><div class="detail-label">Importance</div><div class="detail-value">${d.importance.toFixed(3)}</div></div>
    <div class="detail-row"><div class="detail-label">Recall count</div><div class="detail-value">${d.recall_count}</div></div>
    <div class="detail-row"><div class="detail-label">Attributes</div><div class="detail-value">${attrs}</div></div>
    <div class="detail-row"><div class="detail-label">Outgoing edges (${outEdges.length})</div>${outHtml}</div>
    <div class="detail-row"><div class="detail-label">Incoming edges (${inEdges.length})</div>${inHtml}</div>
  `;
}

// ── Search highlight ──────────────────────────────────────────────
document.getElementById("search-input").addEventListener("input", function() {
  const q = this.value.toLowerCase();
  nodeG.selectAll("circle").attr("opacity", d => {
    if (!q) return 1;
    const hit = String(d.id).includes(q) || (d.label || "").toLowerCase().includes(q) ||
                (d.namespace_id || "").toLowerCase().includes(q) ||
                (d.entity_id || "").toLowerCase().includes(q);
    return hit ? 1 : 0.15;
  });
});

// ── Zoom & pan ────────────────────────────────────────────────────
const zoom = d3.zoom().scaleExtent([0.05, 4])
  .on("zoom", (event) => g.attr("transform", event.transform));
svg.call(zoom);
document.getElementById("btn-reset").onclick = () =>
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(width/2, height/2).scale(0.8));
document.getElementById("btn-auto").onclick = () => {
  simulation.alpha(0.5).restart();
};

// ── Handle window resize ──────────────────────────────────────────
window.addEventListener("resize", () => {
  const w = window.innerWidth - 280;
  const h = window.innerHeight - 50;
  simulation.force("center", d3.forceCenter(w / 2, h / 2));
});

// ── Simulation tick ───────────────────────────────────────────────
simulation.on("tick", () => {
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);

  linkLabel
    .attr("x", d => ((d.source.x + d.target.x) / 2))
    .attr("y", d => ((d.source.y + d.target.y) / 2) - 3);

  nodeG.attr("transform", d => `translate(${d.x},${d.y})`);
});

}); // end window.load
</script>
</body>
</html>"""


def visualize(db,
              output_path: str = "feather_graph.html",
              namespace_filter: str = "",
              entity_filter: str = "",
              title: str = "Feather DB — Context Graph") -> str:
    """
    Generate a self-contained HTML file with an interactive D3 force graph.

    Args:
        db:               Feather DB instance
        output_path:      Where to write the HTML file (default: feather_graph.html)
        namespace_filter: Only include records from this namespace (empty = all)
        entity_filter:    Only include records from this entity (empty = all)
        title:            Page title

    Returns:
        Absolute path to the generated HTML file.
    """
    graph_data = export_graph(db, namespace_filter, entity_filter)
    data_json  = json.dumps(graph_data, separators=(",", ":"))
    # Escape </script> so it can't prematurely close the inline script block
    data_json  = data_json.replace("</", "<\\/")

    # Inline D3 so the HTML works offline (no CDN dependency)
    _d3_paths = [
        os.path.join(os.path.dirname(__file__), "d3.min.js"),
        "/tmp/d3.min.js",
    ]
    d3_src = None
    for p in _d3_paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as _f:
                d3_src = _f.read()
            break
    if d3_src is None:
        # fallback to CDN
        d3_src = ""  # will be replaced with CDN tag below
        _html = _HTML_TEMPLATE.replace(
            "<script>__D3_INLINE__</script>",
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'
        )
    else:
        _html = _HTML_TEMPLATE.replace("__D3_INLINE__", d3_src)

    html = _html.replace("__GRAPH_DATA__", data_json)

    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html)

    node_count = len(graph_data["nodes"])
    edge_count = len(graph_data["edges"])
    print(f"[feather_db] Context graph exported → {abs_path}")
    print(f"             {node_count} nodes  ·  {edge_count} edges")
    return abs_path
