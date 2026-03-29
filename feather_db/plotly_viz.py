"""
feather_db.plotly_viz — Plotly network graph for Feather DB.

Returns a plotly.graph_objects.Figure that works natively in Gradio via gr.Plot(),
Jupyter notebooks, and standalone HTML export. No iframe/embedding issues.

Usage:
    from feather_db.plotly_viz import plot_graph

    fig = plot_graph(db)                        # all nodes
    fig = plot_graph(db, namespace="acme")      # filtered
    fig.show()                                  # standalone browser
    fig.write_html("graph.html")                # export
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — one colour per entity_type attribute value
# Falls back gracefully for unknown types
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_COLORS = {
    # agent logic
    "campaign":           "#a78bfa",   # violet
    "campaign_aggregate": "#7c3aed",   # dark violet
    "signal":             "#38bdf8",   # sky blue
    "trigger":            "#f97316",   # orange
    "root_cause":         "#ef4444",   # red
    "action":             "#22c55e",   # green
    "scoring_band":       "#fbbf24",   # amber
    "operational":        "#94a3b8",   # slate
    # real data
    "ad":                 "#34d399",   # emerald
    # generic feather types
    "fact":               "#60a5fa",
    "preference":         "#fbbf24",
    "event":              "#34d399",
    "conversation":       "#f472b6",
}
_DEFAULT_COLOR = "#64748b"

_SCALING_COLORS = {
    "Highly Favoured":   "#22c55e",
    "Normal":            "#60a5fa",
    "Suppressed":        "#f97316",
    "Highly Suppressed": "#ef4444",
}

_EDGE_COLORS = {
    "caused_by":             "#f97316",
    "resolved_by":           "#22c55e",
    "has_trigger":           "#a78bfa",
    "tracks_signal":         "#38bdf8",
    "weighted_signal":       "#fbbf24",
    "activates_trigger":     "#fb923c",
    "signals_underspend":    "#ef4444",
    "signals_ctr_issue":     "#f472b6",
    "signals_cvr_issue":     "#fb923c",
    "exhibits_fatigue":      "#dc2626",
    "top_performer_risk":    "#facc15",
    "belongs_to_campaign":   "#6b7280",
    "feeds_campaign":        "#8b5cf6",
    "scores_to":             "#94a3b8",
    "governed_by":           "#64748b",
    "correlates_with":       "#475569",
    "related_to":            "#64748b",
    "caused_by":             "#f97316",
}
_DEFAULT_EDGE_COLOR = "#475569"


def _node_color(meta) -> str:
    if meta is None:
        return _DEFAULT_COLOR
    et = meta.get_attribute("entity_type") if hasattr(meta, "get_attribute") else ""
    if et:
        # For ad nodes, colour by scaling index
        if et == "ad":
            si = meta.get_attribute("scaling_index") if hasattr(meta, "get_attribute") else ""
            return _SCALING_COLORS.get(si, _ENTITY_COLORS["ad"])
        return _ENTITY_COLORS.get(et, _DEFAULT_COLOR)
    # Fallback: use ContextType int
    type_map = {0: "#60a5fa", 1: "#fbbf24", 2: "#34d399", 3: "#f472b6"}
    return type_map.get(getattr(meta, "type", None), _DEFAULT_COLOR)  # type: ignore


def _edge_color(rel_type: str) -> str:
    return _EDGE_COLORS.get(rel_type, _DEFAULT_EDGE_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
# Layout — spring (Fruchterman-Reingold) via networkx, circular fallback
# ─────────────────────────────────────────────────────────────────────────────

def _layout(node_ids: list, edges: list) -> dict:
    """Return {node_id: (x, y)} using networkx spring layout."""
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        for e in edges:
            G.add_edge(e["source"], e["target"])
        pos = nx.spring_layout(G, k=2.5 / math.sqrt(max(len(node_ids), 1)),
                               iterations=80, seed=42)
        return {nid: (float(pos[nid][0]), float(pos[nid][1])) for nid in node_ids}
    except ImportError:
        # Circular fallback
        n = len(node_ids)
        return {
            nid: (math.cos(2 * math.pi * i / max(n, 1)),
                  math.sin(2 * math.pi * i / max(n, 1)))
            for i, nid in enumerate(node_ids)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def plot_graph(
    db,
    namespace: str = "",
    entity: str = "",
    title: str = "Feather DB — Context Graph",
    height: int = 720,
    show_edge_labels: bool = True,
    max_edge_label_nodes: int = 80,
) -> "plotly.graph_objects.Figure":  # type: ignore
    """
    Build an interactive Plotly network graph from a Feather DB instance.

    Args:
        db:                    Open Feather DB instance.
        namespace:             Filter to this namespace_id (empty = all).
        entity:                Filter to this entity_id (empty = all).
        title:                 Figure title.
        height:                Figure height in pixels.
        show_edge_labels:      Show rel_type labels on edges (auto-disabled for large graphs).
        max_edge_label_nodes:  Auto-disable edge labels above this node count.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    # ── Pull raw graph data from DB ──────────────────────────────────────────
    raw = db.export_graph_json(namespace, entity)
    import json
    data = json.loads(raw)

    g_nodes = data["nodes"]   # [{id, label, type, importance, ...}, ...]
    g_edges = data["edges"]   # [{source, target, rel_type, weight}, ...]

    if not g_nodes:
        fig = go.Figure()
        fig.update_layout(title="No nodes to display", paper_bgcolor="#0f1117",
                          plot_bgcolor="#0f1117", font_color="#e2e8f0", height=height)
        return fig

    node_ids = [n["id"] for n in g_nodes]
    node_id_set = set(node_ids)

    # Filter edges to only those with both endpoints present
    valid_edges = [e for e in g_edges
                   if e["source"] in node_id_set and e["target"] in node_id_set]

    # ── Compute layout ────────────────────────────────────────────────────────
    pos = _layout(node_ids, valid_edges)

    # ── Build metadata lookup ─────────────────────────────────────────────────
    meta_lookup = {}
    for nid in node_ids:
        meta_lookup[nid] = db.get_metadata(nid)

    # ── Edge traces — one trace per rel_type for legend grouping ─────────────
    from collections import defaultdict
    edges_by_rel: dict = defaultdict(list)
    for e in valid_edges:
        edges_by_rel[e["rel_type"]].append(e)

    traces = []
    show_labels = show_edge_labels and len(g_nodes) <= max_edge_label_nodes

    for rel_type, rel_edges in edges_by_rel.items():
        color = _edge_color(rel_type)
        ex, ey, label_x, label_y, label_text = [], [], [], [], []

        for e in rel_edges:
            sx, sy = pos[e["source"]]
            tx, ty = pos[e["target"]]
            ex += [sx, tx, None]
            ey += [sy, ty, None]
            if show_labels:
                label_x.append((sx + tx) / 2)
                label_y.append((sy + ty) / 2)
                label_text.append(rel_type.replace("_", " "))

        # Edge line trace
        traces.append(go.Scatter(
            x=ex, y=ey,
            mode="lines",
            line=dict(color=color, width=1.2),
            opacity=0.55,
            hoverinfo="skip",
            showlegend=True,
            legendgroup=rel_type,
            name=rel_type.replace("_", " "),
        ))

        # Edge label trace (mid-point annotations)
        if show_labels and label_x:
            traces.append(go.Scatter(
                x=label_x, y=label_y,
                mode="text",
                text=label_text,
                textfont=dict(size=7, color=color),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=rel_type,
            ))

    # ── Node traces — one trace per entity_type for legend grouping ───────────
    nodes_by_type: dict = defaultdict(list)
    for n in g_nodes:
        meta = meta_lookup.get(n["id"])
        et = ""
        if meta and hasattr(meta, "get_attribute"):
            et = meta.get_attribute("entity_type") or ""
        if not et:
            type_labels = {0: "fact", 1: "preference", 2: "event", 3: "conversation"}
            et = type_labels.get(n.get("type", 0), "fact")
        nodes_by_type[et].append((n, meta))

    for et, node_list in nodes_by_type.items():
        xs, ys, sizes, colors, hover_texts, node_texts = [], [], [], [], [], []
        for n, meta in node_list:
            nid = n["id"]
            x, y = pos[nid]
            xs.append(x)
            ys.append(y)

            # Size: importance × stickiness
            imp = n.get("importance", 0.5)
            recall = n.get("recall_count", 0)
            stickiness = 1 + math.log1p(recall)
            size = max(8, min(28, 6 + imp * 12 * min(stickiness, 2)))
            sizes.append(size)

            node_color = _node_color(meta)
            # Ad nodes: use scaling index colour
            if et == "ad" and meta and hasattr(meta, "get_attribute"):
                si = meta.get_attribute("scaling_index") or ""
                node_color = _SCALING_COLORS.get(si, _ENTITY_COLORS["ad"])
            colors.append(node_color)

            # Label (short, shown on graph)
            label = n.get("label") or str(nid)
            if len(label) > 22:
                label = label[:20] + "…"
            node_texts.append(label)

            # Hover text (rich detail)
            content = n.get("content", "")
            if len(content) > 160:
                content = content[:158] + "…"

            attrs_html = ""
            if meta and hasattr(meta, "get_attribute"):
                key_attrs = ["entity_type", "scaling_index", "conversion_status",
                             "engagement_status", "fatigue_status",
                             "total_spend", "last_7_days_spend", "ad_id",
                             "campaign_name", "adset_name"]
                parts = []
                for k in key_attrs:
                    v = meta.get_attribute(k)
                    if v and v != et:
                        parts.append(f"<b>{k}:</b> {v}")
                if parts:
                    attrs_html = "<br>" + "<br>".join(parts)

            hover_texts.append(
                f"<b>{n.get('label', str(nid))[:60]}</b><br>"
                f"<i>type: {et}  ·  id: {nid}</i><br>"
                f"importance: {imp:.2f}  ·  recalls: {recall}<br>"
                f"ns: {n.get('namespace_id','—')}  ·  entity: {n.get('entity_id','—')}<br>"
                f"{content}"
                f"{attrs_html}"
            )

        base_color = _ENTITY_COLORS.get(et, _DEFAULT_COLOR)

        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color="#1e293b", width=1.5),
                symbol="circle",
            ),
            text=node_texts,
            textposition="bottom center",
            textfont=dict(size=9, color="#cbd5e1"),
            hovertext=hover_texts,
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor="#1a1d27",
                bordercolor="#4a5568",
                font=dict(size=11, color="#e2e8f0"),
            ),
            showlegend=True,
            legendgroup=et,
            name=et,
            legendgrouptitle_text=None,
        ))

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"{title}  <span style='font-size:12px;color:#64748b'>"
                 f"{len(g_nodes)} nodes · {len(valid_edges)} edges</span>",
            font=dict(size=15, color="#a78bfa"),
            x=0.01,
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13151f",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False, zeroline=False, showgrid=False, range=None),
        yaxis=dict(visible=False, zeroline=False, showgrid=False,
                   scaleanchor="x", scaleratio=1),
        hovermode="closest",
        legend=dict(
            bgcolor="#1a1d27",
            bordercolor="#2d3748",
            borderwidth=1,
            font=dict(size=10, color="#94a3b8"),
            itemsizing="constant",
            orientation="v",
            x=1.0,
            xanchor="left",
            y=1.0,
            yanchor="top",
            title=dict(text="<b>Node / Edge types</b>", font=dict(size=10, color="#64748b")),
        ),
        dragmode="pan",
        newshape=dict(line_color="#a78bfa"),
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    return fig
