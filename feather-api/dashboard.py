"""
Feather DB — Dashboard
=======================
A full management + visualization UI for any Feather DB deployment.

Tabs:
  Overview    — health, namespace stats, importance distribution chart
  Records     — browse, add, update importance, delete
  Search      — semantic search with score chart + record cards
  Graph       — interactive Plotly force-graph of the namespace
  Raw API     — paste any endpoint and see the JSON response
"""

import os, json, math, random, time
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Config — env vars or UI overrides
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BASE = os.getenv("FEATHER_API_BASE", "http://feather-api:8000")
DEFAULT_KEY  = os.getenv("FEATHER_API_KEY",  "feather-9ad2c644da0d76a253b9326bd4d15d16")
DEFAULT_NS   = os.getenv("FEATHER_NS",       "demo")
DIM          = int(os.getenv("FEATHER_DIM",  "768"))


# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────

def _headers(key):
    return {"X-API-Key": key, "Content-Type": "application/json"}

def _get(base, key, path):
    try:
        r = requests.get(f"{base}{path}", headers=_headers(key), timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}

def _post(base, key, path, body):
    try:
        r = requests.post(f"{base}{path}", headers=_headers(key), json=body, timeout=15)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}

def _put(base, key, path, body):
    try:
        r = requests.put(f"{base}{path}", headers=_headers(key), json=body, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}

def _delete(base, key, path):
    try:
        r = requests.delete(f"{base}{path}", headers=_headers(key), timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}

def _rand_vec(seed=None, dim=DIM):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm if norm > 0 else v).tolist()

def _list_records(base, key, ns, k=50, modality="text"):
    """Try new /records endpoint, fall back to random-search sampling."""
    status, resp = _get(base, key, f"/v1/{ns}/records?k={k}&modality={modality}")
    if status == 200:
        return resp.get("results", [])
    # fallback — search with multiple seeds
    seen, out = set(), []
    for seed in range(6):
        s, r = _post(base, key, f"/v1/{ns}/search",
                     {"vector": _rand_vec(seed * 137), "k": k, "modality": modality})
        if s == 200:
            for item in r.get("results", []):
                if item["id"] not in seen:
                    seen.add(item["id"])
                    m = item.get("metadata", {})
                    if m.get("attributes", {}).get("_deleted") != "true":
                        out.append(item)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────

def overview(base, key):
    hs, health = _get(base, key, "/health")
    ns_s, ns_r  = _get(base, key, "/v1/namespaces")

    if hs != 200:
        return f"❌ Cannot reach `{base}`\n\n{health}", None, None

    namespaces = ns_r.get("namespaces", [])

    md = [
        f"## Feather DB — `{base}`",
        f"| | |","|--|--|",
        f"| **Status** | ✅ {health.get('status','?')} |",
        f"| **Version** | `{health.get('version','?')}` |",
        f"| **Namespaces** | {len(namespaces)} loaded |",
        "",
    ]

    ns_stats = []
    for ns in namespaces:
        _, stats = _get(base, key, f"/v1/namespaces/{ns}/stats")
        dim = stats.get("dim", "?") if isinstance(stats, dict) else "?"
        ns_stats.append({"name": ns, "dim": dim})

    if ns_stats:
        md += ["### Namespaces", "| Name | Dim |", "|------|-----|"]
        for s in ns_stats:
            md.append(f"| `{s['name']}` | {s['dim']} |")

    # Pie chart of namespaces
    if namespaces:
        counts, labels = [], []
        for ns in namespaces:
            recs = _list_records(base, key, ns, k=30)
            counts.append(len(recs))
            labels.append(ns)
        pie = go.Figure(go.Pie(
            labels=labels, values=counts,
            hole=0.45,
            marker_colors=["#a78bfa","#34d399","#60a5fa","#f97316","#fbbf24"],
            textinfo="label+value",
        ))
        pie.update_layout(
            title="Records per namespace (sampled)",
            paper_bgcolor="#0f1117", font_color="#e2e8f0",
            height=320, margin=dict(l=10,r=10,t=50,b=10),
            showlegend=False,
        )
    else:
        pie = None

    # Importance distribution from first namespace
    imp_fig = None
    if namespaces:
        ns = namespaces[0]
        recs = _list_records(base, key, ns, k=50)
        if recs:
            importances = [r["metadata"].get("importance", 0) for r in recs]
            imp_fig = go.Figure(go.Histogram(
                x=importances, nbinsx=10,
                marker_color="#a78bfa", opacity=0.8,
            ))
            imp_fig.update_layout(
                title=f"Importance distribution — `{ns}`",
                paper_bgcolor="#0f1117", plot_bgcolor="#13151f",
                font_color="#e2e8f0", height=280,
                xaxis=dict(title="Importance", gridcolor="#2d3748"),
                yaxis=dict(title="Count", gridcolor="#2d3748"),
                margin=dict(l=20,r=20,t=50,b=40),
            )

    return "\n".join(md), pie, imp_fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Records
# ─────────────────────────────────────────────────────────────────────────────

def browse_records(base, key, ns, modality, k):
    recs = _list_records(base, key, ns, k=int(k), modality=modality)
    if not recs:
        return f"No records in `{ns}` / `{modality}`."

    lines = [f"**{len(recs)} records** — namespace `{ns}` · modality `{modality}`\n"]
    lines += ["| ID | Content | Entity | Source | Importance |",
              "|----|---------|--------|--------|------------|"]
    for r in recs:
        m = r.get("metadata", {})
        content = (m.get("content") or "—")[:55]
        lines.append(
            f"| `{r['id']}` | {content} | {m.get('entity_id','—')} "
            f"| {m.get('source','—')} | {m.get('importance',0):.2f} |"
        )
    return "\n".join(lines)


def add_record(base, key, ns, rec_id, modality, content, source, entity,
               importance, tags, use_random):
    if not ns.strip():      return "❌ Namespace required."
    if not str(rec_id).strip(): return "❌ ID required."

    vec = _rand_vec(int(rec_id)) if use_random else _rand_vec()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    status, resp = _post(base, key, f"/v1/{ns}/vectors", {
        "id": int(rec_id), "vector": vec, "modality": modality,
        "metadata": {
            "content": content, "source": source or "dashboard",
            "namespace_id": ns, "entity_id": entity or "record",
            "tags_json": json.dumps(tag_list),
            "importance": float(importance),
            "attributes": {"modality": modality},
        },
    })
    if status == 201:
        return f"✅ Added id=`{rec_id}` to `{ns}/{modality}`"
    return f"❌ {status}: {json.dumps(resp)}"


def delete_record(base, key, ns, rec_id):
    if not str(rec_id).strip(): return "❌ ID required."
    status, resp = _delete(base, key, f"/v1/{ns}/records/{int(rec_id)}")
    if status == 200:
        return f"✅ Deleted id=`{rec_id}` from `{ns}` (soft-delete — importance set to 0)"
    if status == 404:
        return f"⚠️ Record `{rec_id}` not found in `{ns}`"
    return f"❌ {status}: {json.dumps(resp)}"


def update_importance(base, key, ns, rec_id, new_imp):
    if not str(rec_id).strip(): return "❌ ID required."
    status, resp = _put(base, key, f"/v1/{ns}/records/{int(rec_id)}/importance",
                        {"importance": float(new_imp)})
    if status == 200:
        return f"✅ Updated id=`{rec_id}` importance → `{new_imp:.2f}`"
    return f"❌ {status}: {json.dumps(resp)}"


def get_record(base, key, ns, rec_id):
    if not str(rec_id).strip(): return "❌ ID required."
    status, resp = _get(base, key, f"/v1/{ns}/records/{int(rec_id)}")
    if status == 200:
        return f"```json\n{json.dumps(resp, indent=2)}\n```"
    return f"❌ {status}: {json.dumps(resp)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Search
# ─────────────────────────────────────────────────────────────────────────────

def search(base, key, ns, modality, k, query_seed):
    vec = _rand_vec(int(query_seed))
    status, resp = _post(base, key, f"/v1/{ns}/search",
                         {"vector": vec, "k": int(k), "modality": modality})
    if status != 200:
        return f"❌ {status}: {json.dumps(resp)}", None

    results = resp.get("results", [])
    if not results:
        return "No results.", None

    # Cards
    cards = [f"**{len(results)} results** — `{ns}` / `{modality}`\n"]
    for i, r in enumerate(results, 1):
        m = r.get("metadata", {})
        imp = m.get("importance", 0)
        bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        cards.append(
            f"**{i}. id=`{r['id']}`** &nbsp;&nbsp; score `{r['score']:.4f}` `{bar}`\n"
            f"> {(m.get('content') or '—')[:120]}\n"
            f"*entity: {m.get('entity_id','—')} · source: {m.get('source','—')} "
            f"· importance: {imp:.2f}*\n"
        )

    # Score chart
    ids    = [str(r["id"]) for r in results]
    scores = [round(r["score"], 4) for r in results]
    conts  = [(r.get("metadata", {}).get("content") or "")[:35] + "…" for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ids, x=scores, orientation="h",
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        customdata=conts,
        hovertemplate="<b>id %{y}</b><br>%{customdata}<br>score: %{x:.4f}<extra></extra>",
        marker=dict(
            color=scores, colorscale=[[0,"#2d3748"],[0.5,"#7c3aed"],[1,"#a78bfa"]],
            line=dict(color="#4a5568", width=1),
        ),
    ))
    fig.update_layout(
        title=f"Similarity scores — top {k}",
        paper_bgcolor="#0f1117", plot_bgcolor="#13151f",
        font_color="#e2e8f0",
        height=max(320, len(results) * 52 + 80),
        xaxis=dict(title="Score", range=[0, 1.05], gridcolor="#2d3748"),
        yaxis=dict(title="ID", autorange="reversed"),
        margin=dict(l=20, r=60, t=50, b=40),
    )

    return "\n".join(cards), fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Graph
# ─────────────────────────────────────────────────────────────────────────────

_MODALITY_COLORS = {
    "text":  "#a78bfa",
    "image": "#34d399",
    "audio": "#fbbf24",
}
_DEFAULT_COLOR = "#60a5fa"


def build_graph(base, key, ns, sample_k, show_edges):
    nodes_map = {}

    for modality in ["text", "image", "audio"]:
        recs = _list_records(base, key, ns, k=int(sample_k), modality=modality)
        for r in recs:
            nid = r["id"]
            if nid not in nodes_map:
                nodes_map[nid] = {
                    "id": nid,
                    "content": (r.get("metadata", {}).get("content") or "")[:50],
                    "entity":  r.get("metadata", {}).get("entity_id", "?"),
                    "importance": r.get("metadata", {}).get("importance", 0.5),
                    "modality": modality,
                    "score": r.get("score", 0.5),
                }

    if not nodes_map:
        return go.Figure().update_layout(
            title="No records found",
            paper_bgcolor="#0f1117", font_color="#e2e8f0", height=600)

    nodes = list(nodes_map.values())
    n = len(nodes)

    # Spring-ish layout with a bit of randomness per node
    rng = random.Random(42)
    for i, nd in enumerate(nodes):
        angle = 2 * math.pi * i / max(n, 1)
        r_val = 0.3 + nd["importance"] * 0.7
        nd["x"] = r_val * math.cos(angle) + rng.gauss(0, 0.05)
        nd["y"] = r_val * math.sin(angle) + rng.gauss(0, 0.05)

    traces = []

    # Edge traces — connect nodes with similar scores
    if show_edges:
        ex, ey = [], []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                if abs(a["score"] - b["score"]) < 0.04 and a["modality"] == b["modality"]:
                    ex += [a["x"], b["x"], None]
                    ey += [a["y"], b["y"], None]
        if ex:
            traces.append(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color="#2d3748", width=0.8),
                hoverinfo="skip", showlegend=False,
            ))

    # Node traces per modality
    for mod in ["text", "image", "audio"]:
        mod_nodes = [nd for nd in nodes if nd["modality"] == mod]
        if not mod_nodes:
            continue
        color = _MODALITY_COLORS.get(mod, _DEFAULT_COLOR)
        traces.append(go.Scatter(
            x=[nd["x"] for nd in mod_nodes],
            y=[nd["y"] for nd in mod_nodes],
            mode="markers+text",
            marker=dict(
                size=[max(10, nd["importance"] * 26) for nd in mod_nodes],
                color=color,
                opacity=0.85,
                line=dict(color="#0f1117", width=2),
            ),
            text=[f"#{nd['id']}" for nd in mod_nodes],
            textfont=dict(size=8, color="#e2e8f0"),
            textposition="bottom center",
            hovertext=[
                f"<b>id {nd['id']}</b><br>"
                f"{nd['content']}<br>"
                f"entity: {nd['entity']}<br>"
                f"modality: {nd['modality']}<br>"
                f"importance: {nd['importance']:.2f}"
                for nd in mod_nodes
            ],
            hoverinfo="text",
            hoverlabel=dict(bgcolor="#1a1d27", bordercolor="#4a5568",
                            font=dict(size=11, color="#e2e8f0")),
            name=f"{mod} ({len(mod_nodes)})",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Namespace graph — `{ns}`  ({n} nodes)",
        paper_bgcolor="#0f1117", plot_bgcolor="#13151f",
        font_color="#e2e8f0", height=620,
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3748", borderwidth=1,
                    font=dict(size=11, color="#94a3b8"), x=1.01, xanchor="left"),
        hovermode="closest",
        dragmode="pan",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Raw API
# ─────────────────────────────────────────────────────────────────────────────

def raw_api(base, key, method, path, body_str):
    path = path.strip()
    try:
        body = json.loads(body_str) if body_str.strip() else {}
    except Exception as e:
        return f"❌ JSON parse error: {e}"

    if method == "GET":
        status, resp = _get(base, key, path)
    elif method == "POST":
        status, resp = _post(base, key, path, body)
    elif method == "PUT":
        status, resp = _put(base, key, path, body)
    elif method == "DELETE":
        status, resp = _delete(base, key, path)
    else:
        return "❌ Unknown method"

    color = "✅" if 200 <= status < 300 else "❌"
    return f"{color} **{status}**\n\n```json\n{json.dumps(resp, indent=2)}\n```"


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Feather DB Dashboard") as app:

    gr.Markdown("# ⚡ Feather DB Dashboard")

    # ── Config bar ────────────────────────────────────────────────────────────
    with gr.Row():
        cfg_base = gr.Textbox(value=DEFAULT_BASE, label="API Base URL", scale=3)
        cfg_key  = gr.Textbox(value=DEFAULT_KEY,  label="API Key",      scale=3, type="password")
        cfg_ns   = gr.Textbox(value=DEFAULT_NS,   label="Namespace",    scale=1)

    gr.Markdown("---")

    with gr.Tabs():

        # ── Overview ──────────────────────────────────────────────────────────
        with gr.Tab("Overview"):
            ov_btn = gr.Button("Refresh", variant="primary")
            ov_md  = gr.Markdown()
            with gr.Row():
                ov_pie = gr.Plot()
                ov_imp = gr.Plot()
            ov_btn.click(overview, inputs=[cfg_base, cfg_key],
                         outputs=[ov_md, ov_pie, ov_imp])
            app.load(overview, inputs=[cfg_base, cfg_key],
                     outputs=[ov_md, ov_pie, ov_imp])

        # ── Records ───────────────────────────────────────────────────────────
        with gr.Tab("Records"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Browse")
                    with gr.Row():
                        br_mod = gr.Dropdown(["text","image","audio"], value="text",
                                             label="Modality")
                        br_k   = gr.Slider(5, 100, value=20, step=5, label="Limit")
                    br_btn = gr.Button("Browse", variant="primary")
                    br_out = gr.Markdown()
                    br_btn.click(browse_records,
                                 inputs=[cfg_base, cfg_key, cfg_ns, br_mod, br_k],
                                 outputs=br_out)

                with gr.Column(scale=1):
                    gr.Markdown("### Get / Delete / Update")
                    rec_id_inp = gr.Number(label="Record ID", precision=0, value=1)
                    with gr.Row():
                        get_btn = gr.Button("Get")
                        del_btn = gr.Button("Delete", variant="stop")
                    new_imp = gr.Slider(0, 1, value=0.8, step=0.05,
                                        label="New importance")
                    upd_btn = gr.Button("Update importance")
                    rec_out = gr.Markdown()

                    get_btn.click(get_record,
                                  inputs=[cfg_base, cfg_key, cfg_ns, rec_id_inp],
                                  outputs=rec_out)
                    del_btn.click(delete_record,
                                  inputs=[cfg_base, cfg_key, cfg_ns, rec_id_inp],
                                  outputs=rec_out)
                    upd_btn.click(update_importance,
                                  inputs=[cfg_base, cfg_key, cfg_ns, rec_id_inp, new_imp],
                                  outputs=rec_out)

            gr.Markdown("### Add Record")
            with gr.Row():
                add_id   = gr.Number(label="ID", value=int(time.time()) % 100000, precision=0)
                add_mod  = gr.Dropdown(["text","image","audio"], value="text", label="Modality")
                add_imp  = gr.Slider(0, 1, value=0.8, step=0.05, label="Importance")
            add_content = gr.Textbox(label="Content",
                                      placeholder="e.g. A photo of a red sports car")
            with gr.Row():
                add_src   = gr.Textbox(value="dashboard", label="Source")
                add_etype = gr.Textbox(value="record",    label="Entity type")
                add_tags  = gr.Textbox(value="",          label="Tags (comma-separated)")
            add_rnd = gr.Checkbox(value=True, label="Auto-generate vector (seed = ID)")
            add_btn = gr.Button("Add Record", variant="primary")
            add_out = gr.Markdown()
            add_btn.click(add_record,
                          inputs=[cfg_base, cfg_key, cfg_ns, add_id, add_mod,
                                  add_content, add_src, add_etype,
                                  add_imp, add_tags, add_rnd],
                          outputs=add_out)

        # ── Search ────────────────────────────────────────────────────────────
        with gr.Tab("Search"):
            gr.Markdown(
                "Search with a random query vector. "
                "Change **Query seed** to explore different regions of the vector space."
            )
            with gr.Row():
                sr_mod  = gr.Dropdown(["text","image","audio"], value="text", label="Modality")
                sr_k    = gr.Slider(1, 30, value=8, step=1, label="Top K")
                sr_seed = gr.Slider(0, 999, value=3, step=1, label="Query seed")
            sr_btn = gr.Button("Search", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    sr_txt = gr.Markdown()
                with gr.Column(scale=2):
                    sr_plt = gr.Plot()
            sr_btn.click(search,
                         inputs=[cfg_base, cfg_key, cfg_ns, sr_mod, sr_k, sr_seed],
                         outputs=[sr_txt, sr_plt])

        # ── Graph ─────────────────────────────────────────────────────────────
        with gr.Tab("Graph"):
            gr.Markdown(
                "Plots sampled records as a force graph. "
                "Node size = importance. Colour = modality."
            )
            with gr.Row():
                gr_k     = gr.Slider(5, 60, value=20, step=5, label="Sample size")
                gr_edges = gr.Checkbox(value=True, label="Show edges")
            gr_btn = gr.Button("Build Graph", variant="primary")
            gr_plt = gr.Plot()
            gr_btn.click(build_graph,
                         inputs=[cfg_base, cfg_key, cfg_ns, gr_k, gr_edges],
                         outputs=gr_plt)

        # ── Raw API ───────────────────────────────────────────────────────────
        with gr.Tab("Raw API"):
            gr.Markdown(
                "Fire any API endpoint directly. "
                "Use this to explore or debug.\n\n"
                "**Example paths:** `/health` · `/v1/namespaces` · "
                "`/v1/demo/records/1` · `/v1/demo/save`"
            )
            with gr.Row():
                raw_method = gr.Dropdown(["GET","POST","PUT","DELETE"],
                                          value="GET", label="Method", scale=1)
                raw_path   = gr.Textbox(value="/health", label="Path", scale=4)
            raw_body = gr.Code(value="{}", language="json", label="Request body (POST/PUT)")
            raw_btn  = gr.Button("Send", variant="primary")
            raw_out  = gr.Markdown()
            raw_btn.click(raw_api,
                          inputs=[cfg_base, cfg_key, raw_method, raw_path, raw_body],
                          outputs=raw_out)

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("DASHBOARD_PORT", "7863")),
    share=False,
)
