"""
Feather DB API — Live Test UI
==============================
Test the deployed API at http://20.219.140.90:8000

Tabs:
  1. Add Vector   — add text or image records with custom metadata
  2. Search       — semantic search with modality filter + score chart
  3. Browse       — view all records in a namespace
  4. Graph        — Plotly network of linked records
  5. Health       — API status and namespace stats
"""

import json
import math
import numpy as np
import gradio as gr
import requests
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE      = "http://20.219.140.90:8000"
API_KEY   = "feather-9ad2c644da0d76a253b9326bd4d15d16"
HEADERS   = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
DIM       = 768


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_vec(seed: int | None = None) -> list:
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _post(path, payload):
    try:
        r = requests.post(f"{BASE}{path}", headers=HEADERS, json=payload, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}


def _get(path):
    try:
        r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Add Vector
# ─────────────────────────────────────────────────────────────────────────────

def add_vector(namespace, record_id, modality, content, source, entity_type,
               importance, tags, use_random_vec, raw_vec_str):
    if not namespace.strip():
        return "❌ Namespace is required."
    if not record_id:
        return "❌ Record ID is required."

    if use_random_vec:
        vec = _rand_vec(int(record_id))
    else:
        try:
            vec = [float(x.strip()) for x in raw_vec_str.split(",")]
            if len(vec) != DIM:
                return f"❌ Vector must be {DIM} dimensions, got {len(vec)}."
        except Exception as e:
            return f"❌ Vector parse error: {e}"

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    payload = {
        "id": int(record_id),
        "vector": vec,
        "modality": modality,
        "metadata": {
            "content": content,
            "source": source or "api-test-ui",
            "namespace_id": namespace,
            "entity_id": entity_type,
            "tags_json": json.dumps(tag_list),
            "importance": float(importance),
            "attributes": {"modality": modality, "entity_type": entity_type},
        },
    }

    status, resp = _post(f"/v1/{namespace}/vectors", payload)
    if status == 201:
        return f"✅ Added id={record_id} to namespace=`{namespace}` modality=`{modality}`\n\n```json\n{json.dumps(resp, indent=2)}\n```"
    return f"❌ Status {status}\n\n```json\n{json.dumps(resp, indent=2)}\n```"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Search
# ─────────────────────────────────────────────────────────────────────────────

def search_vectors(namespace, modality, k, search_seed, use_seed):
    if not namespace.strip():
        return "❌ Namespace required.", None

    vec = _rand_vec(int(search_seed) if use_seed else None)
    payload = {"vector": vec, "k": int(k), "modality": modality}
    status, resp = _post(f"/v1/{namespace}/search", payload)

    if status != 200:
        return f"❌ Status {status}\n{json.dumps(resp, indent=2)}", None

    results = resp.get("results", [])
    if not results:
        return "No results found.", None

    # Text output
    lines = [f"**{len(results)} results** from namespace=`{namespace}` modality=`{modality}`\n"]
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        lines.append(
            f"**{i}. id={r['id']}** — score `{r['score']:.4f}`\n"
            f"   {meta.get('content', '—')[:100]}\n"
            f"   *entity: {meta.get('entity_id','—')} · importance: {meta.get('importance', 0):.2f}*\n"
        )

    # Bar chart
    ids     = [str(r["id"]) for r in results]
    scores  = [r["score"] for r in results]
    labels  = [r.get("metadata", {}).get("content", "")[:30] + "…" for r in results]
    colors  = [f"rgba(167,139,250,{0.4 + s*0.6})" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=ids,
        orientation="h",
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        customdata=labels,
        hovertemplate="<b>id %{y}</b><br>%{customdata}<br>score: %{x:.4f}<extra></extra>",
        marker_color=colors,
        marker_line_color="#7c3aed",
        marker_line_width=1,
    ))
    fig.update_layout(
        title=f"Search results — {modality} modality",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13151f",
        font_color="#e2e8f0",
        height=max(300, len(results) * 55 + 80),
        xaxis=dict(title="Similarity score", range=[0, 1.05], gridcolor="#2d3748"),
        yaxis=dict(title="Record ID", autorange="reversed"),
        margin=dict(l=20, r=60, t=50, b=40),
    )

    return "\n".join(lines), fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Browse
# ─────────────────────────────────────────────────────────────────────────────

def browse_namespace(namespace, browse_k):
    """Fetch records by searching with a random vector (retrieves a broad sample)."""
    if not namespace.strip():
        return "❌ Namespace required."

    results_text = []
    for modality in ["text", "image"]:
        payload = {"vector": _rand_vec(42), "k": int(browse_k), "modality": modality}
        status, resp = _post(f"/v1/{namespace}/search", payload)
        if status == 200 and resp.get("results"):
            results_text.append(f"\n### Modality: `{modality}` ({len(resp['results'])} records)")
            results_text.append("| ID | Content | Entity | Importance | Score |")
            results_text.append("|----|---------|---------| -----------|-------|")
            for r in resp["results"]:
                m = r.get("metadata", {})
                content = m.get("content", "—")[:60]
                results_text.append(
                    f"| {r['id']} | {content} | {m.get('entity_id','—')} "
                    f"| {m.get('importance', 0):.2f} | {r['score']:.4f} |"
                )

    if not results_text:
        return f"No records found in namespace `{namespace}`."

    status, ns_resp = _get(f"/v1/namespaces/{namespace}/stats")
    header = f"## Namespace: `{namespace}`\n"
    if status == 200:
        header += f"dim={ns_resp.get('dim')} · path={ns_resp.get('db_path')}\n"

    return header + "\n".join(results_text)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Link & Graph
# ─────────────────────────────────────────────────────────────────────────────

def link_records(namespace, from_id, to_id):
    if not namespace.strip():
        return "❌ Namespace required."
    status, resp = _post(f"/v1/{namespace}/records/{int(from_id)}/link",
                         {"to_id": int(to_id)})
    if status == 200:
        return f"✅ Linked {from_id} → {to_id} in `{namespace}`"
    return f"❌ Status {status}: {json.dumps(resp)}"


def build_graph_from_api(namespace, sample_k):
    """Sample records then draw a graph showing similarity relationships."""
    nodes, edges_data = [], []
    seen_ids = set()

    for modality in ["text", "image"]:
        payload = {"vector": _rand_vec(7), "k": int(sample_k), "modality": modality}
        status, resp = _post(f"/v1/{namespace}/search", payload)
        if status != 200:
            continue
        for r in resp["results"]:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                nodes.append({
                    "id": r["id"],
                    "content": r.get("metadata", {}).get("content", "")[:40],
                    "modality": modality,
                    "score": r["score"],
                    "importance": r.get("metadata", {}).get("importance", 0.5),
                })

    if not nodes:
        return go.Figure().update_layout(title="No records found",
                                          paper_bgcolor="#0f1117", font_color="#e2e8f0")

    # Simple circular layout
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / max(n, 1)
        node["x"] = math.cos(angle)
        node["y"] = math.sin(angle)

    # Edge: connect nodes with score > 0.7 (similarity threshold)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i < j and abs(a["score"] - b["score"]) < 0.05:
                edges_data.append((a, b))

    # Build figure
    traces = []
    for a, b in edges_data:
        traces.append(go.Scatter(
            x=[a["x"], b["x"], None], y=[a["y"], b["y"], None],
            mode="lines", line=dict(color="#4a5568", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    colors = {"text": "#a78bfa", "image": "#34d399"}
    for mod in ["text", "image"]:
        mod_nodes = [nd for nd in nodes if nd["modality"] == mod]
        if not mod_nodes:
            continue
        traces.append(go.Scatter(
            x=[nd["x"] for nd in mod_nodes],
            y=[nd["y"] for nd in mod_nodes],
            mode="markers+text",
            marker=dict(
                size=[max(12, nd["importance"] * 22) for nd in mod_nodes],
                color=colors[mod],
                line=dict(color="#1e293b", width=1.5),
            ),
            text=[nd["content"][:20] + "…" for nd in mod_nodes],
            textposition="bottom center",
            textfont=dict(size=9, color="#cbd5e1"),
            hovertext=[
                f"<b>id {nd['id']}</b><br>{nd['content']}<br>"
                f"modality: {nd['modality']}<br>score: {nd['score']:.4f}"
                for nd in mod_nodes
            ],
            hoverinfo="text",
            name=f"{mod} records",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Records graph — namespace `{namespace}`",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13151f",
        font_color="#e2e8f0",
        height=560,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3748", borderwidth=1,
                    font=dict(size=11, color="#94a3b8")),
        hovermode="closest",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Health
# ─────────────────────────────────────────────────────────────────────────────

def get_health():
    status, resp = _get("/health")
    _, ns_resp   = _get("/v1/namespaces")

    namespaces = ns_resp.get("namespaces", []) if isinstance(ns_resp, dict) else []

    lines = [
        f"## API Health",
        f"**Status:** {'✅ ' + resp.get('status','?') if status == 200 else '❌ unreachable'}",
        f"**Version:** `{resp.get('version','?')}`",
        f"**Endpoint:** `{BASE}`",
        f"**Namespaces loaded:** {resp.get('namespaces_loaded', 0)}",
        "",
    ]

    if namespaces:
        lines.append("### Namespaces")
        for ns in namespaces:
            _, stats = _get(f"/v1/namespaces/{ns}/stats")
            dim = stats.get("dim", "?") if isinstance(stats, dict) else "?"
            lines.append(f"  · `{ns}` — dim={dim}")
    else:
        lines.append("*No namespaces loaded yet — add some vectors first.*")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Feather DB — API Test UI") as app:
    gr.Markdown(f"""
# Feather DB — Live API Test
**Endpoint:** `{BASE}` · **Version:** v0.7.0 · Connected with API key

Test add, search, browse, link and visualise — text and image modalities.
""")

    with gr.Tabs():

        # ── Tab 1: Add ────────────────────────────────────────────────────────
        with gr.Tab("Add Vector"):
            gr.Markdown("### Add a record to the API")
            with gr.Row():
                ns_add   = gr.Textbox(value="demo", label="Namespace")
                rid_add  = gr.Number(value=10, label="Record ID", precision=0)
                mod_add  = gr.Dropdown(["text", "image", "audio"], value="text", label="Modality")
            content_add = gr.Textbox(label="Content", placeholder="e.g. A beautiful sunset over the mountains")
            with gr.Row():
                source_add  = gr.Textbox(value="api-test-ui", label="Source")
                etype_add   = gr.Textbox(value="fact", label="Entity type")
                imp_add     = gr.Slider(0, 1, value=0.8, step=0.05, label="Importance")
            tags_add = gr.Textbox(value="test", label="Tags (comma-separated)")
            with gr.Row():
                use_rnd  = gr.Checkbox(value=True, label="Use random vector (seed = Record ID)")
                vec_str  = gr.Textbox(label="Or paste vector (768 floats, comma-separated)",
                                       placeholder="0.12, 0.34, ...", visible=False)
            use_rnd.change(lambda x: gr.update(visible=not x), inputs=use_rnd, outputs=vec_str)
            add_btn = gr.Button("Add Record", variant="primary")
            add_out = gr.Markdown()
            add_btn.click(add_vector,
                          inputs=[ns_add, rid_add, mod_add, content_add,
                                  source_add, etype_add, imp_add, tags_add, use_rnd, vec_str],
                          outputs=add_out)

        # ── Tab 2: Search ─────────────────────────────────────────────────────
        with gr.Tab("Search"):
            gr.Markdown("### Semantic search")
            with gr.Row():
                ns_srch  = gr.Textbox(value="demo", label="Namespace")
                mod_srch = gr.Dropdown(["text", "image", "audio"], value="text", label="Modality")
                k_srch   = gr.Slider(1, 20, value=5, step=1, label="Top K")
            with gr.Row():
                seed_srch    = gr.Number(value=3, label="Query seed (random vector)", precision=0)
                use_seed_srch = gr.Checkbox(value=True, label="Use fixed seed (reproducible)")
            srch_btn = gr.Button("Search", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    srch_txt = gr.Markdown()
                with gr.Column(scale=2):
                    srch_plt = gr.Plot()
            srch_btn.click(search_vectors,
                           inputs=[ns_srch, mod_srch, k_srch, seed_srch, use_seed_srch],
                           outputs=[srch_txt, srch_plt])

        # ── Tab 3: Browse ─────────────────────────────────────────────────────
        with gr.Tab("Browse"):
            gr.Markdown("### Browse all records in a namespace")
            with gr.Row():
                ns_brw   = gr.Textbox(value="demo", label="Namespace")
                k_brw    = gr.Slider(5, 50, value=20, step=5, label="Sample size")
            brw_btn = gr.Button("Browse", variant="primary")
            brw_out = gr.Markdown()
            brw_btn.click(browse_namespace, inputs=[ns_brw, k_brw], outputs=brw_out)

        # ── Tab 4: Graph ──────────────────────────────────────────────────────
        with gr.Tab("Graph"):
            gr.Markdown("### Link records + visualise the graph")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Link two records**")
                    ns_lnk   = gr.Textbox(value="demo", label="Namespace")
                    from_lnk = gr.Number(value=1, label="From ID", precision=0)
                    to_lnk   = gr.Number(value=2, label="To ID", precision=0)
                    lnk_btn  = gr.Button("Link", variant="secondary")
                    lnk_out  = gr.Markdown()
                    lnk_btn.click(link_records,
                                  inputs=[ns_lnk, from_lnk, to_lnk], outputs=lnk_out)
                with gr.Column(scale=2):
                    gr.Markdown("**Visualise namespace graph**")
                    ns_grph  = gr.Textbox(value="demo", label="Namespace")
                    k_grph   = gr.Slider(5, 30, value=13, step=1, label="Sample size")
                    grph_btn = gr.Button("Build Graph", variant="primary")
                    grph_plt = gr.Plot()
                    grph_btn.click(build_graph_from_api,
                                   inputs=[ns_grph, k_grph], outputs=grph_plt)

        # ── Tab 5: Health ─────────────────────────────────────────────────────
        with gr.Tab("Health"):
            gr.Markdown("### API health and namespace stats")
            hlth_btn = gr.Button("Refresh", variant="primary")
            hlth_out = gr.Markdown()
            hlth_btn.click(get_health, outputs=hlth_out)
            app.load(get_health, outputs=hlth_out)

app.launch(server_name="127.0.0.1", server_port=7862, share=False)
