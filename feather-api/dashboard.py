"""
Feather DB — Professional Dashboard
=====================================
HoneyHive-inspired dark UI with:
  - Left sidebar navigation
  - Top stats bar
  - Graph with observations panel
  - Search with clear + filter chips
  - Records browser with detail panel
  - Raw API explorer
"""

import os, json, math, random, time
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BASE = os.getenv("FEATHER_API_BASE", "http://localhost:8000")
DEFAULT_KEY  = os.getenv("FEATHER_API_KEY",  "feather-9ad2c644da0d76a253b9326bd4d15d16")
DEFAULT_NS   = os.getenv("FEATHER_NS",       "demo")
DIM          = int(os.getenv("FEATHER_DIM",  "768"))

# ─────────────────────────────────────────────────────────────────────────────
# CSS — HoneyHive dark theme
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Base ── */
:root {
  --bg0: #080810;
  --bg1: #0e0e1a;
  --bg2: #13131f;
  --bg3: #1a1a2a;
  --bg4: #21213a;
  --border: #252540;
  --border2: #2e2e50;
  --accent: #7c3aed;
  --accent2: #a78bfa;
  --accent3: #6d28d9;
  --green: #34d399;
  --red: #f87171;
  --yellow: #fbbf24;
  --blue: #60a5fa;
  --pink: #f472b6;
  --text0: #f1f5f9;
  --text1: #cbd5e1;
  --text2: #94a3b8;
  --text3: #64748b;
}

body, .gradio-container, footer { background: var(--bg0) !important; color: var(--text0) !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
footer { display: none !important; }
.svelte-1gfkn6j { gap: 0 !important; }

/* ── Header bar ── */
#feather-header {
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  padding: 10px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  position: sticky;
  top: 0;
  z-index: 100;
}
#feather-logo {
  font-size: 18px;
  font-weight: 700;
  color: var(--accent2);
  letter-spacing: -0.3px;
  white-space: nowrap;
}
.header-sep { color: var(--text3); margin: 0 4px; }
#feather-header .config-inputs { display: flex; gap: 8px; flex: 1; align-items: center; }
#feather-header input, #feather-header select {
  background: var(--bg2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 6px !important;
  color: var(--text0) !important;
  font-size: 12px !important;
  height: 32px !important;
  padding: 0 10px !important;
}
#feather-header label { display: none !important; }

/* ── Stats bar ── */
#stats-bar {
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  padding: 8px 20px;
  display: flex;
  gap: 24px;
  align-items: center;
}
.stat-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text2);
}
.stat-chip .stat-val {
  font-weight: 600;
  color: var(--text0);
  font-variant-numeric: tabular-nums;
}
.stat-chip .stat-dot {
  width: 7px; height: 7px; border-radius: 50%;
  display: inline-block;
}

/* ── Layout: sidebar + content ── */
#main-layout { display: flex !important; height: calc(100vh - 90px); gap: 0 !important; }

/* ── Sidebar ── */
#sidebar {
  width: 200px;
  min-width: 200px;
  background: var(--bg1);
  border-right: 1px solid var(--border);
  padding: 16px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  overflow-y: auto;
}
#sidebar-section-label {
  font-size: 10px;
  font-weight: 600;
  color: var(--text3);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 8px 10px 4px;
}
/* Style radio group as sidebar nav */
#nav-radio .wrap { flex-direction: column !important; gap: 2px !important; }
#nav-radio label {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text2) !important;
  border: none !important;
  background: transparent !important;
  transition: all 0.15s !important;
  width: 100% !important;
}
#nav-radio label:hover {
  background: var(--bg3) !important;
  color: var(--text0) !important;
}
#nav-radio label.selected, #nav-radio label[data-selected] {
  background: var(--bg4) !important;
  color: var(--accent2) !important;
}
#nav-radio input[type=radio] { display: none !important; }
#nav-radio .svelte-1gfkn6j { border: none !important; background: transparent !important; }

/* ── Content area ── */
#content-area {
  flex: 1;
  overflow-y: auto;
  background: var(--bg0);
  padding: 20px;
}

/* ── Panel card ── */
.panel-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 14px;
}
.panel-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 10px;
}

/* ── Tabs hidden (we use sidebar nav) ── */
.tabs { border: none !important; background: transparent !important; }
.tab-nav { display: none !important; }
.tabitem { padding: 0 !important; background: transparent !important; border: none !important; }

/* ── Inputs ── */
input[type=text], input[type=password], input[type=number],
textarea, select, .block {
  background: var(--bg2) !important;
  border-color: var(--border2) !important;
  color: var(--text0) !important;
  border-radius: 8px !important;
}
label span { color: var(--text2) !important; font-size: 12px !important; }

/* ── Buttons ── */
button.primary, .primary button {
  background: var(--accent) !important;
  border: none !important;
  color: #fff !important;
  border-radius: 7px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 16px !important;
  transition: background 0.15s !important;
}
button.primary:hover { background: var(--accent3) !important; }
button.secondary {
  background: var(--bg3) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text1) !important;
  border-radius: 7px !important;
  font-size: 13px !important;
  padding: 8px 14px !important;
}
button.stop { background: #7f1d1d !important; border: none !important; color: #fca5a5 !important; border-radius: 7px !important; }

/* ── Markdown ── */
.prose, .md { color: var(--text1) !important; }
.prose h3, .md h3 { color: var(--text0) !important; font-size: 14px !important; margin-bottom: 8px; }
.prose table, .md table { width: 100%; font-size: 12px; border-collapse: collapse; }
.prose td, .md td, .prose th, .md th {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--text1);
  text-align: left;
}
.prose th, .md th { color: var(--text3); font-weight: 600; font-size: 11px; text-transform: uppercase; }
.prose tr:hover td, .md tr:hover td { background: var(--bg3); }
code { background: var(--bg3) !important; color: var(--accent2) !important; border-radius: 4px; padding: 2px 5px; font-size: 11px; }

/* ── Observations sidebar (graph page) ── */
#obs-panel {
  width: 260px;
  min-width: 260px;
  background: var(--bg1);
  border-left: 1px solid var(--border);
  padding: 14px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.obs-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}
.obs-item {
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  color: var(--text1);
}
.obs-item:last-child { border-bottom: none; }
.obs-item .obs-label { color: var(--text2); font-size: 11px; margin-bottom: 2px; }
.obs-item .obs-val { font-weight: 600; color: var(--text0); }
.obs-item .obs-bar {
  height: 3px; background: var(--bg4); border-radius: 2px; margin-top: 4px;
}
.obs-item .obs-bar-fill {
  height: 3px; border-radius: 2px; background: var(--accent2);
}
.tag-chip {
  display: inline-block;
  background: var(--bg4);
  border: 1px solid var(--border2);
  color: var(--accent2);
  border-radius: 4px;
  padding: 2px 7px;
  font-size: 10px;
  margin: 2px;
}

/* ── Search filter chips ── */
.filter-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.filter-chip {
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: 100px;
  padding: 4px 12px;
  font-size: 11px;
  color: var(--text2);
  cursor: pointer;
}
.filter-chip.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

/* ── Record cards ── */
.rec-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 8px;
  transition: border-color 0.15s;
}
.rec-card:hover { border-color: var(--accent); }
.rec-card .rec-id { font-size: 11px; color: var(--accent2); font-weight: 600; }
.rec-card .rec-content { font-size: 13px; color: var(--text0); margin: 4px 0; }
.rec-card .rec-meta { font-size: 11px; color: var(--text3); }
.imp-bar { height: 2px; background: var(--bg4); border-radius: 1px; margin-top: 6px; }
.imp-fill { height: 2px; border-radius: 1px; }

/* ── Dropdown ── */
.dropdown, select { color: var(--text0) !important; background: var(--bg2) !important; }

/* ── Slider ── */
input[type=range] { accent-color: var(--accent2) !important; }

/* ── Checkbox ── */
input[type=checkbox] { accent-color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 2px; }

/* ── Graph container ── */
#graph-wrap { flex: 1; display: flex; gap: 0; height: calc(100vh - 90px); overflow: hidden; }
#graph-main { flex: 1; overflow: hidden; padding: 16px; }

/* ── Code block ── */
.code-block { background: var(--bg3) !important; border-radius: 8px !important; border: 1px solid var(--border) !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# API client helpers
# ─────────────────────────────────────────────────────────────────────────────
def _h(key): return {"X-API-Key": key, "Content-Type": "application/json"}

def _get(base, key, path):
    try:
        r = requests.get(f"{base}{path}", headers=_h(key), timeout=10)
        return r.status_code, r.json()
    except Exception as e: return 0, {"error": str(e)}

def _post(base, key, path, body):
    try:
        r = requests.post(f"{base}{path}", headers=_h(key), json=body, timeout=15)
        return r.status_code, r.json()
    except Exception as e: return 0, {"error": str(e)}

def _put(base, key, path, body):
    try:
        r = requests.put(f"{base}{path}", headers=_h(key), json=body, timeout=10)
        return r.status_code, r.json()
    except Exception as e: return 0, {"error": str(e)}

def _delete(base, key, path):
    try:
        r = requests.delete(f"{base}{path}", headers=_h(key), timeout=10)
        return r.status_code, r.json()
    except Exception as e: return 0, {"error": str(e)}

def _rand_vec(seed=None, dim=DIM):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).tolist()

def _list_records(base, key, ns, k=50, modality="text"):
    status, resp = _get(base, key, f"/v1/{ns}/records?k={k}&modality={modality}")
    if status == 200:
        return resp.get("results", [])
    seen, out = set(), []
    for seed in range(6):
        s, r = _post(base, key, f"/v1/{ns}/search",
                     {"vector": _rand_vec(seed * 137), "k": k, "modality": modality})
        if s == 200:
            for item in r.get("results", []):
                if item["id"] not in seen:
                    seen.add(item["id"])
                    if item.get("metadata", {}).get("attributes", {}).get("_deleted") != "true":
                        out.append(item)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Stats bar
# ─────────────────────────────────────────────────────────────────────────────
def get_stats(base, key):
    hs, health = _get(base, key, "/health")
    ns_s, ns_r = _get(base, key, "/v1/namespaces")
    if hs != 200:
        return "❌ offline", "—", "—", "—", "—"
    namespaces = ns_r.get("namespaces", [])
    version = health.get("version", "?")
    status_txt = f"✅ v{version}"
    ns_count = str(len(namespaces))

    # Try to get dim from first namespace
    dim_txt = "—"
    if namespaces:
        _, stats = _get(base, key, f"/v1/namespaces/{namespaces[0]}/stats")
        if isinstance(stats, dict):
            dim_txt = str(stats.get("dim", "?"))

    return status_txt, ns_count, dim_txt, str(health.get("namespaces_loaded", "?")), version

# ─────────────────────────────────────────────────────────────────────────────
# Overview page
# ─────────────────────────────────────────────────────────────────────────────
def overview(base, key):
    hs, health = _get(base, key, "/health")
    ns_s, ns_r = _get(base, key, "/v1/namespaces")
    if hs != 200:
        return f"❌ Cannot reach `{base}`", None, None

    namespaces = ns_r.get("namespaces", [])

    # Namespace pie
    pie = None
    if namespaces:
        counts, labels, colors_pie = [], [], []
        palette = ["#7c3aed","#34d399","#60a5fa","#f97316","#fbbf24","#f472b6","#a78bfa"]
        for i, ns in enumerate(namespaces):
            recs = _list_records(base, key, ns, k=30)
            counts.append(max(len(recs), 1))
            labels.append(ns)
            colors_pie.append(palette[i % len(palette)])
        pie = go.Figure(go.Pie(
            labels=labels, values=counts, hole=0.55,
            marker=dict(colors=colors_pie, line=dict(color="#080810", width=2)),
            textinfo="label+value",
            textfont=dict(size=11, color="#e2e8f0"),
        ))
        pie.update_layout(
            title=dict(text="Records per namespace", font=dict(size=13, color="#94a3b8")),
            paper_bgcolor="#0e0e1a", font_color="#e2e8f0",
            height=300, margin=dict(l=10,r=10,t=45,b=10), showlegend=False,
            annotations=[dict(text=f"<b>{len(namespaces)}</b><br>ns", x=0.5, y=0.5,
                              font=dict(size=14, color="#a78bfa"), showarrow=False)]
        )

    # Importance histogram — first namespace
    imp_fig = None
    if namespaces:
        ns = namespaces[0]
        recs = _list_records(base, key, ns, k=80)
        if recs:
            imps = [r["metadata"].get("importance", 0) for r in recs]
            imp_fig = go.Figure(go.Histogram(
                x=imps, nbinsx=12,
                marker=dict(color="#7c3aed", opacity=0.85,
                            line=dict(color="#a78bfa", width=0.5)),
            ))
            imp_fig.update_layout(
                title=dict(text=f"Importance distribution — {ns}", font=dict(size=13, color="#94a3b8")),
                paper_bgcolor="#0e0e1a", plot_bgcolor="#13131f",
                font_color="#e2e8f0", height=280,
                xaxis=dict(title="Importance", gridcolor="#1a1a2a", range=[0,1.05]),
                yaxis=dict(title="Count", gridcolor="#1a1a2a"),
                margin=dict(l=30,r=20,t=45,b=40),
            )

    ns_details = []
    for ns in namespaces:
        _, stats = _get(base, key, f"/v1/namespaces/{ns}/stats")
        dim = stats.get("dim","?") if isinstance(stats, dict) else "?"
        ns_details.append((ns, dim))

    rows = "\n".join(f"| `{n}` | {d} | text |" for n,d in ns_details)
    md = f"""### Connected to `{base}`

| Field | Value |
|-------|-------|
| Status | ✅ {health.get('status','?')} |
| Version | `{health.get('version','?')}` |
| Namespaces loaded | **{len(namespaces)}** |

### Namespaces
| Name | Dim | Modality |
|------|-----|----------|
{rows}
"""
    return md, pie, imp_fig

# ─────────────────────────────────────────────────────────────────────────────
# Records page
# ─────────────────────────────────────────────────────────────────────────────
def browse_records(base, key, ns, modality, k):
    recs = _list_records(base, key, ns, k=int(k), modality=modality)
    if not recs:
        return f"No records found in `{ns}` / `{modality}`."

    lines = [f"**{len(recs)} records** — `{ns}` · `{modality}`\n",
             "| ID | Content | Entity | Source | Importance |",
             "|----|---------|--------|--------|------------|"]
    for r in recs:
        m = r.get("metadata", {})
        content = (m.get("content") or "—")[:60].replace("|","·")
        imp = m.get("importance", 0)
        imp_bar = "█" * int(imp * 8) + "░" * (8 - int(imp * 8))
        lines.append(
            f"| `{r['id']}` | {content} | `{m.get('entity_id','—')}` "
            f"| {m.get('source','—')} | {imp:.2f} {imp_bar} |"
        )
    return "\n".join(lines)

def add_record(base, key, ns, rec_id, modality, content, source, entity, importance, tags, use_random):
    if not ns.strip():         return "❌ Namespace required."
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
    if status == 201: return f"✅ Added `id={rec_id}` → `{ns}/{modality}`"
    return f"❌ {status}: {json.dumps(resp)}"

def delete_record(base, key, ns, rec_id):
    if not str(rec_id).strip(): return "❌ ID required."
    status, resp = _delete(base, key, f"/v1/{ns}/records/{int(rec_id)}")
    if status == 200:  return f"✅ Deleted `id={rec_id}` (soft-delete: importance→0)"
    if status == 404:  return f"⚠️ `id={rec_id}` not found in `{ns}`"
    return f"❌ {status}: {json.dumps(resp)}"

def update_importance(base, key, ns, rec_id, new_imp):
    if not str(rec_id).strip(): return "❌ ID required."
    status, resp = _put(base, key, f"/v1/{ns}/records/{int(rec_id)}/importance",
                        {"importance": float(new_imp)})
    if status == 200: return f"✅ `id={rec_id}` importance → `{new_imp:.2f}`"
    return f"❌ {status}: {json.dumps(resp)}"

def get_record(base, key, ns, rec_id):
    if not str(rec_id).strip(): return "❌ ID required."
    status, resp = _get(base, key, f"/v1/{ns}/records/{int(rec_id)}")
    if status == 200: return f"```json\n{json.dumps(resp, indent=2)}\n```"
    return f"❌ {status}: {json.dumps(resp)}"

# ─────────────────────────────────────────────────────────────────────────────
# Search page
# ─────────────────────────────────────────────────────────────────────────────
def search(base, key, ns, modality, k, query_seed):
    vec = _rand_vec(int(query_seed))
    status, resp = _post(base, key, f"/v1/{ns}/search",
                         {"vector": vec, "k": int(k), "modality": modality})
    if status != 200:
        return f"❌ {status}: {json.dumps(resp)}", None, ""

    results = resp.get("results", [])
    if not results:
        return "No results.", None, ""

    # Result cards markdown
    cards = [f"**{len(results)} results** — `{ns}` · `{modality}` · seed `{query_seed}`\n"]
    for i, r in enumerate(results, 1):
        m = r.get("metadata", {})
        score = r["score"]
        imp   = m.get("importance", 0)
        score_pct = int(score * 10)
        bar = "█" * score_pct + "░" * (10 - score_pct)
        entity = m.get("entity_id", "—")
        source = m.get("source", "—")
        content = (m.get("content") or "—")[:140]
        cards.append(
            f"**{i}.** `id={r['id']}` &nbsp; score `{score:.4f}` `{bar}`\n"
            f"> {content}\n"
            f"<small>entity: `{entity}` · source: `{source}` · importance: `{imp:.2f}`</small>\n"
        )

    # Score bar chart
    ids    = [str(r["id"]) for r in results]
    scores = [round(r["score"], 4) for r in results]
    conts  = [(r.get("metadata", {}).get("content") or "")[:40] + "…" for r in results]
    imps   = [r.get("metadata", {}).get("importance", 0) for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ids, x=scores, orientation="h",
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
        customdata=[[c, f"{imp:.2f}"] for c, imp in zip(conts, imps)],
        hovertemplate=(
            "<b>id %{y}</b><br>%{customdata[0]}<br>"
            "score: <b>%{x:.4f}</b><br>importance: %{customdata[1]}<extra></extra>"
        ),
        marker=dict(
            color=scores,
            colorscale=[[0,"#1e1e3a"],[0.4,"#4c1d95"],[0.7,"#7c3aed"],[1,"#a78bfa"]],
            line=dict(color="#2e2e50", width=0.5),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0e0e1a", plot_bgcolor="#13131f",
        font_color="#e2e8f0",
        height=max(300, len(results) * 44 + 80),
        xaxis=dict(title="Similarity score", range=[0,1.1],
                   gridcolor="#1a1a2a", tickfont=dict(size=10)),
        yaxis=dict(title="", autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=10, r=70, t=20, b=40),
        hovermode="y unified",
    )

    # Observations HTML panel
    top_score = scores[0] if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    top_imp   = imps[0] if imps else 0
    obs_html = f"""
<div style="padding:4px 0">
  <div class="obs-title">⚡ Search Observations</div>
  <div class="obs-item">
    <div class="obs-label">Top score</div>
    <div class="obs-val">{top_score:.4f}</div>
    <div class="obs-bar"><div class="obs-bar-fill" style="width:{int(top_score*100)}%"></div></div>
  </div>
  <div class="obs-item">
    <div class="obs-label">Avg score</div>
    <div class="obs-val">{avg_score:.4f}</div>
    <div class="obs-bar"><div class="obs-bar-fill" style="width:{int(avg_score*100)}%; background:#34d399"></div></div>
  </div>
  <div class="obs-item">
    <div class="obs-label">Top result importance</div>
    <div class="obs-val">{top_imp:.2f}</div>
    <div class="obs-bar"><div class="obs-bar-fill" style="width:{int(top_imp*100)}%; background:#fbbf24"></div></div>
  </div>
  <div class="obs-item">
    <div class="obs-label">Results returned</div>
    <div class="obs-val">{len(results)} / {k}</div>
  </div>
  <div class="obs-item">
    <div class="obs-label">Namespace · Modality</div>
    <div class="obs-val">{ns} · {modality}</div>
  </div>
</div>"""

    return "\n".join(cards), fig, obs_html

def clear_search():
    return "", None, "", 3  # clear cards, chart, obs, reset seed

# ─────────────────────────────────────────────────────────────────────────────
# Graph page
# ─────────────────────────────────────────────────────────────────────────────
_MOD_COLORS = {"text": "#7c3aed", "visual": "#34d399", "image": "#34d399",
               "audio": "#fbbf24", "video": "#60a5fa"}
_ENT_PALETTE = ["#a78bfa","#34d399","#60a5fa","#f97316","#fbbf24","#f472b6","#e879f9"]

def build_graph(base, key, ns, sample_k, modality):
    recs = _list_records(base, key, ns, k=int(sample_k), modality=modality)
    if not recs:
        empty = go.Figure()
        empty.update_layout(
            paper_bgcolor="#0e0e1a", plot_bgcolor="#13131f",
            font_color="#94a3b8", height=580,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="No records found — try a different namespace or modality",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(size=14, color="#64748b"))],
        )
        return empty, "No data."

    n = len(recs)
    rng = random.Random(42)

    # Assign positions — radial + jitter, grouped by entity
    entities = list(dict.fromkeys(r.get("metadata", {}).get("entity_id", "?") for r in recs))
    ent_idx  = {e: i for i, e in enumerate(entities)}

    nodes = []
    for i, r in enumerate(recs):
        m = r.get("metadata", {})
        eid   = m.get("entity_id", "?")
        ei    = ent_idx.get(eid, 0)
        group_angle = 2 * math.pi * ei / max(len(entities), 1)
        spread = 2 * math.pi * i / max(n, 1)
        radius = 0.3 + m.get("importance", 0.5) * 0.65
        x = radius * math.cos(spread) + rng.gauss(0, 0.04)
        y = radius * math.sin(spread) + rng.gauss(0, 0.04)
        nodes.append({
            "id": r["id"], "x": x, "y": y,
            "content": (m.get("content") or "")[:60],
            "entity": eid,
            "source": m.get("source", "—"),
            "importance": m.get("importance", 0.5),
            "recall": m.get("recall_count", 0),
            "links": m.get("links", []),
            "color": _ENT_PALETTE[ei % len(_ENT_PALETTE)],
            "size": max(10, m.get("importance", 0.5) * 28),
        })

    # Edge traces from actual links
    ex, ey = [], []
    id_to_pos = {nd["id"]: (nd["x"], nd["y"]) for nd in nodes}
    for nd in nodes:
        for link_id in nd["links"]:
            if link_id in id_to_pos:
                x0, y0 = id_to_pos[nd["id"]]
                x1, y1 = id_to_pos[link_id]
                ex += [x0, x1, None]
                ey += [y0, y1, None]

    traces = []
    if ex:
        traces.append(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="#2e2e50", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))

    # Nodes per entity group (color by entity)
    for ei, eid in enumerate(entities):
        grp = [nd for nd in nodes if nd["entity"] == eid]
        if not grp: continue
        color = _ENT_PALETTE[ei % len(_ENT_PALETTE)]
        traces.append(go.Scatter(
            x=[nd["x"] for nd in grp],
            y=[nd["y"] for nd in grp],
            mode="markers",
            marker=dict(
                size=[nd["size"] for nd in grp],
                color=color, opacity=0.88,
                line=dict(color="#080810", width=1.5),
                symbol="circle",
            ),
            text=[f"#{nd['id']}" for nd in grp],
            hovertext=[
                f"<b>id {nd['id']}</b><br>"
                f"{nd['content']}<br>"
                f"<span style='color:#94a3b8'>entity: {nd['entity']}</span><br>"
                f"<span style='color:#94a3b8'>source: {nd['source']}</span><br>"
                f"importance: <b>{nd['importance']:.2f}</b>  ·  "
                f"recalled: <b>{nd['recall']}</b>x"
                for nd in grp
            ],
            hoverinfo="text",
            hoverlabel=dict(bgcolor="#1a1a2a", bordercolor="#2e2e50",
                            font=dict(size=11, color="#e2e8f0")),
            name=f"{eid[:20]} ({len(grp)})",
            showlegend=len(entities) <= 8,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
        font_color="#e2e8f0",
        height=580,
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, scaleanchor="x", range=[-1.3, 1.3]),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="#13131f", bordercolor="#1e1e2e", borderwidth=1,
                    font=dict(size=10, color="#94a3b8"),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        hovermode="closest",
        dragmode="pan",
    )

    # Observations panel data
    if nodes:
        by_imp = sorted(nodes, key=lambda x: x["importance"], reverse=True)
        top3   = by_imp[:3]
        avg_imp = sum(nd["importance"] for nd in nodes) / len(nodes)
        total_links = sum(len(nd["links"]) for nd in nodes)
        most_recalled = sorted(nodes, key=lambda x: x["recall"], reverse=True)[:2]
        ent_counts = {}
        for nd in nodes:
            ent_counts[nd["entity"]] = ent_counts.get(nd["entity"], 0) + 1
        top_entities = sorted(ent_counts.items(), key=lambda x: x[1], reverse=True)[:4]

        top3_html = "\n".join(
            f'<div class="obs-item">'
            f'<div class="obs-label">#{nd["id"]} · imp {nd["importance"]:.2f}</div>'
            f'<div class="obs-val" style="font-size:11px">{nd["content"][:50]}…</div>'
            f'<div class="obs-bar"><div class="obs-bar-fill" style="width:{int(nd["importance"]*100)}%"></div></div>'
            f'</div>'
            for nd in top3
        )
        ent_html = "\n".join(
            f'<span class="tag-chip">{e[:18]} ({c})</span>'
            for e, c in top_entities
        )
        recall_html = "\n".join(
            f'<div class="obs-item"><div class="obs-label">id {nd["id"]}</div>'
            f'<div class="obs-val">{nd["recall"]}x recalled</div></div>'
            for nd in most_recalled
        )

        obs = f"""
<div>
  <div class="obs-title">📊 Nodes: {n} &nbsp;·&nbsp; Edges: {total_links}</div>

  <div class="obs-title" style="margin-top:14px">⭐ Top by Importance</div>
  {top3_html}

  <div class="obs-title" style="margin-top:14px">🔁 Most Recalled</div>
  {recall_html if recall_html else '<div class="obs-item" style="color:#64748b">No recall data</div>'}

  <div class="obs-title" style="margin-top:14px">🗂 Entities</div>
  <div style="margin-top:4px">{ent_html}</div>

  <div class="obs-title" style="margin-top:14px">📈 Avg Importance</div>
  <div class="obs-item">
    <div class="obs-val">{avg_imp:.3f}</div>
    <div class="obs-bar"><div class="obs-bar-fill" style="width:{int(avg_imp*100)}%"></div></div>
  </div>

  <div class="obs-title" style="margin-top:14px">🔗 Graph Edges</div>
  <div class="obs-item"><div class="obs-val">{total_links} links</div></div>
</div>"""
    else:
        obs = "<div style='color:#64748b'>No data</div>"

    return fig, obs

# ─────────────────────────────────────────────────────────────────────────────
# Raw API page
# ─────────────────────────────────────────────────────────────────────────────
def raw_api(base, key, method, path, body_str):
    path = path.strip()
    try:
        body = json.loads(body_str) if body_str.strip() else {}
    except Exception as e:
        return f"❌ JSON parse error: {e}"
    if   method == "GET":    status, resp = _get(base, key, path)
    elif method == "POST":   status, resp = _post(base, key, path, body)
    elif method == "PUT":    status, resp = _put(base, key, path, body)
    elif method == "DELETE": status, resp = _delete(base, key, path)
    else: return "❌ Unknown method"
    tag = "✅" if 200 <= status < 300 else "❌"
    return f"{tag} **HTTP {status}**\n\n```json\n{json.dumps(resp, indent=2)}\n```"

# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Feather DB") as app:

    # ── Header ───────────────────────────────────────────────────────────────
    with gr.Row(elem_id="feather-header"):
        gr.HTML('<span id="feather-logo">🪶 Feather DB</span>'
                '<span class="header-sep">/</span>')
        with gr.Row(elem_classes=["config-inputs"]):
            cfg_base = gr.Textbox(value=DEFAULT_BASE, label="Base URL",
                                  placeholder="http://...", scale=3, min_width=180)
            cfg_key  = gr.Textbox(value=DEFAULT_KEY, label="API Key",
                                  type="password", scale=3, min_width=160)
            cfg_ns   = gr.Textbox(value=DEFAULT_NS, label="Namespace",
                                  placeholder="demo", scale=1, min_width=80)

    # ── Stats bar ─────────────────────────────────────────────────────────────
    with gr.Row(elem_id="stats-bar"):
        stats_html = gr.HTML("""
<span class="stat-chip"><span class="stat-dot" style="background:#34d399"></span>
  Status <span class="stat-val" id="s-status">—</span></span>
<span class="stat-chip">Namespaces <span class="stat-val" id="s-ns">—</span></span>
<span class="stat-chip">Dim <span class="stat-val" id="s-dim">—</span></span>
<span class="stat-chip">Loaded <span class="stat-val" id="s-loaded">—</span></span>
""")
        refresh_stats_btn = gr.Button("↻ Refresh", scale=0, size="sm", variant="secondary")

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(elem_id="main-layout"):

        # ── Sidebar ───────────────────────────────────────────────────────────
        with gr.Column(elem_id="sidebar", scale=0, min_width=200):
            gr.HTML('<div id="sidebar-section-label">Navigate</div>')
            nav = gr.Radio(
                choices=["Overview", "Records", "Search", "Graph", "Raw API"],
                value="Overview",
                label="",
                elem_id="nav-radio",
            )

        # ── Content ───────────────────────────────────────────────────────────
        with gr.Column(elem_id="content-area", scale=1):

            with gr.Tabs(elem_id="main-tabs") as tabs:

                # ── Overview ─────────────────────────────────────────────────
                with gr.Tab("Overview", id=0):
                    with gr.Row():
                        ov_btn = gr.Button("Refresh Overview", variant="primary", scale=0)
                    ov_md = gr.Markdown()
                    with gr.Row():
                        ov_pie = gr.Plot(show_label=False)
                        ov_imp = gr.Plot(show_label=False)
                    ov_btn.click(overview, [cfg_base, cfg_key], [ov_md, ov_pie, ov_imp])
                    app.load(overview, [cfg_base, cfg_key], [ov_md, ov_pie, ov_imp])

                # ── Records ───────────────────────────────────────────────────
                with gr.Tab("Records", id=1):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### Browse Records")
                            with gr.Row():
                                br_mod = gr.Dropdown(["text","visual","image","audio"],
                                                     value="text", label="Modality", scale=1)
                                br_k   = gr.Slider(5, 100, value=20, step=5,
                                                   label="Limit", scale=2)
                                br_btn = gr.Button("Browse", variant="primary", scale=0)
                            br_out = gr.Markdown()
                            br_btn.click(browse_records,
                                         [cfg_base,cfg_key,cfg_ns,br_mod,br_k], br_out)

                        with gr.Column(scale=1, min_width=220):
                            gr.Markdown("### Record Actions")
                            rec_id_inp = gr.Number(label="Record ID", precision=0, value=1)
                            with gr.Row():
                                get_btn = gr.Button("Get",    variant="secondary")
                                del_btn = gr.Button("Delete", variant="stop")
                            new_imp = gr.Slider(0, 1, value=0.8, step=0.05, label="Importance")
                            upd_btn = gr.Button("Update importance", variant="primary")
                            rec_out = gr.Markdown()
                            get_btn.click(get_record,     [cfg_base,cfg_key,cfg_ns,rec_id_inp], rec_out)
                            del_btn.click(delete_record,  [cfg_base,cfg_key,cfg_ns,rec_id_inp], rec_out)
                            upd_btn.click(update_importance, [cfg_base,cfg_key,cfg_ns,rec_id_inp,new_imp], rec_out)

                    gr.Markdown("---\n### Add Record")
                    with gr.Row():
                        add_id   = gr.Number(label="ID", value=int(time.time())%100000, precision=0)
                        add_mod  = gr.Dropdown(["text","visual","image","audio"],
                                               value="text", label="Modality")
                        add_imp  = gr.Slider(0, 1, value=0.8, step=0.05, label="Importance")
                    add_content = gr.Textbox(label="Content", placeholder="Description of this record…")
                    with gr.Row():
                        add_src   = gr.Textbox(value="dashboard", label="Source")
                        add_etype = gr.Textbox(value="record",    label="Entity")
                        add_tags  = gr.Textbox(value="",          label="Tags (comma-separated)")
                    add_rnd = gr.Checkbox(value=True, label="Auto-generate vector (seed = ID)")
                    with gr.Row():
                        add_btn = gr.Button("Add Record", variant="primary")
                        add_save_btn = gr.Button("Save namespace to disk", variant="secondary")
                    add_out = gr.Markdown()
                    add_btn.click(add_record,
                                  [cfg_base,cfg_key,cfg_ns,add_id,add_mod,
                                   add_content,add_src,add_etype,add_imp,add_tags,add_rnd],
                                  add_out)
                    add_save_btn.click(
                        lambda b, k, ns: f"✅ Saved" if _post(b, k, f"/v1/{ns}/save", {})[0]==200 else "❌ Save failed",
                        [cfg_base, cfg_key, cfg_ns], add_out
                    )

                # ── Search ────────────────────────────────────────────────────
                with gr.Tab("Search", id=2):
                    with gr.Row():
                        sr_mod  = gr.Dropdown(["text","visual","image","audio"],
                                              value="text", label="Modality", scale=1)
                        sr_k    = gr.Slider(1, 30, value=8, step=1, label="Top K", scale=2)
                        sr_seed = gr.Slider(0, 999, value=3, step=1,
                                            label="Query seed", scale=2)
                        sr_btn  = gr.Button("Search", variant="primary", scale=0)
                        sr_clr  = gr.Button("Clear",  variant="secondary", scale=0)

                    with gr.Row():
                        with gr.Column(scale=1):
                            sr_txt = gr.Markdown(label="Results")
                        with gr.Column(scale=2):
                            sr_plt = gr.Plot(show_label=False)

                    sr_obs = gr.HTML()

                    sr_btn.click(search,
                                 [cfg_base,cfg_key,cfg_ns,sr_mod,sr_k,sr_seed],
                                 [sr_txt, sr_plt, sr_obs])
                    sr_clr.click(clear_search, [], [sr_txt, sr_plt, sr_obs, sr_seed])

                # ── Graph ─────────────────────────────────────────────────────
                with gr.Tab("Graph", id=3):
                    # Controls row
                    with gr.Row():
                        gr_mod  = gr.Dropdown(["text","visual","image","audio"],
                                              value="text", label="Modality", scale=1)
                        gr_k    = gr.Slider(5, 80, value=30, step=5, label="Sample size", scale=2)
                        gr_btn  = gr.Button("Build Graph", variant="primary", scale=0)

                    # Graph + observations side panel
                    with gr.Row(elem_id="graph-wrap"):
                        with gr.Column(elem_id="graph-main", scale=1):
                            gr_plt = gr.Plot(show_label=False)
                        with gr.Column(elem_id="obs-panel", scale=0, min_width=260):
                            gr.HTML('<div class="obs-title">📋 Observations</div>')
                            gr_obs = gr.HTML("<div style='color:#64748b;font-size:12px'>Build a graph to see observations.</div>")

                    gr_btn.click(build_graph,
                                 [cfg_base, cfg_key, cfg_ns, gr_k, gr_mod],
                                 [gr_plt, gr_obs])

                # ── Raw API ───────────────────────────────────────────────────
                with gr.Tab("Raw API", id=4):
                    gr.Markdown(
                        "### API Explorer\n"
                        "Fire any endpoint directly.\n\n"
                        "**Quick paths:** `/health` · `/v1/namespaces` · "
                        "`/v1/{ns}/records/{id}` · `/v1/{ns}/save`"
                    )
                    with gr.Row():
                        raw_method = gr.Dropdown(["GET","POST","PUT","DELETE"],
                                                  value="GET", label="Method", scale=0, min_width=100)
                        raw_path   = gr.Textbox(value="/health", label="Path", scale=4)
                        raw_btn    = gr.Button("Send →", variant="primary", scale=0)
                    raw_body = gr.Code(value="{}", language="json",
                                       label="Request body (POST/PUT)", elem_classes=["code-block"])
                    raw_out  = gr.Markdown()
                    raw_btn.click(raw_api,
                                  [cfg_base,cfg_key,raw_method,raw_path,raw_body],
                                  raw_out)

    # ── Sidebar nav → tab switch ───────────────────────────────────────────────
    _NAV_MAP = {"Overview": 0, "Records": 1, "Search": 2, "Graph": 3, "Raw API": 4}
    nav.change(lambda v: gr.Tabs(selected=_NAV_MAP.get(v, 0)), nav, tabs)

    # ── Stats refresh ─────────────────────────────────────────────────────────
    def _update_stats(base, key):
        hs, health = _get(base, key, "/health")
        ns_s, ns_r = _get(base, key, "/v1/namespaces")
        if hs != 200:
            dot_color = "#f87171"
            status_txt = "offline"
            ns_count = "—"
            dim_txt = "—"
            loaded = "—"
        else:
            dot_color = "#34d399"
            status_txt = f"v{health.get('version','?')}"
            namespaces = ns_r.get("namespaces", [])
            ns_count = str(len(namespaces))
            loaded = str(health.get("namespaces_loaded", "?"))
            dim_txt = "—"
            if namespaces:
                _, stats = _get(base, key, f"/v1/namespaces/{namespaces[0]}/stats")
                if isinstance(stats, dict):
                    dim_txt = str(stats.get("dim", "?"))
        return f"""
<span class="stat-chip">
  <span class="stat-dot" style="background:{dot_color}"></span>
  Status <span class="stat-val">{status_txt}</span>
</span>
<span class="stat-chip">Namespaces <span class="stat-val">{ns_count}</span></span>
<span class="stat-chip">Dim <span class="stat-val">{dim_txt}</span></span>
<span class="stat-chip">Loaded <span class="stat-val">{loaded}</span></span>
"""

    refresh_stats_btn.click(_update_stats, [cfg_base, cfg_key], stats_html)
    app.load(_update_stats, [cfg_base, cfg_key], stats_html)

# When run directly
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("DASHBOARD_PORT", "7863")),
        share=False,
        css=CSS,
        theme=gr.themes.Base(),
    )
