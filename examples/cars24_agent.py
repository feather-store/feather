"""
Cars24 — Meta Performance Agent in Feather DB
==============================================
Full agent logic knowledge graph (36 nodes, 60+ edges) + 28 real ad rows
with attribution stored as Feather node attributes and performance-derived
edges back to the agent signal/trigger/root-cause layer.

Embedder modes
--------------
  • Default  — deterministic hash bag-of-words (256-dim, zero deps)
  • OpenAI   — set OPENAI_API_KEY env var (text-embedding-3-small, 1536-dim)

Gradio tabs
-----------
  1. Graph        — interactive D3 force graph (full context web)
  2. Query        — semantic search across all nodes
  3. Scoring      — live metric sliders → P1/P2/P3 + graph-derived diagnosis
  4. Context Chain — query → BFS expansion through graph
  5. Ad Intel     — real ad-level table, filter by campaign / scaling index
  6. DB Health    — node counts, memory tiers, edge summary
"""

import json
import math
import os
import tempfile
import time

import gradio as gr
import numpy as np

import feather_db
from feather_db import DB, FilterBuilder, RelType
from feather_db.graph import export_graph
from feather_db.plotly_viz import plot_graph
from feather_db.memory import MemoryManager


# ─────────────────────────────────────────────────────────────────────────────
# Embedder — hash (default) or OpenAI (set OPENAI_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if OPENAI_API_KEY:
    try:
        from openai import OpenAI as _OAI
        _oai_client = _OAI(api_key=OPENAI_API_KEY)
        EMBED_DIM = 1536

        def embed(text: str) -> np.ndarray:
            resp = _oai_client.embeddings.create(
                model="text-embedding-3-small", input=text[:8192]
            )
            v = np.array(resp.data[0].embedding, dtype=np.float32)
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        print("✓ Using OpenAI text-embedding-3-small (1536-dim)")
    except ImportError:
        OPENAI_API_KEY = ""
        print("⚠ openai package not installed — falling back to hash embedder")

if not OPENAI_API_KEY:
    EMBED_DIM = 256

    def embed(text: str) -> np.ndarray:
        """Deterministic bag-of-words hash embedding (no model needed)."""
        tokens = text.lower().split()
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for tok in tokens:
            h = abs(hash(tok)) % EMBED_DIM
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    print(f"✓ Using hash embedder ({EMBED_DIM}-dim)")


# ─────────────────────────────────────────────────────────────────────────────
# Agent logic node definitions
# ─────────────────────────────────────────────────────────────────────────────

NODES = [
    # id, entity_type, namespace, content, importance
    # ── Campaign ──────────────────────────────────────────────────────────────
    (1,  "campaign",   "cars24", "Cars24 Meta Performance Agent. Objective: maximise BC conversions. Budget: ₹4.8Cr/month. Granularity: city level. Formula: inventory × city-rate. Monitoring: daily.", 1.0),

    # ── Signals ───────────────────────────────────────────────────────────────
    (10, "signal",     "cars24", "Spend Pacing: ratio of actual spend vs planned daily budget. Negative deviation means underspend. Used to detect budget delivery failures.", 0.9),
    (11, "signal",     "cars24", "CPM (Cost Per Mille): cost per 1000 impressions. Rising CPM signals auction competition, algorithm changes, or creative fatigue.", 0.9),
    (12, "signal",     "cars24", "CTR (Click-Through Rate): impressions to clicks ratio. Low CTR indicates creative is not resonating or audience mismatch.", 0.9),
    (13, "signal",     "cars24", "CVR (Conversion Rate): link clicks to BC conversions. Low CVR means post-click drop-off — landing page, funnel or audience intent problem.", 0.9),

    # ── Triggers ──────────────────────────────────────────────────────────────
    (20, "trigger",    "cars24", "Trigger 1 — Underspend: spend is below daily target. Signal weights: Spend Pacing 50%, CTR 30%, CPM 20%. Score 70-100 = P1 act today.", 0.95),
    (21, "trigger",    "cars24", "Trigger 2 — Drop in BCs: daily BC count falls below target. Diagnose by funnel layer. Signal weights: CPM 35%, CTR 35%, CVR 20%, Spend Pacing 10%.", 0.95),
    (22, "trigger",    "cars24", "Trigger 3 — High CPBC: cost per BC elevated, CVR below baseline. Signal weights: CVR 45%, CPM 35%, CTR 20%. Indicates product mismatch or cheap audience.", 0.95),

    # ── Root causes ───────────────────────────────────────────────────────────
    (30, "root_cause", "cars24", "Budget mismanagement: planned budget error or intentional pause causing underspend.", 0.8),
    (31, "root_cause", "cars24", "Ad set delivery constraints: restrictive audience or low bid preventing budget from clearing.", 0.8),
    (32, "root_cause", "cars24", "High CPM from competition: festival season auction pressure, or Meta algorithm change increasing delivery cost.", 0.8),
    (33, "root_cause", "cars24", "Learning phase suppression: too many new creatives launched simultaneously, reducing delivery efficiency.", 0.8),
    (34, "root_cause", "cars24", "Best performing creative paused: top-performing ad accidentally or intentionally paused, causing BC drop.", 0.8),
    (35, "root_cause", "cars24", "Creative fatigue: existing creative not resonating with audience — low CTR indicates concept is worn out.", 0.8),
    (36, "root_cause", "cars24", "Landing page / post-click issue: link click drop not explained by CTR — destination URL broken or page slow.", 0.8),
    (37, "root_cause", "cars24", "Product mismatch: ads reaching wrong users — audience too broad or wrong intent tier, causing low CVR.", 0.8),
    (38, "root_cause", "cars24", "Cheap low-intent audience: targeting optimised for volume over intent — high CTR but poor CVR and high CPBC.", 0.8),

    # ── Actions ───────────────────────────────────────────────────────────────
    (50, "action",     "cars24", "Verify campaign and ad set statuses: check for accidental pauses, budget caps hit, or billing issues.", 0.85),
    (51, "action",     "cars24", "Recalculate budget: use inventory × city-rate formula to reset correct daily targets per city.", 0.85),
    (52, "action",     "cars24", "Expand audience: widen targeting if CTR is low and delivery is constrained. Check for overlap before expanding.", 0.85),
    (53, "action",     "cars24", "Reinstate best performing creative immediately: do not wait — paused top performer is the fastest lever.", 0.85),
    (54, "action",     "cars24", "Creative refresh: launch 1-2 new creatives at a time to avoid triggering learning phase at scale.", 0.85),
    (55, "action",     "cars24", "Escalate to website team: if destination URL is correct but link clicks are dropping, post-click issue is outside media team scope.", 0.85),
    (56, "action",     "cars24", "Tighten targeting: cut cheap low-intent segments, audit product-audience fit, raise intent floor.", 0.85),
    (57, "action",     "cars24", "Call customers: when BC funnel drop-off is unexplained — qualitative signal to understand conversion barrier.", 0.85),
    (58, "action",     "cars24", "Revive old top-performing ad: run alongside new test creative — provides delivery stability during experiments.", 0.85),
    (59, "action",     "cars24", "Check Meta algorithm changes: delivery anomalies or CPM spikes may be platform-side, not campaign-side.", 0.85),
    (60, "action",     "cars24", "Strategic slow-spend during festival season: high CPM expected — deliberate choice, not a trigger for action.", 0.85),
    (61, "action",     "cars24", "Add higher-intent audience segments: when CTR drops or CVR is low. Verify no audience overlap before launch.", 0.85),

    # ── Scoring bands ─────────────────────────────────────────────────────────
    (70, "scoring_band", "cars24", "P1 — Act Today: score 70-100. Immediate action required. Deviation is significant and compounding.", 1.0),
    (71, "scoring_band", "cars24", "P2 — Monitor Closely: score 40-69. Trend is concerning but not yet critical. Daily check required.", 0.8),
    (72, "scoring_band", "cars24", "P3 — Healthy: score 0-39. Metrics within acceptable range. Routine monitoring only.", 0.6),

    # ── Operational rules ─────────────────────────────────────────────────────
    (80, "operational", "cars24", "Adding budget: trigger when BC below target but CPM stable and delivery healthy. Increase at city level proportional to inventory. Avoid during learning phase.", 0.9),
    (81, "operational", "cars24", "Audience or ad set additions: trigger when CTR drops or CVR low due to audience mismatch. Add higher-intent segments only.", 0.9),
    (82, "operational", "cars24", "Creative changes: launch 1-2 new creatives at a time. Always revive old top performers alongside new tests.", 0.9),
    (83, "operational", "cars24", "New campaigns: trigger at start of month, on weak funnel metrics, or to recreate a best-performer. Avoid mid-month without clear rationale.", 0.9),
]


# ─────────────────────────────────────────────────────────────────────────────
# Agent logic edge definitions
# ─────────────────────────────────────────────────────────────────────────────

EDGES = [
    # Campaign → triggers
    (1, 20, "has_trigger",    1.0),
    (1, 21, "has_trigger",    1.0),
    (1, 22, "has_trigger",    1.0),

    # Campaign → signals
    (1, 10, "tracks_signal",  1.0),
    (1, 11, "tracks_signal",  1.0),
    (1, 12, "tracks_signal",  1.0),
    (1, 13, "tracks_signal",  1.0),

    # Trigger 1 (Underspend) signal weights
    (20, 10, "weighted_signal", 0.50),
    (20, 12, "weighted_signal", 0.30),
    (20, 11, "weighted_signal", 0.20),

    # Trigger 2 (BC Drop) signal weights
    (21, 11, "weighted_signal", 0.35),
    (21, 12, "weighted_signal", 0.35),
    (21, 13, "weighted_signal", 0.20),
    (21, 10, "weighted_signal", 0.10),

    # Trigger 3 (High CPBC) signal weights
    (22, 13, "weighted_signal", 0.45),
    (22, 11, "weighted_signal", 0.35),
    (22, 12, "weighted_signal", 0.20),

    # Trigger 1 → root causes
    (20, 30, "caused_by", 0.9),
    (20, 31, "caused_by", 0.8),
    (20, 32, "caused_by", 0.6),

    # Trigger 2 → root causes
    (21, 32, "caused_by", 0.9),
    (21, 33, "caused_by", 0.8),
    (21, 34, "caused_by", 0.9),
    (21, 35, "caused_by", 0.8),
    (21, 36, "caused_by", 0.7),
    (21, 38, "caused_by", 0.6),

    # Trigger 3 → root causes
    (22, 37, "caused_by", 0.9),
    (22, 38, "caused_by", 0.9),
    (22, 32, "caused_by", 0.5),

    # Root causes → actions
    (30, 50, "resolved_by", 0.9),
    (30, 51, "resolved_by", 0.9),
    (31, 52, "resolved_by", 0.8),
    (31, 51, "resolved_by", 0.7),
    (32, 59, "resolved_by", 0.8),
    (32, 60, "resolved_by", 0.7),
    (33, 54, "resolved_by", 0.9),
    (34, 53, "resolved_by", 1.0),
    (35, 54, "resolved_by", 0.9),
    (35, 58, "resolved_by", 0.8),
    (36, 55, "resolved_by", 0.9),
    (37, 56, "resolved_by", 0.9),
    (37, 57, "resolved_by", 0.7),
    (38, 56, "resolved_by", 0.9),
    (38, 61, "resolved_by", 0.8),

    # Triggers → scoring bands
    (20, 70, "scores_to", 0.8),
    (20, 71, "scores_to", 0.6),
    (20, 72, "scores_to", 0.4),
    (21, 70, "scores_to", 0.8),
    (21, 71, "scores_to", 0.6),
    (22, 70, "scores_to", 0.8),
    (22, 71, "scores_to", 0.6),

    # Actions → operational rules
    (51, 80, "governed_by", 0.9),
    (52, 81, "governed_by", 0.9),
    (61, 81, "governed_by", 0.8),
    (54, 82, "governed_by", 0.9),
    (58, 82, "governed_by", 0.8),
    (53, 83, "governed_by", 0.7),

    # Signal cross-relationships
    (11, 12, "correlates_with", 0.7),
    (12, 13, "correlates_with", 0.6),
    (11, 13, "correlates_with", 0.5),
]


# ─────────────────────────────────────────────────────────────────────────────
# Real ad data — 28 rows from Hawky.ai / Dhan campaign (2026-03-19)
# Fields: ad_id used directly as node ID (valid uint64)
# ─────────────────────────────────────────────────────────────────────────────

AD_DATA = [
    {'row': 1,  'ad_id': '120242809046790732', 'adset_id': '120242809046820732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_Investing_Achieve your 2026 financial goals_English_Green_Medium_NA_No_No_190202026', 'adset_name': 'SB_Acquisition_Android_Mix_MF+Investing_FT_19022026', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 4800, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Up', 'engagement_index': 1.0096, 'engagement_status': 'Normal Engagement', 'conversion_status': 'No conversions', 'conversion_index': 0, 'fatigue_status': 'Healthy', 'total_spend': 392.58, 'last_7_days_spend': 222.42, 'number_of_days_live': 19, 'scaling_ratio': 0.0624, 'cost_per_result_trend': 'No conversions'},
    {'row': 2,  'ad_id': '120242732898140732', 'adset_id': '120242732643040732', 'campaign_id': '120242731363430732', 'ad_name': 'SB_Static_Trading_Iceberg3_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Trading_FT_16022026_Lookalike_PULifetime2%', 'campaign_name': 'SB_Acquisition_Android_Trading_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5000, 'scaling_index': 'Highly Favoured', 'scaling_trend': 'Stable', 'engagement_index': 0.9837, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 0.8847, 'fatigue_status': 'Healthy', 'total_spend': 51240.10, 'last_7_days_spend': 13348.66, 'number_of_days_live': 29, 'scaling_ratio': 1.858, 'cost_per_result_trend': 'Stable'},
    {'row': 3,  'ad_id': '120242732643060732', 'adset_id': '120242732643040732', 'campaign_id': '120242731363430732', 'ad_name': 'SB_Static_Trading_Iceberg2_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Trading_FT_16022026_Lookalike_PULifetime2%', 'campaign_name': 'SB_Acquisition_Android_Trading_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5000, 'scaling_index': 'Suppressed', 'scaling_trend': 'Scaling Down', 'engagement_index': 1.0892, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 1.10, 'fatigue_status': 'Healthy', 'total_spend': 18202.73, 'last_7_days_spend': 2522.54, 'number_of_days_live': 29, 'scaling_ratio': 0.66, 'cost_per_result_trend': 'Stable'},
    {'row': 4,  'ad_id': '120242732643050732', 'adset_id': '120242732643040732', 'campaign_id': '120242731363430732', 'ad_name': 'SB_Static_Trading_Iceberg_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Trading_FT_16022026_Lookalike_PULifetime2%', 'campaign_name': 'SB_Acquisition_Android_Trading_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Gradually Scaling Down', 'engagement_index': 0.9628, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Low Efficiency', 'conversion_index': 1.6071, 'fatigue_status': 'Healthy', 'total_spend': 13297.62, 'last_7_days_spend': 2497.11, 'number_of_days_live': 30, 'scaling_ratio': 0.4821, 'cost_per_result_trend': 'Stable'},
    {'row': 5,  'ad_id': '120242732417530732', 'adset_id': '120242731363440732', 'campaign_id': '120242731363430732', 'ad_name': 'SB_Static_Trading_P&LExit2_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Trading_FT_16022026_Trade', 'campaign_name': 'SB_Acquisition_Android_Trading_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5500, 'scaling_index': 'Normal', 'scaling_trend': 'Stable', 'engagement_index': 1.1358, 'engagement_status': 'Above Avg Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 0.9743, 'fatigue_status': 'Healthy', 'total_spend': 59858.87, 'last_7_days_spend': 6889.14, 'number_of_days_live': 29, 'scaling_ratio': 0.9743, 'cost_per_result_trend': 'Constantly Declining'},
    {'row': 6,  'ad_id': '120242731363420732', 'adset_id': '120242731363440732', 'campaign_id': '120242731363430732', 'ad_name': 'SB_Static_Trading_P&LExit_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Trading_FT_16022026_Trade', 'campaign_name': 'SB_Acquisition_Android_Trading_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5500, 'scaling_index': 'Normal', 'scaling_trend': 'Gradually Scaling Up', 'engagement_index': 0.8356, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 1.0257, 'fatigue_status': 'Healthy', 'total_spend': 63018.29, 'last_7_days_spend': 19237.24, 'number_of_days_live': 29, 'scaling_ratio': 1.0257, 'cost_per_result_trend': 'Constantly Improving'},
    {'row': 7,  'ad_id': '120242730205540732', 'adset_id': '120242730205510732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_TVDhan_Trade Plan1_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_TVDhan_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Stable', 'engagement_index': 1.3129, 'engagement_status': 'High Engagement', 'conversion_status': 'No conversions', 'conversion_index': 0, 'fatigue_status': 'Healthy', 'total_spend': 5652.24, 'last_7_days_spend': 506.80, 'number_of_days_live': 29, 'scaling_ratio': 0.1655, 'cost_per_result_trend': 'No conversions'},
    {'row': 8,  'ad_id': '120242730205530732', 'adset_id': '120242730205510732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_TVDhan_Market Replay_English_Green_HIgh_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_TVDhan_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Up', 'engagement_index': 1.0976, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 0.9372, 'fatigue_status': 'Healthy', 'total_spend': 10134.67, 'last_7_days_spend': 3469.77, 'number_of_days_live': 28, 'scaling_ratio': 0.312, 'cost_per_result_trend': 'Constantly Improving'},
    {'row': 9,  'ad_id': '120242730205520732', 'adset_id': '120242730205510732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_TVDhan_Dhancharts_English_Green_HIgh_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_TVDhan_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 5000, 'scaling_index': 'Highly Favoured', 'scaling_trend': 'Stable', 'engagement_index': 0.9725, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 0.9421, 'fatigue_status': 'Healthy', 'total_spend': 86589.65, 'last_7_days_spend': 17427.51, 'number_of_days_live': 29, 'scaling_ratio': 2.5352, 'cost_per_result_trend': 'Stable'},
    {'row': 10, 'ad_id': '120242728550180732', 'adset_id': '120242727336540732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Video_MTF_MTF4_Hinglish_White_Medium_40secs_Yes_Female_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_MTF_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'video', 'os': 'Android', 'budget_amount': 7000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Stable', 'engagement_index': 0.7566, 'engagement_status': 'Low Engagement', 'conversion_status': 'No conversions', 'conversion_index': 0, 'fatigue_status': 'Healthy', 'total_spend': 3238.23, 'last_7_days_spend': 1239.24, 'number_of_days_live': 27, 'scaling_ratio': 0.1232, 'cost_per_result_trend': 'No conversions'},
    {'row': 11, 'ad_id': '120242728432240732', 'adset_id': '120242727336540732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Video_MTF_MTF3_Hinglish_Blue_Medium_51secs_Yes_Female_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_MTF_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'video', 'os': 'Android', 'budget_amount': 7000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Stable', 'engagement_index': 0.6930, 'engagement_status': 'Low Engagement', 'conversion_status': 'Low Efficiency', 'conversion_index': 1.2735, 'fatigue_status': 'Early Fatigue', 'total_spend': 4894.20, 'last_7_days_spend': 492.98, 'number_of_days_live': 28, 'scaling_ratio': 0.179, 'cost_per_result_trend': 'Declining'},
    {'row': 12, 'ad_id': '120242728300820732', 'adset_id': '120242727336540732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_MTF_upto4x leverage 12.49%_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_MTF_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 7000, 'scaling_index': 'Highly Favoured', 'scaling_trend': 'Stable', 'engagement_index': 0.9641, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Low Efficiency', 'conversion_index': 1.2357, 'fatigue_status': 'Healthy', 'total_spend': 66484.77, 'last_7_days_spend': 10937.25, 'number_of_days_live': 30, 'scaling_ratio': 2.2396, 'cost_per_result_trend': 'Constantly Improving'},
    {'row': 13, 'ad_id': '120242728186520732', 'adset_id': '120242727336540732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_MTF_MTF upto 1cr_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_MTF_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 7000, 'scaling_index': 'Suppressed', 'scaling_trend': 'Stable', 'engagement_index': 1.0258, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Efficient Converter', 'conversion_index': 0.6634, 'fatigue_status': 'Healthy', 'total_spend': 15296.70, 'last_7_days_spend': 4418.17, 'number_of_days_live': 30, 'scaling_ratio': 0.5153, 'cost_per_result_trend': 'Constantly Improving'},
    {'row': 14, 'ad_id': '120242728112350732', 'adset_id': '120242727336540732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_MTF_MTF starting at12.49%_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_MTF_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 7000, 'scaling_index': 'Suppressed', 'scaling_trend': 'Stable', 'engagement_index': 0.8507, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Very Low Efficiency', 'conversion_index': 2.725, 'fatigue_status': 'Healthy', 'total_spend': 20945.37, 'last_7_days_spend': 2826.69, 'number_of_days_live': 30, 'scaling_ratio': 0.7056, 'cost_per_result_trend': 'Declining'},
    {'row': 15, 'ad_id': '120242727978370732', 'adset_id': '120242727336540732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_MTF_Convert to MTF_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_MTF_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 7000, 'scaling_index': 'Highly Favoured', 'scaling_trend': 'Scaling Up', 'engagement_index': 1.1207, 'engagement_status': 'Above Avg Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 0.8161, 'fatigue_status': 'Healthy', 'total_spend': 81548.47, 'last_7_days_spend': 20731.62, 'number_of_days_live': 30, 'scaling_ratio': 2.747, 'cost_per_result_trend': 'Improving'},
    {'row': 16, 'ad_id': '120242727336550732', 'adset_id': '120242727336540732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_MTF_Margin funding_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_Mix_MTF_FT_16020226', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 7000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Gradually Scaling Down', 'engagement_index': 0.8182, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Efficient Converter', 'conversion_index': 0.7318, 'fatigue_status': 'Healthy', 'total_spend': 11250.07, 'last_7_days_spend': 1539.95, 'number_of_days_live': 30, 'scaling_ratio': 0.379, 'cost_per_result_trend': 'Constantly Improving'},
    {'row': 17, 'ad_id': '120242726984720732', 'adset_id': '120242609779170732', 'campaign_id': '120242609779160732', 'ad_name': 'SB_Video_MTF_MTF2_Hinglish_Blue_Medium_38secs_Yes_Female_160202026', 'adset_name': 'SB_Acquisition_Android_MTF_OMS_Signal_OMSActive_PAN_16022026', 'campaign_name': 'SB_Acquisition_Android_Mix_OMS_Manual_16022026', 'media_type': 'video', 'os': 'Android', 'budget_amount': 24000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Down', 'engagement_index': 0.3637, 'engagement_status': 'Very Low Engagement', 'conversion_status': 'No conversions', 'conversion_index': 0, 'fatigue_status': 'Healthy', 'total_spend': 474.27, 'last_7_days_spend': 6.74, 'number_of_days_live': 15, 'scaling_ratio': 0.0095, 'cost_per_result_trend': 'No conversions'},
    {'row': 18, 'ad_id': '120242726824760732', 'adset_id': '120242609779170732', 'campaign_id': '120242609779160732', 'ad_name': 'SB_Video_MTF_MTF1_Hinglish_Blue_Medium_55secs_Yes_Female_160202026', 'adset_name': 'SB_Acquisition_Android_MTF_OMS_Signal_OMSActive_PAN_16022026', 'campaign_name': 'SB_Acquisition_Android_Mix_OMS_Manual_16022026', 'media_type': 'video', 'os': 'Android', 'budget_amount': 24000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Down', 'engagement_index': 0.4797, 'engagement_status': 'Very Low Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 1.0746, 'fatigue_status': 'Early Fatigue', 'total_spend': 1343.62, 'last_7_days_spend': 13.10, 'number_of_days_live': 17, 'scaling_ratio': 0.0202, 'cost_per_result_trend': 'Declining'},
    {'row': 19, 'ad_id': '120242726587850732', 'adset_id': '120242609779170732', 'campaign_id': '120242609779160732', 'ad_name': 'SB_Static_MTF_MTF on Dhan_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_MTF_OMS_Signal_OMSActive_PAN_16022026', 'campaign_name': 'SB_Acquisition_Android_Mix_OMS_Manual_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 24000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Down', 'engagement_index': 0.9633, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 0.8116, 'fatigue_status': 'Healthy', 'total_spend': 1014.77, 'last_7_days_spend': 16.08, 'number_of_days_live': 19, 'scaling_ratio': 0.015, 'cost_per_result_trend': 'Declining'},
    {'row': 20, 'ad_id': '120242726156180732', 'adset_id': '120242609779170732', 'campaign_id': '120242609779160732', 'ad_name': 'SB_Static_MTF_1700+stocks_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_MTF_OMS_Signal_OMSActive_PAN_16022026', 'campaign_name': 'SB_Acquisition_Android_Mix_OMS_Manual_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 24000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Stable', 'engagement_index': 1.4410, 'engagement_status': 'High Engagement', 'conversion_status': 'No conversions', 'conversion_index': 0, 'fatigue_status': 'Healthy', 'total_spend': 2585.73, 'last_7_days_spend': 214.69, 'number_of_days_live': 22, 'scaling_ratio': 0.0283, 'cost_per_result_trend': 'No conversions'},
    {'row': 21, 'ad_id': '120242725687580732', 'adset_id': '120242609779170732', 'campaign_id': '120242609779160732', 'ad_name': 'SB_Static_MTF_Get 4x leverage_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_MTF_OMS_Signal_OMSActive_PAN_16022026', 'campaign_name': 'SB_Acquisition_Android_Mix_OMS_Manual_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 24000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Stable', 'engagement_index': 1.5042, 'engagement_status': 'High Engagement', 'conversion_status': 'No conversions', 'conversion_index': 0, 'fatigue_status': 'Healthy', 'total_spend': 552.25, 'last_7_days_spend': 31.45, 'number_of_days_live': 20, 'scaling_ratio': 0.0068, 'cost_per_result_trend': 'No conversions'},
    {'row': 22, 'ad_id': '120242725609470732', 'adset_id': '120242609779170732', 'campaign_id': '120242609779160732', 'ad_name': 'SB_Static_MTF_1cr in margin2_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_MTF_OMS_Signal_OMSActive_PAN_16022026', 'campaign_name': 'SB_Acquisition_Android_Mix_OMS_Manual_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 24000, 'scaling_index': 'Highly Favoured', 'scaling_trend': 'Stable', 'engagement_index': 0.9783, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 0.966, 'fatigue_status': 'Healthy', 'total_spend': 643762.57, 'last_7_days_spend': 118761.59, 'number_of_days_live': 30, 'scaling_ratio': 3.808, 'cost_per_result_trend': 'Improving'},
    {'row': 23, 'ad_id': '120242609779150732', 'adset_id': '120242609779170732', 'campaign_id': '120242609779160732', 'ad_name': 'SB_Static_MTF_1cr in margin1_English_Green_Medium_NA_No_No_160202026', 'adset_name': 'SB_Acquisition_Android_MTF_OMS_Signal_OMSActive_PAN_16022026', 'campaign_name': 'SB_Acquisition_Android_Mix_OMS_Manual_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 24000, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Up', 'engagement_index': 1.9836, 'engagement_status': 'High Engagement', 'conversion_status': 'Low Efficiency', 'conversion_index': 1.7315, 'fatigue_status': 'Healthy', 'total_spend': 45465.21, 'last_7_days_spend': 12519.84, 'number_of_days_live': 30, 'scaling_ratio': 0.2689, 'cost_per_result_trend': 'Constantly Improving'},
    {'row': 24, 'ad_id': '120242583260050732', 'adset_id': '120242583260030732', 'campaign_id': '120242583259970732', 'ad_name': 'SB_Video1_English_White_Medium_NA_No_No_05032026', 'adset_name': 'SB_Acquisition_iOS_Video_Installs_Custom__05032026', 'campaign_name': 'SB_Acquisition_Exp1_Storyads_iOS_Mix_Installs_05032026', 'media_type': 'video', 'os': 'iOS', 'budget_amount': 2500, 'scaling_index': 'Normal', 'scaling_trend': 'Stable', 'engagement_index': 1.0, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Normal Converter', 'conversion_index': 1.0, 'fatigue_status': 'Healthy', 'total_spend': 30088.26, 'last_7_days_spend': 15016.79, 'number_of_days_live': 13, 'scaling_ratio': 1.0, 'cost_per_result_trend': 'Stable'},
    {'row': 25, 'ad_id': '120242809046810732', 'adset_id': '120242809046820732', 'campaign_id': '120242727336560732', 'ad_name': 'SB_Static_Investing_Thematic ETFs_English_Green_Medium_NA_No_No_190202026', 'adset_name': 'SB_Acquisition_Android_Mix_MF+Investing_FT_19022026', 'campaign_name': 'SB_Acquisition_Android_Mix_FT_16022026', 'media_type': 'image', 'os': 'Android', 'budget_amount': 4800, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Up', 'engagement_index': 1.2702, 'engagement_status': 'High Engagement', 'conversion_status': 'Superstar Converter', 'conversion_index': 0.1415, 'fatigue_status': 'Healthy', 'total_spend': 559.58, 'last_7_days_spend': 390.38, 'number_of_days_live': 22, 'scaling_ratio': 0.0648, 'cost_per_result_trend': 'Improving'},
    {'row': 26, 'ad_id': '120242583260010732', 'adset_id': '120242583259990732', 'campaign_id': '120242583259970732', 'ad_name': 'SB_Static1_English_Blue_High_NA_No_No_05032026', 'adset_name': 'SB_Acquisition_iOS_Static_Installs_Custom__05032026', 'campaign_name': 'SB_Acquisition_Exp1_Storyads_iOS_Mix_Installs_05032026', 'media_type': 'image', 'os': 'iOS', 'budget_amount': 2500, 'scaling_index': 'Normal', 'scaling_trend': 'Stable', 'engagement_index': 0.8922, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Superstar Converter', 'conversion_index': 0.4816, 'fatigue_status': 'Healthy', 'total_spend': 9729.31, 'last_7_days_spend': 5373.85, 'number_of_days_live': 13, 'scaling_ratio': 1.2369, 'cost_per_result_trend': 'Declining'},
    {'row': 27, 'ad_id': '120242583260000732', 'adset_id': '120242583259990732', 'campaign_id': '120242583259970732', 'ad_name': 'SB_Static4_English_Blue_High_NA_No_No_05032026', 'adset_name': 'SB_Acquisition_iOS_Static_Installs_Custom__05032026', 'campaign_name': 'SB_Acquisition_Exp1_Storyads_iOS_Mix_Installs_05032026', 'media_type': 'image', 'os': 'iOS', 'budget_amount': 2500, 'scaling_index': 'Highly Suppressed', 'scaling_trend': 'Scaling Up', 'engagement_index': 0.7598, 'engagement_status': 'Low Engagement', 'conversion_status': 'No conversions', 'conversion_index': 0, 'fatigue_status': 'Healthy', 'total_spend': 877.70, 'last_7_days_spend': 792.52, 'number_of_days_live': 11, 'scaling_ratio': 0.1269, 'cost_per_result_trend': 'No conversions'},
    {'row': 28, 'ad_id': '120242583259980732', 'adset_id': '120242583259990732', 'campaign_id': '120242583259970732', 'ad_name': 'SB_Static2_Hinglish_White_Medium_NA_No_No_05032026', 'adset_name': 'SB_Acquisition_iOS_Static_Installs_Custom__05032026', 'campaign_name': 'SB_Acquisition_Exp1_Storyads_iOS_Mix_Installs_05032026', 'media_type': 'image', 'os': 'iOS', 'budget_amount': 2500, 'scaling_index': 'Highly Favoured', 'scaling_trend': 'Stable', 'engagement_index': 1.0463, 'engagement_status': 'Normal Engagement', 'conversion_status': 'Low Efficiency', 'conversion_index': 1.4738, 'fatigue_status': 'Healthy', 'total_spend': 14887.55, 'last_7_days_spend': 6605.13, 'number_of_days_live': 12, 'scaling_ratio': 1.9179, 'cost_per_result_trend': 'Declining'},
]

# Campaign aggregate nodes (IDs 100-103 — safe range, far from agent logic IDs 1-83)
CAMPAIGN_NODES = {
    '120242727336560732': (100, 'SB_Acquisition_Android_Mix_FT_16022026',                   'Android Mix FT'),
    '120242731363430732': (101, 'SB_Acquisition_Android_Trading_FT_16022026',                'Android Trading FT'),
    '120242609779160732': (102, 'SB_Acquisition_Android_Mix_OMS_Manual_16022026',            'Android Mix OMS Manual'),
    '120242583259970732': (103, 'SB_Acquisition_Exp1_Storyads_iOS_Mix_Installs_05032026',   'iOS Storyads Exp1'),
}


# ─────────────────────────────────────────────────────────────────────────────
# Build the database
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(tempfile.gettempdir(), "cars24_agent_v2.feather")


def build_db() -> DB:
    db = DB.open(DB_PATH, dim=EMBED_DIM)

    # ── Agent logic nodes ────────────────────────────────────────────────────
    for nid, etype, ns, content, imp in NODES:
        meta = feather_db.Metadata()
        meta.content      = content
        meta.namespace_id = ns
        meta.entity_id    = etype
        meta.importance   = imp
        meta.type         = feather_db.ContextType.FACT
        meta.set_attribute("entity_type", etype)
        meta.set_attribute("node_label",  content.split(":")[0].strip()[:40])
        db.add(id=nid, vec=embed(content), meta=meta)

    for from_id, to_id, rel, weight in EDGES:
        db.link(from_id, to_id, rel_type=rel, weight=weight)

    # ── Campaign aggregate nodes ─────────────────────────────────────────────
    for camp_id, (node_id, full_name, short_name) in CAMPAIGN_NODES.items():
        ads_in_camp = [a for a in AD_DATA if a['campaign_id'] == camp_id]
        total_spend = sum(a['total_spend'] for a in ads_in_camp)
        n_suppressed = sum(1 for a in ads_in_camp if 'Suppressed' in a['scaling_index'])
        n_favoured   = sum(1 for a in ads_in_camp if 'Favoured'   in a['scaling_index'])
        content = (
            f"Campaign: {full_name}. "
            f"{len(ads_in_camp)} ads, total spend ₹{total_spend:,.0f}. "
            f"Highly Favoured: {n_favoured}, Suppressed/Highly Suppressed: {n_suppressed}."
        )
        meta = feather_db.Metadata()
        meta.content      = content
        meta.namespace_id = "cars24"
        meta.entity_id    = "campaign_aggregate"
        meta.importance   = 0.9
        meta.type         = feather_db.ContextType.FACT
        meta.set_attribute("entity_type",   "campaign_aggregate")
        meta.set_attribute("node_label",    short_name)
        meta.set_attribute("campaign_id",   camp_id)
        meta.set_attribute("total_spend",   str(round(total_spend, 2)))
        meta.set_attribute("ad_count",      str(len(ads_in_camp)))
        meta.set_attribute("n_favoured",    str(n_favoured))
        meta.set_attribute("n_suppressed",  str(n_suppressed))
        db.add(id=node_id, vec=embed(content), meta=meta)
        # Link campaign aggregate → master campaign node
        db.link(node_id, 1, rel_type="feeds_campaign", weight=0.8)

    # ── Ad-level nodes ────────────────────────────────────────────────────────
    ingest_ads(db)

    db.save()
    print(f"✓ DB built: {len(db.get_all_ids())} nodes")
    return db


def ingest_ads(db: DB) -> None:
    """Create one Feather node per ad with full attribution and derive signal edges."""
    for ad in AD_DATA:
        ad_id_int = int(ad['ad_id'])
        camp_id   = ad['campaign_id']
        camp_node_id = CAMPAIGN_NODES[camp_id][0]

        # Short display label
        label = ad['ad_name'][:50]

        content = (
            f"Ad: {ad['ad_name']}. "
            f"Campaign: {ad['campaign_name']}. "
            f"Adset: {ad['adset_name']}. "
            f"Media: {ad['media_type']}, OS: {ad['os']}. "
            f"Scaling: {ad['scaling_index']} (ratio {ad['scaling_ratio']:.2f}). "
            f"Conversion: {ad['conversion_status']}. "
            f"Engagement: {ad['engagement_status']}. "
            f"Fatigue: {ad['fatigue_status']}. "
            f"Total spend: ₹{ad['total_spend']:,.2f}, "
            f"Last 7d: ₹{ad['last_7_days_spend']:,.2f}. "
            f"Days live: {ad['number_of_days_live']}."
        )

        # Importance: higher for favoured or high-spend ads
        if ad['scaling_index'] == 'Highly Favoured':
            imp = 0.95
        elif ad['scaling_index'] in ('Normal', 'Suppressed'):
            imp = 0.75
        else:  # Highly Suppressed
            imp = 0.60

        meta = feather_db.Metadata()
        meta.content      = content
        meta.namespace_id = "cars24"
        meta.entity_id    = "ad"
        meta.importance   = imp
        meta.type         = feather_db.ContextType.EVENT
        meta.set_attribute("entity_type",          "ad")
        meta.set_attribute("node_label",           label)
        meta.set_attribute("ad_id",                ad['ad_id'])
        meta.set_attribute("adset_id",             ad['adset_id'])
        meta.set_attribute("campaign_id",          camp_id)
        meta.set_attribute("ad_name",              ad['ad_name'])
        meta.set_attribute("adset_name",           ad['adset_name'])
        meta.set_attribute("campaign_name",        ad['campaign_name'])
        meta.set_attribute("media_type",           ad['media_type'])
        meta.set_attribute("os",                   ad['os'])
        meta.set_attribute("budget_amount",        str(ad['budget_amount']))
        meta.set_attribute("scaling_index",        ad['scaling_index'])
        meta.set_attribute("scaling_trend",        ad['scaling_trend'])
        meta.set_attribute("engagement_status",    ad['engagement_status'])
        meta.set_attribute("conversion_status",    ad['conversion_status'])
        meta.set_attribute("fatigue_status",       ad['fatigue_status'])
        meta.set_attribute("total_spend",          str(ad['total_spend']))
        meta.set_attribute("last_7_days_spend",    str(ad['last_7_days_spend']))
        meta.set_attribute("number_of_days_live",  str(ad['number_of_days_live']))
        meta.set_attribute("scaling_ratio",        str(ad['scaling_ratio']))
        meta.set_attribute("cost_per_result_trend", ad['cost_per_result_trend'])

        db.add(id=ad_id_int, vec=embed(content), meta=meta)

        # ── Derive signal / trigger / root-cause edges ──────────────────────

        # Suppressed delivery → signals underspend (signal node 10)
        if ad['scaling_index'] == 'Highly Suppressed':
            db.link(ad_id_int, 10, rel_type="signals_underspend", weight=0.90)
            db.link(ad_id_int, 20, rel_type="activates_trigger",  weight=0.75)  # Underspend trigger

        elif ad['scaling_index'] == 'Suppressed':
            db.link(ad_id_int, 10, rel_type="signals_underspend", weight=0.60)

        # No conversions → BC drop trigger (node 21)
        if ad['conversion_status'] == 'No conversions':
            db.link(ad_id_int, 21, rel_type="activates_trigger",  weight=0.85)
            db.link(ad_id_int, 13, rel_type="signals_cvr_issue",  weight=0.80)

        # Low / Very Low engagement → CTR signal (node 12)
        if 'Low' in ad['engagement_status'] or 'Very Low' in ad['engagement_status']:
            db.link(ad_id_int, 12, rel_type="signals_ctr_issue",  weight=0.80)

        # Efficiency problems → High CPBC trigger (node 22)
        if ad['conversion_status'] in ('Low Efficiency', 'Very Low Efficiency'):
            db.link(ad_id_int, 22, rel_type="activates_trigger",  weight=0.80)

        # Early Fatigue → creative fatigue root cause (node 35)
        if ad['fatigue_status'] == 'Early Fatigue':
            db.link(ad_id_int, 35, rel_type="exhibits_fatigue",   weight=0.95)

        # Top performers (Highly Favoured) → risk of being paused (node 34)
        if ad['scaling_index'] == 'Highly Favoured':
            db.link(ad_id_int, 34, rel_type="top_performer_risk", weight=0.70)

        # All ads link up to their campaign aggregate node
        db.link(ad_id_int, camp_node_id, rel_type="belongs_to_campaign", weight=0.85)

    print(f"✓ {len(AD_DATA)} ads ingested with performance-derived edges")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring engine
# ─────────────────────────────────────────────────────────────────────────────

def normalise_signal(actual, baseline, invert=False):
    if baseline == 0:
        return 0
    pct = (actual - baseline) / abs(baseline) * 100
    if invert:
        pct = -pct
    return max(0, min(100, pct))


def compute_scores(spend_pct, cpm_pct, ctr_pct, cvr_pct):
    spend_dev = normalise_signal(spend_pct, 100, invert=True)
    cpm_dev   = normalise_signal(cpm_pct,   100)
    ctr_dev   = normalise_signal(100, ctr_pct)
    cvr_dev   = normalise_signal(100, cvr_pct)

    t1 = spend_dev * 0.50 + ctr_dev * 0.30 + cpm_dev * 0.20
    t2 = cpm_dev   * 0.35 + ctr_dev * 0.35 + cvr_dev * 0.20 + spend_dev * 0.10
    t3 = cvr_dev   * 0.45 + cpm_dev * 0.35 + ctr_dev * 0.20

    def band(s):
        if s >= 70: return "🔴 P1 — Act Today"
        if s >= 40: return "🟡 P2 — Monitor Closely"
        return "🟢 P3 — Healthy"

    return {
        "signals": {
            "spend_deviation": round(spend_dev, 1),
            "cpm_deviation":   round(cpm_dev,   1),
            "ctr_deviation":   round(ctr_dev,   1),
            "cvr_deviation":   round(cvr_dev,   1),
        },
        "trigger_1_underspend":  {"score": round(t1, 1), "band": band(t1)},
        "trigger_2_bc_drop":     {"score": round(t2, 1), "band": band(t2)},
        "trigger_3_high_cpbc":   {"score": round(t3, 1), "band": band(t3)},
        "top_trigger": max([
            ("Underspend",  t1),
            ("BC Drop",     t2),
            ("High CPBC",   t3),
        ], key=lambda x: x[1]),
    }


def diagnose(scores: dict, db: DB) -> str:
    top_name, top_score = scores["top_trigger"]
    if top_score < 40:
        return "✅ All metrics healthy — no action required."

    trigger_map = {"Underspend": 20, "BC Drop": 21, "High CPBC": 22}
    tid = trigger_map[top_name]

    edges = db.get_edges(tid)
    cause_ids = [e.target_id for e in edges if e.rel_type == "caused_by"]
    causes = []
    for cid in cause_ids[:3]:
        meta = db.get_metadata(cid)
        if meta:
            causes.append(meta.content.split(".")[0])

    action_ids = set()
    for cid in cause_ids[:3]:
        for e in db.get_edges(cid):
            if e.rel_type == "resolved_by":
                action_ids.add(e.target_id)

    actions = []
    for aid in list(action_ids)[:4]:
        meta = db.get_metadata(aid)
        if meta:
            actions.append(meta.content.split(":")[0].strip())

    # Find real ads triggering this condition
    trigger_rel_map = {
        "Underspend": "activates_trigger",
        "BC Drop":    "activates_trigger",
        "High CPBC":  "activates_trigger",
    }
    incoming = db.get_incoming(tid)
    flagged_ads = []
    for e in incoming:
        if e.rel_type == "activates_trigger":
            m = db.get_metadata(e.source_id)
            if m and m.get_attribute("entity_type") == "ad":
                name = m.get_attribute("ad_name") or m.get_attribute("node_label") or str(e.source_id)
                flagged_ads.append(f"• {name[:60]} — {m.get_attribute('scaling_index')}, {m.get_attribute('conversion_status')}")

    lines = [
        f"**Top Trigger: {top_name}** (score {top_score:.0f}/100)",
        f"{scores['trigger_1_underspend']['band']}  Trigger 1 — Underspend  ({scores['trigger_1_underspend']['score']:.0f})",
        f"{scores['trigger_2_bc_drop']['band']}  Trigger 2 — BC Drop  ({scores['trigger_2_bc_drop']['score']:.0f})",
        f"{scores['trigger_3_high_cpbc']['band']}  Trigger 3 — High CPBC  ({scores['trigger_3_high_cpbc']['score']:.0f})",
        "",
        "**Likely root causes:**",
        *[f"• {c}" for c in causes],
        "",
        "**Recommended actions:**",
        *[f"• {a}" for a in actions],
    ]
    if flagged_ads:
        lines += ["", f"**Real ads activating this trigger ({len(flagged_ads)}):**"]
        lines += flagged_ads[:6]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic query
# ─────────────────────────────────────────────────────────────────────────────

def query_db(query_text: str, k: int, entity_filter: str, db: DB) -> str:
    if not query_text.strip():
        return "Enter a query to search the agent context."

    f = FilterBuilder().namespace("cars24")
    if entity_filter and entity_filter != "all":
        f = f.entity(entity_filter)

    results = db.search(embed(query_text), k=k, filter=f.build())
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        meta    = r.metadata
        etype   = meta.get_attribute("entity_type")
        label   = meta.get_attribute("node_label")
        e_count = len(db.get_edges(r.id))
        lines.append(
            f"**{i}. [{etype.upper()}] {label}**  (score: {r.score:.3f}, id: {r.id}, {e_count} edges)\n"
            f"   {meta.content[:200]}{'...' if len(meta.content) > 200 else ''}\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Context chain display
# ─────────────────────────────────────────────────────────────────────────────

def context_chain_view(query_text: str, hops: int, db: DB) -> str:
    if not query_text.strip():
        return "Enter a query."
    result = db.context_chain(embed(query_text), k=3, hops=hops)
    if not result.nodes:
        return "No chain found."

    lines = ["**Context chain:**\n"]
    for node in result.nodes:
        meta  = db.get_metadata(node.id)
        etype = meta.get_attribute("entity_type") if meta else "?"
        label = meta.get_attribute("node_label")  if meta else str(node.id)
        lines.append(f"  {'→ ' * node.hop}[hop {node.hop}] **{label}** `{etype}`  score={node.score:.3f}")

    if result.edges:
        lines.append("\n**Edges traversed:**")
        for e in result.edges[:10]:
            s = db.get_metadata(e.source)
            t = db.get_metadata(e.target)
            sl = s.get_attribute("node_label") if s else str(e.source)
            tl = t.get_attribute("node_label") if t else str(e.target)
            lines.append(f"  {sl} —[{e.rel_type}]→ {tl}  (w={e.weight:.2f})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ad Intelligence tab helpers
# ─────────────────────────────────────────────────────────────────────────────

def ad_intel_table(campaign_filter: str, scaling_filter: str) -> str:
    rows = AD_DATA
    if campaign_filter != "All":
        rows = [a for a in rows if a['campaign_name'] == campaign_filter
                or CAMPAIGN_NODES.get(a['campaign_id'], (None, None, campaign_filter))[2] == campaign_filter]
    if scaling_filter != "All":
        rows = [a for a in rows if a['scaling_index'] == scaling_filter]

    if not rows:
        return "No ads match the selected filters."

    lines = [
        f"**{len(rows)} ads** | Total spend: ₹{sum(a['total_spend'] for a in rows):,.0f}\n",
        "| # | Ad Name | Scaling | Conv Status | Engagement | Fatigue | Spend ₹ | 7d Spend ₹ |",
        "|---|---------|---------|-------------|------------|---------|---------|------------|",
    ]
    for a in rows:
        name = a['ad_name'][:40] + ('...' if len(a['ad_name']) > 40 else '')
        lines.append(
            f"| {a['row']} | {name} | {a['scaling_index']} | {a['conversion_status']} "
            f"| {a['engagement_status']} | {a['fatigue_status']} "
            f"| ₹{a['total_spend']:,.0f} | ₹{a['last_7_days_spend']:,.0f} |"
        )
    return "\n".join(lines)


def ad_graph_diagnosis(db: DB) -> str:
    """Summarise which ads are connected to which agent triggers via graph edges."""
    trigger_names = {20: "T1 Underspend", 21: "T2 BC Drop", 22: "T3 High CPBC"}
    signal_names  = {10: "Spend Pacing", 11: "CPM", 12: "CTR", 13: "CVR"}
    rc_names      = {35: "Creative Fatigue", 34: "Top Performer Paused"}

    lines = ["## Ad → Agent Signal Map\n"]

    for tid, tname in trigger_names.items():
        incoming = db.get_incoming(tid)
        ads = [e for e in incoming if e.rel_type == "activates_trigger"]
        if ads:
            lines.append(f"**{tname}** — {len(ads)} ads flagged:")
            for e in ads[:8]:
                m = db.get_metadata(e.source_id)
                if m and m.get_attribute("entity_type") == "ad":
                    lines.append(f"  • {m.get_attribute('ad_name', '')[:55]}  [{m.get_attribute('scaling_index')}]")
            lines.append("")

    for sid, sname in signal_names.items():
        incoming = db.get_incoming(sid)
        ads = [e for e in incoming if e.rel_type.startswith("signals_")]
        if ads:
            lines.append(f"**Signal: {sname}** — {len(ads)} ads contributing signal:")
            for e in ads[:5]:
                m = db.get_metadata(e.source_id)
                if m and m.get_attribute("entity_type") == "ad":
                    lines.append(f"  • {m.get_attribute('ad_name', '')[:55]}  [{m.get_attribute('conversion_status')}]")
            lines.append("")

    # Fatigue
    incoming_35 = db.get_incoming(35)
    fatigue_ads = [e for e in incoming_35 if e.rel_type == "exhibits_fatigue"]
    if fatigue_ads:
        lines.append(f"**Root Cause: Creative Fatigue** — {len(fatigue_ads)} ads:")
        for e in fatigue_ads:
            m = db.get_metadata(e.source_id)
            if m:
                lines.append(f"  • {m.get_attribute('ad_name', '')[:55]}")
        lines.append("")

    # Top performers
    incoming_34 = db.get_incoming(34)
    top_perf = [e for e in incoming_34 if e.rel_type == "top_performer_risk"]
    if top_perf:
        lines.append(f"**Top Performers (watch for accidental pause)** — {len(top_perf)} ads:")
        for e in top_perf:
            m = db.get_metadata(e.source_id)
            if m:
                lines.append(f"  • {m.get_attribute('ad_name', '')[:55]}  spend=₹{m.get_attribute('total_spend')}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Graph HTML
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_graph(db: DB):
    return plot_graph(db, title="Cars24 Meta Performance Agent — Context Graph", height=720)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app
# ─────────────────────────────────────────────────────────────────────────────

def launch():
    db = build_db()
    graph_fig = build_plotly_graph(db)

    ENTITY_TYPES = ["all", "campaign", "campaign_aggregate", "signal", "trigger",
                    "root_cause", "action", "scoring_band", "operational", "ad"]
    CAMPAIGN_NAMES = ["All"] + sorted(set(a['campaign_name'] for a in AD_DATA))
    SCALING_OPTIONS = ["All", "Highly Favoured", "Normal", "Suppressed", "Highly Suppressed"]

    with gr.Blocks(title="Cars24 — Meta Performance Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
# Cars24 — Meta Performance Agent · Feather DB v0.7.0

Living context graph: **{n_logic} agent logic nodes** + **{n_ads} real ads** (28 rows, 4 campaigns, ₹{total_spend:.1f}L total spend)
> Attribution, signal edges and performance data stored as a queryable knowledge graph.
""".format(
            n_logic=len(NODES),
            n_ads=len(AD_DATA),
            total_spend=sum(a['total_spend'] for a in AD_DATA) / 100000,
        ))

        with gr.Tabs():

            # ── Tab 1: Graph ──────────────────────────────────────────────────
            with gr.Tab("Graph"):
                gr.Markdown("""
### Full Context Graph
Agent logic + campaign aggregates + all 28 ad nodes linked via performance-derived edges.
`has_trigger` · `weighted_signal` · `caused_by` · `resolved_by` · `activates_trigger` · `signals_*` · `belongs_to_campaign`

**Pan/zoom** freely · **Hover** any node for full details · **Click legend** to hide/show types
Ad nodes coloured by scaling index: 🟢 Highly Favoured · 🔵 Normal · 🟠 Suppressed · 🔴 Highly Suppressed
""")
                graph_plot = gr.Plot(value=graph_fig)
                refresh_btn = gr.Button("Refresh Graph", variant="secondary")
                refresh_btn.click(lambda: build_plotly_graph(db), outputs=graph_plot)

            # ── Tab 2: Query ──────────────────────────────────────────────────
            with gr.Tab("Query"):
                gr.Markdown("""
### Semantic Query
Search across all agent logic and real ad nodes.
- *"what to do when CTR drops"* · *"highly suppressed ads"*
- *"creative fatigue root cause"* · *"top performer risk"*
- *"MTF video ads"* · *"iOS campaign performance"*
""")
                with gr.Row():
                    query_input = gr.Textbox(label="Query", placeholder="e.g. what to do when CTR drops", scale=4)
                    k_slider    = gr.Slider(1, 15, value=5, step=1, label="Top K", scale=1)
                with gr.Row():
                    etype_dd  = gr.Dropdown(ENTITY_TYPES, value="all", label="Filter by type")
                    query_btn = gr.Button("Search", variant="primary")
                query_out = gr.Markdown()

                query_btn.click(
                    lambda q, k, et: query_db(q, k, et, db),
                    inputs=[query_input, k_slider, etype_dd],
                    outputs=query_out
                )
                query_input.submit(
                    lambda q, k, et: query_db(q, k, et, db),
                    inputs=[query_input, k_slider, etype_dd],
                    outputs=query_out
                )

            # ── Tab 3: Scoring Simulator ───────────────────────────────────────
            with gr.Tab("Scoring Simulator"):
                gr.Markdown("""
### Live Scoring Simulator
Paste today's actual vs baseline metrics. Framework computes trigger scores and
pulls real flagged ads from the graph to show which creatives are driving each issue.
""")
                with gr.Row():
                    spend_sl = gr.Slider(0, 150, value=100, step=1, label="Spend % of planned (100 = on track)")
                    cpm_sl   = gr.Slider(50, 300, value=100, step=1, label="CPM % of baseline (100 = on track)")
                with gr.Row():
                    ctr_sl   = gr.Slider(10, 200, value=100, step=1, label="CTR % of baseline (100 = on track)")
                    cvr_sl   = gr.Slider(10, 200, value=100, step=1, label="CVR % of baseline (100 = on track)")
                score_btn  = gr.Button("Run Scoring", variant="primary")
                score_out  = gr.Markdown()

                def run_scoring(spend, cpm, ctr, cvr):
                    scores    = compute_scores(spend, cpm, ctr, cvr)
                    raw_json  = json.dumps(scores["signals"], indent=2)
                    diagnosis = diagnose(scores, db)
                    return f"```json\n{raw_json}\n```\n\n{diagnosis}"

                score_btn.click(run_scoring, inputs=[spend_sl, cpm_sl, ctr_sl, cvr_sl], outputs=score_out)

            # ── Tab 4: Context Chain ───────────────────────────────────────────
            with gr.Tab("Context Chain"):
                gr.Markdown("""
### Context Chain
Vector search → n-hop BFS expansion through the graph.

- *"underspend"* 2 hops → trigger → root cause → action chain
- *"highly suppressed"* 1 hop → see which triggers these ads activate
- *"creative fatigue"* 2 hops → root cause → recommended actions
""")
                with gr.Row():
                    chain_input = gr.Textbox(label="Start query", placeholder="e.g. underspend", scale=4)
                    hops_sl     = gr.Slider(1, 3, value=2, step=1, label="Hops", scale=1)
                chain_btn   = gr.Button("Trace Chain", variant="primary")
                chain_out   = gr.Markdown()

                chain_btn.click(
                    lambda q, h: context_chain_view(q, h, db),
                    inputs=[chain_input, hops_sl],
                    outputs=chain_out
                )
                chain_input.submit(
                    lambda q, h: context_chain_view(q, h, db),
                    inputs=[chain_input, hops_sl],
                    outputs=chain_out
                )

            # ── Tab 5: Ad Intelligence ─────────────────────────────────────────
            with gr.Tab("Ad Intel"):
                gr.Markdown("""
### Ad Intelligence
28 real ads from Dhan/Hawky.ai campaign snapshot (2026-03-19).
Attribution stored as Feather node attributes. Signal edges derived from performance data.
""")
                with gr.Row():
                    camp_dd    = gr.Dropdown(CAMPAIGN_NAMES, value="All", label="Campaign")
                    scaling_dd = gr.Dropdown(SCALING_OPTIONS, value="All", label="Scaling Index")
                    filter_btn = gr.Button("Filter", variant="primary")
                table_out  = gr.Markdown()

                filter_btn.click(
                    lambda c, s: ad_intel_table(c, s),
                    inputs=[camp_dd, scaling_dd],
                    outputs=table_out
                )

                gr.Markdown("---\n### Ad → Agent Signal Map\nWhich real ads are activating triggers and signals?")
                diag_btn = gr.Button("Run Ad Graph Diagnosis", variant="secondary")
                diag_out = gr.Markdown()

                diag_btn.click(lambda: ad_graph_diagnosis(db), outputs=diag_out)

            # ── Tab 6: DB Health ──────────────────────────────────────────────
            with gr.Tab("DB Health"):
                gr.Markdown("### Knowledge Graph Health")
                health_btn = gr.Button("Run Health Check", variant="primary")
                health_out = gr.Markdown()

                def run_health():
                    report     = MemoryManager.health_report(db)
                    node_count = len(db.get_all_ids())
                    edge_count = sum(len(db.get_edges(nid)) for nid in db.get_all_ids())

                    type_counts = {}
                    for nid in db.get_all_ids():
                        meta = db.get_metadata(nid)
                        if meta:
                            et = meta.get_attribute("entity_type")
                            type_counts[et] = type_counts.get(et, 0) + 1

                    lines = [
                        f"**Total nodes:** {node_count}",
                        f"**Total edges:** {edge_count}",
                        f"**Avg importance:** {report.get('avg_importance', 0):.2f}",
                        "",
                        "**Nodes by type:**",
                        *[f"  · {k}: {v}" for k, v in sorted(type_counts.items())],
                        "",
                        "**Memory tiers:**",
                        f"  · Hot: {report['hot_count']}  Warm: {report['warm_count']}  Cold: {report['cold_count']}",
                        "",
                        "**Ad data summary:**",
                        f"  · Ads: {len(AD_DATA)}  Campaigns: {len(CAMPAIGN_NODES)}",
                        f"  · Total spend: ₹{sum(a['total_spend'] for a in AD_DATA):,.0f}",
                        f"  · Highly Favoured: {sum(1 for a in AD_DATA if a['scaling_index'] == 'Highly Favoured')}",
                        f"  · Highly Suppressed: {sum(1 for a in AD_DATA if a['scaling_index'] == 'Highly Suppressed')}",
                        f"  · Early Fatigue: {sum(1 for a in AD_DATA if a['fatigue_status'] == 'Early Fatigue')}",
                        f"  · No conversions: {sum(1 for a in AD_DATA if a['conversion_status'] == 'No conversions')}",
                    ]
                    return "\n".join(lines)

                health_btn.click(run_health, outputs=health_out)

    app.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    launch()
