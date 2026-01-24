import os
import time
import json
import numpy as np
from feather_db import DB, Metadata, ContextType, ScoringConfig, FilterBuilder

# -----------------------------------------------------------------------------
# 1. Deterministic Mock Embedding
# -----------------------------------------------------------------------------
# We'll map specific keywords to vector dimensions to simulate "meaning".
# Dim 0: "work/business"
# Dim 1: "personal/social"
# Dim 2: "tech/coding"
# Dim 3: "urgent/important"
# Dim 4: "finance/money"
DIM = 5

def mock_embed(text):
    text = text.lower()
    vec = np.zeros(DIM, dtype=np.float32)
    
    if any(w in text for w in ["meeting", "work", "campaign", "project", "quarter", "deadline"]):
        vec[0] += 1.0
    if any(w in text for w in ["party", "dinner", "friend", "birthday", "movie"]):
        vec[1] += 1.0
    if any(w in text for w in ["code", "python", "cpp", "db", "api", "bug", "feature"]):
        vec[2] += 1.0
    if any(w in text for w in ["urgent", "asap", "critical", "blocking"]):
        vec[3] += 1.0
    if any(w in text for w in ["budget", "cost", "price", "invoice", "salary"]):
        vec[4] += 1.0
        
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        # random small noise if no keywords match, to avoid zero vectors issues
        vec = np.random.rand(DIM).astype(np.float32) * 0.1
        
    return vec

# -----------------------------------------------------------------------------
# 2. Dataset Generation
# -----------------------------------------------------------------------------
NOW = int(time.time())
DAY = 86400

DATASET = [
    # Recent highly important work
    {
        "id": 1,
        "text": "Urgent bug fix needed for API auth in Phase 2",
        "type": ContextType.FACT,
        "source": "slack:dev-team",
        "importance": 1.0,
        "age_days": 0.1, # Just now
        "tags": ["bug", "api", "v2"]
    },
    # Older work meeting
    {
        "id": 2,
        "text": "Q1 Planning meeting notes: focus on stability",
        "type": ContextType.EVENT,
        "source": "gcal",
        "importance": 0.8,
        "age_days": 30, # 1 month ago
        "tags": ["planning", "meeting"]
    },
    # Personal info
    {
        "id": 3,
        "text": "User prefers dark mode and high contrast",
        "type": ContextType.PREFERENCE,
        "source": "settings_ui",
        "importance": 1.0,
        "age_days": 100, # Old but permanent preference
        "tags": ["ui", "a11y"]
    },
    # Code snippet
    {
        "id": 4,
        "text": "Python script to migrate database schema",
        "type": ContextType.FACT,
        "source": "github",
        "importance": 0.5,
        "age_days": 2,
        "tags": ["python", "migration"]
    },
    # Irrelevant noise
    {
        "id": 5,
        "text": "Dinner receipt for pizza party",
        "type": ContextType.EVENT,
        "source": "email",
        "importance": 0.1,
        "age_days": 5,
        "tags": ["finance", "food"]
    },
    {
        "id": 6,
        "text": "Invoice for cloud hosting services",
        "type": ContextType.FACT,
        "source": "email",
        "importance": 0.9,
        "age_days": 1,
        "tags": ["finance", "cloud"]
    },
]

# -----------------------------------------------------------------------------
# 3. Setup and Population
# -----------------------------------------------------------------------------
DB_PATH = "test_db_advanced.feather"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

print(f"🚀 Initializing Feather DB at {DB_PATH} with dim={DIM}")
db = DB.open(DB_PATH, DIM)

print(f"📥 Ingesting {len(DATASET)} diverse records...")
for item in DATASET:
    meta = Metadata()
    meta.content = item["text"]
    meta.type = item["type"]
    meta.source = item["source"]
    meta.importance = item["importance"]
    meta.timestamp = int(NOW - (item["age_days"] * DAY))
    meta.tags_json = json.dumps(item["tags"])
    
    vec = mock_embed(item["text"])
    db.add(item["id"], vec, meta)

# -----------------------------------------------------------------------------
# 4. Scenario Testing
# -----------------------------------------------------------------------------

def run_query(test_name, query_text, filter_obj=None, scoring_obj=None):
    print(f"\n🧪 Test: {test_name}")
    print(f"   Query: '{query_text}'")
    q_vec = mock_embed(query_text)
    
    # Use the passed configuration directly
    real_scoring = scoring_obj
        
    results = db.search(q_vec, k=3, filter=filter_obj, scoring=real_scoring)
    
    for i, r in enumerate(results):
        age_days = (NOW - r.metadata.timestamp) / DAY
        print(f"   {i+1}. [Score: {r.score:.4f}] [Age: {age_days:.1f}d] [{r.metadata.type}] {r.metadata.content}")

# Scenario A: Baseline Similarity (Is "bug" related to "code"?)
# Expect: Bug fix (ID 1) and Migration script (ID 4) to appear.
run_query("Baseline Context Retrieval (Similarity Only)", "coding bug issue", scoring_obj=ScoringConfig(half_life=365, weight=0.0))

# Scenario B: Finding the most URGENT recent item
# Expect: The "Urgent bug fix" (ID 1) should be #1 significantly because of recency + importance.
run_query("Urgent Recent Items (High Temporal Weight)", "urgent issue", scoring_obj=ScoringConfig(half_life=1, weight=0.5))

# Scenario C: "Recall" context from a month ago
# Expect: Q1 Planning meeting (ID 2).
run_query("Memory Recall (Older Events)", "planning meeting", scoring_obj=ScoringConfig(half_life=60, weight=0.1))

# Scenario D: Filtering for specific source (e.g. only 'email')
# Expect: Invoice (ID 6) and Pizza receipt (ID 5).
fb = FilterBuilder()
email_filter = fb.types([ContextType.FACT, ContextType.EVENT]).source("email").build()
run_query("Source Filtering (Emails only)", "money cost", filter_obj=email_filter)

# Scenario E: Filtering for Preferences (User personalization)
# Expect: Only ID 3.
pref_filter = fb.types(ContextType.PREFERENCE).build()
run_query("Personalization (Preferences only)", "ui setting", filter_obj=pref_filter)

print("\n✅ Verification Complete.")
