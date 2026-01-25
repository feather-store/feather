# Phase 2: Context Engine Features

Feather DB v0.2.0+ transforms from a simple vector database into a **context-aware engine** with rich metadata, temporal scoring, and advanced filtering capabilities.

## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Metadata Storage](#metadata-storage)
- [Time-Weighted Retrieval](#time-weighted-retrieval)
- [Advanced Filtering](#advanced-filtering)
- [Python API](#python-api)
- [CLI Usage](#cli-usage)
- [Performance](#performance)

---

## Overview

Phase 2 adds three major capabilities:

1. **Metadata Storage**: Attach structured context to every vector
2. **Time-Weighted Retrieval**: Prioritize recent or important information
3. **Advanced Filtering**: Search within specific subsets of your data

These features enable use cases like:
- Personal AI assistants with long-term memory
- Marketing campaign analytics with temporal trends
- Multi-tenant applications with source-based isolation
- Preference learning systems

---

## Core Concepts

### Context Types

Every record can be classified into one of four types:

```python
from feather_db import ContextType

ContextType.FACT          # Objective information (e.g., "User lives in NYC")
ContextType.PREFERENCE    # User preferences (e.g., "Prefers dark mode")
ContextType.EVENT         # Time-bound occurrences (e.g., "Meeting at 3pm")
ContextType.CONVERSATION  # Dialogue history (e.g., chat messages)
```

### Metadata Structure

```python
from feather_db import Metadata

meta = Metadata()
meta.timestamp = 1706140800      # Unix timestamp
meta.importance = 0.9            # 0.0 to 1.0
meta.type = ContextType.FACT
meta.source = "user_profile"    # Origin identifier
meta.content = "User prefers email notifications"
meta.tags_json = '["notification", "preference"]'
```

---

## Metadata Storage

### Adding Records with Metadata

```python
import time
from feather_db import DB, Metadata, ContextType

db = DB.open("my_context.feather", dim=384)

# Add a fact
fact_meta = Metadata()
fact_meta.timestamp = int(time.time())
fact_meta.importance = 1.0
fact_meta.type = ContextType.FACT
fact_meta.source = "onboarding"
fact_meta.content = "User signed up for premium plan"

db.add(id=1, vector=embedding, metadata=fact_meta)

# Add a preference
pref_meta = Metadata()
pref_meta.timestamp = int(time.time())
pref_meta.importance = 0.8
pref_meta.type = ContextType.PREFERENCE
pref_meta.source = "settings_ui"
pref_meta.content = "Dark mode enabled"

db.add(id=2, vector=embedding, metadata=pref_meta)
```

### Persistence

Metadata is automatically saved with the binary format (v2):

```python
db.save()  # Persists vectors + metadata

# Later...
db2 = DB.open("my_context.feather", dim=384)
results = db2.search(query_vector, k=5)
for r in results:
    print(f"{r.metadata.content} (importance: {r.metadata.importance})")
```

---

## Time-Weighted Retrieval

### Temporal Decay Scoring

Combine semantic similarity with recency bias:

```python
from feather_db import ScoringConfig

# Configure exponential decay
scoring = ScoringConfig(
    half_life=86400 * 7,  # 7 days in seconds
    weight=0.5            # 50% similarity, 50% recency
)

results = db.search(query_vector, k=10, scoring=scoring)
```

### How It Works

The final score is computed as:

```
final_score = (1 - weight) * similarity + weight * recency_factor

where:
  recency_factor = importance * exp(-λ * age)
  λ = ln(2) / half_life
```

### Use Cases

**Recent events matter more:**
```python
# Find urgent tasks (short half-life, high weight)
urgent_scoring = ScoringConfig(half_life=86400, weight=0.7)
tasks = db.search(query, k=5, scoring=urgent_scoring)
```

**Long-term preferences:**
```python
# Stable preferences (long half-life, low weight)
pref_scoring = ScoringConfig(half_life=86400 * 365, weight=0.1)
prefs = db.search(query, k=3, scoring=pref_scoring)
```

---

## Advanced Filtering

### Filter by Type

```python
from feather_db import FilterBuilder

# Only search facts
filter = FilterBuilder().types(ContextType.FACT).build()
results = db.search(query_vector, k=10, filter=filter)
```

### Filter by Source

```python
# Only data from a specific integration
filter = FilterBuilder().source("slack:eng-team").build()
results = db.search(query_vector, k=10, filter=filter)
```

### Filter by Timestamp Range

```python
import time

# Last 30 days
thirty_days_ago = int(time.time()) - (86400 * 30)
filter = FilterBuilder().after(thirty_days_ago).build()
results = db.search(query_vector, k=10, filter=filter)
```

### Filter by Importance

```python
# High-importance items only
filter = FilterBuilder().min_importance(0.8).build()
results = db.search(query_vector, k=10, filter=filter)
```

### Combining Filters

```python
# Complex query: recent high-importance events from calendar
filter = (FilterBuilder()
    .types(ContextType.EVENT)
    .source("google_calendar")
    .after(int(time.time()) - 86400 * 7)
    .min_importance(0.7)
    .build())

results = db.search(query_vector, k=5, filter=filter)
```

---

## Python API

### Complete Example

```python
import time
import numpy as np
from feather_db import DB, Metadata, ContextType, ScoringConfig, FilterBuilder

# Initialize
db = DB.open("context.feather", dim=128)

# Add diverse records
records = [
    {
        "id": 1,
        "content": "User prefers morning meetings",
        "type": ContextType.PREFERENCE,
        "importance": 1.0,
        "source": "calendar_analysis",
        "age_days": 100
    },
    {
        "id": 2,
        "content": "Urgent: Q1 budget review tomorrow",
        "type": ContextType.EVENT,
        "importance": 1.0,
        "source": "email",
        "age_days": 0.1
    }
]

NOW = int(time.time())
for rec in records:
    meta = Metadata()
    meta.timestamp = NOW - int(rec["age_days"] * 86400)
    meta.importance = rec["importance"]
    meta.type = rec["type"]
    meta.source = rec["source"]
    meta.content = rec["content"]
    
    vector = np.random.rand(128).astype(np.float32)
    db.add(rec["id"], vector, meta)

# Search with time-weighting
scoring = ScoringConfig(half_life=86400 * 7, weight=0.5)
query = np.random.rand(128).astype(np.float32)
results = db.search(query, k=5, scoring=scoring)

for r in results:
    age_days = (NOW - r.metadata.timestamp) / 86400
    print(f"[{r.metadata.type}] {r.metadata.content}")
    print(f"  Score: {r.score:.3f} | Age: {age_days:.1f} days")

db.save()
```

---

## CLI Usage

### Adding with Metadata

```bash
# Create a vector file
python -c "import numpy as np; np.save('vec.npy', np.random.rand(128))"

# Add with full metadata
feather add my_db.feather 1 \
  --npy vec.npy \
  --content "User completed onboarding" \
  --source "app_events" \
  --context-type 0 \
  --importance 0.9 \
  --timestamp $(date +%s)
```

### Searching with Filters

```bash
# Search only facts from a specific source
feather search my_db.feather \
  --npy query.npy \
  --k 10 \
  --type-filter 0 \
  --source-filter "app_events"
```

---

## Performance

### Benchmarks (10,000 records, dim=128)

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Ingestion | ~1,280 rec/s | - |
| Simple Search (k=10) | - | 0.08 ms |
| Filtered Search | - | 0.20 ms |
| Complex (Filter + Score) | - | 0.68 ms |

**Database Size**: ~5.5 MB for 10k records

### Optimization Tips

1. **Use appropriate half-life**: Shorter half-life = faster decay
2. **Filter early**: Apply filters before scoring when possible
3. **Batch operations**: Add multiple records before calling `save()`
4. **Index size**: Larger `ef_construction` = better recall, slower build

---

## Migration from v0.1.x

### Breaking Changes

- Binary format changed from v1 to v2 (metadata support)
- `search()` now returns `SearchResult` objects instead of tuples
- Old v1 databases can still be read (backward compatible)

### Upgrade Path

```python
# Old code (v0.1.x)
ids, distances = db.search(query, k=5)

# New code (v0.2.x)
results = db.search(query, k=5)
for r in results:
    print(f"ID: {r.id}, Score: {r.score}")
    print(f"Content: {r.metadata.content}")
```

---

## Next Steps

- **Phase 3 Ideas**: Multi-vector records, automatic pruning, session management
- **Integrations**: LangChain, LlamaIndex adapters
- **Deployment**: Docker images, cloud-native builds

For more examples, see the `examples/` directory in the repository.
