# Feather DB Phase 3: The Living Context Engine
**Official Feature & Capability Guide**

Feather DB has evolved from a simple vector store into a **Multimodal, Graph-Based Intelligence Layer**. This document outlines the core capabilities and input specifications for v0.3.0.

---

## 1. Core Capabilities

### 🪶 Multimodal Pockets (The "One-to-Many" Record)
A single Entity ID (e.g., `100`) can now hold multiple vector signals from different modalities. You no longer need separate IDs for the "image" and "text" of the same object.

*   **Functionality**: Store optional named vectors (`visual`, `text`, `audio`) under one Record ID.
*   **Use Case**: A "Campaign" entity contains its Hero Image (Visual Vector) and its Copy (Text Vector).
*   **Search**: You can query the database using *any* modality to find the same underlying entity.

### 🕸️ Contextual Graph (The "Memory Web")
Records can be explicitly linked to represent relationships, causality, or attribution.

*   **Functionality**: `db.link(source_id, target_id)` creates a directed edge.
*   **Use Case**: Link a `Creative Asset` -> `Conversion Event`.
*   **Traversal**: When retrieving a record, you immediately get its "Context Links" (e.g., seeing that an image is linked to a high-value purchase).

### 🧠 Living Context (Adaptive Decay)
The system tracks detailed usage metrics for every record to emulate human memory.

*   **Functionality**: `db.touch(id)` increments a `recall_count` and updates `last_recalled_at`.
*   **Stickiness**: Records with high recall decay significantly slower ("Core Memories") than one-off records ("Noise").
*   **Automatic Decay**: The `Scorer` automatically down-weights older, un-accessed memories during retrieval.

---

## 2. Input Formats & Data Types

### Basic Inputs
| Field | Type | Description |
| :--- | :--- | :--- |
| **ID** | `uint64` | Unique 64-bit integer identifier for the Entity. |
| **Vector** | `float32[]` | The embedding vector. Dimension must match the index config (default 768). |
| **Modality** | `string` | **[NEW]** The signal type. Standard values: `"text"`, `"visual"`, `"audio"`. Defaults to `"text"`. |

### Metadata Structure
The `Metadata` object attaches structured context to the vector.

| Field | Type | Description |
| :--- | :--- | :--- |
| `content` | `string` | Raw text, JSON summary, or human-readable label. |
| `source` | `string` | URI of the data source (e.g., `s3://...`, `mongo://...`). |
| `type` | `ContextType` | Enum: `FACT`, `EVENT`, `PREFERENCE`, `CONVERSATION`. |
| `tags_json`| `json_string`| List of categorical tags (e.g., `["summer", "video"]`). |
| `creation` | `int64` | Timestamp of the original event (used for decay). |

### Phase 3 Special Fields (Automatic)
*   `links`: `List[uint64]` — Array of IDs this record points to.
*   `recall_count`: `uint32` — Number of times this record was relevant.
*   `last_recalled_at`: `uint64` — Timestamp of last access.

---

## 3. Python API Reference (v0.3.0)

### Ingestion (Multimodal)
```python
# Add Visual Signal to Entity 100
db.add(id=100, vec=image_vec, meta=m, modality="visual")

# Add Text Signal to Entity 100 (Same ID!)
db.add(id=100, vec=text_vec, meta=m, modality="text")
```

### Linking (Graph)
```python
# Link Entity 100 to Event 999
db.link(from_id=100, to_id=999)
```

### Retrieval (Cross-Modal)
```python
# Search Visual Index
results = db.search(query_vec, k=5, modality="visual")

# Result Object
print(results[0].id)             # 100
print(results[0].score)          # Similarity score (decayed)
print(results[0].metadata.links) # [999] -> Context/Attribution!
```

---

## 4. Limits & Performance
*   **Max Graph Links**: Variable (typically <1000 per node recommended for performance).
*   **Ingestion Rate**: ~3,000 multimodal items/sec (Python), ~12,000/sec (C++).
*   **Graph Traversal Speed**: Instant (Embedded Pointer lookup).
*   **Consistency**: Immediate (Graph links are available to queries instantly).
