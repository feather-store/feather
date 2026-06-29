# Vector Dimensions in Feather

How Feather handles embedding dimensions, and how to use any dim flexibly.

## TL;DR

- **Any dimension works** — 384, 512, 768, 1024, 1536, … up to 1,048,576. No power-of-two requirement.
- A namespace's dimension is **defined by its data**: the first vector you insert
  fixes it (or pin it explicitly at create time).
- **Mix freely across namespaces.** One namespace can be 1024, another 768,
  another 384 — all on the same instance at the same time.
- **One dim per modality** inside a namespace. The first vector locks it; later
  vectors of a different length get a clean `400`.
- The recommended pattern is **one namespace per embedding model / dimension**.

## The one rule

A namespace has exactly **one dimension per modality**, and it's set the moment
the first vector lands in that modality (the index is built at that size).

```
empty namespace          → no fixed dim yet
+ first vector (dim 1024) → namespace is now 1024 for that modality
+ another vector (1024)   → ✓ accepted
+ another vector (768)    → ✗ 400  "vector dim 768 != index dim 1024"
```

This is inherent to how the ANN index stores vectors — every vector in an index
must be the same length. It is **not** a server-wide setting; each namespace
decides its own.

## Three ways dimension is flexible

### 1. Different dim per namespace

Run as many different-dimension namespaces as you want, simultaneously:

| Namespace | Dim  | Typical model                     |
|-----------|------|-----------------------------------|
| `proj_a`  | 1024 | BGE-large, GTE-large, Cohere v3   |
| `proj_b`  | 1536 | OpenAI text-embedding-3-small     |
| `proj_c`  | 768  | many sentence-transformers        |
| `proj_d`  | 384  | MiniLM / small models             |

No reconfiguration — each namespace adopts its own dim from its data.

### 2. Different dim per modality, in one namespace

Each modality is an independent index, so a single namespace can hold:

```
text   → 768
image  → 512
audio  → 1024
```

Search within a modality only (you can't search text vectors against the image
index — different spaces).

### 3. Any value

384, 512, 768, 1000, 1020, 1024, 1536, 3072 … all valid. (Dimensions that are a
multiple of 16 — e.g. 768, 1024, 1536 — use the widest SIMD path and are
marginally fastest; everything else is fully supported and exact.)

## Creating a namespace at a specific dim

You don't have to pre-declare a dim — the first inserted vector defines it. But
you can pin it up front so the dashboard reports it immediately and clients get
an early `400` on mismatch:

```bash
# Pin the dimension at create time
curl -X POST "$BASE/v1/namespaces" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"name": "proj_a", "dim": 1024}'

# Or just start inserting — the first vector fixes the dim
curl -X POST "$BASE/v1/proj_a/vectors" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"id": 1, "vector": [/* 1024 floats */], "metadata": {"content": "…"}}'
```

## Importing vectors (no silent reshaping)

`POST /v1/{ns}/import` and `POST /v1/{ns}/ingest_text` **never pad or truncate**
your vectors to make them fit. If an item's vector length disagrees with the
namespace's established dim, that item is rejected with a clear per-item error
and the rest still import. This guarantees your embeddings are stored exactly as
produced.

```jsonc
// import response
{ "inserted": 9, "skipped": 1, "embedded": 0,
  "errors": ["item 4: dim mismatch: got 900, expected 1024"] }
```

## What you can't do (and why)

- **Change a namespace's dim after it has data.** The index is built at that
  dimension. To switch, create a new namespace (or upload a fresh `.feather`)
  and re-ingest.
- **Auto-embed into a dim the model doesn't produce.** Server-side embedding
  outputs whatever the configured model outputs; a mismatch is rejected honestly
  rather than reshaped. Supplying your own vectors has no such tie — that's the
  fully flexible path.

## Recommended pattern

**One namespace per embedding model / dimension.** Keep each model's vectors in
their own namespace at that model's native dimension. This keeps every index
internally consistent, lets different teams/projects use different models on the
same instance, and means you never have to think about a global default.
