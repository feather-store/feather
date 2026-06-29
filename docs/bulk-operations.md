# Bulk Operations in Feather

Delete, insert, and upload many records in one call — instead of looping
single-record requests.

## TL;DR

- **Bulk delete:** `POST /v1/{ns}/records/batch_delete` — many ids and/or a whole
  `entity_id` in one call, one save.
- **Bulk insert:** `POST /v1/{ns}/import` — many records (with vectors, or
  auto-embedded from `metadata.content`).
- **Bulk upload:** `POST /v1/admin/upload` — adopt a whole local `.feather` file
  as a namespace.
- **Never loop single-record `DELETE`/`POST` over a large namespace** — each
  single-record write re-saves the entire namespace, so N calls = N full saves.

## Bulk delete

```
POST /v1/{ns}/records/batch_delete
{
  "ids":       [101, 102, 103],   // optional: explicit record ids
  "entity_id": "APR26045",        // optional: also delete every record with this entity_id
  "cascade":   false              // optional: prune graph edges to deleted ids (default false)
}
```

The delete set is `ids` ∪ (all records whose `metadata.entity_id` matches). The
whole batch runs under **one** namespace lock and is followed by **one** save.

```jsonc
// response
{ "namespace": "pocketus", "requested": 3, "deleted": 3,
  "not_found": 0, "edges_pruned": 0,
  "hint": "run POST /compact to reclaim space" }
```

```bash
# delete a list of ids
curl -X POST "$BASE/v1/pocketus/records/batch_delete" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"ids": [101, 102, 103]}'

# delete every record for one entity
curl -X POST "$BASE/v1/pocketus/records/batch_delete" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"entity_id": "APR26045"}'
```

Notes:
- Up to **100,000 ids** per call.
- Ids that don't exist (or are already deleted) are counted in `not_found`, not
  an error — so retries are safe.
- `cascade` is **off by default** for speed; deletes are soft (reclaimed by
  `POST /v1/{ns}/compact`). Graph export already ignores edges to deleted
  records, so leave `cascade` off unless you specifically need the edge lists
  cleaned immediately.
- Dashboard: **Maintenance → Bulk delete** (ids and/or entity_id + cascade).

### ⚠️ Don't loop single-record DELETE

`DELETE /v1/{ns}/records/{id}` saves the **entire** namespace on every call. A
loop that deletes thousands of records one at a time will re-serialize a large
namespace thousands of times and can stall the server. Always collect the ids
and send **one** `batch_delete`.

```python
# BAD — N full saves of the namespace, can wedge the server
for rid in ids:
    requests.delete(f"{BASE}/v1/{ns}/records/{rid}", headers=H)

# GOOD — one lock, one save
requests.post(f"{BASE}/v1/{ns}/records/batch_delete",
              headers=H, json={"ids": ids})
```

## Bulk insert

```
POST /v1/{ns}/import
{
  "items": [
    { "id": 1, "vector": [/* dim floats */], "metadata": { "content": "…", "entity_id": "E1" } },
    { "id": 2, "metadata": { "content": "embed me server-side" } }
  ],
  "modality": "text"
}
```

Each item has an `id` and **either** a precomputed `vector` (must match the
namespace dim — see [dimensions](dimensions.md)) **or** `metadata.content`, which
is embedded server-side via the configured provider. Items are inserted as one
batch. Vectors are stored exactly as given — no padding/truncation; a dim
mismatch is reported per item, not silently reshaped.

```jsonc
// response
{ "inserted": 9, "skipped": 1, "embedded": 0,
  "errors": ["item 4: dim mismatch: got 900, expected 1024"] }
```

### Importing many batches (large datasets)

Import is **fast and constant-time per batch** regardless of how big the
namespace gets: it appends to a write-ahead log instead of re-writing the whole
`.feather` on every call. A full save happens automatically at most once every
~30s per namespace (and on shutdown), so your data is always durable.

When you finish a bulk-load session, force one final full save so the file is
fully compacted — either set `"flush": true` on your last batch, or call:

```bash
curl -X POST "$BASE/v1/{ns}/flush" -H "X-API-Key: $KEY"
```

```python
# stream a large dataset in batches; flush once at the end
for i, chunk in enumerate(batches):
    last = (i == len(batches) - 1)
    requests.post(f"{BASE}/v1/{ns}/import", headers=H,
                  json={"items": chunk, "flush": last})
```

## Bulk upload (whole `.feather`)

```
POST /v1/admin/upload        (multipart form)
  file=@local.feather
  namespace=my_ns
  overwrite=false
```

Adopts a locally-built `.feather` file as a cloud namespace — the vectors, graph,
attributes, and prebuilt index come over intact (no re-embedding). The file is
streamed to disk, its magic + format version are validated, then atomically moved
into place. Returns `409` if the namespace exists (unless `overwrite=true`).

```bash
curl -X POST "$BASE/v1/admin/upload" \
  -H "X-API-Key: $KEY" \
  -F "file=@./my_data.feather" -F "namespace=my_ns"
```

Dashboard: **Namespaces → Import .feather**.

## Which to use

| Goal | Endpoint |
|------|----------|
| Delete a set of ids / a whole entity | `POST /v1/{ns}/records/batch_delete` |
| Delete everything under a `namespace_id` | `POST /v1/{ns}/purge` |
| Insert/upsert many records | `POST /v1/{ns}/import` |
| Move a local DB to the cloud as-is | `POST /v1/admin/upload` |
| Reclaim space after deletes | `POST /v1/{ns}/compact` |
