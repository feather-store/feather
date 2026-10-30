# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### RemoteDB keeps its connection open

- **The default transport reuses one connection per thread.** It used
  `urlopen`, which opens a new connection per request: a TCP handshake, plus
  a TLS one over https, on every call. Now standard-library `http.client`
  with keep-alive, still no dependency. Measured on localhost 0.26 → 0.14 ms
  a call; over a network each call saves the handshake round trips. An idle
  connection the server closed is detected before reuse and replaced; if it
  closes between that check and the send, an idempotent request (GET, PUT,
  DELETE) is sent once more on a fresh connection and a POST raises, since
  it may have been applied. A proxy from the
  environment still goes through `urlopen`. `RemoteDB.close()` now closes
  the connections it opened. `tests/test_remote_keepalive.py`.

### Performance gate

- **`benchmarks/perf_gate.py`** turns the measurements behind this release's
  speedups into checks, in ~5 s: a repeated embed never reaches the provider,
  an open with text costs < 2.5× one without (12× before lazy BM25),
  `RemoteDB` makes one connection for many calls, and search / keyword
  search / a cold `Pocket.hot()` stay inside loose budgets. Every check is a
  count or a same-run ratio, so runner speed cancels out. Runs as section 10
  of `scripts/verify.sh` and in CI (ubuntu, 3.12); `--json` writes results.

### Faster opens: BM25 built on first use

- **The keyword index is built by the first keyword or hybrid search, not
  by open.** Building it was ~0.8 s of a ~0.9 s open at 100k records with
  text, paid by every reader process and every Cloud namespace reopen, while
  `Pocket` and `FeatherStore` with an embedder only run vector search and
  never query it. Measured at 100k × 384 with ~50-word records: open
  920 → 130 ms; the first keyword search then takes the ~0.8 s once, later
  ones 2.5 ms. A handle that never runs a keyword query never holds the
  index in memory either. `tests/test_bm25_lazy.py`.

### Smaller BM25 index

- **BM25 postings are 8 bytes, down from 16.** A posting names its document
  by a u32 doc number instead of the u64 id, and the document length is
  stored once per doc instead of in every posting. Measured at 100k docs of
  ~50 words (4.2M postings): a reader's heap after open 211 → 167 MB, query
  latency unchanged. Every process that opens the file builds its own index,
  so this is per reader. The index is never serialised: no format change.

### Client-side embedding cache

- **`make_embedder()` / `default_embedder()` cache provider calls.** An agent
  re-embeds the same query and memory text turn after turn; each miss is a
  100–300 ms provider round trip, against a sub-millisecond search. Provider
  embedders (used by `feather-agent`, `FeatherStore`, `feather-serve`) now keep
  a thread-safe LRU keyed by the text's SHA-256 — `FEATHER_EMBED_CACHE`
  entries, default 2048, `0` disables — the same knob and default the Cloud
  API's `EmbeddingProvider` already had. Failures are not cached; hits return
  a copy; `embed.stats` counts calls and hits. `tests/test_embedder_cache.py`.
- **`local` embed provider.** `FEATHER_EMBED_PROVIDER=local` (or
  `--embed-provider local`) embeds on the CPU with a small ONNX model via
  `fastembed` (`pip install feather-db[local]`; default
  `BAAI/bge-small-en-v1.5`, 384-d): 4–10 ms a text measured, against
  100–300 ms for an API call, with no key and no network after the first
  model download. Opt-in; vectors from different models are not comparable,
  so choose one per database.

### Valid time (bitemporal records)

- **`Metadata.valid_at`: when it happened, beside when it was recorded.**
  A session on 05/14 saying "last Saturday I went to a wedding" is
  recorded 05/14 and valid 05/06; `timestamp` could only hold one of them.
  0 means not given, and `valid_time` then falls back to the timestamp.
  Stored as a reserved attribute, like the native key: no format change,
  the WAL carries it, and a build without it writes it back intact.
- **Indexed and filterable.** `DB.newest(..., by_valid=True)` walks a
  `(valid time, id)` index (O(log n + limit)); `SearchFilter.valid_after` /
  `valid_before` and `FilterBuilder.valid_between()` filter on it. Over
  HTTP: `valid_at` on metadata, `valid_after`/`valid_before` on search,
  hybrid and keyword, `by_valid` on `GET /newest`; `RemoteDB` matches.
- **Episodes order by an index walk.** `FeatherStore` maps a numeric
  `valid_at` in an item's value to the engine field, and `AgentMemory`
  writes one per episode, so `episodes.recent()` / `all()` no longer read
  and sort the namespace. A namespace with episodes from before 0.22
  falls back to the old key scan.

### AgentMemory

- **`ingest_facts(facts)` routes extracted facts by lifetime:** events to
  episodes (one per entity and day, so a retold event is not a second
  one), preferences to `preferences`, everything else to `facts` with
  supersession. Each write carries the fact's valid time, not the ingest
  time.
- **`facts.state(..., at=)` and `preferences.set(..., at=)` take a valid
  time.** A superseded fact's `until` is when its successor became true,
  not when the successor was written, and an older fact ingested late
  goes into the history in date order instead of replacing the current
  value. An older preference does not roll back a newer one.

### Extractors

- **`SessionFactExtractor`: dated facts from a whole session in one call.**
  `FactExtractor` reads one text at a time and is never told when it was
  said, so "last Tuesday" stayed relative. The new extractor reads a
  session per call (cheaper than per turn, and each turn is read with the
  rest), gets the session date, makes every date absolute, and keys each
  fact by `(subject, attribute)` with the turn it came from.
  `link_validity()` chains facts with the same key: each keeps its
  valid-from date and gains a valid-until instead of being overwritten;
  `timeline()` renders the values that changed. Exported from
  `feather_db.extractors`.

- **Session facts carry `kind`, `entity`, `amount`, `own` (extractor
  0.2.0).** `kind` is event / state / preference / other; `entity` names
  an event's specific instance the same way each time it comes up;
  `own` separates what the user did from what they mentioned, asked about
  or were told. `ledger(facts)` renders events one row per distinct
  entity, every date it was mentioned on, others' marked NOT the user's
  own; `profile(chains)` the latest value of each preference. The version
  bump invalidates cached v0.1 extractions.

### LongMemEval bench: the reader sees every retrieved session

- **The context budget is filled breadth first.** All 500 contexts sat at
  the 160k-char cap: sessions were admitted best-first at the fullest
  rendering that fit, so ~14 of ~22 retrieved sessions reached the reader.
  Now every session is admitted at its most compact rendering, then
  upgraded in rank order while the budget allows (`lme_context.render`).
- **Admission is bounded per session (`EVIDENCE` level).** Multi-round
  retrieval merges ~100 hit turns per round, so a top session carried 10+
  hits and cost ~25k chars even at its compact rendering: 160k admitted
  12-15 of ~25 sessions and cut golds retrieved at ranks 10-24. Every
  session is now admitted at its best hit ±2 turns (or HITS_ONLY, if
  cheaper), ~3k chars at most, before any is expanded.
- **`--facts session` adds an event ledger and a preference profile.**
  Count questions get the ledger of the events most similar to the
  question (`LEDGER_K` = 60, kind-filtered search) with every mention of
  each; every question gets the user's preferences (`PROFILE_K` = 30).
  Both sit above the sessions with their reading rules in their headers,
  so `ANSWER_SYSTEM`, and every run without facts, is unchanged. Records
  gain `ledger` and `profile`.
- **A LongMemEval number can no longer mean something else.** `--limit`
  defaulted to 5 and `--split all` did not override it, so
  `lme_final.sh` scored 5 questions (a "final" 1.000) and `lme_dev.sh`
  refused to start; `--limit` now defaults to every question. A question
  that crashed was dropped from the denominator (an outage on hard
  questions raised the score); it is now a record scored wrong, bucket
  `run_error`, as is an answerer error. Every result carries
  `reportable` / `not_reportable_because` (subset, split, oracle,
  substring judge, hash embedder, missing or non-official judge,
  errors, uncommitted code) and prints it. `python -m bench.lme_eval
  --check RUN.jsonl N` fails a questions file that is short, duplicated,
  errored or unjudged, and both scripts run it. LLM retries back off for
  ~2 minutes (was ~4 s, short of a rate-limit window). In the routed
  reader a failed check keeps the draft instead of failing the question.
- **Measured: the routed reader is worse; `--reader verified` keeps
  what survived.** All 500, official judge: routed + map-reduce 0.794 vs
  0.904 (75 answers lost, 20 gained, McNemar p < 0.001). The separate
  premise check flipped 81 answers to "not mentioned", 46 of them right
  (it demands every premise from a user turn, so questions about what
  the assistant said fail; unanswerable questions are built from
  near-miss entities, so no word-overlap guard separates good flips from
  bad: a guard that stops the 46 also stops 25 of 27 good ones).
  Map-reduce lost 11 net on aggregations (98 -> 87 right); the sliced
  per-route prompts lost on preferences. Abstention re-search + quoted
  verify changed one answer, to a right one, and every quote was real.
  `--reader verified` is the single prompt plus that verify step;
  `routed` is kept only to reproduce the result.
- **Routed reader (`--reader routed`, `bench/lme_reader.py`).** One
  prompt with six procedures made the premise check and the abstention
  rule pull against each other: the 0.904 run returned "not mentioned"
  on 11 answerable questions (9 with all evidence in context) and
  answered 5 unanswerable ones. Now one short classify call routes each
  question to its procedure alone (sliced from `ANSWER_SYSTEM`, so the
  text is the measured one). An abstention is searched again (paraphrased
  queries + BM25 on the key nouns) and verified by a call that must quote
  the answering turn; the answer stands only if the quote is found in what
  it was shown. Every other factual answer, including a recovered one,
  gets its premises checked by their own call (SUPPORTED with a quote, or
  UNSUPPORTED). `--reader single` (the default) is unchanged. Records gain
  `reader` (route, draft, research, verify, premises).
- **Map-reduce covers every aggregation.** With `--count-map`, "most /
  least / in what order" questions are mapped too and reduced by
  `AGG_REDUCE_SYSTEM`; counts keep `REDUCE_SYSTEM`. With `--facts
  session` the reduce also reads the event ledger rows.
- **New failure bucket `packing_miss`:** every gold session was retrieved
  but one was cut by the budget. `retrieval_miss` now means a gold was
  never retrieved. `summarize()` recomputes buckets, so older JSONL is
  relabelled: the 0.904 baseline's 8 retrieval misses are 7 packing, 1
  retrieval.
- **Shortening cuts assistant prose, not user turns.** User turns are 13%
  of the text and 94% of the evidence; the compact renderings used to clip
  them to 300 chars or drop them when they were not next to a hit. Each
  record now has `evidence_levels`, and the summary `evidence_shortened`.
- **Answer prompt:** counting writes an IN/OUT ledger (time window, the
  user's own, duplicates, latest running total); "unknown" requires a
  second search under other words; the two weeks before the current date
  are listed with weekdays. A date-difference answer given without
  `DAYS_BETWEEN` is sent back once, and the tool also reports rounded
  months and years.
- **`--count-map retrieved|haystack`: counts by map-reduce.** Each session
  is read whole by its own short call that lists every candidate instance
  with an absolute date and a status (did / plans / asked / assistant said
  / running total) and decides nothing; one reduce call reads only those
  lists and applies the window, ownership, duplicate and running-total
  rules, with `DAYS_BETWEEN`. Each read is ~10k chars instead of one 160k
  read of a shortened context. `haystack` maps every session (an upper
  bound with no retrieval). Off by default; per-record `count_map` stats,
  and a failed map step falls back to the single read.
- **Premise check in the answer prompt.** A factual answer first lists
  what the question presupposes and finds where the user says each; a
  premise found only in an assistant reply, or replaced by something else,
  or a missing operand, makes the answer "not mentioned". Every example in
  the prompts is invented: the earlier ones were lifted from failing test
  questions, which tunes on test (a test now fails on any of them).
- **`--map-provider` / `--map-model`**: the count map calls on a cheap
  model under a strong answerer.
- **Gemini 2.5 Pro as answerer no longer disables multi-round retrieval.**
  Calls of ≤512 tokens asked for `thinking_budget=0`, which Pro rejects;
  the planner's failure was swallowed as "no follow-ups". Pro now gets
  its minimum budget (128).
- **`scripts/lme_dev.sh`** runs one-change ablations on the dev split
  (current, countmap, strong, facts, votes, tokens), each compared with
  the baseline; **`scripts/lme_final.sh`** refuses a dirty tree, runs all
  500 once and prints the headline and the held-out 400. `pip install
  feather-db[bench]` adds the tokenizer dependencies.
- **A frozen dev/test split** (`bench/datasets/longmemeval_s_dev.json`, 100
  of 500, stratified by type × abstention, picked by sha1 of the id).
  `--split dev|test|all`. Tune on dev; the headline stays all 500.
- **`--context-tokens N`** budgets the context in the answer model's
  tokens (Gemini's local tokenizer, `bench/tokens.py`; needs
  `sentencepiece` + `protobuf`). A 160k-char context was 34–40k tokens.
- **Results record the exact command and whether the tree was dirty**
  (untracked code counts): `commit` alone did not reproduce a run made
  on uncommitted changes.
- **`python -m bench.lme_eval A.jsonl B.jsonl` compares two runs** question
  by question: exact McNemar p and the paired SE, overall and per type,
  and the ids each run fixed or broke. One run's score has a ±1.3-point
  standard error at n=500; a headline difference is not evidence.

### Deleted HNSW slots are no longer reused

- **`add()` after `forget()` could make other records unfindable.** Reusing
  a deleted slot ran hnswlib's `updatePoint`, which re-prunes every old
  neighbour of the slot and could drop another record's last incoming link.
  Under forget + add churn, 44 of 24,000 new and 30 of 16,000 untouched
  records were unreachable even by their own vector at `ef=400`. A new
  record now always takes a fresh slot, which brings that back to a clean
  build's rate (~0.02%, inherent to HNSW pruning). Tombstones stay as
  routing nodes until `compact()`; use `set_auto_compact()` to bound them.
  Reverses 0.19's "Deleted HNSW slots are reused".

### Cloud API

- **`POST /v1/{ns}/ingest_file` stores a file by what is in it, not its
  name.** Text is read as is; images, video, audio and PDFs are described by
  a multimodal model (`app/describe.py`, Gemini: objects and colours, people
  and their clothing, the setting, every word of visible text, every spoken
  word). The description is the record's content (BM25) and is embedded with
  the configured provider. The file name is kept as the `file_name`
  attribute and never searched. Records are keyed `file:<sha256>`, so the
  same file uploaded again updates one record and reuses its description
  (`refresh=true` re-reads it). Files over `FEATHER_DESCRIBE_INLINE_MB` (14)
  go through Gemini's File API and are deleted from it afterwards. Config:
  `FEATHER_DESCRIBE_API_KEY` (or `GEMINI_API_KEY` / `GOOGLE_API_KEY`, or a
  Gemini embedding key), `FEATHER_DESCRIBE_MODEL` (default
  `gemini-2.5-flash`), `FEATHER_MAX_INGEST_FILE_MB` (512). Its own admission
  class, `FEATHER_MAX_INFLIGHT_FILES` (8). `pypdf`, when installed, adds a
  PDF's full text layer.
- **`hybrid_search` without a `vector` embeds `query` on the server**, with
  the same model that embedded the records, so `{"query": "blue car"}` is a
  complete content search. The admin UI's hybrid search now does this when
  an embedder is configured (it used to send a seeded random vector), and has
  an "Upload creative" button that shows what was understood.
- **A namespace move no longer answers 500 mid-handover.** A request that
  reached the target shard while the source still held the file's write
  lock got a 500 (a 502 through the gateway). It is now a retryable 503
  `namespace_held` with Retry-After.
- **A request whose handle is closed under it retries up to 5 times**, not
  once. With a short `FEATHER_IDLE_CLOSE_S`, a loaded host could lose the
  reopened handle as well.

### Three slowdowns against 0.21 removed

A full benchmark run of this branch against master (v0.21.0) on an M1
(`benchmarks/runs/2026-10-28-m1-vs-master/`) found three things slower. All
three are fixed.

- **`export_graph_json(namespace)` was ~4× slower (7.6 → 29 ms at 100k records).**
  It walked the whole store three times for one namespace. It now reads the
  namespace / entity index, so the cost is O(matches): 4 ms. Forgotten records
  are no longer exported (the "a forgotten record is gone" contract), and
  nodes come out in id order.
- **Several writers beside readers were slower than one.** The phase-fair lock
  admitted one writer per reader phase, so every write paid a full round of
  reader wake-ups. 8 readers + 4 `link()` writers did 2.6k writes/s, against
  9.8k with a single writer. A writer phase now runs the writers already
  queued back to back, for at most 100 µs, before the readers' turn:
  6–13k writes/s. Reads in that mix drop ~20% and remain 3–4× master's, since
  master's lock prefers writers. Expensive writes (`add()`, ~0.5 ms) still get
  one phase each, so reader latency beside them is unchanged; a count bound
  instead of a time bound cut those reads 9.8k → 0.7k/s.
- **`GET /records?limit=100` was 4.9 → 7.1 ms.** Each record became a pydantic
  model built from ~17 bound properties, more since rows carry their edges.
  The route now uses `Metadata.to_dict()`, the C++ row the search fast path
  already uses (same JSON). Now 5.0–5.9 ms against master's 5.4–6.5.

Checked and not a regression: int8 search. The benchmark that showed it
slower converted a float modality without saving. Set the same way on both
builds (50k × 768, ef=50), master's int8 runs at 1.40 ms with recall@10 0.659,
and the branch at 0.51 ms with 0.832.

### Sharding: rendezvous hashing, live moves, lazy namespaces

**Adding a shard moved ~80% of namespaces.** Ownership was `crc32(ns) % N`, so
every namespace's remainder changed with N. It is now rendezvous hashing over
stable shard ids (`FEATHER_SHARD_IDS` / `FEATHER_SHARD_ID`; the old
`FEATHER_SHARD_COUNT` / `FEATHER_SHARD_INDEX` mean ids `"0".."N-1"`).
- Adding a fifth shard to four moves 20% of 20,000 test namespaces, all to the
  new shard, against 80% before.
- Load stays within 10% across shards.
- Upgrading moves ~N/(N+1) namespaces once. With a shared volume that is a
  clean restart; `FEATHER_SHARD_HASH=modulo` keeps the old rule meanwhile.
- Shard ids are strings: `owner_shard` in a 421 is now `"1"`, not `1`.

**A placement table overrides the hash**, so one hot namespace can move alone,
with no restart: `POST /v1/admin/placement/move` on the gateway.
- **Each shard keeps the table** under an epoch and persists it in its data
  dir. It adopts only a newer epoch, from `PUT /v1/admin/placement`.
- **The move.** The target takes the table first. The source takes it next:
  it drains in-flight writes, checkpoints and closes the namespace, which
  releases the file. The gateway then routes to the target, and requests in
  the gap are 421-redirected there.
- **Tested** with a writer hammering the namespace throughout: every write
  landed exactly once, then the namespace moved back and its override was
  removed.
- **Needs a data volume both shards see**: a target that cannot see the file
  is refused (409).

**Every namespace opened at startup and stayed open.** A shard is limited by
memory, and that memory was the sum of every tenant it had ever held.
- Namespaces now open on their first request and close after
  `FEATHER_IDLE_CLOSE_S` idle (default 900). Past `FEATHER_MAX_OPEN` (default
  unlimited), the least recently used close first.
- Measured (`benchmarks/idle_namespaces.py`, 40 tenants × 5k × 256): 564 MB
  with all of them open, 77 MB idle, 116 MB with 4 hot. That is 73 → 354
  tenants per GB, at 27 ms per reopen.
- `/health` reports `namespaces_loaded` (open now) and `namespaces_known`.
  `/v1/admin/overview` counts closed namespaces from the stats they had when
  they closed (`_namespace_stats.json`), without opening them.
- **A request racing a close** (it took the handle just before) is run once
  more on a fresh handle. The engine now raises `feather_db.ClosedError`,
  before the call does anything, and the closer holds the namespace's write
  lock.
- **A bug the race test found on the way:** the write paths raised a plain
  `RuntimeError` for a closed handle, so a write racing a close returned a 500
  instead of being re-run.

### The contracts, stated and tested (CONTRACTS.md)

Four promises, each with its limits and the tests that enforce it.

- **Exact results by default.** Namespace graphs, added in this release, are
  now opt-in (`FEATHER_NS_ANN_MIN` / `set_namespace_ann_min`). Filtered
  search is an exact scan again unless asked otherwise. Unfiltered vector
  search is the one approximate query by design.
- **One writer, many readers**, as since 0.20, now with a contract test that
  uses real processes.
- **Read-only means a consistent snapshot. This was false.** Under a writer
  saving every 7 adds, 187 of 7,395 read-only opens saw a torn view: the old
  base plus later WAL records, missing the writes in between. A commit could
  finish, and delete `<wal>.old`, between the reader's read of the base and its
  read of the WALs.
  - Every WAL file now starts with a `WALGEN` record (the base generation its
    writes follow), written in the same `write()` as the header.
  - A reader checks that base → `<wal>.old` → `<wal>` is unbroken and retries
    with backoff when it is not.
  - A broken chain that does not change between two attempts is the files as
    they are (a crash), and is loaded.
  - Old builds skip the new record.
  - Result: 0 torn opens in 13,000.
- **One feature set in every mode.** `feather_db.RemoteDB` is the embedded `DB`
  over the Cloud API: same methods, arguments, return attributes and
  exceptions.
  - `feather_db.remote.PARITY` maps every public `DB` method to its route.
    `tests/test_parity.py` fails on a method without one, a missing route, or
    different arguments.
  - `tests/test_contracts.py` runs the same tests against both modes (36 in
    all).
  - New routes:
    - `GET /by_key`, `GET /newest`, `GET /ids` and `GET /metadata_namespaces`;
    - `POST /records/{id}/touch`, `POST /forget_expired`, `POST /auto_link`
      and `POST /vacuum`;
    - `GET /admin/engine` (every engine statistic and setting);
    - `PUT /admin/settings` (ef, auto-compaction, namespace graphs, on-disk
      and in-RAM int8).
  - `/batch` gains `upsert`, records named by key (`key`, `to_key`) and
    `require_target`.
  - Search bodies take `types`.
  - Metadata carries `key`, `edges`, `recall_count` and `last_recalled_at` in
    both directions.

**Behaviour changes, from making the modes agree:**

- `DB.get_metadata()` returns `None` for a forgotten record. It used to return
  a `"_forgotten"` shell until the next save, so the answer changed across a
  reopen and disagreed with the API's 404. `get_metadata(id, include_dead=True)`
  shows the tombstone.
- `DB.link()` returns whether it linked: `False` when the source does not
  exist, where it used to return nothing and do nothing. The API's single-edge
  route now matches it: a forward-referenced target is allowed again, and a
  missing source is a 404.
- API metadata: an omitted (null) `timestamp` or `namespace_id` takes the
  default, now or the URL namespace. A given `0` or `""` is stored as given,
  as embedded stores it. It used to be read as "omitted". Import follows the
  same rule and now also takes `key`, `confidence` and `ttl`.
- `PUT /records/{id}` replaces edges and key when the body gives them, and
  keeps them when it omits them.
- `DELETE /records/{id}?cascade=false` forgets without pruning edges into the
  record, as embedded `forget()` does. The default still prunes.
- `purge()` stamps the records whose edges it prunes in id order, from a base
  the PURGE WAL record now carries. A crash used to recover them at different
  versions than they had.

### `newest()` walks a time index; packet manifests record real versions

**`FeatherStore.newest()` read and sorted the whole namespace.** Asking for
the 10 most recent items took time linear in the namespace: 12 ms at 10k items
and 152 ms at 100k.
- The engine now keeps each namespace's live records in a `(timestamp, id)`
  ordered index, maintained wherever the secondary indexes are and rebuilt on
  load.
- `db.newest(namespace, limit, prefix=False, before=None)` walks it from the
  end and returns ids newest first. With `prefix` it does a k-way merge over
  every namespace below. `before` pages by a timestamp or by the
  `(timestamp, id)` of the previous page's last record.
- `FeatherStore.newest()` uses it and skips expired items as it pages:
  0.04 ms for the 10 newest at 1k, 10k or 100k items (3,879× faster at 100k).
  A custom `key=` (episodes order by event time) cannot use the index and
  still scans.
- Ties within one second fall back to id order, which follows write order for
  records written since native keys landed.

**Packet manifests recorded `recall_count` as each ref's "version".** Every
search moved it and an edit of the rule did not, so a stored manifest could
neither show that a rule was unchanged nor that it had changed.
- Refs now carry the record's version (`Metadata.version`, `PocketItem.version`).
  It moves on every write to the record, never on a read, and is never reused.
  The values are increasing stamps from the DB's clock, not 1, 2, 3.
- The manifest also lists the refs as structured data (`"refs"`).
- `PacketBuilder.revalidate(manifest)` replays one. It reports each ref that is
  missing or now at another version (`was`, `now`), and returns
  `may_mutate=False` when a required ref is stale.
- Required refs from a `resolve_required` store are checked against that
  store's versions. A rule returned without one is listed as `unverified`.

### Multi-tenant search: a namespace-prefix index and per-namespace ANN

`FeatherStore.search(("org", "user42"), query=...)` ranked the whole file,
took the top `max(8 × limit, 50)`, and kept whatever was under the prefix. With
enough other tenants, every hit belonged to someone else and the user's own
search came back `[]`. `AgentMemory.recall(kinds=...)` did the same one level
down: it fetched `4 × limit` and discarded the other kinds. Filtered search
inside one namespace was exact but scanned every record in it: 4.8 ms for a
50k-record tenant, against 0.22 ms for an unfiltered search (100k × 384).

- **`SearchFilter.namespace_prefix`** (`FilterBuilder().namespace_prefix(p)`):
  `p` and every namespace below it, with `.` separating levels. `acme.u1`
  matches `acme.u1.prefs` but not `acme.u10`. It is indexed like
  `namespace_id`: a sorted set of namespace names turns the prefix into a
  range, so a search ranks inside those namespaces. It works for vector,
  keyword and hybrid search, and in the Cloud API's search bodies.
- **`FeatherStore` searches inside the prefix** in one engine call. It grows
  `k` only for what the engine cannot filter: expired items, unindexed items
  and LangGraph value filters. `AgentMemory.recall(kinds=...)` searches each
  kind's namespace and merges the results. Measured with 200 tenants of five
  items each: five of the user's own hits.
- **Per-namespace ANN.** A namespace with at least `FEATHER_NS_ANN_MIN` (4096)
  records in a modality gets its own HNSW graph.
  - **No vector copies.** Its nodes are 8-byte references to the modality's
    own slots (`SlotRefSpace`), about 160 B a node, scored by the modality's
    kernel on the same bytes, int8 included.
  - **Built lazily.** The first filtered search that needs a graph queues a
    background build; that search, and any others until the build finishes,
    scan as before. The build runs in chunks under the shared lock, so it
    holds up no writer, and is not persisted: a reopen rebuilds it on demand.
  - **Kept current by writes.** A record that is added, moved in or given a
    new vector is linked in. One that dies or moves out is marked, so a
    graph's live nodes are exactly its namespace's records, and a
    namespace-only search walks it with no filter callback. Other predicates
    filter the walk. Compaction and int8 conversion renumber slots, so they
    drop the graphs, which then rebuild on demand.
  - **A prefix mixes both.** A search over several namespaces walks the ones
    that have a graph, scans the rest, and merges.

  Measured on 100k × 384 embedding-shaped data, k=10
  (`benchmarks/namespace_search.py`):

  | namespace | records | scan | graph | recall@10 |
  |---|---|---|---|---|
  | half the file | 50,000 | 4.88 ms | 0.34 ms (14.6×) | 1.000 |
  | a tenth | 10,000 | 0.73 ms | 0.17 ms (4.4×) | 1.000 |
  | a hundredth | 1,000 | 0.08 ms | (scanned) | 1.000 |

  A 50k-record graph builds in about 6 s in the background. The graph walk
  uses twice the modality's `ef`: it replaces an exact scan, so it buys recall
  back. On hard 32-d Gaussian data that took recall from 0.93 to 0.995.

**Behaviour change: filtered search in a large namespace is approximate.** It
used to be an exact scan of every candidate. Below `FEATHER_NS_ANN_MIN`
records, and for filters that do not confine to namespaces, it still is. It
remains complete: a walk that comes back short of k falls back to the scan.
To get exact everywhere, use `FEATHER_PREFILTER_MODE=exact` (now also turns
the graphs off), `FEATHER_NS_ANN_MIN=0`, or `db.set_namespace_ann_min(0)` per
handle. `db.namespace_ann_info()` and `db.wait_for_namespace_graphs()` report
on and wait for the builds.

### Native string keys: no more ids hashed from a key

`Pocket`, the packet builder and `FeatherStore` (so `AgentMemory` and
`feather-agent` too) placed each record at a 53-bit SHA-1 of (namespace, key).
Two keys that hashed alike shared one record. The second write silently
overwrote the first, and both keys then read back the second memory. The
packet builder never checked the record it found, so a required compliance
rule could resolve to whatever the colliding key had stored. A key also could
not be searched: BM25 indexed only the content, so the memory stored under
`allergy` was not found by searching "allergy".

- **`Metadata.key`.** A key is unique among the live records of its
  `namespace_id`, and the engine refuses a second record with the same one
  (`DuplicateKeyError`).
  - `db.upsert(meta, vec=None)` writes the record keyed
    `(meta.namespace_id, meta.key)` and returns `(id, version)`.
  - `db.id_for(key, namespace)` and `db.get_by_key(key, namespace)` read it.
  - In a `WriteBatch`, `upsert` writes one, and `update` / `forget` / `link`
    accept a key (`"k"` or `(namespace, "k")`) wherever they took an id. Each
    op sees the keys the ops before it bind and free.
- **Ids are allocated, not hashed.** They come from the version clock, offset
  by 2^52, so an id is never handed out twice. That holds even after a delete
  and a reload, so an edge left pointing at a deleted id never starts pointing
  at a stranger. Ids stay below 2^53, exact in JSON. Eight threads upserting
  one new key make one record.
- **BM25 indexes the key** with the content: "allergy" and "allergies" now
  find the memory stored under `allergy`. The `feather-agent` persona test that
  pinned the old behaviour was updated, as its docstring asked.
- **No format change.** The key is serialized as a reserved attribute (`"\0key"`),
  so v9, v10 and the WAL carry it unchanged, and a build without keys keeps it
  as an ordinary attribute and writes it back intact. Tests cover crash replay,
  save, vacuum, compaction and `FEATHER_FORMAT=9`.
- **Existing files.** `feather_db.keys` is the single place that addresses
  records by key.
  - `resolve()` still finds a record written at its hashed id, but only if the
    record says it is that key. Under a collision the hashed id holds another
    key's record, which the old code returned.
  - A legacy record is adopted under the id it already has the first time it
    is written, so edges into it stay valid.
  - `FeatherStore` adopts all of a file's legacy records when it opens it, and
    a `Pocket` adopts those of its own scope, which makes their keys
    searchable. `keys.migrate(db)` does it on demand.
- **Cloud API.** `PUT /records/{id}` keeps the record's key, as it keeps its
  edges, and records show `key`. A PUT without `namespace_id` no longer moves
  the record out of its namespace: it defaults to the URL's, as
  `POST /vectors` always did. Before, such a PUT dropped the record from every
  filtered search and listing of the namespace.
- `pocket.scoped_id()` still exists and returns the legacy hash, but nothing
  places records with it any more.

### Cloud API: typed errors, 429 backpressure, nothing blocking the event loop

A client used to get the same kind of answer whether it should retry, go to
another shard, or stop. `ingest_text` answered `400` ("your fault") when the
embedding provider was rate-limited. The write pool's queue had no bound, so
under overload a request waited until the client timed out, and the client's
retry landed on top of the backlog. That is how a slow minute turns into an
outage. An upload copied its file and loaded the database on the event loop,
so every other request on that worker waited for the whole load.

- **Every error says what to do.** The body is `{"detail", "error",
  "retryable", ...}`; `detail` is unchanged.
  - Retry: `429 overloaded`, `503 unavailable` / `write_failed` /
    `namespace_loading` / `upstream_unavailable`, with `Retry-After`.
  - Go elsewhere: `421 wrong_shard`, which names `owner_shard`.
  - Fix the request: every other 4xx (`not_found`, `invalid_request`,
    `version_conflict`, ...).
  - Server failure: `500 internal`, `502 upstream_error`. These are not
    retryable: repeating a request that broke the server is how one bad
    request becomes a storm.
- **Backpressure.** Past a limit on requests in flight, a request is refused at
  once with `429` and `Retry-After`. Reads, writes and uploads are counted
  separately (`FEATHER_MAX_INFLIGHT_READS` / `_WRITES` / `_UPLOADS`, default
  256 / 64 / 2), so a write burst never refuses searches. `/health` reports
  the in-flight count, the limit and the rejections per class. Measured with a
  limit of 2: four slow ingests got two `200`s and two `429`s, while a search
  in the middle got `200`.
- **Embedding-provider failures.** A rate-limited or unreachable provider is
  a `503 upstream_unavailable`, which is retryable. A provider that refuses the
  key or the model is a `502 upstream_error`. Neither is a `400` any more.
- **Off the event loop:**
  - the upload's copy, its database load and its record count;
  - the checkpointer pass, including saves, reopens and WAL stats;
  - opening every namespace at startup, and closing them all at shutdown;
  - decoding bodies over 256 KiB;
  - hashing bodies over 1 MiB for Idempotency-Key.

  With a 1.5 s load, `/health`, a read of another namespace and the creation
  of a new namespace each answered in under 0.5 s during the upload. Before,
  they all waited for the whole load.
- **The load is also outside `DBManager`'s global lock.** That lock serialised
  every namespace's open, recovery and delete behind a large load, even with
  the loop free. The namespace being adopted is reserved meanwhile: its file
  inode is claimed, so an alias is still refused. Requests to it get
  `503 namespace_loading`.
- **The gateway follows a 421 once** to the shard it names. That happens
  mid-reshard, when the gateway's `FEATHER_SHARDS` disagrees with the shards'
  count. A 421 is refused before any work, so this is safe for every method.
  Its own failures are typed too:
  - an unreachable shard is a retryable `503 shard_unreachable`, since the
    request was never sent;
  - a shard that took the request and never answered is a `504 timeout`,
    not retryable, since it may have been applied.
- **`mcp_remote` retries only what is safe to repeat.** It retries 429/503
  responses to GETs, read-only POSTs (searches) and writes carrying an
  `Idempotency-Key`, which ingest and link now send, reusing the same key on
  every attempt. It also retries a request that never reached the server.
  Retries are bounded (3), with exponential backoff and jitter, and honour
  `Retry-After`. The errors are typed: `OverloadedError`, `UnavailableError`,
  `WrongShardError` (`.owner_shard`), `RequestError`, `ServerError`, and
  `ConnectionFailed` (`.sent`). Tool results carry `error_type` and
  `retryable`. A server too old for keys gets the write once without one.

**Fixed along the way.**
- **An overwriting upload that failed to load lost the old namespace's latest
  writes.** The old handle was closed without a checkpoint and its WAL
  deleted, so the rollback restored a backup without them. The WAL now moves
  with the `.bak` and comes back on rollback.
- **`POST /records/{id}/link` ignored `rel_type` and `weight`.** `mcp_remote`
  had been sending `rel_type` all along; it was silently dropped.

### Compare-and-set versions, atomic write batches, idempotency keys

Two agents updating one memory overwrote each other silently: each read the
record, changed it, and wrote it back, and the second write erased the first.
Several records that belong together, such as an extracted fact with its entity
links and its provenance, were written one call at a time. A reader could see
the fact before its links, and a crash between the calls left half a graph
behind. A retried request was applied twice. `ingest_text` mints a random id,
so each retry of it created another record.

- **Every record has a `version`** (`Metadata.version`, `DB.version(id)`). Every
  write to the record changes it. Versions come from one clock per DB, so a
  version is never reused, not even by a record re-created after a delete.
  Searches don't change it. A missing or forgotten record is `0`.
- **`DB.put_if(id, meta, expected_version, vec=None)`** replaces the record only
  if its version is still the one that was read. `0` means the record must not
  exist yet. Otherwise it raises `VersionConflictError`, whose `.id`,
  `.expected` and `.actual` attributes let the caller re-read and retry. Eight
  threads × 50 read-modify-write increments of one counter end at exactly 400.
- **`DB.write(WriteBatch)`** applies `put` / `update` / `link` / `forget` ops as
  one unit. Every check runs first, against the state before the batch: every
  `expected_version`, the dims, and that both ends of each link exist once the
  ops before it are applied. A failed check writes nothing, not even to the
  WAL. The batch is logged as one WAL group in one write (`BEGIN` … `COMMIT`)
  and applied under one exclusive lock. A reader never sees part of a batch:
  without one, a reader polling while facts were added and then linked saw
  184 facts with no link out of 200. A crash part-way through the write drops
  the group whole: the test cuts the WAL at 12 points inside a batch.
- **`DB.write(batch, idempotency_key=…, fingerprint=…, note=…)`**: a repeat of
  the same key returns the first `BatchResult` (`replayed=True`, with the
  stored `note`) and applies nothing. The same key with another fingerprint
  raises `IdempotencyKeyReusedError`. The key is written in the same WAL group
  as its batch, so a retry after a crash is recognised too. Keys expire after
  `FEATHER_IDEMPOTENCY_TTL_S` (default 24 h). Past `FEATHER_IDEMPOTENCY_MAX`
  keys (default 100,000) the oldest are dropped, and a retry of a dropped key
  is applied again. `DB.idempotent_result(key, fingerprint)` looks a key up
  without writing.
- **Cloud API.**
  - `GET /records/{id}` returns `version` and an `ETag`.
  - `PUT /records/{id}` and `PUT /records/{id}/importance` take `If-Match` and
    answer `412` (with the current `ETag`) instead of overwriting a change the
    caller never saw.
  - `POST /v1/{ns}/batch` is the atomic batch over HTTP. A conflict is a `409`
    that names the record and both versions.
  - `Idempotency-Key` works on every record write: `POST /vectors`,
    `ingest_text`, `PUT /records/{id}` and `/importance`, `POST` and `DELETE`
    `/link`, `DELETE /records/{id}`, `batch_delete` and `/batch`. A retry gets
    the first response, with `Idempotent-Replayed: true`. That includes a
    DELETE retried after it succeeded, which used to answer `404`. Reusing a
    key with a different request is a `422`.
  - `import`, `seed`, `upload` and namespace create/delete refuse a key with
    `400` rather than ignore it. Routes that are idempotent by nature, such as
    purge, compact and save, accept a key and ignore it.
  - `DELETE /records/{id}` and `batch_delete` now forget and prune edges in one
    atomic batch, so a crash can no longer leave the pruning half done.
  - `DELETE /link` reads the record under the namespace lock: a link added
    between its read and its write-back used to be overwritten.
  - On a `feather-db` wheel older than this, both headers get a `501`; nothing
    ignores them silently.

**Behaviour changes.** `POST /records/{id}/link` answers `404` when either
record is missing. It used to answer `200 "linked"` and either link nothing
(missing source) or leave a dangling edge (missing target).
`PUT /records/{id}/importance` on a missing record is a `404` instead of a
`200` that did nothing.

**Format.** v10 gains two optional sections: `VERSIONS` (7), which holds each
written record's version and the clock, and `IDEMPOTENCY` (8). The WAL gains
`STAMP`, `BEGIN` and `COMMIT` records, written in the same `fwrite` as the
change they belong to. Neither the file format nor the WAL version changes.
- **Downgrading.** A build without these records skips the new sections and WAL
  ops but still replays every data record, so downgrading loses versions, keys
  and all-or-nothing, never data.
- **No versions on disk.** A file written by v3–v9, or with `FEATHER_FORMAT=9`,
  carries no versions. Its clock restarts at `time << 20`, which stays above
  any version handed out before, so a stale `expected_version` still conflicts.
- **Overhead.** Each WAL write costs 25 more bytes, and each commit 16 bytes
  per changed record.

### Cloud API: namespace names are validated, never rewritten

`DBManager` built a namespace's file name by stripping every character except
letters, digits, `-` and `_`, but kept its handles under the name as given. So
`ni.ke`, `n.i.k.e` and `nike` were separate namespaces, each with its own
handle, on **one file**. Every checkpoint wrote its own handle's view over the
others'. Reproduced: three writes acknowledged with a `201` (`nike`, `ni.ke`,
`nike`), and two left on disk. `DELETE /v1/namespaces/ni.ke` deleted `nike`'s
data. A name that stripped to nothing was an unhandled `ValueError`, so a `500`.

- **A name is either valid or a `400`.** Nothing is opened or created for an
  invalid name, and the error names the offending characters. The allowed set
  is unchanged (as `str.isalnum()` decides, plus `-` and `_`), so every name the
  old code left alone, and every file it ever created, works exactly as before.
- **Clients that sent such a name** (a dotted name, say) now get a `400`. Their
  data was never under that name: it is in the namespace with those characters
  removed (`ni.ke` → `nike`), together with that namespace's own records.
- **A valid name that the filesystem folds onto another open namespace's file**
  is a `409` that names the other namespace. That happens with case on macOS
  and Windows (`NIKE` and `nike`) and with Unicode normalisation on APFS.
  Before, it was the same silent shared file. Names are compared by the file
  they open, not by a guessed normalisation, so on Linux `Nike` and `nike`
  remain two namespaces, as they always were.
- **Names too long for `<ns>.feather.wal.old`** (over 239 bytes) are refused.
  Before, the namespace opened, and every checkpoint failed when the WAL was
  rotated.
- **A `.feather` file in the data dir whose name is not valid** (only possible
  by copying it in by hand) is skipped at startup with a log line. Before, it
  was listed under its own name while the API served a different, empty file
  from the stripped name.

### `DB.open` and every blocking call release the GIL

A slow open, a read queued behind a writer on the data lock, or `storage_stats()`
waiting out a save used to hold the Python GIL the whole time, freezing every
Python thread in the process: an API event loop, a heartbeat. Measured: a 88 ms
open stalled other threads 89 ms; a read waiting behind `add_batch` 324 ms →
328 ms; dropping a writer during a compaction 1121 ms → 1122 ms. All three now
stall other threads for no longer than a scheduler tick.

- **Released:** `open`, `touch`, `get_metadata`, `get_edges`, `get_incoming`,
  `get_vector`, `all_ids`, `modality_names`, `ids_in_namespace`,
  `ids_for_entity`, `ids_with_attribute`, `namespace_size`, `list_namespaces`,
  `set/get_auto_compact`, `set_quantized`, `is_quantized`, `set_int8_ram`,
  `is_int8_ram`, `storage_stats`, `wal_size`, `wal_fsync_count`,
  `is_compacting`, `size`, `dim`, `set_ef`, `get_ef` (the writes, searches,
  `save`, `close` and `compact` already were).
- **Dropping the last reference** (`~DB`: joins a running compaction, then
  checkpoints) runs without the GIL when built with pybind11 ≥ 3. Older pybind11
  has no way to ask for it, and still builds, with the previous behaviour.
- **Opens of one file stay serialised.** The GIL used to do that by accident.
  Without it, two threads opening one path for writing could both miss the
  in-process lock registry; the loser was refused as "another process (pid
  unknown)", or both loads cut the WAL tail at once. The engine now does it
  itself: writer opens take turns per path, and the registry lookup, `flock` and
  insert are one critical section. A close racing an open now releases the
  `flock` before the registry entry disappears; before, the open could admit
  itself against a lock nobody held. Read-only opens never wait.
- No API change; results are identical.

### In-RAM int8 stops costing recall: a trained scale, and a float re-rank from disk

`set_int8_ram()` stored vectors under a fixed scale `max_abs / 127`, default
`max_abs = 1.0`. Unit-norm embeddings rarely have a component above 0.25, so most
of the 255 levels went unused, and recall dropped to 0.936 on embedding-shaped
data (384-d). Now:

- **The scale is trained from the data.** `set_int8_ram(modality)` with no
  `max_abs` picks the clip that minimises mean squared quantization error over a
  sample of the vectors. A larger clip costs rounding error on every
  component; a smaller one clamps the few beyond it, so outlier dimensions get
  clamped instead of fitted. A modality holding ≥ 1000 live vectors
  (`FEATHER_INT8_TRAIN_MIN`) converts at once. One with fewer stays float32
  until it reaches that, and converts on that write; `add_batch` converts inside the chunk
  that crosses the threshold. Exact scan, recall@10 at 384-d: 0.906 (clip 1.0)
  → 0.978 (trained); with outlier dimensions 0.941 → 0.956.
- **Search re-ranks by the float32 originals, read from the file.** This
  covers `search`, `hybrid_search` and `context_chain`. With `refine=2` (the
  default), the graph returns the top `k × 2` by int8 distance.
  Each one is then re-scored by exact float L2 against its original. The
  originals are not held in RAM: they are written to the file as a new
  optional section, `FULLVEC` (type 6), and read back with one positioned
  read per candidate. Before a save, they are read from the WAL record that
  logged the write, so a bulk load holds no float copies either. Scores are
  the float scores, and `get_vector()` returns the original rather than the
  dequantized copy. `refine=0` keeps no originals: smallest file, int8 recall.
- **Converting keeps the graph.** `set_int8_ram()` on a non-empty float
  modality now converts it in place (it used to raise). Links, levels, labels
  and delete marks are copied slot for slot, and only the vectors are
  re-encoded. Its originals stay in RAM until the next `save()`. An int8
  modality's scale is fixed: calling again only changes `refine`.
- **`db.int8_info(modality)`** reports whether a modality is int8 or still
  waiting to train, its clip, the refine factor, and where its originals are:
  in the file, still in the WAL, or copied in RAM.
- **The re-rank widens the beam too.** A refine query walks the graph with
  `ef × refine`. Re-ranking alone plateaued at recall 0.980–0.985 for refine
  2–8 at 100k, because int8 distances misjudge which neighbour to expand, and true
  neighbours never entered a float-width beam. A float-built graph walked in
  int8 lost the same, so the fault is in the walk, not the graph.
- **Measured** (`benchmarks/int8_refine.py`, 100k × 384, embedding-shaped,
  1000 queries, ef 50, warm page cache; memory is the physical footprint):

  | | recall@10 | p50 / p99 ms | memory after open | build peak | file |
  |---|---|---|---|---|---|
  | float32 | 0.9988 | 0.128 / 0.337 | 217 MB | 279 MB | 167 MB |
  | int8, clip 1.0 (old default) | 0.9390 | 0.071 / 0.143 | 108 MB | 179 MB | 58 MB |
  | int8, trained, refine 0 | 0.9534 | 0.071 / 0.149 | 108 MB | 176 MB | 58 MB |
  | **int8, trained, refine 2** | **1.0000** | 0.142 / 0.255 | **108 MB** | 186 MB | 204 MB |

  The re-rank costs latency (about float's) and disk (the originals), not RAM.
  On a cold page cache each re-ranked candidate is a disk read: 20 per k=10
  query.
- **Format.** FULLVEC is flagged optional, so a v10 build without it opens the
  file and uses the int8 index alone. If such a build later changes a slot,
  its GRAPH record voids that slot's earlier FULLVEC on load, so a stale
  original is never used. The MODALITY section gains two trailing fields
  (`refine`, waiting-to-train) that older builds skip. `FEATHER_FORMAT=9`
  writes no originals.
- **Behaviour change:** `set_int8_ram("m")` with no `max_abs` used to mean a
  clip of 1.0. It now means "trained", and the modality stays float32 until it
  holds 1000 vectors. Pass `max_abs=1.0` for the old behaviour.

### A failed WAL write or fsync is fatal to the handle

Every one of these used to be ignored, so the write was reported as saved:
`wal_append()` never looked at `fwrite`/`fflush`; a WAL that could not be
opened made the mutation go ahead with **no record at all**; and `fsync`'s
return code was dropped for the WAL and the checkpoint data file alike.

- **The first failure poisons the handle**, as PostgreSQL PANICs on an fsync
  error. That call raises `RuntimeError`, and so does every later write and
  `save()`, until the file is reopened. Reads keep working. Retrying is what
  loses data: a record appended after a torn one never replays, and an fsync
  retried after a failure can succeed without the pages that failed
  ("fsyncgate"). `db.write_error()` returns the cause (`None` while healthy).
- **Group commit tells every writer.** When a leader's fsync fails, every
  writer it covered raises. None of them returns as if its write were durable.
- **`close()` on a poisoned handle releases it without a checkpoint** and does
  not raise (it runs from `__exit__` while the original error propagates).
  The WAL stays; the reopen replays it and cuts a torn tail.
- **WAL appends are unbuffered, one `fwrite` per record.** With stdio
  buffering, a failed append left the unwritten tail in the buffer, and
  `fclose()` later completed the record that had been reported as failed.
- **`add_batch` logs a whole chunk before applying any of it**, so a failed
  append leaves none of that chunk in memory (it used to leave metadata
  without vectors).
- **Only a failure that lost bytes poisons.** An fsync that ran and failed,
  or a WAL write that failed partway. Not being able to open the WAL (for
  example EMFILE) refuses that one write and applies nothing; the next write
  tries again. A filesystem that cannot fsync at all (`EINVAL`, `ENOTSUP`,
  `ENOSYS`: some FUSE and network mounts) is treated as synced, exactly as
  before, so a service on such a mount keeps running. A directory that cannot
  be opened is still skipped, and `EBADF` from a directory fsync is
  tolerated, as in PostgreSQL. EINTR is retried. The load-time header
  repair stays best-effort, so a file that opened before still opens.
- A write that raised can still be on disk after the reopen, if its bytes
  reached the disk before the fsync failed. It is never the other way round:
  a write that returned is durable.
- **`feather_db.WriteFailedError`** (a `RuntimeError`, C++ `feather::WriteFailed`)
  is what the failing call and every later refusal raise, so a caller can tell
  "reopen and retry" from any other error without matching messages.
- **Cloud API recovers on its own.** The failed write is a `503` with
  `Retry-After` and `"retryable": true`, not a `500`. `DBManager.get()` and
  the background checkpointer reopen the namespace, which replays its WAL.
  The new handle is opened before the old one is retired, and the old one
  is never `close()`d, so requests already reading from it finish normally.
  If the reopen fails (the disk is still broken), reads keep being served
  from the old handle. On an engine wheel without `WriteFailedError` the app
  still starts and behaves as before. Retries are throttled to one per `FEATHER_RECOVER_RETRY_S` (default
  5 s), not one per request. `save_all()` no longer stops at the first
  namespace that raises; it returns `{namespace: error}`.
- **Cloud API accepts v10 uploads.** `/v1/admin/upload` checked the header
  against `MAX_FORMAT_VERSION = 9`, so every file this build writes was
  rejected with a 400.
- **No C++ exception crosses the C ABI.** `src/feather_core.cpp` had no
  try/catch, so any engine error in the Rust CLI (a dimension mismatch, a
  WAL failure) aborted the process with SIGABRT. Every entry point now
  catches it and returns `-1` (`nullptr` from `feather_open`), with the
  message in `feather_last_error()`. The CLI exits 1 and prints the reason,
  including why an open was refused (e.g. the file lock). The return type
  went from `void` to `int`, which is binary-compatible with callers built
  against the old declarations.
- **The `feather-db-cli` library API is unchanged.** `DB::open` still returns
  `Option`, and `add`/`link`/`touch`/`save`/`search` keep their signatures.
  On an engine error they now panic with the message, where they used to
  abort the process. New `try_*` forms return `Result`, and the CLI uses
  them. The pre-change `main.rs` builds unmodified against this crate.
- `feather_db.merge` no longer swallows `WriteFailedError` while copying
  edges, so a merge into a failed target is never reported as complete.

### A corrupt file is an error, never a crash

18 of 300 single-byte corruptions of a v9 file used to kill the process
(SIGBUS), read-only opens included. In 60,000 fuzzed opens across v9 (saved
graph and rebuild path), v10 and the WAL — with v10 checksums and WAL CRCs
recomputed so the corruption reaches the parsers — none now crashes, hangs or
exhausts memory; each opens or raises `RuntimeError`.

- **v9 graph loader** (`loadIndexStream`): the whole header is validated
  before it sizes anything. `count × slot_size` was checked before
  `slot_size` was, so a forged count could wrap the product under the
  remaining-bytes guard, allocate the base layer at the wrapped size and fill
  it at the real one. HNSW parameters must match what the constructor derives
  from `M`; every link list is bounded by the bytes left; every read is
  checked.
- **Graphs are validated before search follows them**, in v9 and v10 alike:
  a link to a missing node, a link into a layer its target does not have, or
  a node above the entry point is corruption.
- **Metadata strings grow as their bytes arrive**, so a corrupt 32-bit
  length no longer zero-fills a 4 GB string before the read fails; a record
  cut short is an error instead of a half-filled field.
- **The v2–v9 reader checks every read** (short reads used to leave `dim`
  and `element_count` holding stack garbage), bounds the modality count and
  the v2 `dim`.
- **WAL records that pass their CRC but do not decode are refused** instead
  of applied; an ADD's `dim` is bounded by its own payload.
- **Slot counts are 32-bit.** v10's GRAPH and VECTORS sections store them as
  u32, like the internal ids they count. *v10 files written by earlier
  builds of this unreleased branch do not open.*
- `tests/test_corrupt_files.py`: a seeded fuzz in the suite, plus a deeper
  run on a rotating seed in CI (`FEATHER_FUZZ_SEED` reproduces a failure,
  `FEATHER_FUZZ_CASES` scales it).

### File format v10: append-only commits

A save now writes what changed since the last save instead of rewriting the
whole file. The file is two checksummed header slots followed by typed,
length-prefixed, checksummed sections. Each commit appends its sections,
makes them durable, then switches the header. Layout: CLAUDE.md §3.2 and
`include/format_v10.h`.

Measured at 100k × 384 (`benchmarks/save_cost.py`, save after 100 adds and
100 importance updates):

| | v9 | v10 |
|---|---|---|
| bytes written per save | 176.5 MB | 0.42 MB |
| save time | 105 ms | 0.9 ms |
| `add()` stall while a save runs (max) | 117 ms | 8 ms (3 ms without fsync contention) |

- **No torn or orphaned files.** An incremental commit never rewrites a byte
  that a valid header covers. A crash at any point leaves at least one valid
  header whose sections are all on disk. A torn header fails its XXH64
  checksum and the other slot is used. A torn commit is cut back by the next
  writer, which also retires the abandoned header. Every crash state is
  rebuilt byte-for-byte in `tests/test_format_v10.py`.
- **Corruption is refused, not misread.** If the newest commit fails its
  checksum and `<wal>.old` is present (a crash mid-commit), the load falls
  back to the previous commit and the WAL redoes the torn one. With no
  `<wal>.old`, the bytes went bad after the commit succeeded. Falling back
  would silently drop committed records, so the open raises and leaves the
  file untouched; `FEATHER_RECOVER_OLDER=1` opens the previous commit
  deliberately. An unknown mandatory section type or a newer version also
  raises. Every count read off disk is bounded by the bytes
  behind it, and every graph link is checked against the slot count.
- **What changed is tracked at the source.** hnswlib marks every slot it
  writes, and an insert's ~M rewired neighbours cost ~140 B each (links
  only). Every metadata mutation marks its record. A search hit costs a
  20-byte salience record, not the whole record.
- **New:** `db.vacuum()` rewrites the file now. `db.storage_stats()` reports
  format, generation, committed bytes, live estimate and the last commit's
  size.

### close() frees memory; Python owns the DB object

The binding held `DB` with `py::nodelete`, so nothing a handle allocated was
ever freed. That was ~50 MB per open/close of a 39 MB file (10 cycles:
+493 MB), which is how the API and MCP servers slowly ran out of memory.

- **`close()` frees** the graphs (or their mappings), metadata and every
  index immediately. Measured over 40 open/close cycles: memory stays flat at
  about one handle's worth (the allocator keeping freed pages for reuse)
  instead of growing ~50 MB a cycle.
- **Any call on a closed handle raises** `RuntimeError` ("DB is closed").
  Writes already did; reads used to answer from memory that is now freed.
- **Dropping the last reference destroys the object.** A writer that was
  never closed is checkpointed and its lock released at that point, as
  `~DB()` always did in C++; a reader never writes. Before, `del db` released
  nothing until the process exited. Prefer `close()`: it happens when you
  choose, not at garbage collection, and not with the GIL held.
- **Behaviour change for tests and scripts:** a writer handle discarded
  without `close()` now checkpoints, so its WAL disappears. Four tests relied
  on the leak to keep it; they now say what they mean (`close(save=False)`,
  or `os._exit(0)` in a child process).

### The graph is saved even when records are deleted

One `forget()` used to make the v9 writer drop the HNSW graph from the file
and store bare vectors, so every later open rebuilt the graph. Measured at
100k × 384 after a single `forget()`:

| | before | after |
|---|---|---|
| v9 file (`FEATHER_FORMAT=9`), writer open | 53 s | 0.10 s |
| v9 file, read-only open | 63 s | 0.17 s |
| v10 file (default) | 0.16 s | 0.16 s (v10 already kept the graph) |

The graph is now saved with its delete marks, which the loader has always
restored. Only an on-disk-quantized modality still stores bare vectors.

- **Fixed: `_deleted` records were searchable, and v10 turned them into
  ghosts.** Killing a record with the `_deleted` attribute (through
  `update_metadata`, `add` or `add_batch`, as the Cloud API and
  `FeatherStore` do) left its vector live: unfiltered vector search returned
  it. A save dropped the record but v10 kept the node, so after a reload it
  came back as a hit with empty metadata. A dead record's vectors are now
  marked deleted on every path, including WAL replay, exactly as
  `forget()` does. Re-adding the id with a vector revives it.
- **Repaired on open:** a live graph node whose record is missing or dead is
  marked deleted when the file is opened (5–13 ms at 100k), so files written
  before this fix lose their ghosts. A writer commits the repair at the next
  save.

### Read-only handles share one copy of the vectors (mmap)

A full rewrite now stores each modality's vectors as one contiguous `VECTORS`
section on a 64 KiB boundary. `DB.open(..., read_only=True)` maps it
copy-on-write instead of copying it, so reader processes share one copy
through the page cache. Measured at 100k × 384 with 10 concurrent readers
(`benchmarks/readers_memory.py`):

| | per reader | 10 readers |
|---|---|---|
| vectors-only DB | 227 → 74 MB | 2,268 → 891 MB (incl. 154 MB shared once) |
| ~300-char agent text per record | 355 → 213 MB | 3,547 → 2,283 MB |

What mapping cannot reach is heap built at open: metadata, BM25 postings
(rebuilt from content on every open), secondary indexes. So the saving is
1.5–2.5×, not 10×; getting further needs those in mappable form too.

- **Open time.** BM25 rebuild now stems each distinct word once per open
  instead of every occurrence: 2.1 s → 0.72 s at 100k records with text
  (synthetic text with a small vocabulary; real text gains less). A
  vectors-only DB opens in ~75 ms instead of 170 ms.
- **hnswlib's slot layout was split.** Level 0 now holds `[links][label]`,
  and vectors live in their own array. Search results are identical and
  throughput rose ~10% (denser link lists). v9 streams are converted on
  write and read, so v9 files stay byte-identical.
- Writers still load into the heap (a long-lived writer must not pin a
  vacuumed inode). `FEATHER_MMAP=0` turns mapping off; Windows reads as
  before. `storage_stats()["mapped_bytes"]`.
- A save also rewrites the file once vectors appended since the last image
  exceed it (`FEATHER_IMAGE_RATIO`, default 1). Those are private to every
  reader; doubling keeps the rewrite cost O(1) per vector.
- **Do not overwrite a `.feather` in place** (`cp` onto it, `truncate`)
  while readers have it open: a mapped reader gets SIGBUS. Feather never
  does this itself; replace files by rename.

### Breaking / behaviour changes (format v10)
- **Older builds cannot read a v10 file.** The first `save()` (including
  `close()`) after opening a v3–v9 file rewrites it as v10. Older builds
  refuse it cleanly: offset 8 holds `0xFFFFFFFF`, which they read as the
  metadata count and reject against the file size. `FEATHER_FORMAT=9` keeps
  writing v9 snapshots for a fleet that still runs pre-v10 readers.
- **Superseded bytes stay until a vacuum.** Earlier versions of updated or
  forgotten records stay in the file until it is vacuumed. Vacuum runs
  automatically when the file exceeds `FEATHER_VACUUM_RATIO` (default 2) ×
  live data + `FEATHER_VACUUM_MIN_BYTES` (default 4 MiB). It also runs at the
  next save after `purge()`, `compact()` or a `set_quantized()` change, and on
  `vacuum()`. `purge()` + `save()` still erases the purged records' metadata
  from the file. Vectors of forgotten records remain as deleted graph nodes
  until `compact()`, as they do in memory.
- **Cold load is ~30% slower** (75 → 99 ms at 50k × 384), because every byte
  is checksummed and every graph link validated.
- **On-disk-quantized files are ~2× smaller than float at 128-d, not >2.5×.**
  v10 also persists their graph links, so they load without the rebuild that
  v9 needed (the gap shrinks with dim: links are ~132 B per node).

### Fixed (found while verifying v10; also present in v9)
- **WAL replay of `update_metadata()` kept the old edges.** In memory,
  `update_metadata()` replaces the record, edges included. Replay used the
  add rule ("no edges given → keep the existing ones"), so after a crash an
  edge the caller had removed came back.
  `tests/test_wal_recovery.py` now kills a writer after three updates (drop
  every edge, keep a subset, add one) and checks that the reopened graph
  matches the one the process was serving.
- **`PUT /v1/{ns}/records/{id}` dropped every edge of the record.** The
  request model has no edges field, and `update_metadata()` replaces the
  whole record, so a PUT that changed one attribute unlinked the record from
  everything. The route now carries the stored edges over, read under the
  namespace lock. Edges are changed through `/link` and `DELETE /link`, which
  still remove them, the last one included. The engine keeps its replace rule
  because unlink and the delete cascade depend on it.
- **Forgotten records behaved differently in memory than after a reload.**
  Every save drops dead records, but in memory `link()` still added edges
  from a forgotten record, and re-adding a forgotten id inherited the dead
  shell's edges. After a reload or crash replay those edges were gone. Now
  `link()` from a forgotten record is a no-op (as `auto_link` already was),
  and a re-added id starts fresh.
- **The Rust CLI never rebuilt after a C++ change.** The `cc` crate prints
  `rerun-if-env-changed`, and once a build script prints any `rerun-if-*`
  line Cargo stops watching sources. After `scripts/sync-cpp.sh`, a
  `cargo build` linked the old engine. `build.rs` now declares
  `rerun-if-changed=cpp`.
- **The CLI is tested end to end** (`tests/test_cli_v10.py`, skipped unless
  the binary is built). The Rust binary and the Python binding read and write
  the same v10 files in both directions. Also covered: lock refusal, WAL
  recovery across the two, the v9 upgrade, torn-tail repair and refusal of
  damaged data.
- Known, not fixed: after a crash, the WAL's `PURGE` does not prune edges
  pointing at records forgotten *before* the last save. Memory counts their
  shells as namespace members; the file has no shells. The edge left behind
  dangles. A fix needs the purged id list in the WAL record.
### Fixed — post-merge review of the concurrency work

Three defects found reviewing PR #8 and fixed on master directly after it
merged. Each has a regression test.

- **`hybrid_search()` returned soft-deleted records that `search()` filtered
  out.** `is_dead_meta()` is meant to be one rule applied by every search
  path, but the vector-side filter functor inside `hybrid_search` never got
  it, so with a filter set the BM25 half dropped a tombstone and the vector
  half fused it straight back in through RRF. Measured on a three-record
  namespace: `search() -> [1]`, `keyword_search() -> [1]`,
  `hybrid_search() -> [1, 2]`. A `_deleted` record is not `markDelete`d in
  HNSW — only `forget()` does that — so this check was the only thing
  keeping it out of the graph walk.
- **A corrupt dim in a v1 WAL sized an allocation before it was validated.**
  `replay_wal`'s ADD handler read `dim32` off disk and used it for both
  `std::vector<float> vec(dim32)` and `get_or_create_index(modality, dim32)`
  before the only bound (`dim32 != m_idx.dim`) ran. `dim32 * 4` is unsigned,
  so `0x40000000` wrapped the read length to zero while still reserving
  4 GB. v2 records are CRC-protected; v1 WALs (<=0.17.0) are not and are
  deliberately still replayed, so one corrupt byte was enough to OOM a
  memory-capped container on startup. `dim32` is now bounded against the
  bytes the record actually carries, before either allocation — the same
  rule the `plen` guard above it already applied.
- **The BM25 tie-break comment stated both rules and matched neither.** Two
  consecutive lines claimed ties go to the smaller id and to the higher id.
  The code does the latter; `search()` does the former. Behaviour is
  unchanged — the comment now says so, and says why the two are not unified.


### Concurrency, compaction and search-path optimizations

Concurrency and durability. See `CONCURRENCY.md` for the problems,
`CONCURRENCY_BASELINE.md` for the before numbers and `CONCURRENCY_RESULTS.md`
for the after numbers.

### Breaking / behaviour changes
- **`close(save=False)`.** The inter-process file lock itself shipped in
  0.20.0 (exclusive for writers, shared for `read_only=True`, reentrant within
  one process, `FEATHER_LOCK=0` to disable). This work builds on it:
  - `db.close()` now also stops the background compactor before it
    checkpoints and releases the lock.
  - `db.close(save=False)` releases the handle without checkpointing, so the
    WAL is kept. Tests use it to simulate a crash.
  - `db.closed` is a property alias of `is_closed()`.
  - Python never runs `~DB()`, so `del db` does **not** release the lock.
- **Group commit.** A write can become visible to concurrent readers a few
  milliseconds before its fsync completes. The writer still returns only
  once the write is durable. `FEATHER_WAL_STRICT=1` restores the old
  "never visible before durable" behaviour.
- **Auto-compaction runs on a background thread.** Call
  `wait_for_compaction()` to block until it finishes.
- **`add_batch` applies records in chunks** (`FEATHER_BATCH_CHUNK`), so
  readers may observe a partially applied batch.
- **Cloud API: `FEATHER_EMBED_STRICT_DIM=1` (opt-in).** With it, a provider
  dimension mismatch returns 400 instead of being padded or truncated, and
  OpenAI/Azure `text-embedding-3-*` and Gemini are asked for the configured
  dimension natively. Without it, nothing changes: vectors are fitted to the
  configured dim as before, so existing namespaces keep ranking consistently.
- **Cloud API: single-record mutations no longer save the whole file.** A
  background checkpointer saves based on WAL size and age. Data stays durable
  through the WAL.
- **New WAL op `PURGE`.** Pre-0.19 builds skip it on replay.

### Fixed
- **`save()` durability:**
  - The new base file is fsynced before it is renamed into place, and the
    directory is fsynced after.
  - Write errors (for example a full disk) are checked. Before, a truncated
    snapshot could replace the file and the WAL was then deleted.
- **Writer starvation.** The engine's lock is now a phase-fair
  `FairSharedMutex` instead of glibc's reader-preferring rwlock. Before, 8
  readers at 768-d cut a writer from 191 to 0.6 writes/s.
- **The WAL fsync no longer runs while holding the exclusive lock**
  (group commit). One fsync covers every concurrent writer.
- **`add_batch` releases the lock between chunks.** Before, readers dropped
  99% while batches ran.
- **`compact()` builds without the lock and swaps at the end.** Before it
  froze the DB for 40–86 s.
- **Deleted HNSW slots are reused.** The vendored hnswlib was patched so
  that re-adding a forgotten id works.
- **`update_metadata` is O(edges of the record).** It used to be O(all
  edges): 13 ms at 500k edges.
- **`auto_link` and `purge` are WAL-logged.** `auto_link` also runs its kNN
  pass under the shared lock.
- **`save()` rotates the WAL and makes the base file durable outside the
  lock**, so writers aren't blocked for the whole save. Snapshot readers use
  a long-shared mode, so queries keep flowing during a checkpoint.
- **A wrong-dimension `add()` is rejected before it reaches the WAL.**
  Before, it was replayed on the next open, reading past the vector.
- **`add()` with edges now updates the reverse edge index.**
- **Array strides are handled.** Non-contiguous NumPy arrays passed to
  `add`, `search`, `context_chain` and `hybrid_search` are now read
  correctly.
- **`IngestPipeline` no longer restarts its id counters per instance.** A
  new pipeline on an existing DB used to overwrite earlier records. It now
  continues after the highest stored id and re-learns stored entities.
- **Internal self-queries no longer inflate `recall_count`.** This covers
  ContextEngine sampling, ContradictionDetector, pipeline candidate lookup,
  MMR over-fetch and tool over-fetch.
- **`feather-serve` now starts.** `stdio_server` was called with the server
  object as `stdin`.
- **The LlamaIndex adapters are exported again**, as
  `FeatherVectorStoreIndex` and `FeatherReader`.
- **`merge()` closes the source DB.**

### Performance: search
- **Distance kernels are chosen at runtime** (`include/feather_simd.h`):
  AVX-512F, AVX2+FMA, SSE2 or scalar on x86-64, and NEON on arm64. Each wide
  kernel carries its own `target` attribute, so one portable binary uses the
  best ISA the machine has.
  - PyPI wheels were built SSE-only (`FEATHER_SIMD=sse`), so pip users never
    got the AVX kernels. arm64 had no SIMD distance at all.
  - Build modes: `FEATHER_SIMD=auto` (new default), `native` or `none`.
  - `FEATHER_SIMD_RUNTIME` caps the level at run time.
  - `feather_db.core.simd_info()` reports the kernel in use.
  - int8 in-RAM distance also has AVX2 and NEON kernels.
- **Filtered search:**
  - The exact scan uses the index's SIMD distance function directly on the
    stored vector. Before, it copied each candidate's vector and used a
    scalar loop.
  - Large, unselective candidate sets go through filtered HNSW with `ef`
    raised for that one call (`searchKnnEf`). The route is picked by a cost
    model (`candidates × selectivity² > C × k`), and the exact scan is the
    fallback if the walk comes up short. This route is approximate, so it is
    opt-in (`FEATHER_PREFILTER_MODE=auto`); the default stays exact.
  - Candidate sets are built as vectors instead of re-hashed sets.
  - Only the top-k results get their metadata copied, on every search path.
- **`search()` rejects a query of the wrong dimension** instead of reading
  past its buffer.
- **In-RAM int8 is now faster than float as well as 4× smaller.**
  - The int8 AVX2 kernel processes 32 bytes per step, computing the difference
    exactly via saturating subtracts.
  - A new float-query × int8-row kernel (AVX2 and NEON) speeds up the int8
    filtered scan.
  - Measured unfiltered p50: 1.43× faster than float at 768-d and 1.82× at
    1536-d.
- **Atomic reader fast path in `FairSharedMutex`, and per-thread slots in the
  HNSW visited-list pool.** Neither takes a mutex per read any more. Both are
  verified with a ThreadSanitizer stress test (`tests/cpp/lock_stress.cpp`).
  Measured throughput was unchanged on this machine; see
  CONCURRENCY_RESULTS.md §8.
- **Cloud API: hot-route request bodies are decoded with msgspec**
  (`feather-api/app/fastparse.py`). The pydantic models remain the OpenAPI
  schemas. Bulk import is 10–20% faster; search is unchanged within noise.

### Added
- **Python bindings:**
  - `close(save=True)`, the `closed` property and context-manager support
  - `wait_for_compaction(timeout)`, `is_compacting()`, `wal_size()`,
    `wal_fsync_count()`
  - `search(..., with_cosine=True)`, which fills `SearchResult.cosine`
  - `SearchResult.to_dict()` and `Metadata.to_dict()`
- **Cloud API: request options and transport:**
  - `vector_b64` (base64 float32) as an alternative to `vector`
  - `include_metadata=false` on search routes
  - Responses serialized with orjson; `raw_score` cosine computed in C++
- **Cloud API: execution:**
  - A separate write thread pool (`FEATHER_WRITE_THREADS`)
  - Async `ingest_text` with an embedding micro-batcher
  - Pooled httpx clients, provider-side batch embedding and an LRU embedding
    cache
  - A pure ASGI metrics middleware
  - A dev-only `mock` embedding provider
- **Cloud API: deletes** prune edges through the reverse index, in
  O(in-degree), for records in any modality.
- **Sharding:**
  - `FEATHER_SHARD_COUNT` and `FEATHER_SHARD_INDEX`; a shard answers 421 for
    namespaces it doesn't own.
  - New `feather-gateway/` proxy, `docker-compose.sharded.yml` and
    `docs/deploy-sharding.md`.
- **Benchmarks:** `benchmarks/concurrency_bench.py` and
  `benchmarks/api_concurrency_bench.py`.
- **Tests:** file lock, group commit, background compaction, slot reuse,
  fair lock, API features, gateway, and Python-layer fixes.

---

### Fixed: found by review of the concurrency rewrite (several also on 0.21.0)
Each has a regression test (`tests/test_wal_and_lock_edges.py` unless noted).
- **A read-only handle could delete the writer's data.** `purge()` and
  `auto_link()` began WAL-logging without `check_writable()`; a reader shares
  the writer's WAL, so its `purge()` was replayed by the writer and the records
  were gone. `update_importance()` and `forget_expired()` had the same hole.
  All four now raise on a read-only handle, as every other mutation does.
- **WAL replay dropped links.** Replay overwrote metadata on ADD/UPDATE, while
  the live path keeps existing edges when the new metadata has none: re-adding
  a linked record and crashing lost the link, and replaying `<wal>.old` twice
  was not idempotent.
- **A torn WAL tail hid every later write.** Replay stopped at the torn record,
  but new records were appended after it — acknowledged, then unreachable by
  the next recovery. A writable open now cuts the tail; a read-only open never
  touches the writer's WAL.
- **The file lock leaked** when two same-process handles on one path closed in
  open order: the descriptor now belongs to the per-process registry entry.
- **Filtered search is exact again by default.** The cost model's graph walk
  for large unselective filters is approximate (recall@10 0.86 on a 50%
  namespace filter; with `scoring=` an important record could miss the top-k).
  It is now opt-in: `FEATHER_PREFILTER_MODE=auto`. The default exact route is
  still the new in-place SIMD scan.
- `set_ef()` during a background compaction was undone at the swap;
  `auto_link()` could link to a record purged during its unlocked kNN pass;
  a new WAL's directory entry was never fsynced.
- **Rollback safety:** `purge()` also logs a FORGET per record ahead of the new
  PURGE op, so a 0.21.0 build replaying the WAL after a crash leaves the
  records deleted instead of bringing them back.

### Fixed: Cloud API and gateway (review)
- `"metadata": {}` on add stored timestamp 0 (1970) instead of now.
- One invalid text in an `/ingest_text` micro-batch failed every request in the
  batch, across tenants: a failed batch is now retried text by text, so each
  caller sees only its own outcome. Shutdown fails queued requests instead of
  leaving them hanging.
- Imports that embed server-side ran on the 8-thread write pool and could stall
  every write on the server (2.7 s for an unrelated add); they use the shared
  pool, as before.
- A 421 (sharded) upload left its staged file (up to 256 MB) behind.
- 422 `detail` for a non-object `metadata`, and bodies with `NaN`, now match
  0.21.0 exactly.
- Gateway: namespaces named `admin` or `namespaces` were routed to the wrong
  shard; a non-string name on create was a 500; a broadcast every shard
  refused (e.g. 401) was reported as 502.
- Docker image: Feather Desk (`/app`) was not copied in and answered 404.

### Fixed: Rust CLI
- **Use-after-free on every string argument** (master since v0.3): `map_or`
  moved each `CString` into the closure and returned a pointer to it after it
  was freed. `--content`/`--source` were stored as garbage bytes and `search`
  received a garbage modality and printed nothing.

### Added
- Cloud API: `confidence_gte` on `/search`, `/keyword_search` and
  `/hybrid_search` (optional; omitted, nothing changes).

### Fixed: silent overwrites in agent memory
- **`PocketSession.note()` keys collided.** They were `time_ns() % 100000`,
  and `time_ns()` ticks in microseconds on macOS, leaving 100 possible keys:
  300 notes in one session produced 41 distinct keys, so most notes silently
  replaced others. Keys are now random (`n<12 hex>`).
- **Session ids could collide** for the same reason (hash of the clock), and
  two sessions sharing an id share — and drop — one scratch scope. Now random.
- **`episodes.record()` overwrote a same-second repeat.** The key is
  `<unix second>_<slug>`, so recording one event twice in a second (a retry)
  replaced the first, despite episodes being append-only. A taken key now gets
  `#2`, `#3`…; an uncontested key is unchanged.

### Added: `FEATHER_WAL_FULLFSYNC=1` (macOS)
Plain `fsync()` on macOS leaves data in the drive's write cache, so a power cut
can lose writes the WAL acknowledged. The variable switches the WAL and
checkpoints to `F_FULLFSYNC`. Opt-in because it costs ~150× per sync (25 µs →
3.7 ms measured); the default is unchanged. `test_concurrent_writers_share_fsyncs`
uses it so group commit is observable on macOS (~140 fsyncs per 640 writes,
against 611–640 without), instead of failing there.

### Kept compatible with existing Cloud API clients
Diffed against master over 99 requests (every route, the filters, errors,
auth): identical status codes and bodies.
- msgspec decodes with `strict=False`, so pydantic's coercions still work:
  `"k": "5"`, `5.0` for an int, `"true"`, numeric strings in `vector`. Strict
  mode returned 422 for all of them.
- A 422 carries the exact `detail` pydantic produced (`loc`, `msg`, `type`,
  `input`), rebuilt on the error path only.
- `keyword_search` and hybrid fusion break score ties toward the higher id
  again, as they always did; the rewrite had flipped it.

## [0.21.0] — 2026-10-12

### Feather Desk, and `confidence` / `ttl` over HTTP

`feather-web/` is a context workspace mounted at `/app`, distinct from the
operator console at `/admin`: scopes are the dock, trust is the card's edge
colour, confidence is a bar rather than a number, and a record with a `ttl`
counts down so an expiring run finding looks different from a standing rule.

Building it surfaced that neither `confidence` nor `ttl` was reachable through
the Cloud API at all. `MetadataIn` sets `extra="forbid"`, so a POST carrying
either was rejected with 422, and `MetadataOut` omitted both. That made this
release's own `confidence_gte` incoherent — a filter for a field the API
refused to accept — and left no way to write an expiring record, which is what
a short-lived agent run scope needs. Both are now accepted (validated:
confidence in [0,1], ttl >= 0) and returned on get, list and search. Defaults
are unchanged, so nothing starts expiring.

Everything here landed after the v0.20.0 tag. The headline is a correctness fix
for a defect that **was live in 0.20.0**: `recent()` and `len()` returned wrong
answers past a handful of records.

### `confidence_gte` — filter on how certain a record is

`Metadata.confidence` has been stored and persisted since Phase 2 but was never
in `SearchFilter`, so a caller could record how sure it was about a fact and
then had no way to ask for only the sure ones.

Deliberately a separate axis from `importance`: importance is how much a record
matters, confidence is how likely it is to be true. A critical rule believed
weakly and a trivial fact known for certain are not the same thing, and
collapsing them loses the distinction that makes either useful. Unset records
default to `1.0`, so records written before anyone used the field do not vanish
the moment someone filters on it.

```python
f = FilterBuilder().namespace("brand.b1.shared.findings").min_confidence(0.9).build()
```

### `recent()` and `all()` returned an arbitrary N, not the newest N

Found by measuring at 2000 memories instead of the 2 the tests used. With two
episodes, "the newest 1 of an arbitrary 8" and "the newest 1" are the same
answer, so the tests passed over a real defect.

`AgentMemory.recent()`/`all()` asked `search()` for `limit*4` items and sorted
those, but `search()` has no ordering — it materialises every candidate under
the prefix and slices in id order. At 300 episodes, `recent(5)` returned events
299, 298, 296, 295, 293: two real episodes silently skipped. `len()` was capped
at the 1000 items it fetched, so 1200 records reported 1000.

Episodes also ordered by *write* time. The engine timestamp is whole seconds, so
300 writes inside one second tie and `all(3)` returned 298, 297, 296. Episodes
now sort on the event time encoded in the key, so a backdated `at=` also sorts
where it belongs.

New on `FeatherStore`, outside `BaseStore`: `scan()` (live rows, never parsing
the value JSON), `count()` (a count, not a fetch — the engine's
`namespace_size()` counts tombstones until compaction), `newest(n, key=)`
(parses only the n that survive the sort). At 2000 memories: `recent(10)`
17.59 → 3.97 ms, `len()` 17.67 → 2.59 ms, `summary()` 17.00 → 2.30 ms.

### Readers take no lock

A shared lock conflicts with the writer's exclusive one, so every reader process
was refused while the writer merely held the handle open — not even while
saving. That blocks the single-writer/many-reader topology locking was added to
support. Readers are now never blocked.

Nothing is given up, because the integrity was never coming from the lock:
`save_vectors()` renames a complete temp file over the original, so `open()`
returns a whole inode and a reader keeps a consistent snapshot until it reopens;
WAL records are CRC-checked so replay of a log being appended to yields a clean
prefix; and a read-only handle never clears the writer's WAL (verified: a
crashed writer's 20 unsaved records survive a read-only open). Readers trade
**freshness, not integrity** — reopen to advance.

### Context packets — required context guaranteed, or the packet is degraded

`Pocket.hot()` fits a budget by heat, which is right for a working set and wrong
for an operating constraint: a rule saying "never make medical claims" is not
more relevant for being recalled often, and must not fall out because forty
creative notes outranked it.

`PacketBuilder` puts required refs in first, judges the required set **whole**
(dropping rules one at a time would hand a caller a partial constraint set with
no signal), and raises `RequiredContextUnavailable` with the arithmetic when
they are missing or will not fit. `allow_degraded=True` returns a packet with
`may_mutate` False instead. Everything dropped is listed in `omitted` with a
reason. `manifest()` records **references and versions, never content**, so a
decision replays while still being revalidated against live scope. Real token
counting via tiktoken when installed; `exact_tokens` says which was used.

`resolve_required=` lets an authoritative store own the constraints. When given
it is authoritative with **no pocket fallback** — otherwise index lag reads as a
missing rule and blocks every mutation, and a stale Feather copy of a
just-changed rule becomes enforceable.

Two bugs it surfaced: a repeated required key counted as *missing* (so listing a
rule twice blocked every mutation), and a forgotten record still resolving — a
retracted rule satisfied its own requirement and kept authorising mutations.

### Embedder defaults from the environment

`FEATHER_EMBED_PROVIDER` / `FEATHER_EMBED_MODEL` now feed `FeatherStore` and
`feather-agent` when no `embed=` is passed. A provider that is set but
unbuildable **raises** rather than degrading to keyword search: BM25 does no
stemming, so a paraphrased query returns "nothing found" and the caller cannot
tell a misconfiguration from a memory that was never stored.

### cp314 wheels

cibuildwheel 2.23 → 4.2.1. 2.23 predates Python 3.14, so adding `cp314-*` to it
built nothing. cp38 drops out (4.x will not build it; EOL since 2024-10) —
`requires-python` stays `>=3.8` and the sdist still installs there.

---

## [0.20.0] — 2026-10-04

Agent memory, end to end: a tiered pocket per agent, a typed store for the five
kinds of thing an agent remembers, and an MCP surface that is now tested over
the actual protocol rather than by calling its own functions.

### AgentMemory — the five kinds of thing an agent stores

`FeatherStore` is a correct `BaseStore`, and deliberately generic: it will store
anything under any namespace you invent. An agent memory is not generic. Five
kinds get stored, they have different lifetimes, and the difference is not
decorative — write an episode with upsert semantics and you have destroyed the
history that made it an episode; write a preference append-only and the agent
holds six contradictory answers to "how do they want to be contacted" with no
way to tell which is current.

| kind | semantics | namespace |
|---|---|---|
| preferences | revised in place | `(org, user, "preferences")` |
| episodes | append-only, time-keyed | `(org, user, "episodes")` |
| facts | corrected, keeps `supersedes` | `(org, brand, "facts")` |
| procedures | versioned as `name@N` | `(org, agent, "procedures")` |
| entities | merged, org-wide | `(org, "entities")` |
| observations | cites its evidence | `(org, user, "observations")` |

The kind determines the write semantics, so the kind is now in the API rather
than in the caller's head.

It also owns namespace **ordering**. Search takes a prefix, so
`(org, subject, kind)` makes "everything I know about this user" a prefix read
while `(org, kind, subject)` makes it a filter over every user in the org — same
data, and the second does not scale. A caller cannot invert it. A kind whose
subject was never supplied raises naming the missing argument instead of
surfacing as an opaque tuple error three frames down.

`examples/langgraph_agent_memory.py` adds memory to an existing LangGraph agent
by changing one line (`builder.compile(store=...)`). It runs offline with a
bag-of-words stand-in embedder, and is **executed by a test** rather than read.

### The MCP server is now tested over the protocol, not by calling itself

The previous tests called `server.call_tool()` directly. That tests the Python
functions; it does not test the server. The JSON Schema the SDK derives from the
type hints, argument validation, the initialize handshake, result envelopes and
error signalling were all skipped — and that layer is exactly where an MCP
server breaks for a user while every unit test stays green. The old
`mcp_server` module is the proof: it passed its own tests and raised on import
against SDK 2.0.

`tests/mcp_harness.py` runs a real `ClientSession` over the SDK's in-memory
transport — the same JSON-RPC messages Claude Desktop sends over stdio, with no
subprocess. 23 protocol tests and 12 persona tests (a personal assistant across
four restarts with a correction and a forget; a pinned safety fact surviving 40
junk memories under budget pressure; an agency with inherited policy and brand
isolation; a 50-agent fleet).

CI and `verify.sh` now **fail if the MCP suite skips** rather than runs.
`importorskip` turns a missing SDK into a green skip, which is precisely how the
old server shipped broken with CI passing.

### `recall()` ranked by heat, so it ignored the query

A pinned record has heat 1.0 by definition, so an org compliance policy was
returned first for "which tiktok creative performed best" exactly as readily as
for "what should I not claim". Heat answers *what should I be carrying*;
relevance answers *what did you ask for*; only `hot()` may conflate them.
Scores are now normalised per scope (each scope is searched separately, so raw
scores are not comparable across them) and ranked by relevance with heat as a
0.05 tiebreaker — at 0.15 a pinned row still beat one scoring 0.04 higher.

### `build(db=...)` — the correct pattern for several agents

Two `DB.open` calls on one path are two independent instances with independent
in-memory state, and whichever saves last overwrites the other's records
(measured: 1 of 2 survived). So giving each agent its own MCP server on a shared
file silently loses memory. `build(db=...)` runs a scope against an
already-open database; one DB with many scopes is safe. A test asserts the
collision **still happens**, so if it ever stops failing, file locking landed.

### Inter-process file locking — the silent multi-process data loss is closed

Two processes opening one `.feather`, each writing, each saving, both exiting 0
— and 20 of 20 records from one of them gone. No error, no log. Each DB holds
the whole dataset in RAM and `save_vectors()` rewrites the file from that view,
so the second save writes a view that never contained the first's records.

This is not fixable by serialising the saves: a mutex would only decide whose
records vanish. True concurrent writing needs merge-on-save or a paged storage
engine, which is a format change. So what ships is enforcement of the model that
**is** safe — one writer, many readers — with a loud refusal in place of silent
loss:

- `DB.open()` takes an exclusive advisory lock (`flock`) for the handle's
  lifetime. A second **process** is refused, naming the holder's pid and the
  single-writer model.
- `DB.open(..., read_only=True)` takes a *shared* lock, so several reader
  processes may hold the file at once. Every mutation on a read-only handle
  raises at the call site — not at save time, because a handle that accepts
  `add()` and only complains on checkpoint has already lied to the caller. A
  read-only close never rewrites the file. (Changed after release — see
  Unreleased: a shared lock still conflicts with the writer's exclusive one,
  so readers were refused while a writer merely held the file open.) — not at save time, because a handle that
  accepts `add()` and only complains on checkpoint has already lied to the
  caller. A read-only close never rewrites the file.
- `db.close()` checkpoints, releases the WAL handle and drops the lock. It is
  **required**, not decorative: the Python binding holds `DB` with
  `py::nodelete`, so `~DB()` never runs from Python and without `close()` the
  lock would live until process exit. `with DB.open(...) as db:` also works.
- The lock is **reentrant within one process**, so a program may still reopen
  its own file — a strict lock would have refused that, since `del db` cannot
  release one. Same-process double-open therefore remains the documented footgun
  it already was (use one handle and separate agents by namespace, or
  `build(db=...)`); what is now impossible is the cross-process case, where you
  collide with a process you cannot see.
- `FEATHER_LOCK=0` disables it, because `flock` is unreliable on NFS and some
  container overlay mounts. The caller then owns the consequence.
- A crashed writer does not strand the file: `flock` is released by the kernel
  on process death, and a leftover `.lock` file is not itself a lock.

11 tests, including the original report reproduced end to end across two real
processes.

> **Note for mixed Python/CLI use:** the Rust CLI now also participates. If a
> long-running Python process holds a database, `feather add --db that.feather`
> is refused rather than silently discarding one side's writes.


### FeatherStore now honours the whole `BaseStore` contract
Four behaviours the LangGraph interface specifies that the first implementation
either silently ignored or actively inverted. Each was found by exercising the
contract rather than re-reading the code.

- **TTL was unreachable.** `BaseStore.put()` refuses a `ttl` unless the subclass
  sets `supports_ttl`, and raises *before* reaching `batch()` — so Feather's own
  `ttl` field and `forget_expired()` could not be used through the LangGraph API
  even though both already worked. Declared, and enforced on read.
- **A sub-second TTL became permanent.** LangGraph's `ttl` is in *minutes*,
  Feather stores whole *seconds*, and `0` means "never expires" — so
  `int(0.001 * 60)` silently turned an expiring memory into a permanent one.
  Now rounded and floored at one second: a positive TTL can never mean forever.
- **`index=False` was ignored.** The placeholder vector is identical for every
  unindexed record, so without a marker they matched each other perfectly and an
  `index=False` item came back as the *top* semantic hit — the opposite of what
  was asked. Unindexed records are now excluded from semantic results while
  remaining stored and listable.
- **`list_namespaces` ignored its match conditions.** Prefix, suffix and `*`
  wildcards are now applied — and applied to the *full* namespace before
  `max_depth` truncation, because truncating first lets `("a","b","c")` match a
  suffix condition on `("b",)` at depth 2, a path the caller never stored.

10 tests; **7 fail against the previous implementation.**


### Feather over MCP, rebuilt for SDK 2.0 — and shaped like agent memory
The existing `feather-serve` is **broken against the current SDK**: `Server.list_tools`
was removed in MCP 2.0, so `create_server()` raises before it can start. It also
exposed 16 low-level database verbs — a model handed `feather_add_intel`,
`feather_mmr_search` and `feather_consolidate` has to work out which one means
"remember this", and it usually decides not to.

`feather-agent` replaces it with a surface shaped like what an agent does, using
each MCP capability for what it is actually good at:

- **Tools** (`remember`, `recall`, `context`, `forget`, `memory_stats`) — the
  model calls these when it decides to. Good for recall, weak for capture.
- **Prompts** (`/remember`, `/what-do-you-know`, `/catch-up`) — **user-invoked**,
  which is the reliable capture path precisely because it does not depend on the
  model choosing to act. This is the capability the old server never implemented.
- **Resources** (`feather://context`, `feather://stats`) — attachable, so the
  working set costs no tool call and no round trip.

Backed by `Pocket`, so an agent reached over MCP gets the same hot/warm/cache
tiering and scope inheritance: `--scope hawky.brand_a.creative` reads its own
memory plus `hawky.brand_a` and `hawky`, and sibling agents cannot see each
other. Memory survives a server restart because it is one file.

    feather-agent --db agent.feather --scope hawky.brand_a.creative

The `mcp` extra is pinned to **>=2.0.0** (0.9.0 predates both the stateless
protocol core and the `MCPServer` API), and CI installs it on 3.10+ so this
adapter cannot break silently again. 17 tests, including one that asserts the
legacy server is still broken — if it ever passes, something changed and the
replacement should be revisited.

### Concurrency test no longer flakes on a loaded machine
`test_concurrent_search_scales` took a single timing sample, which failed once
in three runs on an otherwise-passing build. Scheduler noise and CPU contention
can only make a timing run *slower*, so it now takes the best of three rounds —
the fastest round is the honest estimate of what the lock permits.


### Agent pocket memory — hot / warm / cache tiering per agent
An agent does not want "the top k for a query". It wants the things it should be
carrying right now, sized to the context budget it has left. Answering that by
hand is what every agent codebase ends up doing badly.

- **Feather already measured the signal and nobody read it as a tier.** Every
  search hit increments `recall_count`, and `stickiness = 1 + ln(1+recall_count)`
  already slows decay for what an agent keeps returning to. `Pocket` interprets
  those counters instead of leaving them to re-rank search.
- **Three tiers, one file.** HOT is the working set — fits a token budget,
  ordered by heat, and costs no query, no embedding and no model call. WARM is
  everything else in scope, reachable by `recall()`, and a hit **promotes** it so
  the working set follows real usage. CACHE is keyed results with a TTL, in the
  same store with the same expiry.
- **Scopes are hierarchical and agents inherit.** A pocket at
  `("hawky","brand_a","creative")` reads its own memory plus `("hawky","brand_a")`
  and `("hawky",)` — its notes AND the brand rules AND the org policy, in one
  budget, ranked together. Writes always land in the pocket's own scope, so no
  agent can mutate a sibling's memory; the agent's own key shadows an inherited
  one. Inherited memory is discounted 30% per level, because fleet-wide recall
  counts say little about whether *this* agent needs it now.
- **Sessions separate learning from thinking.** `with p.session() as s:` gives
  scratch its own scope — real memory during the run, dropped at the end unless
  `s.keep()` promotes it. Scratch is capped at 40% of budget, because everything
  written this run has age ~0 and would otherwise evict the rules it is supposed
  to follow. `__exit__` cleans up on the exception path, and `maintain()` reaps
  scratch from processes that were killed outright.
- **`hot()` is memoised against a write generation.** It is O(namespace) —
  measured 171 ms at 10,000 memories, far too slow for a per-turn call. Repeat
  calls are now **0.001 ms**, recomputed on any write through any pocket on the
  same DB. `recall()` invalidates too: it looks like a read but increments
  `recall_count`, and without that the cache served a view predating the
  promotion it had just performed.

Measured: 50 agents × 40 memories → per-agent `hot()` 0.2 ms median; 16
concurrent agents × 30 ops → no errors; 200 sessions × 5 scratch notes → 0
leaked scopes, flat RSS; 2,000 cache lookups over 50 keys → 170 µs each,
computed exactly 50 times.

**Known limitation:** single-process only. Two processes writing one `.feather`
still destroy each other's writes silently (measured: 0/20 records survived from
one of two agent processes). Since agents are usually separate processes, file
locking is the prerequisite for a fleet sharing one file — tracked separately.

55 tests in `tests/test_pocket.py`. Suite 318 → 373.

## [0.19.0] — 2026-09-29

Feather becomes long-term memory for LangGraph agents, and two engine defects
that undermined it are fixed.

### FeatherStore — Feather as a LangGraph `BaseStore`
- LangGraph has two memory sockets: `BaseCheckpointSaver` for per-run graph
  state, and `BaseStore` for **long-term, cross-thread** memory. Feather is the
  second one, and `BaseStore` needs exactly two methods — `batch()`/`abatch()`.
- The existing adapters (`FeatherVectorStore`, `FeatherRetriever`) make Feather
  *a search index an agent queries*. A `BaseStore` makes it *the thing an agent
  remembers into* — same engine, different position in the stack.
- Record ids are `sha1(namespace + key)`, **deterministic on purpose**: `put()`
  on the same key must overwrite in place, since Feather's `id` is a true upsert
  key. The existing MCP helper seeds ids with `time_ns()`; copying that would
  have grown the store forever without ever updating anything.
- `created_at` is preserved across updates while `updated_at` moves, so an agent
  can tell when it first learned something from when it last confirmed it.
- Three behaviours are worked around in Python until format v10 lands, and all
  three are things v10 was already scoped to fix: namespace **prefix** search
  (the engine's namespace index is exact-match), operator filters (`$gt`/`$lte`
  over string attributes), and the second timestamp.
- `examples/langgraph_agent_memory.py` — a runnable agent whose memory survives
  across processes in a 3 KB file, with no key, server or container.
- Install with `pip install feather-db[langgraph]`.

### Query dimension was never validated (memory safety)
- Every distance kernel reads exactly `dim` floats from the query pointer and
  trusted the caller for the length. A **short** query was therefore an
  out-of-bounds read — measured, a 256-dim query against a 512-dim index read
  **1 KB past the end of the buffer** and returned a confident-looking 0.0076. A
  long query silently compared a truncated prefix.
- The Cloud API guarded this at the HTTP edge; the C++ core and the Python
  bindings did not, so every embedded user — the entire point of the product —
  was unguarded.
- `search`, `hybrid_search` and `context_chain` now raise `ValueError` naming
  both dimensions and the modality.

### `auto_link`'s threshold now means cosine
- It was compared against `1/(1+L2_squared)`, which is not a similarity: the
  documented default of 0.80 actually required cosine ≥ 0.875, and on
  unnormalised embeddings it was unreachable. Measured, **20 documents at true
  cosine 0.9971 produced ZERO links, silently** — and `auto_link` is the
  primitive the clustering and entity-graph work sits on.
- The cosine is computed from the vectors rather than derived from the distance,
  because `cos = 1 − L2sq/2` only holds for unit vectors and nothing in Feather
  enforces normalisation. Edge weights are now the true cosine too, which
  matters because they flow into `context_chain` scoring.
- Same corpus, after: **300 links** at every threshold from 0.80 to 0.99.

### Tests
`tests/test_query_dim_guard.py` (7), `tests/test_auto_link_threshold.py` (4) and
`tests/test_langgraph_store.py` (14). The first two were verified against the
pre-fix engine: **11 of 15 fail**. CI installs langgraph on 3.10+ so the store
tests run rather than skip. Suite 303 → 318 (+14 more on 3.10+).


---

## [0.18.2] — 2026-09-01

Release plumbing. In 0.18.1 the macOS wheel legs and the CLI binaries built for
the first time; this fixes the remaining Linux leg.

- **manylinux2014 can no longer install a modern numpy.** The Linux wheel *built*
  and passed `auditwheel repair`; the cibuildwheel **test** step then failed,
  because there is no manylinux2014 numpy wheel for cp312+, so pip fell back to
  the sdist and hit `NumPy requires GCC >= 10.3` against manylinux2014's GCC 9.
  Linux wheels now build on **`manylinux_2_28`** (glibc 2.28 — Debian 10,
  Ubuntu 18.10, CentOS 8 and later; GCC 12), which is where numpy itself moved.
  Pinning an old numpy instead would only defer the same failure.
- Created the `pypi` GitHub environment the publish job declares. Publishing uses
  **Trusted Publishing (OIDC)**, so no token is stored — but the environment has
  to exist and a matching trusted publisher must be registered on PyPI.


---

## [0.18.1] — 2026-09-01

Release plumbing only — no engine changes. Fixes the reason PyPI has never
received a binary wheel.

- **`macos-13` runners were retired by GitHub on 2025-12-04.** `wheels.yml` and
  `release.yml` still requested that label, and a job asking for a retired runner
  **queues forever rather than failing** — one sat pending 21 hours before being
  cancelled. Both now target `macos-15-intel` for x86_64.
- **`cibuildwheel` bumped v2.21 → v2.23.** The pinned version referenced a
  manylinux image that has since been removed from quay.io, failing the Linux leg
  independently of the macOS one.
- **The SIMD gate read the host architecture, not the target.** `setup.py` keyed
  x86 intrinsics off `platform.machine()`, so a cross-compiled build pulled
  `immintrin.h` into an arm64 wheel (*"This header is only meant to be used on
  x86 and x64 architecture"*). It now prefers `ARCHFLAGS` / `_PYTHON_HOST_PLATFORM`
  and treats a universal2 build as arch-neutral. `CIBW_ENVIRONMENT` is scoped
  per-leg as a second line of defence.
- **`publish-rust.yml` reported success when it published nothing.** The step was
  `cargo publish || echo "publish skipped (already published?)"`, which swallowed
  every failure — including an unset `CARGO_REGISTRY_TOKEN`, which is what
  actually happened on v0.18.0: the job went green and the crate was never
  published. It now fails loudly on a missing token and only tolerates a genuine
  "already uploaded" response. (Same silent-success failure mode the Cloud API was
  fixed for in 0.18.0, in our own release plumbing.)

**Still required before the crate can publish:** add a `CARGO_REGISTRY_TOKEN`
repository secret.


---

## [0.18.0] — 2026-08-31

Six data-correctness fixes, four of which could lose or corrupt data, plus the
integration feedback from two teams running Feather in anger. Everything below
was reproduced before it was fixed and verified by reverting the fix and
watching the new tests fail.

### Also in this release
- **The Rust CLI's vendored engine is now generated, not drifted.**
  `feather-cli/cpp/` had fallen 764 lines behind `include/` and contained none of
  the WAL recovery, fsync, checksum, failed-open or salience fixes — a published
  CLI would still have lost a bulk import on crash under a version whose notes
  said otherwise. `scripts/sync-cpp.sh` regenerates it and CI now fails on any
  diff. (The copy exists because `cargo package` will not follow paths outside
  the package root, so `build.rs` cannot compile against `../../include`.)


### Search no longer mutates what it returns, and the real cosine is available
Two further findings from the same integration report, benchmarked against
Qdrant 1.19.0 holding identical L2-normalised vectors with numpy as the
reference.

- **Reads performed writes.** `recall_count` incremented on every record a search
  *touched*, not the ones it returned. The pre-filtered path scans the whole
  indexed candidate set, so a `k=5` query over a 400-record namespace credited
  **all 400** (measured); with scoring enabled the HNSW path fetched `k*3` and
  credited 15 for 5 returned. Since `stickiness = 1 + ln(1 + recall_count)` is the
  entire basis of adaptive decay, uniform inflation makes stickiness a constant
  and collapses decay to recency x importance — **irreversibly**, because only the
  inflated counter is persisted.
  Salience now applies **after** truncation, so a recall means "this record was
  returned". Measured: filtered `k=5` went 400 -> 5 touched, unfiltered with
  scoring 15 -> 5.
- **`track=false`** added to search (`record_salience` on the C++ `search()` and
  the Python binding). Evaluation, benchmarking and monitoring traffic can now
  read without perturbing the state it measures. Default stays true.
- **`raw_score=true`** returns `cosine`: the exact cosine similarity, computed
  from the stored vector. The default `score` is `1/(1+L2_squared)` optionally
  blended with time decay — correct for ranking (the reporting team measured
  **100% top-10 agreement with exact cosine on every query**) but it is not a
  similarity, is off by 0.046-0.077 against one, and cannot be calibrated away
  because the error changes sign. Verified exact to 3e-9.
  Both `score` and `cosine` now carry docstrings saying which is which.

4 new tests; suite 299 -> 303.


### Cloud API — stop failing silently (integration feedback)
Seven issues reported by a team integrating Feather as the vector store for
creative analysis (~1,260 records at 768d). Every one shared a shape: the server
answered 200 and the caller found out much later, or never.

- **`import` reported success when it stored nothing** *(data loss)*. A backfill
  that wrote zero vectors returned a flat `200 OK` with `inserted: 0`. Now
  **400** when nothing landed and **207 Multi-Status** when only some items did,
  with a `partial` flag on the body. A caller checking the HTTP status is doing
  the normal thing; the status has to carry the outcome.
- **Unknown metadata keys were dropped silently** *(data loss)*. Top-level
  metadata is a fixed schema, and anything else — `creative_hash`,
  `creative_url` — was accepted, counted as inserted, and simply absent on read
  back. The reporting team only noticed after building a UI and seeing every
  media URL come back empty. Unknown keys are now **rejected by name**, on both
  `/import` (400) and `/vectors` (422), with the error pointing at `attributes`
  as the supported place for custom fields.
- **Stored vectors could not be read back** *(functional gap)*. No route exposed
  them, so "find records like this one" and any offline clustering had to
  **re-embed text that had already been embedded** — an API call and a cost per
  lookup. Added `?include_vector=true` to `GET /records/{id}`.
- **A zero vector returned ranked results** *(correctness)*. An empty string
  embeds to exactly this, so it is a plausible client bug rather than a
  hypothetical. A zero-norm vector has no direction, but the engine ranks by L2
  and returned confident-looking rows (measured: top score 0.500). Now a 400.
- **Single-record GET had a different shape from every other route.** It
  returned the metadata object bare while list and search nest it under
  `metadata`, so client code written for one silently returned nothing on the
  other. `id` and a `metadata` mirror are now included **additively** — the flat
  fields stay, so existing callers are unaffected.
- **Repeated import errors are collapsed.** A misconfigured provider produced one
  copy of the same message per item; a 1,000-item import returned 1,000
  identical strings. Duplicates now fold into one line with a count.
- **The `k` cap is documented** on the field rather than only surfacing as a bare
  422 from pydantic.

`tests/test_api_records.py` gains 7 tests for these; **all 7 fail against the
pre-fix code**. Suite 292 -> 299.


### Tombstones spent the caller's page budget — and the first fix made it worse
- Filed by the dev team against 0.16.0. `GET /records` counted deleted records
  against `limit`, so a paging client received mostly ghosts: on a namespace of
  47 live records behind 127 tombstones, `limit=10` returned 2 live rows and
  `limit=100` returned 25. Correctness required over-fetching the entire
  history.
- v0.17.0 excluded tombstones from the listing — but filtered them **after**
  slicing to `limit`. A page whose ids happened to all be tombstones therefore
  came back **empty**: measured, `limit=10/50/100` each returned **0 rows**,
  and only `limit=174` returned the 47 live records. That is worse than the
  original bug, because a caller reads zero rows and concludes the namespace is
  empty rather than under-filled.
- **Fixed** by filtering before truncating: the scan walks past tombstones and
  keeps going until it has `limit` live records or runs out of ids. `has_more`
  and `next_cursor` now come from the scan rather than the raw slice, so a
  trailing run of tombstones no longer advertises a page that isn't there.
- **`include_deleted=true`** added as an explicit opt-in; those rows carry
  `"deleted": true` so no client has to recognise the internal `"_forgotten"`
  sentinel — the team asked for exactly this, and the sentinel is due to be
  replaced by a real flag byte in format v10 anyway.
- Verified end-to-end against a running uvicorn server over real HTTP, not just
  the test client: their exact reproduction, all four documented findings, and
  a full paged walk returning each of 47 live records exactly once across 7
  pages — before and after a server restart.
- Two findings needed no code: `POST /v1/{ns}/compact` already exists, and the
  stats route is `/v1/namespaces/{ns}/stats`, not `/v1/{ns}/stats`.
- 5 new tests in `tests/test_api_records.py`; **3 of 5 fail against the
  pre-fix code**. Suite 287 -> 292.


### A failed `open()` destroyed the file it failed to read (data loss)
- `DB::open()` builds a `unique_ptr<DB>`. When `load_vectors()` threw, the
  pointer unwound straight into `~DB()`, which ran `save_vectors()`
  unconditionally — serialising a **half-parsed** view of the file back over the
  file, then `wal_clear()`ing the log that could have rebuilt it. The failure
  path and the destroy-the-evidence path were the same path.
- **Reproduced**: a `.feather` truncated to 60% left a 16 KB `<path>.tmp`
  behind and then SIGSEGV'd. The original survived only because the crash landed
  *before* `std::rename` — a corruption that failed cleanly would have completed
  the rename and taken the data with it.
- **Fixed** with a `load_complete_` flag set as the last statement of
  `load_vectors()`. The destructor checks it, and `save_vectors()` now throws
  `std::logic_error` rather than silently truncating. Truncated-file opens went
  from **SIGSEGV (rc=139)** to a clean `RuntimeError` with the input
  **byte-identical** and no `.tmp` left behind.
- Only reachable from the `open()` unwind: `bindings/feather.cpp:141` binds `DB`
  with `py::nodelete`, so `~DB()` never runs from Python.

### An absurd metadata count spun instead of failing
- `meta_count` (`feather.h:969`) is a loop trip count and was unbounded — the
  v7+ loader guards `dim` and `element_count` this way, the v3–v5 metadata
  section did not. A forged `0xFFFFFF00` left the process spinning for **over
  ten minutes**, allocating from garbage lengths on a stream that had already
  run out.
- **Fixed** by rejecting any count larger than the remaining bytes can hold —
  arithmetic, not heuristic — plus a read check per record. Now **0 seconds**.
- Found while independently verifying a multi-agent audit; the agents did not
  catch this one.

### `tests/test_open_failure_safety.py` (new, 7 tests)
Truncation at 30/60/90%, no `.tmp` residue, the WAL surviving a failed load,
the absurd-count timeout, and the healthy open/save path still working.
Verified to have teeth: **5 of 7 fail against the pre-fix code**.


### WAL durability — fsync + per-record checksums (foundation)
- **The gap:** `wal_append()` only `flush()`ed. That hands bytes to the kernel,
  which survives the *process* dying but **not the machine or container dying** —
  which is most of what a write-ahead log is for. Every existing recovery test
  SIGKILLs the process, so the suite passed against the weaker guarantee and
  structurally could not have caught this.
- **The other gap:** there was no checksum. The `MAX_WAL_PAYLOAD` length guard
  catches a *short* tail, but it cannot see a record whose bytes were mangled in
  place — right size, wrong contents. Such a record replayed as plausible
  garbage and was then written into the next checkpoint as if it were real:
  silent corruption promoted to durable corruption.
- **WAL format v2**: an 8-byte header (`FWAL` + version) and a CRC32 on every
  record, covering `op + id + plen + payload`. Replay verifies before applying
  and stops at the last record that still checks out.
- **v1 WALs still recover.** Replay sniffs the header — `'F'` (0x46) can never be
  a v1 opcode, so the discrimination is exact, not heuristic. A WAL that a crash
  already left on disk must not be stranded by the upgrade that fixes crashes.
- **fsync policy**: one `fsync` per single-record mutation, and **one per batch**
  for `add_batch()`/`forget_expired()` rather than one per record — the batch
  either lands or its tail is lost, and the per-record CRC is what lets replay
  find that boundary. `FEATHER_WAL_SYNC=0` opts out.
- Measured cost: **`add()` +19%** (0.298s → 0.356s for 2,000 records),
  **`add_batch()` ~0%** (0.071s → 0.072s for 2,000).
- Switched the WAL handle from `std::ofstream` to `std::FILE*`: durability needs
  the underlying descriptor and there is no portable way to reach `fileno()`
  through an `ofstream`.
- `tests/test_wal_durability.py` (new, 8 tests): header/version, record framing
  incl. CRC, a flipped byte caught rather than replayed, intact WAL still
  replaying in full, torn tail, v1 back-compat, and both fsync modes. Verified
  to have teeth — **4 of 8 fail against the pre-fix code**.


---

## [0.17.0] — 2026-08-25

### Cloud — the API refuses to start without `FEATHER_API_KEY` (security)
- **Why now:** `v0.17.0` is the first release to publish an *official* container
  image (`ghcr.io/feather-store/feather-api`). That image sets
  `ENV FEATHER_API_KEY=""`, and `verify_api_key()` read an empty key as "dev
  mode — let everyone in". Anyone following the one-command deploy in the
  README would have got a vector database on a public port with read **and
  delete** access to every namespace, with nothing anywhere saying so. Shipping
  that as the official image would have turned a deployment mistake into a
  distributed default.
- **Fixed:** an unset key is now a startup error, not a silent downgrade to open
  access. Running without auth requires setting `FEATHER_DEV_MODE=1`
  explicitly — a separate variable on purpose, so the image's empty-string
  default can never be the thing that decides whether auth is on. A configured
  key always wins over `FEATHER_DEV_MODE`, so a stray env var in a deploy can't
  open a production host.
- `feather-api/docker-compose.yml` uses `${FEATHER_API_KEY:?…}` so the failure
  surfaces at compose time, naming the missing variable, rather than as a
  container that won't boot.
- `tests/test_api_auth.py` (new, 6 tests) pins it: unset and empty keys both
  refuse to start, dev mode still works, a real key is enforced and outranks dev
  mode, and every destructive route (`POST /vectors`, `DELETE /records/{id}`,
  `batch_delete`, `DELETE /namespaces/{ns}`) 401s unauthenticated with the data
  intact afterwards. Verified to have teeth: 2 fail against the old code.

### CI — official multi-arch Docker image on GHCR
- New `.github/workflows/docker.yml` builds `linux/amd64` + `linux/arm64` and
  publishes to GHCR on every `v*` tag (`<version>`, `<major.minor>`, `<major>`,
  `latest`). `workflow_dispatch` builds `:edge` from the default branch without
  touching stable tags. Auth via the built-in `GITHUB_TOKEN` — no registry
  secrets to configure or leak. Thanks to **@Ajinkgupta** ([#3]).
- `feather-api/docker-compose.yml` pulls the published image by default,
  replacing a stale local `feather-db-api:0.9.0` pin; building from source is
  kept as a commented option for contributors.
- Removed the stale root `Dockerfile` (a v0.1.0 placeholder superseded by the
  multi-stage `feather-api/Dockerfile`).

### Cloud — deleted records kept appearing in `GET /v1/{ns}/records` (data bug)
- **Reported from production:** the listing returned ids that
  `GET`/`DELETE /v1/{ns}/records/{id}` answered **404** for, and the record
  count never dropped after a delete. Root cause: `list_records` filtered on
  **only** the `_deleted` attribute, while `DELETE` calls `db.forget()`, which
  marks a record by setting `source="_forgotten"` and never touches attributes.
  Every *other* endpoint (get / put / delete / batch_delete / stats / schema /
  top_recalled / hierarchy) checked **both** conditions. So deletes genuinely
  succeeded, the listing kept showing the tombstones forever, and callers
  retrying against those ids got a correct 404 and concluded deletion was
  broken. Downstream, pools built from the listing served records for files
  that no longer existed.
- **Fixed** and made structural: there is now exactly one definition of "this
  record is gone" (`_is_dead_meta`), and all **nine** call sites go through it,
  so the endpoints cannot drift apart again.
- `tests/test_api_records.py` (new) pins the invariant that every endpoint
  agrees on which records exist — listing vs. stats vs. per-id fetch vs. all
  three search paths. Verified to have teeth: 4 of 9 fail against the old code.
  CI now installs the Cloud API deps so these actually run.

### Engine — the WAL was discarded for namespaces that had never been saved (data loss)
- **`load_vectors()` returned at `if (!f) return;` — before `replay_wal()`.** A
  namespace with no base `.feather` therefore threw away a complete, on-disk
  WAL. That is exactly the state throttled bulk import leaves a *fresh*
  namespace in for up to `FEATHER_IMPORT_SAVE_INTERVAL_S` (30s default), so a
  crash in that window silently lost every imported record — while the
  changelog promised "durable without a full save".
  **Measured: 0/300 records recovered before, 300/300 after.**
- **A corrupt WAL length field is no longer taken at face value.** A 13-byte
  truncated WAL declaring a 4.29 GB payload drove peak RSS to ~1.5 GB building
  the `std::string` before the read failed — enough to OOM a memory-capped
  container on startup and make the namespace unloadable. Replay now bounds the
  length against `MAX_WAL_PAYLOAD` and the bytes actually remaining, and stops
  at a bad record while keeping everything recovered up to that point.
  **Measured: peak RSS 1,509 MB → 33 MB.** (The base-file loader already
  guarded `dim`/`element_count` this way; the WAL never got the same treatment.)
- **The WAL handle stays open** instead of being reopened for every append —
  that open+close syscall pair was paid once per item inside `add_batch`, i.e.
  on the bulk-ingest hot path. Each record is still flushed, so durability is
  unchanged.
- Derived indexes are now built once after replay rather than twice.

### Engine — `get_all_ids()` no longer uses exceptions for flow control
- It called `read_vector_label()` in a `try`/`catch` per record: every record
  *not* in the queried modality cost a thrown-and-caught C++ exception, and
  every record that *was* cost a full dim-length vector copy that was
  immediately discarded. Now a direct lookup against the label table with the
  same two conditions `getDataByLabel` throws on — identical result set.

### Engine — concurrent search (readers-writer lock + GIL release)
- **Queries were serialised twice over:** every `search`/`keyword_search`/
  `hybrid_search` took the DB's single `std::mutex` **exclusively**, and only
  `add_batch` released the GIL — so no matter how many workers the Cloud API
  ran, the whole process searched on one core.
- `mutex_` is now a **`std::shared_mutex`**. Retrieval and every const accessor
  take it in **shared** mode; only mutations take it exclusively. hnswlib's
  `searchKnn` is `const` and its visited-list pool is internally mutex-guarded,
  so concurrent traversal is safe.
- The one thing a query mutated was the salience touch. `Metadata::recall_count`
  / `last_recalled_at` are now **`mutable std::atomic`** with a `note_recall()`
  helper, so a query records its hit under the shared lock — no deferral, and
  the increment stays immediately visible. Metadata gains explicit copy/move
  (atomics aren't copyable); **the on-disk layout is byte-identical**.
- The binding layer now **releases the GIL** around `search`, `keyword_search`,
  `hybrid_search`, `context_chain`, `add`, `link`, `update_metadata`,
  `update_importance`, `save`, `compact`, `auto_link`, `forget`, `purge`,
  `forget_expired`, `export_graph_json` and `get_all_ids`.
- `save()` no longer blocks queries: `save_vectors()` only reads, so it takes
  the data lock **shared** and serialises saver-against-saver on a dedicated
  `save_mutex_` (they share one `.tmp` path and rename).
- **Measured** (40k×128, 8 threads, M-series): search **5,691 → 28,092 qps**,
  scaling **1.00x → 4.94x** where it was previously flat at ~1.0x.

### Engine — BM25 indexing was O(n²) (cold load of text-bearing DBs)
- `add_to_bm25_index` re-summed **all** of `doc_lengths_` to recompute `avg_dl`
  on **every** document inserted, and `rebuild_bm25_index` calls it once per
  document — so building the index was quadratic, and it is rebuilt on every
  `load()`. Synthetic benchmarks never caught it because their records have no
  `content`. `avg_dl` is now maintained as a running total (`total_dl_`).
- Re-indexing an existing document scanned the **entire vocabulary** to retire
  its old postings. Callers now pass the previously indexed content, so removal
  touches only that document's terms.
- **Measured** (16k records with content): cold load **0.177s → 0.029s** (6.1x),
  and growth is linear (~1.9x per doubling) instead of quadratic (~3.2x).
  Extrapolated to 300k records this is ~60s of per-open rebuild removed.

### Engine — forgotten/purged records leaked into keyword search (bug)
- `forget()`, `forget_expired()` and `purge()` updated the vector index and the
  secondary indexes but **never touched the BM25 index**. A forgotten record
  stayed fully retrievable via `keyword_search`/`hybrid_search`, and a purged
  one came back as a hit with **empty metadata** (its posting outlived its
  metadata). All three now de-index, `rebuild_bm25_index()` skips dead records,
  and both BM25 scorers defensively skip postings with no live metadata.

### Build — header changes no longer produce stale objects
- Practically the whole engine lives in `include/*.h`, but `setup.py` listed
  only the `.cpp` sources, so an incremental `build_ext` silently reused stale
  objects after a header edit — a build that looked fresh but contained none of
  the changes. The extension now declares `depends=glob("include/*.h")`.

### Cloud — fast bulk import (throttled saves instead of one full save per call)
- **`POST /v1/{ns}/import` was O(batches × filesize):** it called `db.save()` on
  every call, and each save re-serializes the *entire* namespace file (plus the
  persisted HNSW graph). Loading 300k+ records as many batches re-wrote the whole
  growing file every batch — so import got slower and slower. `add_batch` is
  WAL-logged, so data is durable without a full save (the WAL is replayed on
  load; `save()` clears it). Import now saves at most once per
  `FEATHER_IMPORT_SAVE_INTERVAL_S` (default 30s) per namespace; `ingest_text` too.
- **`flush`**: pass `"flush": true` on your final import batch — or call
  **`POST /v1/{ns}/flush`** — to force a full save and compact the WAL into the
  `.feather`. (Shutdown still saves everything.)

### Cloud — bulk delete (one save, not N)
- **`POST /v1/{ns}/records/batch_delete`** `{ids?, entity_id?, cascade?}`: delete
  many records under a single namespace lock with **one** `save()`. The
  single-record `DELETE` re-serializes the whole namespace per call, so a
  per-record delete loop over a large namespace saved it N times and could wedge
  the server. Targets are `ids` ∪ (all records with `entity_id`); `cascade`
  (default off) prunes graph edges to the deleted ids in a single sweep. Caps at
  100k ids/call; reports `deleted` / `not_found` / `edges_pruned`.
- Dashboard **Maintenance → Bulk delete** card (ids and/or entity_id, optional
  cascade) wired to the new endpoint.
- (Bulk *insert* / *upload* already exist: `POST /v1/{ns}/import` and
  `POST /v1/admin/upload`.)

### Cloud — dashboard entity_id search filter
- Added a dedicated **Filter — entity_id** input to the search panel, and
  `runSearch` now routes `entity_id`/`namespace_id` attribute-filter keys to the
  top-level filter fields. Filtering search by `entity_id` returned nothing
  before because those are top-level metadata fields, not entries in the
  attributes map — the dashboard had only a `namespace_id` input.

### Cloud — per-namespace dimension (support any embedding dim)
- **A namespace's dimension is now defined by its data, never the server
  default.** The engine already adopted the first inserted vector's length, but
  the API layer compared every vector/query against the server-wide
  `FEATHER_DB_DIM` (768) — so a fresh namespace silently rejected or
  **padded/truncated** non-768 embeddings (1024, 1536, 384, …), quietly
  corrupting them. Now the dim is enforced only once a modality index actually
  exists (`_established_dim`); on an empty namespace the first vector is free to
  define any dimension (1…2²⁰).
- **No more silent padding.** `import` and `ingest_text` no longer pad/truncate
  to 768. A vector whose length disagrees with the namespace's established dim
  is rejected with a clear per-item error; a server-side embedding whose model
  dim doesn't match the namespace is rejected honestly instead of reshaped.
- **`POST /v1/namespaces`** accepts an optional `dim` to pin a new namespace's
  dimension up front (so the dashboard reports it before the first insert and
  clients get an early 400 on mismatch). Without it, the first vector decides.
- Client-supplied vectors are first-class at any dimension; server-side
  embedding stays tied to the configured model's fixed output dim (inherent).

### Cloud — graph endpoint no longer 500s on large/dirty namespaces
- **`GET /v1/{ns}/graph`** guarded: an unfiltered graph over
  `FEATHER_GRAPH_NODE_LIMIT` (4000) returns a friendly "too large — filter by
  namespace_id/entity" payload instead of trying to serialize hundreds of
  thousands of nodes; and a `UnicodeDecodeError`/`ValueError` from
  `export_graph_json` (one record with non-UTF-8 bytes anywhere in the
  namespace) degrades to an empty graph with a "dirty data" note rather than a
  500.

### Cloud — import a local `.feather` straight into the dashboard
- **`POST /v1/admin/upload`** (feather-api): upload a locally-built `.feather`
  file and adopt it as a cloud namespace. The upload is streamed to disk in
  1 MiB chunks (no whole-file buffering), its magic + format version are
  validated (v1–v9), then it's atomically moved into the data dir and served —
  so it appears in the dashboard immediately. The graph, attributes and
  persisted HNSW index come over intact: no re-embedding, no per-record calls.
  Returns `409` if the namespace exists (unless `overwrite=true`), `400` on a
  non-`.feather` payload, `413` over the size cap (`FEATHER_MAX_UPLOAD_MB`,
  default 512).
- **Admin SPA:** an "Import .feather" button on the Namespaces screen picks a
  local file, prompts for a namespace, uploads it, and opens it — the
  "push my local DB to the cloud" flow, end to end.
- `DBManager.adopt()` performs the validated, atomic file adoption (safe even
  when replacing a live namespace, since `DB` is bound `py::nodelete`).

### Fixed — JSON bulk import silently producing empty namespaces
- `POST /v1/{namespace}/import` previously **required** a precomputed `vector`
  of the exact namespace dim on every item; records pasted as plain text (no
  vector) or with a mismatched dim were silently skipped, so the import
  "succeeded" but the namespace stayed empty.
- Items without a `vector` are now **embedded server-side** from
  `metadata.content` using the configured provider (auto-padded/truncated to
  the namespace dim). The response reports how many were `embedded`, and the
  no-provider / empty-item / wrong-dim cases now return clear per-item errors
  instead of a silent skip.

### Hardened the upload + import path
- **Allocation guards on load (`feather.h` / `hnswalg.h`):** a corrupt or forged
  `.feather` header (absurd `dim`, an `element_count`/`max_elements` far larger
  than the file can back) no longer triggers a multi-gigabyte allocation — load
  validates the declared sizes against the actual remaining bytes and rejects
  with a clear error. `loadIndexStream` now fits index capacity to the real data
  instead of trusting the on-disk `max_elements`. Protects every load path, not
  just uploads.
- **Upload size cap** lowered to 256 MB (`FEATHER_MAX_UPLOAD_MB`) — bounds peak
  RAM since the file is loaded into the in-RAM index on adopt.
- **Backup + rollback on adopt:** replacing a namespace first moves the old file
  aside to `<ns>.feather.bak`; the new file is opened exactly once, and if it
  fails to load the namespace is **rolled back to the backup** (never left
  broken). A single unloadable file on disk no longer crashes server startup —
  it's skipped and logged.
- **Concurrent auto-embed:** bulk import now embeds vectorless records through a
  bounded thread pool (`FEATHER_EMBED_CONCURRENCY`, default 8) instead of one
  blocking provider call at a time, so large text imports don't crawl or time
  out.
- **Dim coherence:** `ingest_text` and bulk auto-embed fit the embedding to the
  namespace's actual dim (so a 512-dim uploaded DB, or a provider whose native
  dim differs, still ingests); `add_vector` returns a clean `400` on a dim
  mismatch instead of letting the C++ index throw.

### Fixed — uploaded DB shows "N records" in Maintenance but a blank Records list
- A `.feather` whose vectors live under a **modality other than `"text"`** (e.g.
  an external embedding pipeline names it `"embeddings"`) uploaded fine and
  `db.size()` reported the right count, but the dashboard hardcoded `"text"`
  everywhere — so the Records list, header count, and stats all read an empty
  `"text"` index and showed **0 / blank** while Maintenance showed the real count.
- New core accessors `all_ids()` (every id with metadata, across modalities) and
  `modality_names()`. The dashboard now lists/counts records by metadata id and
  reports the **actual** modalities + primary dim — records show regardless of
  modality name.
- Also fixed a cursor off-by-one that hid the record with **id 0** on the first
  page (`after` now defaults to -1).

## [0.16.0] — 2026-06-18

### Persisted HNSW graph — file format v9 (much faster cold load)
- **`save()` now embeds the prebuilt HNSW graph in the `.feather` file**, so
  `load()` restores it verbatim instead of rebuilding the graph from vectors.
  Cold load stops paying the dominant graph-construction cost on every open.
- **5–6× faster than the parallel rebuild, ~25× faster than serial** on a
  40k×128 DB (random data: 1.8s vs 10.6s parallel / 44s serial; clustered data:
  **48 ms vs 2.7s parallel / 13.4s serial**). The win grows with N — HNSW build
  is super-linear, a sequential read is not.
- **Higher, deterministic recall.** The persisted graph is the serial-build
  graph (recall 0.988), so reload no longer inherits the parallel-rebuild
  recall loss (0.963) — every load is identical and reproducible.
- **Single-file property preserved.** New `saveIndexStream`/`loadIndexStream` in
  the hnswlib fork write/read the graph (header + base layer, which already
  holds the vectors, + per-element link lists) inline in the one `.feather`
  stream — no sidecar files.
- **Transparent + safe fallback.** The graph is persisted only when it holds
  exactly the live set (no `forget()`/`purge()` nodes to filter) and the
  modality isn't on-disk-quantized. Otherwise — and for any v3–v8 file — load
  rebuilds via the existing (parallel) path. `compact()` re-enables the fast
  path. in-RAM int8 modalities persist their int8 graph and round-trip exactly.
- Trade-off: the file grows ~25% (the link lists) in exchange for the load
  speedup. On-disk int8 quantization (`set_quantized()`) is unaffected and still
  shrinks files (it opts out of graph-persist by design).

## [0.15.3] — 2026-06-17

### Adaptive index capacity (major RAM reduction)
- **HNSW indices now grow on demand instead of preallocating 1M elements.**
  Each modality index previously allocated `max_elements = 1,000,000` up front,
  touching hundreds of MB of `link_list_locks_` + base-layer memory per index
  regardless of how many vectors it actually held. Indices now start at
  `INITIAL_MAX_ELEMENTS = 4096` and double via `resizeIndex()` as needed, so RAM
  tracks the real working set.
- **~7.7× less RAM for many-namespace deployments.** A 19-namespace workload
  with ~70 vectors each went from **709 MB → 92 MB** of index overhead; a single
  small DB drops correspondingly. No upper limit — indices grow past the old 1M
  cap if you actually store that many vectors.
- **Growth is reservation-based and thread-safe at the call sites.** `reserve()`
  is invoked before every insert path — single `add()`/WAL replay reserve `+1`;
  `add_batch()` and parallel load reserve the whole batch *before* the
  concurrent `parallel_add` (since `resizeIndex` reallocs buffers and is not
  thread-safe); `compact()` sizes the rebuilt index to the survivor count.
- No format change (still v8) and no API change. Recall, parallel-load speedup
  (4.79×), int8-RAM savings, pre-filtered search, and auto-compaction all
  verified unchanged by the regression suite.

## [0.15.2] — 2026-06-17

### Admin dashboard polish
- **Records table virtualized** — only the ~30 rows in the scroll window are in
  the DOM (spacer rows preserve scroll height), so a heavily paged list stays
  light instead of accumulating thousands of `<tr>`s. Sticky header.
- **API key — "Remember on this device" toggle** — checked → `localStorage`
  (persists, the default, no sign-out surprise); unchecked → `sessionStorage`
  (clears when the tab closes). Reads from either store on load; sign-out clears
  both. Replaces the bare localStorage handling with an explicit, secure choice.
- **Focus-trap in modals & drawer** — added the Alpine Focus plugin and `x-trap`
  to the key dialog, record drawer, new-namespace / import / confirm modals, so
  keyboard focus stays inside the open dialog.

---

## [0.15.1] — 2026-06-16

### Added — Real embedders for the MCP connector
- `feather-serve --embed-provider gemini|openai|voyage|cohere|ollama` builds a
  real client-side embedder so the persona MCP's `feather_recall` is genuinely
  semantic (was a deterministic hash embedder). Gemini `text-embedding-004` is
  native **768-dim** (matches the hosted namespaces); OpenAI `text-embedding-3-small`
  honours `--dim 768` directly. Key via `--embed-key` / `FEATHER_EMBED_API_KEY` /
  `GOOGLE_API_KEY` / `OPENAI_API_KEY`. New module
  `feather_db.integrations.embedders.make_embedder` (urllib-only, dim-fitting).
- Docstring shows Claude Desktop / Code config with a real embedder.

---

## [0.15.0] — 2026-06-14

### Added — In-RAM int8 quantization (file format v8)
- `set_int8_ram(modality, max_abs)` / `is_int8_ram(modality)` — store a
  modality's vectors as **int8 in memory** (not just on disk) via a new
  `Int8L2Space` (global scale = `max_abs/127`, integer-L2 distance). Measured
  **~1.6–1.8× less RAM** at 60k×768 (227 MB → 129 MB) with recall **0.88** vs
  the float baseline. Query/insert quantize transparently; `get_vector`
  dequantizes. Persisted in **file format v8** (per-modality int8-RAM flag +
  scale); v3–v7 files load unchanged.
- Best for embedding-shaped vectors; the global scale resolves moderate dynamic
  ranges well. int8 is lossy by design — a small recall drop trades for memory.

### Fixed
- **`open()` no longer pre-creates an empty "text" index.** That throwaway index
  preallocated ~70 MB (1M-element link locks) and forced `set_int8_ram()` /
  `set_quantized()` to build a *second* index — doubling RAM. The modality is
  now created lazily on the first `add()`; `dim()` falls back to the configured
  default so the Cloud API still validates before any vectors exist.
- `add_point()` reuses a thread-local int8 buffer during parallel inserts
  (avoids per-call allocation churn).

---

## [0.14.0] — 2026-06-11

### Added — MCP remote backend (persona context engine)
- `feather-serve` can now talk to a **deployed Feather Cloud API** over HTTP
  (`--api-url` / `--api-key`, or `FEATHER_API_URL` / `FEATHER_API_KEY`) instead
  of opening a local `.feather` file — so Claude (Desktop or Code) can use a
  hosted, shared Feather instance as a persona context engine. Local `--db`
  mode is unchanged.
- New `feather_db.integrations.mcp_remote.RemoteFeatherTools` — embeds
  **client-side** and calls the plain REST endpoints, so the server needs no
  embedding configuration and any namespace dim works.
- Remote tool set (8): `feather_ingest`, `feather_recall`,
  `feather_keyword_recall`, `feather_context_chain`, `feather_get_record`,
  `feather_link`, `feather_stats`, `feather_list_namespaces`.
- Docstring documents Claude Desktop **and** Claude Code config for both local
  and remote backends.

### Fixed
- **`/v1/{ns}/context_chain` 500** whenever the chain contained edges — the
  endpoint read `e.source_id`/`e.target_id` but the `ContextEdge` binding
  exposes `.source`/`.target`. (Found while testing the MCP connector.)

---

## [0.13.2] — 2026-06-11

### Fixed
- **Admin dashboard sign-out regression**: 0.13.1 moved the API key to
  `sessionStorage`, which silently logged out operators whose key lived in
  `localStorage` (dashboard rendered empty until the key was re-entered).
  Reverted to `localStorage` so the key auto-loads as before. No data was
  affected — this was purely the browser's stored key. (Proper key-handling
  hardening can return later as an opt-in, without breaking existing sessions.)

---

## [0.13.1] — 2026-06-11

Cloud API + admin dashboard — surface the Phase 7–8 capabilities and fix a few gaps.

### Added — Cloud API (`feather-api/`)
- `GET /v1/{ns}/admin/index_stats` — record count, dim, on-disk quantization
  state, auto-compaction threshold, and per-namespace **secondary-index** sizes.
- `PUT /v1/{ns}/admin/auto_compact` `{ratio}` — enable/adjust incremental
  auto-compaction.
- `PUT /v1/{ns}/admin/quantize` `{modality, on}` — toggle on-disk int8
  quantization (persists on save).
- `POST /v1/{ns}/import` now inserts via **`add_batch`** (parallel HNSW build)
  instead of one-at-a-time — much faster bulk import.

### Added — Admin dashboard (`static/admin/`)
- New **Maintenance** tab per namespace: index stats, secondary-index namespace
  sizes, auto-compaction control, on-disk int8 toggle, and compact-now.
- **Keyword** search mode (BM25) alongside Vector / Hybrid; query-text field
  shown for keyword, vector-seed hidden.

### Changed / Fixed
- API key moved from **localStorage → sessionStorage** (no longer persists
  across browser sessions).
- Cmd-K palette "open record" no longer truncates ids > 2^53 (`parseInt`
  removed — id passed through as-is).

---

## [0.13.0] — 2026-06-10

Phase 8 — ingestion & load performance.

### Added — Parallel load
- `load_vectors()` now reads vectors serially (sequential I/O) and rebuilds the
  HNSW graph **in parallel** across a thread pool (hnswlib `addPoint` is
  thread-safe). Measured **~4.5–4.8× faster load** on 8 cores (40k×128: 7.6s →
  1.7s) with **identical recall** (0.950 serial vs 0.950 parallel — both builds
  are valid approximate graphs).
- `FEATHER_LOAD_THREADS` env var caps/sets the pool size (`1` = serial). Small
  modalities (<256 vectors) stay serial to avoid thread overhead.

### Added — Parallel batch ingestion
- `DB.add_batch(ids, vecs, metas=None, modality="text")` — bulk-insert N records
  (`vecs` = N×dim float32) with the **HNSW graph built in parallel** and the GIL
  released during the insert. Measured **~3.4× faster** than a serial `add()`
  loop (40k×128: 15.1s → 4.5s), recall **1.000**. Metadata, secondary indexes,
  and BM25 are maintained exactly as `add()` (so pre-filtered search works over
  batch-ingested data); same-id re-adds and empty/mismatched inputs are handled.

### Changed — SIMD distance kernels enabled on x86
- `setup.py` now compiles the hand-written **SSE/AVX** L2 kernels in `space_l2.h`
  (previously dormant — the `USE_*` macros were never defined, so every distance
  was scalar). Enabled **only on x86_64** with runtime CPU dispatch (`AVXCapable()`)
  — SSE2 is baseline, AVX selected at runtime. On arm64/aarch64 the build is
  unchanged (relies on `-O3` NEON auto-vectorization). Override via
  `FEATHER_SIMD=none|sse|avx|avx512`. Speeds up dense search and the pre-filtered
  exact path on x86 servers/containers (the deployment target).

---

## [0.12.0] — 2026-06-09

### Added — On-disk int8 quantization (file format v7)
- `set_quantized(modality, on=True)` / `is_quantized(modality)` — persist a
  modality's vectors as **int8 + per-vector scale** instead of float32.
  ~**2.5–4× smaller `.feather` files** and faster load I/O; vectors are
  dequantized back to float32 on load, so the in-memory HNSW index and search
  are unchanged. Opt-in per modality, default off.
- Per-vector symmetric scalar quantization (`scale = max|v|/127`). Measured on
  500×128-dim vectors: file **2.87× smaller**, max element error **0.39%**,
  top-10 recall **9/10** vs the float32 baseline.
- **File format bumped to v7** — adds a 1-byte per-modality quantization flag
  (and, when set, an int8 vector section). v3–v6 files load transparently (the
  flag is only read for v7+); a non-quantized v7 save is byte-compatible in
  spirit with v6 plus the flag.
- The quantization choice is persisted in the file, so a reloaded DB keeps
  saving that modality quantized.

> Note: this reduces **disk footprint and load I/O**, not in-RAM size — the HNSW
> index remains float32 in memory. In-RAM int8 indexing is a larger follow-up.

---

## [0.11.0] — 2026-06-09

Phase 7 — query performance. New capabilities land additively under this line.

### Added — Secondary metadata indexes
- **Inverted indexes** on `namespace_id`, `entity_id`, and each `attribute`
  key=value, mirroring the existing `reverse_index_` pattern. Rebuilt on load
  and maintained incrementally on `add`/`update_metadata`/`forget`/`purge`, so
  namespace/entity/attribute lookups go from **O(n) scans to O(matches)** —
  directly attacking the "slow filtered queries / load on large attribute DBs"
  limitation.
- Only **live** records are indexed (forgotten/deleted excluded), so the id
  sets double as ready-made candidate sets for the upcoming pre-filtered ANN
  search.
- New Python methods on `DB`: `ids_in_namespace(ns)`, `ids_for_entity(eid)`,
  `ids_with_attribute(key, value)`, `namespace_size(ns)`, `list_namespaces()`.
- No file-format change — indexes are derived in-memory on load.

### Added — Pre-filtered ANN search
- `search(..., filter=...)` now takes an **exact pre-filtered path** when the
  filter constrains an indexed field (namespace/entity/attribute): it resolves
  the candidate set from the secondary indexes, applies any remaining
  (non-indexed) predicates, and ranks **exactly** over just those vectors.
- Fixes the long-standing **"selective filter returns far fewer than k"** bug —
  HNSW's `ef`-bounded filtered traversal would stop after exploring `ef`
  neighbours and silently under-return. The new path returns up to k matches
  whenever ≥k records match, and is **O(matches)** so a selective filter is also
  faster. Non-indexed-only filters (source/importance/tags/time) keep the
  existing HNSW path. No API or file-format change.

### Added — Incremental auto-compaction
- `set_auto_compact(ratio)` / `get_auto_compact()` — when a modality index's
  deleted/total ratio crosses `ratio` after a `forget`/`purge`/`forget_expired`,
  the index is rebuilt to reclaim the dead vectors (HNSW only marks deletions;
  space was previously reclaimed only via a manual `compact()`). `0` disables
  it (default), so behaviour is unchanged unless opted in.

### Fixed — compact()
- `compact()` now reclaims **forgotten** records too (it previously only removed
  `_deleted=true` records, so `forget()`'d vectors lingered in the index
  forever). The dead definition is unified on `is_dead_meta()` across save,
  indexing, and compaction.
- `compact()` is now **orphan-safe**: it keeps only records that are present and
  live in the metadata store, so vectors whose metadata was hard-deleted by
  `purge()` can no longer be resurrected into the rebuilt index.

---

## [0.10.13] — 2026-06-08

Consolidated patch line on top of the 0.10.0 Cloud Edition release. Packaging
metadata (`pyproject.toml`, `setup.py`, `Cargo.toml`) is now aligned to
`0.10.13` to match `feather_db.__version__` — it had drifted to `0.9.0`.

### Fixed
- **Dangling edges on record delete** — deleting a record now also strips
  inbound/outbound edges that referenced it, so `export_graph_json` and the
  graph view no longer surface phantom nodes (was v0.10.11).
- **Delete persistence** — `forget()`/`purge()` now survive save+reload; the
  C++ `save_vectors()` path was rewritten so deletions are not resurrected on
  load.

### Added
- **Admin SPA (Hawky edition)** — delete-namespace, schema discovery, filter
  chips, edit-record form, embedding-config panel, bulk import, hierarchy view,
  marketing KPI card, ops-per-minute sparkline, top-recalled list, connection
  panel. `Cache-Control: no-cache` on the SPA so deploys are picked up live.
- **Bulk endpoints** — `/purge` and `/compact`.

### Hardened (Cloud API)
- **Search input bounds** — `k` is now validated `1..1000` (and `hops` `0..10`,
  `rrf_k` `1..10000`) on `/search`, `/keyword_search`, `/hybrid_search`,
  `/context_chain`. A negative or absurd `k` previously flowed straight into the
  C++ `size_t` search width and could OOM/crash the worker; it now returns 422.
- **Query-dimension guard** — search endpoints reject a query vector whose
  length ≠ the index dim with a clean `400` instead of risking an out-of-bounds
  read inside the HNSW index.
- **Embedding response robustness** — all six providers now raise the documented
  `RuntimeError` (→ clean error to the client) when an upstream returns an
  error-shaped or otherwise unexpected payload, instead of a bare
  `KeyError`/`IndexError` surfacing as a 500.

### Security
- Removed a leaked production API key and VM IP from previously tracked files.

---

## [0.10.0] — 2026-05-12

### Added — Feather DB Cloud Edition

This release replaces the legacy Gradio dashboard with a custom, brand-aligned
admin SPA and adds the connective tissue needed to run Feather as a managed
context engine for brand customers.

#### `feather-api/static/admin/` — Atlas-style admin SPA
- New single-page admin UI mounted at **`/admin/`** (HTML + Tailwind via CDN +
  Alpine.js + D3, ~2k LOC, zero build step). Served by FastAPI with
  `Cache-Control: no-cache` so each deploy is picked up without manual refresh.
- **Overview screen** — 4 record-count tiles + 4 ops/latency tiles + ops-per-minute
  sparkline + recent-activity feed.
- **Namespaces list** with create-namespace modal.
- **Namespace detail** with 7 tabs: Records · Search · Schema · Hierarchy ·
  Graph · Context · Console.
- **Record drawer** with Metadata + Edges tabs, add-edge form, edit-record form.
- **Marketing profile card** — when a record has marketing-shaped attributes
  (`brand`, `channel`, `campaign`, `ctr`, `roas`), render as a KPI card.
- **Cmd-K command palette** — fuzzy search namespaces, jump to record by id,
  trigger actions.
- **Settings** — connection info (cURL / Python / JS snippets), embedding service
  config, API key management.
- Removed Gradio dashboard (~1300 LOC).
- Brand-aligned per the Feather DB brand book: paper `#fcfaf7` surfaces,
  Geist + Geist Mono + Playfair Display italic type, lime as gesture only,
  ink-black primary CTAs, canonical hawk-feather logo.

#### `feather-api/app/embedding.py` — pluggable embedding service
- **Six providers** with curated model dropdowns:
  - OpenAI (text-embedding-3-small/large, ada-002)
  - Azure OpenAI (endpoint + deployment + api_version fields)
  - Google AI / Gemini (gemini-embedding-001, exp, legacy)
  - Voyage (voyage-3, voyage-3-lite, voyage-large-2, voyage-code-2)
  - Cohere (embed-english-v3, multilingual-v3, light-v3)
  - Ollama (nomic-embed-text, mxbai-embed-large, all-minilm)
- Stored in-memory; API key never echoed back. Padding/truncation to namespace
  dim if a different-dim model is configured.
- New endpoints: `GET/PUT /v1/admin/embedding_config`,
  `GET /v1/admin/embedding_models`.
- **`POST /v1/{ns}/ingest_text`** — server-side embed + store in one call.

#### `feather-api/app/metrics.py` — in-memory observability
- Ring-buffer for HTTP ops (default 2000 events).
- Snapshot endpoint with p50/p95/p99 latency.
- Bucketed time-series for sparkline rendering.
- HTTP middleware records every `/v1/*` request.

#### New backend endpoints
- `POST   /v1/namespaces` — create empty namespace
- `DELETE /v1/namespaces/{ns}` — hard-delete namespace (.feather + WAL)
- `GET    /v1/{ns}/schema` — distinct attribute keys + samples
- `GET    /v1/{ns}/hierarchy` — Brand → … → Creative tree from attributes
- `GET    /v1/{ns}/top_recalled` — records sorted by `recall_count`
- `GET    /v1/{ns}/graph` — D3-shaped nodes + links
- `POST   /v1/{ns}/context_chain` — vector search + BFS expansion
- `POST   /v1/{ns}/purge` — bulk hard-delete by `metadata.namespace_id`
- `POST   /v1/{ns}/compact` — rebuild HNSW dropping soft-deleted records
- `POST   /v1/{ns}/seed` — bulk-insert N random records (test seeding)
- `POST   /v1/{ns}/import` — bulk import inline `{id, vector, metadata}`
- `POST   /v1/{ns}/ingest_text` — embed + store via configured provider
- `GET    /v1/{ns}/records/{id}/edges` — outgoing + incoming edges
- `GET    /v1/admin/{overview,metrics,activity,ops_timeseries,connection_info}`

#### Bug fixes
- **Delete persistence (C++ `save_vectors`)** — `forget()` and `purge()` now
  properly skip records in the persisted file. Previously, forgotten vectors
  resurrected on save+reload as orphans.
- **Big-int id precision** — dashboard preserves >2^53 record ids as strings
  through JSON parse so view/delete works on huge ids. Seeded ids capped at
  2^53 - 1 going forward.
- **Gemini default model** — switched to `gemini-embedding-001`
  (Google deprecated `text-embedding-004`).

#### Removed
- `feather-api/dashboard.py` (Gradio).
- `feather-api/Dockerfile.dashboard`.
- `feather-dashboard` service from `docker-compose.yml`.

---

## [0.9.0] — 2026-04-29

### Added — Agentic Context Engine (Phase 9.1)

#### `feather_db.extractors` — LLM-backed ingestion primitives
- **`FactExtractor`** — converts raw text turns into atomic (subject, predicate, object) triples with confidence scores and temporal anchors. Pluggable on any `LLMProvider`.
- **`EntityResolver`** — canonicalises entity mentions across turns; de-duplicates aliases into stable canonical IDs (e.g. "Alice" / "al" / "alice" → one node).
- **`TemporalParser`** — rule-based ISO-8601 timestamp extraction; stamps `observed_at` / `valid_at` on facts without any LLM call.
- **`OntologyLinker`** — LLM-backed typed-edge inference between facts and entities. Whitelisted `rel_type` set (supports, contradicts, supersedes, etc.). Detect-only — never auto-resolves. `contradicts` / `supersedes` edges require a `rationale` string.
- **`ContradictionResolver`** — two-layer contradiction detection: rule pre-filter (same subject+predicate, numeric tolerance ±5%, temporal classification) + optional LLM severity scoring (`definite` / `probable` / `possible`). Returns `ContradictionFinding` list — never auto-resolves.
- **`ContradictionFinding`** dataclass — `new_fact`, `conflicting_with`, `severity`, `rationale`, `suggested_resolution`, `detected_at`. Audit-trail-friendly.
- Stable item IDs (`f_0`, `f_1` for Facts; `e_0`, `e_1` for Entities) used during LLM calls so edge endpoints can be mapped back to real Feather row IDs post-persistence.

#### `feather_db.pipelines.IngestPipeline` — enhanced
- **`enable_phase2` flag** — gates `OntologyLinker` + `ContradictionResolver`; off by default (OSS), intended for Cloud activation.
- Phase 2 flow: fetch contradiction candidates via `FilterBuilder` before persisting → detect contradictions → `db.link("contradicts", severity_weight)` → apply ontology edges.
- **`IngestStats`** gains `ontology_edges_added`, `contradictions_detected`, `contradictions_by_severity` fields.
- `_apply_ontology_links()` maps stable `f_0`/`e_0` IDs back to real row IDs after persistence.

#### `feather_db.feedback` — append-only audit log
- **`FeedbackEvent`** — dataclass with `kind`, `target_id`, `namespace_id`, `timestamp`, `user_id`, `payload`, `notes`, deterministic `event_id` (sha256[:12]).
- **`FeedbackLog`** — append-only JSONL; `log()`, `iter_events()` (filtered), `aggregate_for_target()`, skips corrupt lines.
- **`feedback_decay_modifier()`** — layered on `recall_count`: retracted→0.0, corrections penalise (0.5^N, cap 3), endorsements boost (1+0.2*N, cap 5), namespace-isolated.

#### `feather_db.hierarchy` — marketing attribution tree
- **`Hierarchy`** class with `MARKETING_HIERARCHY` default (`Brand → Channel → Campaign → AdSet → Ad → Creative`).
- `ancestors()`, `descendants()` (BFS), `common_ancestor()`, `is_descendant_of()`.
- Parent-level ordering enforced on `add()`.

#### `feather_db.reason` — QueryPlanner + PlanExecutor skeleton
- **`QueryPlanner`** — emits a single-step `hybrid_search` plan by default; subclass hook for multi-step plans (Phase 11).
- **`PlanExecutor`** — executes `hybrid_search` + `vector_search`, falls back to vector on hybrid failure, warns on unimplemented step kinds, captures `step_traces`.

#### `feather_db.integrations` — Phase 9 MCP tools
- **`feather_ingest`** MCP tool — runs `IngestPipeline` (Phase 9 structured extraction) when `system_provider` configured; falls back to raw-text storage. Per-call `namespace` override.
- **`feather_recall`** MCP tool — hybrid search with adaptive decay (`ScoringConfig`); `half_life_days`, `time_weight` knobs. Total MCP tools: 14 → 16.
- **`feather-serve` CLI** — new `--system-provider`, `--system-model`, `--namespace` args. Supports `claude`, `openai`, `gemini`, `ollama` system providers.

### Tests
- 108 new tests across 6 test files: `test_extractors_ontology` (20), `test_extractors_contradictions` (19), `test_feedback` (20), `test_hierarchy` (16), `test_reason` (11), `test_pipelines_phase2` (9), `test_mcp_phase9` (20), `test_phase9_smoke` (13). All pass; 0 regressions.

### Changed
- `feather_db.integrations.base.FeatherTools.__init__` accepts `system_provider` + `namespace` params.

---

## [0.8.0] — 2026-04-04

### Added — BM25 Hybrid Search (Phase 6)

#### C++ Core (`include/feather.h`)
- **BM25 inverted index** — in-memory posting list (`term → [{doc_id, term_freq}]`) with document length tracking and adaptive `avg_dl` computation.
- **`tokenize(text)`** — static C++ tokenizer: lowercase, split on non-alphanumeric, stop-word filtered (40 common English words), minimum 2-char tokens.
- **`keyword_search(query, k, filter)`** — Okapi BM25 scoring (`k1=1.2, b=0.75`). IDF formula: `log((N - n_t + 0.5) / (n_t + 0.5) + 1)`. Fully filter-aware. Returns ranked `SearchResult` list.
- **`hybrid_search(vec, query, k, rrf_k, filter, scoring, modality)`** — merges dense vector search and BM25 via **Reciprocal Rank Fusion** (`score = Σ 1/(rrf_k + rank + 1)`, default `rrf_k=60`). Same RRF formula used by Qdrant, Weaviate, Elasticsearch.
- BM25 index auto-built from `metadata_store_` on `load()` — fully **backward-compatible** with v3/v4/v5/v6 `.feather` files.
- `add()` and `update_metadata()` both auto-update the inverted index on content changes.

#### Python Bindings (`bindings/feather.cpp`)
- `db.keyword_search(query, k, filter)` → `list[SearchResult]`
- `db.hybrid_search(vec, query, k, rrf_k, filter, scoring, modality)` → `list[SearchResult]`

#### REST API (`feather-api/`)
- `POST /v1/{namespace}/keyword_search` — BM25 search with full filter support
- `POST /v1/{namespace}/hybrid_search` — hybrid BM25+vector search with RRF

---

## [0.7.0] — 2026-03-21

### Added — Self-Aligned Context Engine (Phase 1)

#### `feather_db.providers` — LLM Provider Abstraction
- **`LLMProvider`** (ABC) — minimal `complete(messages, max_tokens, temperature) -> str` interface shared by all providers
- **`ClaudeProvider`** — Anthropic Claude via `anthropic` SDK. Default: `claude-haiku-4-5-20251001`. Separates `system` from conversation messages per Anthropic API spec.
- **`OpenAIProvider`** — OpenAI Chat Completions API and any compatible endpoint (Groq, Mistral, Together AI, vLLM, LM Studio). `json_mode=True` sets `response_format=json_object`.
- **`OllamaProvider`** — Ollama local server. Subclass of `OpenAIProvider`; default `base_url="http://localhost:11434/v1"`. No API key required.
- **`GeminiProvider`** — Google Gemini via `google-genai` SDK. Uses `response_mime_type="application/json"` for native JSON mode.
- All providers default to `temperature=0.0` for deterministic JSON output.

#### `feather_db.engine` — ContextEngine
- **`ContextEngine`** — self-aligned ingestion pipeline that wraps `DB` with LLM-powered classification.
- **`ingest(text, hint) -> int`** — 10-step pipeline: embed → sample context → LLM classify → apply hint → store → link → episode → watch triggers → contradiction check → auto-save.
- **`ingest_batch(texts, hints) -> list[int]`** — batch ingestion helper.
- LLM JSON schema: `{entity_type, importance, confidence, ttl, namespace, episode_id, suggested_links}`
- **`_heuristic_classify(text)`** — built-in keyword-based fallback; activated when `provider=None` or LLM fails. Fully offline, zero latency.
- 5-stage JSON extraction — direct parse → strip code fences → balanced brace scan → trailing comma repair → heuristic fallback. Robust on small/local models.
- Node ID: SHA-256 of `text[:200] + timestamp + PID` mod 2^50.
- Integrates with `WatchManager`, `EpisodeManager`, `ContradictionDetector` (all optional).

#### New exports in `feather_db`
- `LLMProvider`, `ClaudeProvider`, `OpenAIProvider`, `OllamaProvider`, `GeminiProvider`, `ContextEngine`

#### New example
- `examples/context_engine_demo.py` — auto-detects provider from env vars (Claude → OpenAI → Gemini → Ollama → heuristic), ingests 6 records, runs semantic search and context chain.

### Changed
- `feather_db.__version__` → `"0.7.0"`
- `README.md` — updated to v0.7.0 architecture; added Self-Aligned Context Engine, LLM agent connectors, MCP server, LangChain/LlamaIndex sections.

---

## [0.6.1] — 2026-03-14

### Fixed
- `setup.py`: `-undefined dynamic_lookup` linker flag now only applied on macOS — fixes build from source on Linux (HuggingFace Spaces, manylinux, Ubuntu)

## [0.6.0] — 2026-03-13

### Added — Complete Living Context Engine

#### C++ Core (file format v6)
- **`Metadata.ttl`** (int64_t) — time-to-live in seconds from `timestamp`. `0` = never expires. Enables working/session memory that auto-evicts.
- **`Metadata.confidence`** (float, default 1.0) — epistemic certainty about the stored fact. Separate from `importance` (relevance weight).
- **`DB.forget(id)`** — soft-delete: `markDelete` in HNSW (exits all searches), blanks `content`, sets `importance=0`. Graph shell preserved so incoming edges remain traversable.
- **`DB.purge(namespace_id)`** — hard-delete all nodes in a namespace. Removes from HNSW, `metadata_store_`, reverse index, and outgoing edges of surviving nodes. Returns count.
- **`DB.forget_expired()`** — scans all nodes, calls `forget()` on any where `ttl > 0 && now > timestamp + ttl`. Returns count of expired nodes.
- File format bumped to **v6**. v5, v4, v3 files load transparently via end-of-stream guards.

#### `feather_db.memory` — MemoryManager
- **`why_retrieved(db, id, query_vec)`** — full score breakdown: similarity, stickiness, effective_age_days, recency, importance, confidence, final_score, and human-readable formula string.
- **`health_report(db)`** — orphan nodes, hot/warm/cold tier counts, expired-TTL count, recall histogram, avg_importance, avg_confidence.
- **`search_mmr(db, query_vec, k, diversity)`** — Maximal Marginal Relevance post-processing. Balances relevance vs diversity. `diversity=0` = pure similarity, `diversity=1` = maximum spread.
- **`assign_tiers(db)`** — classifies nodes as hot/warm/cold by recall_count + recency percentile. Optionally writes `tier` attribute back to DB.
- **`consolidate(db, namespace, since_hours, llm_fn)`** — greedy union-find clustering by cosine similarity, generates summary nodes, links originals with `consolidated_into` edge, lowers original importance to 0.3.

#### `feather_db.triggers` — WatchManager + ContradictionDetector
- **`WatchManager.watch(db, query_text, threshold, callback, embed_fn)`** — register a semantic watch that fires `callback(node_id, similarity)` when a new node matches.
- **`WatchManager.check_triggers(db, new_node_id)`** — check all watches against a newly added node. Returns list of match records.
- **`ContradictionDetector.check(db, new_node_id, threshold, auto_link)`** — detect high-similarity nodes from a different source; auto-create `contradicts` edges.
- **`ContradictionDetector.scan_all(db)`** — full-DB contradiction scan (expensive; use sparingly).

#### `feather_db.episodes` — EpisodeManager
- **`begin_episode(db, episode_id, description)`** — create an episode header node; returns deterministic header ID.
- **`add_to_episode(db, node_id, episode_id)`** — tag node with episode_id attribute + link from header via `episode_contains` edge.
- **`get_episode(db, episode_id)`** — ordered member nodes by timestamp.
- **`close_episode(db, episode_id)`** — seals episode, adds `episode_end` edge, writes `episode_status=closed`.
- **`list_episodes(db)`** — all open and closed episodes.

#### `feather_db.merge`
- **`merge(target_db, source_path, conflict_policy)`** — merge two Feather DBs. Policies: `keep_target` (default), `keep_source`, `merge` (union attrs, higher importance/confidence wins). Returns `{merged, skipped, conflicts, edges_added}`.

#### `feather_db.integrations.langchain_compat`
- **`FeatherVectorStore`** — LangChain `VectorStore`: `add_texts()`, `similarity_search()`, `similarity_search_with_score()`, `from_texts()`, `from_documents()`.
- **`FeatherMemory`** — LangChain `BaseMemory`: semantic history retrieval with adaptive decay scoring, `save_context()`, `clear()`.
- **`FeatherRetriever`** — LangChain `BaseRetriever`: wraps `context_chain()` for graph-expanded retrieval.

#### `feather_db.integrations.llamaindex_compat`
- **`FeatherVectorStoreIndex`** — LlamaIndex `VectorStore`: `add()` (BaseNode), `query()` (VectorStoreQuery → VectorStoreQueryResult), `delete()`.
- **`FeatherReader`** — LlamaIndex `BaseReader`: `load_data(db_path, dim)` → list of LlamaIndex Documents.

#### 7 New Agent Tools (all providers — total 14 tools)
- **`feather_forget`** — soft-delete a node by ID
- **`feather_health`** — knowledge graph health report
- **`feather_why`** — score breakdown explaining a retrieval
- **`feather_mmr_search`** — diversity-aware semantic search
- **`feather_consolidate`** — cluster + merge similar nodes
- **`feather_episode_get`** — ordered nodes in a named episode
- **`feather_expire`** — scan and expire all TTL-exceeded nodes

#### `feather_db.integrations.mcp_server` — MCP Server
- **`feather-serve`** CLI entry point — `feather-serve --db my.feather --dim 3072`
- Full MCP server exposing all **14 Feather tools** as first-class MCP tool definitions
- Works with Claude Desktop, Cursor, and any MCP-compatible agent — zero code required
- Claude Desktop config: `{"mcpServers": {"feather": {"command": "feather-serve", "args": ["--db", "my.feather"]}}}`
- `asyncio.run_in_executor` wrapping for non-blocking tool dispatch
- Resource endpoint: `feather://db/info` returns DB stats

### Changed
- `feather_db.__version__` → `"0.6.0"`
- `pyproject.toml`: optional dependency groups `langchain`, `llamaindex`, `mcp`, `all`; `feather-serve` console script entry point
- `feather_db.integrations.__init__`: exports LangChain + LlamaIndex adapters (graceful fallback if deps absent)

---

### Added — LLM Agent Connectors (`feather_db.integrations`)

- **`feather_db/integrations/`** package — production-ready connectors that expose Feather DB as tool-use / function-calling tools for any LLM provider.
- **`ClaudeConnector`** — Feather DB tools in Anthropic `input_schema` format. Drop-in for `client.messages.create(tools=...)`. Includes `run_loop()` for fully autonomous multi-turn agent execution.
- **`OpenAIConnector`** — OpenAI function-calling format. Works with OpenAI (`gpt-4o`), Azure OpenAI, Groq (`llama-3.3-70b-versatile`), Mistral, Together AI, Ollama. Pass `base_url=` for any OpenAI-compatible endpoint.
- **`GeminiConnector`** — Gemini `FunctionDeclaration` + `types.Tool` format. Works with `client.chats.create(config=conn.chat_config())` and `google-genai` SDK.
- **`GeminiEmbedder`** — Multimodal embedder wrapping `models/gemini-embedding-2-preview` (3072-dim). Supports `embed_text()`, `embed_image()`, `embed_video_transcript()`, `embed_any()`. Mock mode (offline) built in.
- **`FeatherTools`** base class — 7 built-in tools available to all connectors:
  - `feather_search` — semantic search with namespace/entity/product filtering
  - `feather_context_chain` — vector search + BFS graph expansion (n hops)
  - `feather_get_node` — full metadata + edge inspection by ID
  - `feather_get_related` — graph neighbour traversal (in/out/both directions)
  - `feather_add_intel` — agent-ingestible new intelligence nodes
  - `feather_link_nodes` — typed weighted edge creation from agent output
  - `feather_timeline` — chronological node list by product or entity type
- All connectors share a single `TOOL_SPECS` definition (single source of truth) — adding a new tool propagates automatically to all three providers.
- Mock embedder fallback — all connectors work fully offline without an API key (useful for testing, CI, local dev).
- Top-level exports: `from feather_db import ClaudeConnector, OpenAIConnector, GeminiConnector, GeminiEmbedder`.
- **Example**: `examples/agent_connectors_demo.py` — complete multi-connector demo with context chain, timeline, write tools, and agent loop for all four providers (Claude, OpenAI, Groq, Gemini). Runs fully offline in mock mode.

---

## [0.5.0] — 2026-02-28

### Added — Context Graph & Living Context Engine

- **`Edge` struct** (C++): `{target_id, rel_type, weight}` — replaces flat `links` list. Backward-compat: old `links` IDs promoted to `Edge(rel="related_to", w=1.0)` on load. Python `meta.links` property still works.
- **Typed, weighted `db.link(from, to, rel_type, weight)`** — 9 built-in relationship types: `related_to`, `derived_from`, `caused_by`, `contradicts`, `supports`, `precedes`, `part_of`, `references`, `multimodal_of`. Free-form strings also accepted.
- **Reverse index** (`IncomingEdge`): in-memory index rebuilt on load; query via `db.get_incoming(id)`.
- **`db.get_edges(id)`** — outgoing edges with type + weight.
- **`db.auto_link(modality, threshold, rel_type, candidates)`** — auto-creates edges between records whose vector similarity exceeds `threshold`. Semantic relationship discovery without manual labeling.
- **`db.context_chain(query, k, hops, modality)`** — vector search + n-hop BFS graph expansion. Returns `ContextChainResult` with scored `ContextNode` list and `ContextEdge` list. Score = `similarity × hop_decay × importance × stickiness`.
- **`db.export_graph_json(namespace_filter, entity_filter)`** — D3/Cytoscape-compatible JSON. Dangling-edge safe: only emits edges where both source and target are in the exported node set.
- **`db.get_vector(id, modality)`** — retrieve raw vector for a given ID and modality.
- **`db.get_all_ids(modality)`** — list all IDs present in a modality index.
- **`feather_db.graph` module**: `export_graph()` → Python dict; `visualize()` → self-contained interactive D3 force-graph HTML (D3 inlined, no CDN dependency, works offline).
- **`feather_db.RelType`** — constant class with all 9 standard relationship type strings.
- **File format v5**: edges appended after attributes section; v3/v4 files load transparently.
- **Interactive Inspector** (`examples/feather_inspector.py`): local HTTP server with force graph, PCA embedding scatter view, top-8 similarity panel, edit and delete via REST API.
- **Example**: `examples/context_graph_demo.py` — full Nike campaign knowledge graph demo with auto-link, context_chain, multimodal pocket, HTML export.

### Changed
- `feather_db.__version__` bumped to `0.5.0`.
- `db.size()` method added — returns total record count.
- `visualize()` now inlines D3.js (280KB) — fully offline, no CDN needed.
- `export_graph_json` filters dangling edges (edges to nodes excluded by namespace filter).
- Force-graph simulation tuned for large graphs (>100 nodes): nodes pre-spread, edge labels hidden, faster alpha decay.

---

## [0.4.0] — 2026-02-20

### Added — Generic Living Context Engine (Namespace + Entity + Attributes)

- **Namespace + Entity + Attributes** (`Metadata` fields): `namespace_id` (partition key — brand, org, tenant), `entity_id` (subject key — user, customer, product), `attributes` (domain-specific KV map `map<string,string>`). C++ core remains fully domain-agnostic.
- **File format v4**: new fields appended after `last_recalled_at` in the binary layout; v3 files load transparently (missing fields default to empty).
- **`DB.update_metadata(id, meta)`**: replace the full metadata record for an existing ID without touching the HNSW index.
- **`DB.update_importance(id, importance)`**: targeted importance update for signal-feedback loops.
- **`meta.set_attribute(key, value)` / `meta.get_attribute(key, default)`**: safe KV helpers that bypass pybind11 map copy gotcha (`meta.attributes['k'] = v` silently does nothing; use these methods instead).
- **Filter fields**: `SearchFilter.namespace_id`, `SearchFilter.entity_id`, `SearchFilter.attributes_match` (all-KV-match semantics).
- **`FilterBuilder` methods**: `.namespace(ns)`, `.entity(eid)`, `.attribute(key, value)` (chainable, accumulates KV pairs).
- **`DomainProfile`** (Python base class): generic typed helpers `set_namespace()`, `set_entity()`, `set_attr()`, `get_attr()`, `to_metadata()`.
- **`MarketingProfile(DomainProfile)`**: digital marketing adapter — `set_brand()`, `set_user()`, `set_channel()`, `set_campaign()`, `set_ctr()`, `set_roas()`, `set_platform()` plus read-only properties.
- **Example**: `examples/marketing_living_context.py` — end-to-end multi-brand demo with namespace/entity/attribute filtering and importance feedback.

### Changed
- `feather_db.__version__` bumped to `0.4.0`.
- `feather_db.__all__` extended with `DomainProfile`, `MarketingProfile`.

---

## [0.3.0] — 2026-02-15

### Added — Multimodal Pockets + Contextual Graph + Living Context

- **Multimodal Pockets**: Support for storing multiple named signals (text, visual, audio) under a single Entity ID. Each modality gets its own independent HNSW index with its own dimensionality.
- **Contextual Graph**: Native variable-length edge lists (`links`) in Metadata for graph relationships.
- **Living Context / Sticky Memory**: Adaptive decay based on `recall_count` — frequently accessed records resist temporal decay. Formula: `stickiness = 1 + log(1 + recall_count)`, `effective_age = age / stickiness`.
- **API**: New `db.link(from, to)` and `db.search(..., modality="visual")` methods.
- **CLI**: Updates to `feather-cli` for multimodal ingestion and linking.

---

## [0.2.1] — 2026-01-20

### Added — Context Engine + Time-Decay Scoring

- **Context Engine**: Structured metadata attributes (type, source, creation_time).
- **Time-Decay Scoring**: `Scorer` class with half-life decay formula.
- **Filtered Search**: `FilterBuilder` and pre-filtering logic in HNSW.
- **Python Bindings**: `feather_db.Metadata` and `feather_db.ContextType` objects.

---

## [0.1.0] — 2025-11-16

### Added — Initial Release

- Core C++17 HNSW implementation with Python (pybind11) and Rust (CLI) bindings.
- Binary `.feather` file format with magic number validation.
- L2 (Euclidean) distance metric with SIMD optimizations (AVX512/AVX/SSE).
- Add rate: 2,000–5,000 vectors/second. Search time: 0.5–1.5ms per query (k=10).

---

[0.5.0]: https://github.com/feather-store/feather/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/feather-store/feather/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/feather-store/feather/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/feather-store/feather/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/feather-store/feather/releases/tag/v0.1.0
