# feather-db-cli

Command-line interface for **[Feather](https://github.com/feather-store/feather)** —
an embedded, single-file vector database + living-context engine (C++17 core with
Python and Rust bindings). Sub-millisecond ANN search via HNSW, a typed context
graph, adaptive decay, and `.feather` single-file persistence (format v9 with a
persisted HNSW graph for fast cold load).

The CLI wraps the C ABI of the Feather core (vendored under `cpp/`, built via
`build.rs`), so it is self-contained — no system Feather install required.

## Install

```bash
cargo install feather-db-cli
```

## Usage

```bash
feather add    --db my.feather --id 1 --vec "0.1,0.2,0.3" --modality text
feather search --db my.feather --vec "0.1,0.2,0.3" --k 5
feather link   --db my.feather --from 1 --to 2
feather save   --db my.feather
```

## Scope

The CLI exposes the core vector + graph operations (`add`, `search`, `link`,
`save`). The richer context-engine surface — namespaces, attributes,
`context_chain`, hybrid retrieval — currently lives in the Python package
(`pip install feather-db`). See the
[main README](https://github.com/feather-store/feather) for the full feature set.

Licensed under MIT.
