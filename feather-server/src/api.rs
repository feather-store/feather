//! The hot routes of feather-api/app/main.py, with the same requests,
//! responses, status codes and error bodies. See README.md for what differs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{Path, RawQuery, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::ffi::{self, BatchOp, Db, EngineError, Filter, HeatQuery, Kind, MetaIn, RawScoring};
use crate::manager::Manager;
use crate::packed::{self, Reader};

pub struct App {
    pub mgr: Manager,
    pub api_key: String,
    /// Bounds concurrent writes, as feather-api's write pool does.
    pub write_slots: tokio::sync::Semaphore,
}

pub type AppState = State<Arc<App>>;

// ── Errors ────────────────────────────────────────────────────────────

/// `{"detail", "error", "retryable", ...}` (feather-api/app/errors.py).
#[derive(Debug)]
pub struct ApiError {
    status: u16,
    detail: Value,
    error: String,
    extra: Vec<(&'static str, Value)>,
    headers: Vec<(header::HeaderName, String)>,
}

impl ApiError {
    pub fn new(status: u16, detail: impl Into<String>, error: &str) -> Self {
        ApiError { status, detail: Value::String(detail.into()), error: error.into(), extra: vec![], headers: vec![] }
    }
    pub fn internal(detail: impl Into<String>) -> Self {
        Self::new(500, detail, "internal")
    }
    pub fn retry_after(mut self, s: u64) -> Self {
        self.headers.push((header::RETRY_AFTER, s.to_string()));
        self
    }
    fn with(mut self, k: &'static str, v: Value) -> Self {
        self.extra.push((k, v));
        self
    }
    /// A 422 with pydantic's detail shape: a list of {type, loc, msg}.
    fn invalid(loc: &[&str], msg: impl Into<String>, kind: &str) -> Self {
        let mut l: Vec<Value> = vec![Value::from("body")];
        l.extend(loc.iter().map(|s| Value::from(*s)));
        ApiError {
            status: 422,
            detail: json!([{"type": kind, "loc": l, "msg": msg.into()}]),
            error: "invalid_request".into(),
            extra: vec![],
            headers: vec![],
        }
    }
    fn value_error(msg: impl Into<String>) -> Self {
        Self::invalid(&[], msg, "value_error")
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        match e.kind {
            Kind::Invalid => ApiError::new(400, e.msg, "invalid_request"),
            Kind::Closed => ApiError::new(503, format!("namespace reopening: {}", e.msg), "namespace_reopening")
                .retry_after(1),
            Kind::WriteFailed => ApiError::new(503, format!("write not saved: {}", e.msg), "write_failed")
                .retry_after(5),
            Kind::VersionConflict => {
                let (id, expected, actual) = e.conflict.unwrap_or_default();
                ApiError::new(409, e.msg, "version_conflict")
                    .with("id", id.into())
                    .with("expected_version", expected.into())
                    .with("actual_version", actual.into())
            }
            Kind::DuplicateKey => ApiError::new(409, e.msg, "duplicate_key"),
            Kind::IdempotencyReused => ApiError::new(
                422,
                "Idempotency-Key was already used for a different request",
                "idempotency_key_reused",
            ),
            Kind::Other => ApiError::internal(e.msg),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let retryable = matches!(self.status, 429 | 503);
        let mut body = serde_json::Map::new();
        body.insert("detail".into(), self.detail);
        body.insert("error".into(), Value::String(self.error));
        body.insert("retryable".into(), Value::Bool(retryable));
        for (k, v) in self.extra {
            body.insert(k.into(), v);
        }
        let mut r = json_response(
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            serde_json::to_vec(&Value::Object(body)).unwrap(),
        );
        for (k, v) in self.headers {
            if let Ok(v) = HeaderValue::from_str(&v) {
                r.headers_mut().insert(k, v);
            }
        }
        r
    }
}

fn json_response(status: StatusCode, body: Vec<u8>) -> Response {
    let mut r = (status, body).into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}

type ApiResult = Result<Response, ApiError>;

fn truncated(_: packed::Truncated) -> ApiError {
    ApiError::internal("engine returned a truncated record")
}

// ── Plumbing ──────────────────────────────────────────────────────────

/// Every route's first step: the API key, then a refusal of what this server
/// does not do and must not silently ignore.
fn auth(app: &App, headers: &HeaderMap, write: bool) -> Result<(), ApiError> {
    check_key(app, headers)?;
    // X-Feather-As-Of reads a past commit (feather-api's _AsOfScope). Served
    // from the live namespace it would be a wrong answer with a 200.
    if headers.contains_key("x-feather-as-of") {
        if write {
            return Err(ApiError::new(
                400,
                "a request with X-Feather-As-Of reads the past and cannot write",
                "read_only",
            ));
        }
        return Err(ApiError::new(
            501,
            "feather-server does not serve X-Feather-As-Of (time travel); the Python Cloud API does",
            "not_implemented",
        ));
    }
    Ok(())
}

fn check_key(app: &App, headers: &HeaderMap) -> Result<(), ApiError> {
    if app.api_key.is_empty() {
        return Ok(()); // dev mode, opted into explicitly at startup
    }
    let given = headers.get("x-api-key").map(|v| v.as_bytes()).unwrap_or(b"");
    let want = app.api_key.as_bytes();
    // Constant time in the key's length.
    let mut diff = (given.len() != want.len()) as u8;
    for (i, w) in want.iter().enumerate() {
        diff |= w ^ given.get(i).copied().unwrap_or(0);
    }
    if diff != 0 {
        return Err(ApiError::new(401, "Invalid or missing X-API-Key", "unauthorized"));
    }
    Ok(())
}

/// A read's engine work, on the blocking pool: an engine call can wait on
/// the namespace's lock or build an index, and must not stall an async
/// worker meanwhile. (Running it inline on the worker measured the same:
/// 0.57 vs 0.59 ms CPU per search, the same saturation rate.)
async fn read<T: Send + 'static>(f: impl FnOnce() -> Result<T, ApiError> + Send + 'static) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f).await.map_err(|e| ApiError::internal(format!("handler panicked: {e}")))?
}

/// A write's engine work: on the blocking pool, at most FEATHER_WRITE_THREADS
/// at once (a write can wait on the WAL fsync).
async fn write<T: Send + 'static>(
    app: &App, f: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    let _slot = app.write_slots.acquire().await.map_err(|_| ApiError::internal("shutting down"))?;
    tokio::task::spawn_blocking(f).await.map_err(|e| ApiError::internal(format!("handler panicked: {e}")))?
}

fn parse<'a, T: Deserialize<'a>>(body: &'a [u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| {
        let msg = e.to_string();
        // serde names an unknown field `x`; pydantic reported it at its loc.
        if let Some(rest) = msg.strip_prefix("unknown field `") {
            if let Some(name) = rest.split('`').next() {
                return ApiError::invalid(&["metadata", name], "Extra inputs are not permitted", "extra_forbidden");
            }
        }
        if e.is_syntax() || e.is_eof() {
            return ApiError::invalid(&[], format!("JSON decode error: {msg}"), "json_invalid");
        }
        ApiError::value_error(msg)
    })
}

fn bool_param(q: &HashMap<String, String>, name: &str) -> Result<bool, ApiError> {
    match q.get(name).map(|s| s.to_ascii_lowercase()) {
        None => Ok(false),
        Some(v) => match v.as_str() {
            "1" | "true" | "t" | "yes" | "y" | "on" => Ok(true),
            "0" | "false" | "f" | "no" | "n" | "off" => Ok(false),
            _ => Err(ApiError {
                status: 422,
                detail: json!([{"type": "bool_parsing", "loc": ["query", name],
                                "msg": "Input should be a valid boolean, unable to interpret input"}]),
                error: "invalid_request".into(),
                extra: vec![],
                headers: vec![],
            }),
        },
    }
}

fn query_map(raw: Option<&str>) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for pair in raw.unwrap_or("").split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        m.insert(percent_decode(k, true), percent_decode(v, true));
    }
    m
}

/// %XX-decode; `plus` also reads '+' as a space (query strings only).
fn percent_decode(s: &str, plus: bool) -> String {
    let b = s.as_bytes();
    let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 3 <= b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push(h << 4 | l);
                i += 3;
                continue;
            }
        }
        out.push(if plus && b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ── Request bodies ────────────────────────────────────────────────────

fn k10() -> i64 { 10 }
fn rrf60() -> i64 { 60 }
fn yes() -> bool { true }
fn text() -> String { "text".into() }
fn score() -> String { "score".into() }

/// The union of fastparse.py's SearchBody, HybridBody and KeywordBody (each
/// route reads its own fields; unknown fields are ignored, as msgspec does).
#[derive(Deserialize)]
struct QueryBody {
    vector: Option<Vec<f32>>,
    vector_b64: Option<String>,
    query: Option<String>,
    #[serde(default = "k10")]
    k: i64,
    #[serde(default = "rrf60")]
    rrf_k: i64,
    #[serde(default = "score")]
    fusion: String,
    #[serde(default = "text")]
    modality: String,
    #[serde(default = "yes")]
    track: bool,
    #[serde(default)]
    raw_score: bool,
    #[serde(default = "yes")]
    include_metadata: bool,
    namespace_id: Option<String>,
    namespace_prefix: Option<String>,
    types: Option<Vec<i64>>,
    entity_id: Option<String>,
    attributes_match: Option<HashMap<String, String>>,
    source: Option<String>,
    source_prefix: Option<String>,
    importance_gte: Option<f64>,
    confidence_gte: Option<f64>,
    tags_contains: Option<Vec<String>>,
    timestamp_after: Option<i64>,
    timestamp_before: Option<i64>,
    valid_after: Option<i64>,
    valid_before: Option<i64>,
    scoring_half_life: Option<f64>,
    scoring_weight: Option<f64>,
    scoring_min: Option<f64>,
    scoring_relevance_floor: Option<f64>,
}

fn decode_b64(s: &str) -> Result<Vec<f32>, ApiError> {
    // Python's b64decode(validate=True): standard alphabet, padding
    // required, trailing bits not checked.
    const ENGINE: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        base64::engine::GeneralPurposeConfig::new().with_decode_allow_trailing_bits(true),
    );
    let raw = ENGINE.decode(s).map_err(|_| ApiError::value_error("`vector_b64` is not valid base64"))?;
    if raw.is_empty() || raw.len() % 4 != 0 {
        return Err(ApiError::value_error("`vector_b64` must decode to a whole number of float32 values"));
    }
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// `check_vector` + `vector_array`: exactly one of the two, or neither when
/// the route allows it.
fn take_vector(vector: Option<Vec<f32>>, b64: Option<&str>, required: bool) -> Result<Option<Vec<f32>>, ApiError> {
    match (vector, b64) {
        (Some(_), Some(_)) => Err(ApiError::value_error("provide exactly one of `vector` or `vector_b64`")),
        (None, None) if required => {
            Err(ApiError::invalid(&["vector"], "Field required", "missing"))
        }
        (None, None) => Ok(None),
        (Some(v), None) => Ok(Some(v)),
        (None, Some(s)) => decode_b64(s).map(Some),
    }
}

fn check_k(k: i64, name: &str, max: i64) -> Result<usize, ApiError> {
    if k < 1 {
        return Err(ApiError::invalid(&[name], "Input should be greater than or equal to 1", "greater_than_equal"));
    }
    if k > max {
        return Err(ApiError::invalid(&[name], format!("Input should be less than or equal to {max}"), "less_than_equal"));
    }
    Ok(k as usize)
}

impl QueryBody {
    /// `_build_filter`: Python truthiness decides what is set — "" and []
    /// are unset, and so is a timestamp bound of 0; valid-time bounds and
    /// the `_gte`s are set whenever given.
    fn filter(&self) -> Result<Filter, ApiError> {
        let s = |o: &Option<String>| o.as_ref().filter(|v| !v.is_empty()).cloned();
        let mut types = vec![];
        for t in self.types.iter().flatten() {
            types.push(u8::try_from(*t).map_err(|_| ApiError::value_error(format!("unknown context type {t}")))?);
        }
        Ok(Filter {
            namespace_id: s(&self.namespace_id),
            namespace_prefix: s(&self.namespace_prefix),
            entity_id: s(&self.entity_id),
            source: s(&self.source),
            source_prefix: s(&self.source_prefix),
            types,
            attributes: self.attributes_match.iter().flatten().map(|(k, v)| (k.clone(), v.clone())).collect(),
            tags: self.tags_contains.clone().unwrap_or_default(),
            importance_gte: self.importance_gte.map(|v| v as f32),
            confidence_gte: self.confidence_gte.map(|v| v as f32),
            timestamp_after: self.timestamp_after.filter(|v| *v != 0),
            timestamp_before: self.timestamp_before.filter(|v| *v != 0),
            valid_after: self.valid_after,
            valid_before: self.valid_before,
        })
    }

    /// `_build_scoring`: scoring only when a half-life or a weight is given.
    fn scoring(&self) -> Option<RawScoring> {
        if self.scoring_half_life.is_none() && self.scoring_weight.is_none() {
            return None;
        }
        Some(RawScoring {
            half_life: self.scoring_half_life.unwrap_or(30.0) as f32,
            weight: self.scoring_weight.unwrap_or(0.3) as f32,
            min: self.scoring_min.unwrap_or(0.0) as f32,
            relevance_floor: self.scoring_relevance_floor.unwrap_or(-1.0) as f32,
        })
    }
}

/// `_check_query_dim`: the dim of an established index, and no zero vector.
fn check_query_dim(db: &Db, q: &[f32], modality: &str) -> Result<(), ApiError> {
    if let Some(d) = db.modality_dim(modality)? {
        if d != 0 && q.len() != d {
            return Err(ApiError::new(
                400,
                format!("Query vector dim {} != index dim {d} for modality '{modality}'", q.len()),
                "invalid_request",
            ));
        }
    }
    if !q.is_empty() && q.iter().all(|x| *x == 0.0) {
        return Err(ApiError::new(
            400,
            "Query vector is all zeros: a zero-norm vector has no direction, so similarity against it is \
             undefined. This usually means an empty string was embedded.",
            "invalid_request",
        ));
    }
    Ok(())
}

// ── Routes ────────────────────────────────────────────────────────────

pub async fn health(State(app): AppState) -> Response {
    let body = json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "server": "feather-server",
        "namespaces_loaded": app.mgr.open_count(),
        "namespaces_known": app.mgr.list_namespaces().len(),
    });
    json_response(StatusCode::OK, serde_json::to_vec(&body).unwrap())
}

pub async fn list_namespaces(State(app): AppState, headers: HeaderMap) -> ApiResult {
    auth(&app, &headers, false)?;
    let body = json!({ "namespaces": app.mgr.list_namespaces() });
    Ok(json_response(StatusCode::OK, serde_json::to_vec(&body).unwrap()))
}

pub async fn search(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, false)?;
    let req: QueryBody = parse(&body)?;
    let k = check_k(req.k, "k", 1000)?;
    let q = take_vector(req.vector.clone(), req.vector_b64.as_deref(), true)?.unwrap_or_default();
    let n = app.mgr.get(&ns, false).await?;
    let filter = req.filter()?;
    let scoring = req.scoring();
    let out = read(move || {
        check_query_dim(&n.db, &q, &req.modality)?;
        let p = n.db.search(&q, k, &filter, scoring.as_ref(), &req.modality, req.track, req.raw_score,
                            req.include_metadata)?;
        packed::results_json(p.bytes()).map_err(truncated)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

pub async fn keyword_search(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, false)?;
    let req: QueryBody = parse(&body)?;
    let Some(query) = req.query.clone() else {
        return Err(ApiError::invalid(&["query"], "Field required", "missing"));
    };
    let k = check_k(req.k, "k", 1000)?;
    let n = app.mgr.get(&ns, false).await?;
    let filter = req.filter()?;
    // keyword_search builds BM25 on its first call: seconds on a large
    // namespace, which is why reads run on the blocking pool.
    let out = read(move || {
        let p = n.db.keyword_search(&query, k, &filter, req.track, req.include_metadata)?;
        packed::results_json(p.bytes()).map_err(truncated)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

pub async fn hybrid_search(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, false)?;
    let req: QueryBody = parse(&body)?;
    let Some(query) = req.query.clone() else {
        return Err(ApiError::invalid(&["query"], "Field required", "missing"));
    };
    let k = check_k(req.k, "k", 1000)?;
    let rrf_k = check_k(req.rrf_k, "rrf_k", 10_000)?;
    if req.fusion != "score" && req.fusion != "rrf" {
        return Err(ApiError::invalid(&["fusion"], "Input should be 'score' or 'rrf'", "literal_error"));
    }
    let Some(q) = take_vector(req.vector.clone(), req.vector_b64.as_deref(), false)? else {
        if query.trim().is_empty() {
            return Err(ApiError::new(422, "give `vector`, or a non-empty `query` to embed", "invalid_request"));
        }
        return Err(ApiError::new(
            501,
            "feather-server does not embed queries: send `vector` (or `vector_b64`), or use the Python \
             Cloud API, which embeds `query` with its configured provider",
            "not_implemented",
        ));
    };
    let n = app.mgr.get(&ns, false).await?;
    let filter = req.filter()?;
    let scoring = req.scoring();
    let out = read(move || {
        check_query_dim(&n.db, &q, &req.modality)?;
        let p = n.db.hybrid_search(&q, &query, k, rrf_k, &filter, scoring.as_ref(), &req.modality, &req.fusion,
                                   req.include_metadata)?;
        packed::results_json(p.bytes()).map_err(truncated)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

pub async fn get_record(
    State(app): AppState, Path((ns, id)): Path<(String, String)>, RawQuery(raw): RawQuery, headers: HeaderMap,
) -> ApiResult {
    auth(&app, &headers, false)?;
    let id: u64 = id.parse().map_err(|_| ApiError {
        status: 422,
        detail: json!([{"type": "int_parsing", "loc": ["path", "record_id"],
                        "msg": "Input should be a valid integer, unable to parse string as an integer"}]),
        error: "invalid_request".into(),
        extra: vec![],
        headers: vec![],
    })?;
    let q = query_map(raw.as_deref());
    let include_vector = bool_param(&q, "include_vector")?;
    let include_deleted = bool_param(&q, "include_deleted")?;
    let modality = q.get("modality").cloned().unwrap_or_else(text);
    let n = app.mgr.get(&ns, false).await?;
    let ns2 = ns.clone();
    let (out, version) = read(move || {
        let not_found = || ApiError::new(404, format!("Record {id} not found in namespace '{ns2}'"), "not_found");
        let p = n.db.get(id, include_deleted)?.ok_or_else(not_found)?;
        let m = packed::read_meta(&mut Reader::new(p.bytes())).map_err(truncated)?;
        if m.is_dead() && !include_deleted {
            return Err(not_found());
        }
        let mut out = Vec::with_capacity(p.bytes().len() * 2 + 64);
        record_fields(&mut out, id, &m);
        if include_vector {
            out.extend_from_slice(b",\"vector\":");
            match n.db.get_vector(id, &modality) {
                Ok(v) if !v.is_empty() => {
                    out.push(b'[');
                    for (i, x) in v.iter().enumerate() {
                        if i > 0 {
                            out.push(b',');
                        }
                        packed::json_f32(&mut out, *x);
                    }
                    out.push(b']');
                }
                _ => out.extend_from_slice(b"null"),
            }
        }
        out.push(b'}');
        Ok((out, m.version))
    })
    .await?;
    let mut r = json_response(StatusCode::OK, out);
    if version != 0 {
        r.headers_mut().insert(header::ETAG, HeaderValue::from_str(&format!("\"{version}\"")).unwrap());
    }
    Ok(r)
}

/// `{**MetadataOut, "id": id, "metadata": MetadataOut` without the closing
/// brace: GET /records/{id} and GET /by_key.
fn record_fields(out: &mut Vec<u8>, id: u64, m: &packed::Meta) {
    out.push(b'{');
    packed::meta_out_fields(out, m);
    out.extend_from_slice(b",\"id\":");
    packed::json_int(out, id);
    out.extend_from_slice(b",\"metadata\":{");
    packed::meta_out_fields(out, m);
    out.push(b'}');
}

#[derive(Deserialize)]
struct GetManyBody {
    ids: Vec<u64>,
    #[serde(default)]
    include_deleted: bool,
}

pub async fn get_many(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, false)?;
    let req: GetManyBody = parse(&body)?;
    if req.ids.len() > 10_000 {
        return Err(ApiError::invalid(&["ids"], "List should have at most 10000 items after validation", "too_long"));
    }
    let n = app.mgr.get(&ns, false).await?;
    let out = read(move || {
        let p = n.db.get_many(&req.ids, req.include_deleted)?;
        let mut r = Reader::new(p.bytes());
        let count = r.u32().map_err(truncated)? as usize;
        let mut out = Vec::with_capacity(p.bytes().len() * 2 + 32);
        out.extend_from_slice(b"{\"records\":[");
        for i in 0..count {
            if i > 0 {
                out.push(b',');
            }
            if r.u8().map_err(truncated)? == 0 {
                out.extend_from_slice(b"null");
                continue;
            }
            let m = packed::read_meta(&mut r).map_err(truncated)?;
            if m.is_dead() && !req.include_deleted {
                out.extend_from_slice(b"null");
                continue;
            }
            out.extend_from_slice(b"{\"id\":");
            packed::json_int(&mut out, req.ids[i]);
            out.extend_from_slice(b",\"metadata\":");
            packed::meta_dict(&mut out, &m);
            out.push(b'}');
        }
        out.extend_from_slice(b"]}");
        Ok(out)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

fn one() -> f64 { 1.0 }
fn related_to() -> String { "related_to".into() }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeIn {
    target_id: u64,
    #[serde(default = "related_to")]
    rel_type: String,
    #[serde(default = "one")]
    weight: f64,
}

/// models.py MetadataIn: unknown keys are refused, omitted fields default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataIn {
    timestamp: Option<i64>,
    #[serde(default = "one")]
    importance: f64,
    #[serde(default, rename = "type")]
    type_: i64,
    #[serde(default)]
    source: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    tags_json: String,
    namespace_id: Option<String>,
    #[serde(default)]
    entity_id: String,
    key: Option<String>,
    edges: Option<Vec<EdgeIn>>,
    recall_count: Option<i64>,
    last_recalled_at: Option<i64>,
    #[serde(default)]
    attributes: HashMap<String, String>,
    #[serde(default = "one")]
    confidence: f64,
    #[serde(default)]
    ttl: i64,
    valid_at: Option<i64>,
}

impl MetadataIn {
    /// `_meta_from_model`: validated, with the API's defaults applied.
    fn into_meta(self, default_ns: &str) -> Result<MetaIn, ApiError> {
        let bad = |loc: &str, msg: String, kind: &str| ApiError::invalid(&["metadata", loc], msg, kind);
        if !(0..=3).contains(&self.type_) {
            return Err(bad("type", "Input should be 0, 1, 2 or 3".into(), "enum"));
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(bad("confidence", "Input should be between 0 and 1".into(), "value_error"));
        }
        for (name, v) in [("ttl", Some(self.ttl)), ("recall_count", self.recall_count),
                          ("last_recalled_at", self.last_recalled_at)] {
            if v.is_some_and(|v| v < 0) {
                return Err(bad(name, "Input should be greater than or equal to 0".into(), "greater_than_equal"));
            }
        }
        let mut edges = vec![];
        for (i, e) in self.edges.unwrap_or_default().into_iter().enumerate() {
            if e.rel_type.chars().count() > 255 {
                return Err(ApiError::invalid(&["metadata", "edges", &i.to_string(), "rel_type"],
                                             "String should have at most 255 characters", "string_too_long"));
            }
            edges.push((e.target_id, e.rel_type, e.weight as f32));
        }
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
        Ok(MetaIn {
            timestamp: self.timestamp.unwrap_or(now),
            importance: self.importance as f32,
            type_: self.type_ as u8,
            source: self.source,
            content: self.content,
            tags_json: self.tags_json,
            namespace_id: self.namespace_id.unwrap_or_else(|| default_ns.to_string()),
            entity_id: self.entity_id,
            key: self.key.filter(|k| !k.is_empty()),
            confidence: self.confidence as f32,
            ttl: self.ttl,
            valid_at: self.valid_at.unwrap_or(0),
            recall_count: self.recall_count.map(|v| v.min(u32::MAX as i64) as u32),
            last_recalled_at: self.last_recalled_at.map(|v| v as u64),
            attributes: self.attributes.into_iter().collect(),
            edges,
        })
    }
}

#[derive(Deserialize)]
struct AddVectorBody {
    id: u64,
    vector: Option<Vec<f32>>,
    vector_b64: Option<String>,
    metadata: Option<MetadataIn>,
    #[serde(default = "text")]
    modality: String,
}

const IDEMPOTENCY_HEADER: &str = "idempotency-key";

/// writes.idempotency: the key, and a fingerprint of everything the request
/// says (sha256 of "METHOD path?query\n" + body), so a retry that reaches
/// either server is recognised.
fn idempotency(method: &Method, uri: &Uri, headers: &HeaderMap, body: &[u8]) -> Result<(String, String), ApiError> {
    let key = headers.get(IDEMPOTENCY_HEADER).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if key.is_empty() {
        return Ok((key, String::new()));
    }
    if key.len() > 255 {
        return Err(ApiError::new(400, "Idempotency-Key is longer than 255 characters", "invalid_request"));
    }
    let mut h = Sha256::new();
    h.update(format!("{} {}?{}\n", method, percent_decode(uri.path(), false), uri.query().unwrap_or("")).as_bytes());
    h.update(body);
    let fp: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok((key, fp))
}

/// The note feather-api stores with an idempotent POST /vectors:
/// `json.dumps({"id", "namespace", "modality"})`, byte for byte (", " and
/// ": " separators, non-ASCII escaped), so the WAL and the file the two
/// servers write are the same and either can replay the other's retry.
fn py_note(id: u64, ns: &str, modality: &str) -> String {
    fn py_str(out: &mut String, s: &str) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (c as u32) > 0x7f => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{:04x}", u));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    let mut out = format!("{{\"id\": {id}, \"namespace\": ");
    py_str(&mut out, ns);
    out.push_str(", \"modality\": ");
    py_str(&mut out, modality);
    out.push('}');
    out
}

pub async fn add_vector(
    State(app): AppState, Path(ns): Path<String>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes,
) -> ApiResult {
    auth(&app, &headers, true)?;
    let (key, fingerprint) = idempotency(&method, &uri, &headers, &body)?;
    let req: AddVectorBody = parse(&body)?;
    let vec = take_vector(req.vector, req.vector_b64.as_deref(), true)?.unwrap_or_default();
    let meta = req.metadata.map(|m| m.into_meta(&ns)).transpose()?;
    let n = app.mgr.get(&ns, true).await?;
    let (id, modality) = (req.id, req.modality);
    let ns2 = ns.clone();
    let (body, version, replayed) = write(&app, move || {
        if let Some(d) = n.db.modality_dim(&modality)? {
            if d != 0 && vec.len() != d {
                return Err(ApiError::new(
                    400,
                    format!("vector dim {} != index dim {d} for modality '{modality}'", vec.len()),
                    "invalid_request",
                ));
            }
        }
        let mut body = json!({"id": id, "namespace": ns2, "modality": modality});
        let note = if key.is_empty() { String::new() } else { py_note(id, &ns2, &modality) };
        let _g = n.write_lock.lock().unwrap();
        // writes.prior: a retry is answered from the stored result before
        // anything else; write_batch checks again atomically either way.
        let prior = if key.is_empty() { None } else { n.db.idempotent_result(&key, &fingerprint)? };
        let (version, replayed, stored) = match prior {
            Some((v, note)) => (v, true, note),
            None => {
                let r = n.db.put(id, &vec, meta.as_ref(), &modality, &ns2, &key, &fingerprint, &note)?;
                (r.version, r.replayed, r.note)
            }
        };
        if replayed && !stored.is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(&stored) {
                body = v;
            }
        }
        if let Value::Object(m) = &mut body {
            m.insert("version".into(), version.into());
        }
        Ok((body, version, replayed))
    })
    .await?;
    let mut r = json_response(StatusCode::CREATED, serde_json::to_vec(&body).unwrap());
    r.headers_mut().insert(header::ETAG, HeaderValue::from_str(&format!("\"{version}\"")).unwrap());
    if replayed {
        r.headers_mut().insert("idempotent-replayed", HeaderValue::from_static("true"));
    }
    Ok(r)
}

// ── Agent-memory routes ───────────────────────────────────────────────
// Everything an agent session sends besides search and get_many, measured on
// AgentMemory / Pocket / PacketBuilder / FeatherStore over RemoteDB: setup
// (admin/engine, by_key, metadata_namespaces, normalize_vectors, POST
// /v1/namespaces), every turn (batch, by_keys, newest, heat_select,
// touch_many, ids, records/{id}) and close (save). A namespace is served by
// one process (its write lock), so a namespace moved here must answer all of
// them or its clients fail with 501.

fn query_error(kind: &str, name: &str, msg: &str) -> ApiError {
    ApiError {
        status: 422,
        detail: json!([{"type": kind, "loc": ["query", name], "msg": msg}]),
        error: "invalid_request".into(),
        extra: vec![],
        headers: vec![],
    }
}

fn int_param(q: &HashMap<String, String>, name: &str) -> Result<Option<i64>, ApiError> {
    match q.get(name) {
        None => Ok(None),
        Some(v) => v.parse::<i64>().map(Some).map_err(|_| {
            query_error("int_parsing", name, "Input should be a valid integer, unable to parse string as an integer")
        }),
    }
}

/// Every value of a repeated query parameter (`?x=a&x=b`).
fn query_all(raw: Option<&str>, name: &str) -> Vec<String> {
    raw.unwrap_or("")
        .split('&')
        .filter(|p| !p.is_empty())
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(k, true) == name).then(|| percent_decode(v, true))
        })
        .collect()
}

fn ok_json(v: &Value) -> Response {
    json_response(StatusCode::OK, serde_json::to_vec(v).unwrap())
}

/// `{"<field>": [id, ...]}` from a packed [u32 n][u64 id]* list.
fn ids_json(field: &str, b: &[u8]) -> Result<Vec<u8>, ApiError> {
    let mut r = Reader::new(b);
    let n = r.u32().map_err(truncated)?;
    let mut out = Vec::with_capacity(16 + n as usize * 12);
    out.extend_from_slice(b"{\"");
    out.extend_from_slice(field.as_bytes());
    out.extend_from_slice(b"\":[");
    for i in 0..n {
        if i > 0 {
            out.push(b',');
        }
        packed::json_int(&mut out, r.u64().map_err(truncated)?);
    }
    out.extend_from_slice(b"]}");
    Ok(out)
}

fn record_id(s: &str) -> Result<u64, ApiError> {
    s.parse().map_err(|_| ApiError {
        status: 422,
        detail: json!([{"type": "int_parsing", "loc": ["path", "record_id"],
                        "msg": "Input should be a valid integer, unable to parse string as an integer"}]),
        error: "invalid_request".into(),
        extra: vec![],
        headers: vec![],
    })
}

#[derive(Deserialize)]
struct CreateNamespaceBody {
    name: String,
    dim: Option<i64>,
}

/// POST /v1/namespaces: create an empty namespace (CreateNamespaceRequest).
pub async fn create_namespace(State(app): AppState, headers: HeaderMap, body: Bytes) -> ApiResult {
    // _IdempotencyGuard: this route cannot honour a key, so one is refused
    // (a bare {"detail"} body, as the middleware sends).
    if headers.contains_key(IDEMPOTENCY_HEADER) {
        let body = json!({"detail": "POST /v1/namespaces does not support Idempotency-Key"});
        return Ok(json_response(StatusCode::BAD_REQUEST, serde_json::to_vec(&body).unwrap()));
    }
    auth(&app, &headers, true)?;
    let req: CreateNamespaceBody = parse(&body)?;
    let chars = req.name.chars().count();
    if chars < 1 {
        return Err(ApiError::invalid(&["name"], "String should have at least 1 character", "string_too_short"));
    }
    if chars > 120 {
        return Err(ApiError::invalid(&["name"], "String should have at most 120 characters", "string_too_long"));
    }
    if !req.name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-') {
        return Err(ApiError::invalid(&["name"], "String should match pattern '^[A-Za-z0-9_\\-]+$'",
                                     "string_pattern_mismatch"));
    }
    if let Some(d) = req.dim {
        if d < 1 {
            return Err(ApiError::invalid(&["dim"], "Input should be greater than or equal to 1", "greater_than_equal"));
        }
        if d > 1_048_576 {
            return Err(ApiError::invalid(&["dim"], "Input should be less than or equal to 1048576", "less_than_equal"));
        }
    }
    let n = app.mgr.get_dim(&req.name, true, req.dim.map(|d| d as usize)).await?;
    let name = req.name.clone();
    let dim = write(&app, move || {
        n.db.save()?;
        Ok(n.db.dim("text")?)
    })
    .await?;
    let body = json!({"name": name, "dim": dim, "created": true});
    Ok(json_response(StatusCode::CREATED, serde_json::to_vec(&body).unwrap()))
}

/// POST /{ns}/save: checkpoint now (what RemoteDB.close() sends).
pub async fn save(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap) -> ApiResult {
    auth(&app, &headers, true)?;
    let (a, ns2) = (app.clone(), ns.clone());
    write(&app, move || a.mgr.save(&ns2).map_err(|e| ApiError::internal(e.msg))).await?;
    Ok(ok_json(&json!({"namespace": ns, "saved": true})))
}

/// GET /{ns}/by_key: the live record holding `key` in `key_namespace`.
pub async fn by_key(
    State(app): AppState, Path(ns): Path<String>, RawQuery(raw): RawQuery, headers: HeaderMap,
) -> ApiResult {
    auth(&app, &headers, false)?;
    let q = query_map(raw.as_deref());
    let key = q.get("key").cloned().ok_or_else(|| query_error("missing", "key", "Field required"))?;
    let kns = q.get("key_namespace").cloned().unwrap_or_default();
    let n = app.mgr.get(&ns, false).await?;
    let out = read(move || {
        let not_found = || ApiError::new(404, format!("no record has key '{key}' in '{kns}'"), "not_found");
        let (id, p) = n.db.lookup_key(&key, &kns)?.ok_or_else(not_found)?;
        let m = packed::read_meta(&mut Reader::new(p.bytes())).map_err(truncated)?;
        let mut out = Vec::with_capacity(p.bytes().len() * 2 + 64);
        record_fields(&mut out, id, &m);
        out.push(b'}');
        Ok(out)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ByKeysBody {
    refs: Vec<Vec<String>>,
}

/// POST /{ns}/by_keys: ids for many (key_namespace, key) refs, null where none.
pub async fn by_keys(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, false)?;
    let req: ByKeysBody = parse(&body)?;
    if req.refs.iter().any(|r| r.len() != 2) {
        return Err(ApiError::value_error("Value error, each ref is [key_namespace, key]"));
    }
    if req.refs.len() > 10_000 {
        return Err(ApiError::new(413, "at most 10,000 keys per request", "too_large"));
    }
    let n = app.mgr.get(&ns, false).await?;
    let out = read(move || {
        let refs: Vec<(String, String)> = req.refs.into_iter().map(|mut r| {
            let k = r.pop().unwrap();
            (r.pop().unwrap(), k)
        }).collect();
        let p = n.db.ids_for_many(&refs)?;
        let mut r = Reader::new(p.bytes());
        let count = r.u32().map_err(truncated)?;
        let mut out = Vec::with_capacity(16 + count as usize * 12);
        out.extend_from_slice(b"{\"ids\":[");
        for i in 0..count {
            if i > 0 {
                out.push(b',');
            }
            let found = r.u8().map_err(truncated)? != 0;
            let id = r.u64().map_err(truncated)?;
            if found {
                packed::json_int(&mut out, id);
            } else {
                out.extend_from_slice(b"null");
            }
        }
        out.extend_from_slice(b"]}");
        Ok(out)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

/// GET /{ns}/newest: ids of the newest live records, newest first.
pub async fn newest(
    State(app): AppState, Path(ns): Path<String>, RawQuery(raw): RawQuery, headers: HeaderMap,
) -> ApiResult {
    auth(&app, &headers, false)?;
    let q = query_map(raw.as_deref());
    let mns = q.get("metadata_namespace").cloned().unwrap_or_default();
    let limit = int_param(&q, "limit")?.unwrap_or(10).clamp(0, 10_000) as usize;
    let prefix = bool_param(&q, "prefix")?;
    let before_ts = int_param(&q, "before_ts")?;
    let before_id = int_param(&q, "before_id")?;
    let by_valid = bool_param(&q, "by_valid")?;
    let before = before_ts.map(|ts| (ts, before_id.unwrap_or(0) as u64));
    let n = app.mgr.get(&ns, false).await?;
    let out = read(move || ids_json("ids", n.db.newest(&mns, limit, prefix, before, by_valid)?.bytes())).await?;
    Ok(json_response(StatusCode::OK, out))
}

/// GET /{ns}/count_live: live, unexpired records of a metadata namespace.
pub async fn count_live(
    State(app): AppState, Path(ns): Path<String>, RawQuery(raw): RawQuery, headers: HeaderMap,
) -> ApiResult {
    auth(&app, &headers, false)?;
    let q = query_map(raw.as_deref());
    let mns = q.get("metadata_namespace").cloned().unwrap_or_default();
    let prefix = bool_param(&q, "prefix")?;
    let exclude = query_all(raw.as_deref(), "exclude_source");
    let n = app.mgr.get(&ns, false).await?;
    let c = read(move || Ok(n.db.count_live(&mns, prefix, &exclude)?)).await?;
    Ok(ok_json(&json!({"count": c})))
}

/// GET /{ns}/ids: live record ids by metadata namespace, entity, attribute,
/// modality, or all of them (list_ids, in its order of precedence).
pub async fn ids(
    State(app): AppState, Path(ns): Path<String>, RawQuery(raw): RawQuery, headers: HeaderMap,
) -> ApiResult {
    auth(&app, &headers, false)?;
    let q = query_map(raw.as_deref());
    let prefix = bool_param(&q, "prefix")?;
    let (which, a, b) = if let Some(v) = q.get("metadata_namespace") {
        (1, v.clone(), String::new())
    } else if let Some(v) = q.get("entity_id") {
        (2, v.clone(), String::new())
    } else if let Some(v) = q.get("attr_key") {
        (3, v.clone(), q.get("attr_value").cloned().unwrap_or_default())
    } else if let Some(v) = q.get("modality") {
        (4, v.clone(), String::new())
    } else {
        (0, String::new(), String::new())
    };
    let n = app.mgr.get(&ns, false).await?;
    let out = read(move || ids_json("ids", n.db.ids(which, &a, &b, prefix)?.bytes())).await?;
    Ok(json_response(StatusCode::OK, out))
}

/// GET /{ns}/metadata_namespaces: every metadata namespace with live records.
pub async fn metadata_namespaces(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap) -> ApiResult {
    auth(&app, &headers, false)?;
    let n = app.mgr.get(&ns, false).await?;
    let out = read(move || {
        let p = n.db.metadata_namespaces()?;
        let mut r = Reader::new(p.bytes());
        let count = r.u32().map_err(truncated)?;
        let mut out = Vec::with_capacity(32 + count as usize * 32);
        out.extend_from_slice(b"{\"namespaces\":[");
        for i in 0..count {
            if i > 0 {
                out.push(b',');
            }
            out.extend_from_slice(b"{\"id\":");
            packed::json_str(&mut out, r.str().map_err(truncated)?);
            out.extend_from_slice(b",\"size\":");
            packed::json_int(&mut out, r.u64().map_err(truncated)?);
            out.push(b'}');
        }
        out.extend_from_slice(b"]}");
        Ok(out)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

fn usage_half_life() -> f64 { 3.0 }
fn unused_floor() -> f64 { 0.4 }
fn inherit_decay() -> f64 { 0.7 }
fn four() -> i64 { 4 }
fn pinned_attr() -> String { "_pocket_pinned".into() }
fn due_attr() -> String { "_pocket_due_at".into() }
fn origin_attr() -> String { "_origin".into() }
fn text_attr() -> String { "_lg_value".into() }
fn origins() -> Vec<String> { ["user", "agent", "tool", "external"].map(String::from).to_vec() }
fn user() -> String { "user".into() }
fn external() -> String { "external".into() }

/// models.py HeatSelectRequest.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeatSelectBody {
    namespaces: Vec<String>,
    now: f64,
    budget: f64,
    half_life_days: f64,
    min_heat: f64,
    trusted: Vec<String>,
    exclude_sources: Vec<String>,
    #[serde(default = "usage_half_life")]
    usage_half_life: f64,
    #[serde(default = "unused_floor")]
    unused_floor: f64,
    #[serde(default = "inherit_decay")]
    inherit_decay: f64,
    #[serde(default = "four")]
    chars_per_token: i64,
    #[serde(default = "pinned_attr")]
    pinned_attr: String,
    #[serde(default = "due_attr")]
    due_attr: String,
    #[serde(default = "origin_attr")]
    origin_attr: String,
    #[serde(default = "text_attr")]
    text_attr: String,
    #[serde(default = "origins")]
    origins: Vec<String>,
    #[serde(default = "user")]
    default_origin: String,
    #[serde(default = "external")]
    unknown_origin: String,
}

/// POST /{ns}/heat_select: Pocket.hot() chosen in the engine,
/// [[id, depth, heat, tokens], ...] in rank order.
pub async fn heat_select(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, false)?;
    let r: HeatSelectBody = parse(&body)?;
    if r.namespaces.len() > 64 {
        return Err(ApiError::invalid(&["namespaces"], "List should have at most 64 items after validation", "too_long"));
    }
    if !(r.budget >= 0.0) {
        return Err(ApiError::invalid(&["budget"], "Input should be greater than or equal to 0", "greater_than_equal"));
    }
    if !(r.half_life_days > 0.0) {
        return Err(ApiError::invalid(&["half_life_days"], "Input should be greater than 0", "greater_than"));
    }
    if r.chars_per_token < 1 {
        return Err(ApiError::invalid(&["chars_per_token"], "Input should be greater than or equal to 1",
                                     "greater_than_equal"));
    }
    let q = HeatQuery {
        namespaces: r.namespaces,
        now: r.now,
        budget: r.budget,
        half_life_days: r.half_life_days,
        min_heat: r.min_heat,
        usage_half_life: r.usage_half_life,
        unused_floor: r.unused_floor,
        inherit_decay: r.inherit_decay,
        chars_per_token: r.chars_per_token,
        pinned_attr: r.pinned_attr,
        due_attr: r.due_attr,
        origin_attr: r.origin_attr,
        text_attr: r.text_attr,
        default_origin: r.default_origin,
        unknown_origin: r.unknown_origin,
        trusted: r.trusted,
        exclude_sources: r.exclude_sources,
        origins: r.origins,
    };
    let n = app.mgr.get(&ns, false).await?;
    let out = read(move || {
        let p = n.db.heat_select(&q)?;
        let mut r = Reader::new(p.bytes());
        let count = r.u32().map_err(truncated)?;
        let mut out = Vec::with_capacity(16 + count as usize * 48);
        out.extend_from_slice(b"{\"picks\":[");
        for i in 0..count {
            if i > 0 {
                out.push(b',');
            }
            out.extend_from_slice(b"[\"");
            packed::json_int(&mut out, r.u64().map_err(truncated)?);
            out.extend_from_slice(b"\",");
            packed::json_int(&mut out, r.u32().map_err(truncated)?);
            out.push(b',');
            packed::json_f64(&mut out, f64::from_bits(r.u64().map_err(truncated)?));
            out.push(b',');
            packed::json_int(&mut out, r.i64().map_err(truncated)?);
            out.push(b']');
        }
        out.extend_from_slice(b"]}");
        Ok(out)
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

#[derive(Deserialize)]
struct IdsBody {
    ids: Vec<u64>,
}

/// POST /{ns}/records/touch_many: count a recall for each id.
pub async fn touch_many(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, false)?;
    let req: IdsBody = parse(&body)?;
    if req.ids.len() > 10_000 {
        return Err(ApiError::invalid(&["ids"], "List should have at most 10000 items after validation", "too_long"));
    }
    let n = app.mgr.get(&ns, false).await?;
    let count = req.ids.len();
    read(move || Ok(n.db.touch_many(&req.ids)?)).await?;
    Ok(ok_json(&json!({"touched": count})))
}

/// POST /{ns}/records/{id}/touch: one recall, without a search.
pub async fn touch(
    State(app): AppState, Path((ns, id)): Path<(String, String)>, headers: HeaderMap,
) -> ApiResult {
    auth(&app, &headers, false)?;
    let id = record_id(&id)?;
    let n = app.mgr.get(&ns, false).await?;
    read(move || Ok(n.db.touch_many(&[id])?)).await?;
    Ok(ok_json(&json!({"id": id, "touched": true})))
}

/// POST /{ns}/normalize_vectors: unit-length vectors, once, then a save.
pub async fn normalize_vectors(
    State(app): AppState, Path(ns): Path<String>, RawQuery(raw): RawQuery, headers: HeaderMap,
) -> ApiResult {
    auth(&app, &headers, true)?;
    let modality = query_map(raw.as_deref()).get("modality").cloned().unwrap_or_else(text);
    let n = app.mgr.get(&ns, false).await?;
    let c = write(&app, move || {
        let c = n.db.normalize_vectors(&modality)?;
        if c > 0 {
            n.db.save()?;
        }
        Ok(c)
    })
    .await?;
    Ok(ok_json(&json!({"normalized": c})))
}

/// GET /{ns}/admin/engine: what embedded Feather reports about itself
/// (RemoteDB.dim(), modality_names(), storage_stats(), ...).
pub async fn engine_state(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap) -> ApiResult {
    auth(&app, &headers, false)?;
    let n = app.mgr.get(&ns, false).await?;
    let v = read(move || {
        let p = n.db.engine_state()?;
        engine_json(p.bytes()).map_err(truncated)
    })
    .await?;
    Ok(ok_json(&v))
}

fn engine_json(b: &[u8]) -> Result<Value, packed::Truncated> {
    let mut r = Reader::new(b);
    let f32v = |x: f32| -> Value { serde_json::Number::from_f64(x as f64).map(Value::Number).unwrap_or(Value::Null) };
    let (size, dim, auto_compact, compacting) = (r.u64()?, r.u64()?, r.f32()?, r.u8()? != 0);
    let ann = json!({"graphs": r.u64()?, "nodes": r.u64()?, "pending": r.u64()?, "min_records": r.u64()?});
    let storage = json!({
        "format": r.u32()?, "generation": r.u64()?, "file_bytes": r.u64()?,
        "live_bytes_estimate": r.u64()?, "last_commit_bytes": r.u64()?, "last_commit_full": r.u8()? != 0,
        "commits": r.u64()?, "vacuums": r.u64()?, "mapped_bytes": r.u64()?,
        "history_files": r.u64()?, "history_bytes": r.u64()?,
    });
    let (wal_size, wal_fsync_count) = (r.u64()?, r.u64()?);
    let mut modalities = serde_json::Map::new();
    for _ in 0..r.u32()? {
        let name = String::from_utf8_lossy(r.str()?).into_owned();
        let (mdim, ef, quantized, int8_ram) = (r.u64()?, r.u64()?, r.u8()? != 0, r.u8()? != 0);
        let int8 = json!({
            "int8": r.u8()? != 0, "training": r.u8()? != 0, "clip": f32v(r.f32()?), "refine": r.u32()?,
            "originals_on_disk": r.u64()?, "originals_in_wal": r.u64()?, "originals_in_ram": r.u64()?,
            "train_min": r.u32()?,
        });
        let bg = f32v(r.f32()?);
        modalities.insert(name, json!({"dim": mdim, "ef": ef, "quantized": quantized, "int8_ram": int8_ram,
                                       "int8": int8, "background_cosine": bg}));
    }
    Ok(json!({
        "size": size, "dim": dim, "modalities": modalities, "auto_compact": f32v(auto_compact),
        "compacting": compacting, "namespace_ann": ann, "storage": storage,
        "wal_size": wal_size, "wal_fsync_count": wal_fsync_count,
    }))
}

/// models.py BatchOpIn.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchOpIn {
    op: String,
    id: Option<u64>,
    key: Option<String>,
    #[serde(default)]
    key_namespace: String,
    to_key: Option<String>,
    #[serde(default)]
    to_key_namespace: String,
    #[serde(default = "yes")]
    require_target: bool,
    vector: Option<Vec<f32>>,
    #[serde(default = "text")]
    modality: String,
    metadata: Option<MetadataIn>,
    to_id: Option<u64>,
    #[serde(default = "related_to")]
    rel_type: String,
    #[serde(default = "one")]
    weight: f64,
    expected_version: Option<i64>,
}

#[derive(Deserialize)]
struct BatchBody {
    ops: Vec<BatchOpIn>,
}

/// POST /{ns}/batch: several records written atomically (write_batch).
pub async fn batch(
    State(app): AppState, Path(ns): Path<String>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes,
) -> ApiResult {
    auth(&app, &headers, true)?;
    let (key, fingerprint) = idempotency(&method, &uri, &headers, &body)?;
    let req: BatchBody = parse(&body)?;
    // Everything pydantic checks, for every op, before any of the route's own.
    if req.ops.is_empty() {
        return Err(ApiError::invalid(&["ops"], "List should have at least 1 item after validation", "too_short"));
    }
    if req.ops.len() > 10_000 {
        return Err(ApiError::invalid(&["ops"], "List should have at most 10000 items after validation", "too_long"));
    }
    struct Checked {
        o: BatchOpIn,
        kind: u8,
        meta: Option<MetaIn>,
        keep: u32,
    }
    let mut ops = Vec::with_capacity(req.ops.len());
    for (i, mut o) in req.ops.into_iter().enumerate() {
        let at = i.to_string();
        let kind = match o.op.as_str() {
            "put" => ffi::OP_PUT,
            "update" => ffi::OP_UPDATE,
            "link" => ffi::OP_LINK,
            "forget" => ffi::OP_FORGET,
            "upsert" => ffi::OP_UPSERT,
            _ => {
                return Err(ApiError::invalid(&["ops", &at, "op"],
                    "Input should be 'put', 'update', 'link', 'forget' or 'upsert'", "literal_error"))
            }
        };
        if o.rel_type.chars().count() > 255 {
            return Err(ApiError::invalid(&["ops", &at, "rel_type"], "String should have at most 255 characters",
                                         "string_too_long"));
        }
        if o.expected_version.is_some_and(|v| v < 0) {
            return Err(ApiError::invalid(&["ops", &at, "expected_version"],
                                         "Input should be greater than or equal to 0", "greater_than_equal"));
        }
        // _keep_identity: what an update leaves out, the live record keeps.
        let keep = match &o.metadata {
            Some(m) => (if m.edges.is_none() { ffi::FB_KEEP_EDGES } else { 0 })
                | (if m.key.is_none() { ffi::FB_KEEP_KEY } else { 0 })
                | (if m.valid_at.is_none() { ffi::FB_KEEP_VALID_AT } else { 0 }),
            None => 0,
        };
        let meta = o.metadata.take().map(|m| m.into_meta(&ns)).transpose()?;
        ops.push(Checked { o, kind, meta, keep });
    }
    let n = app.mgr.get(&ns, true).await?;
    let ns2 = ns.clone();
    let (out, replayed) = write(&app, move || {
        let mut batch = Vec::with_capacity(ops.len());
        for (i, Checked { o, kind, meta, keep }) in ops.into_iter().enumerate() {
            let bad = |m: &str| ApiError::new(400, format!("ops[{i}]: {m}"), "invalid_request");
            let key_ref = o.key.clone().map(|k| (o.key_namespace.clone(), k));
            if kind != ffi::OP_PUT && kind != ffi::OP_UPSERT && key_ref.is_none() && o.id.is_none() {
                return Err(bad("give id or key"));
            }
            // `if o.vector`: an empty list is no vector.
            let vec = o.vector.filter(|v| !v.is_empty());
            let mut op = BatchOp {
                kind,
                flags: if o.expected_version.is_some() { ffi::FB_EXPECTED } else { 0 },
                id: o.id.unwrap_or(0),
                key_ref: None,
                to_key_ref: None,
                expected: o.expected_version.map(|v| v as u64),
                meta: None,
                vec: None,
                modality: o.modality,
                to_id: 0,
                rel_type: o.rel_type,
                weight: o.weight as f32,
            };
            match kind {
                ffi::OP_PUT => {
                    if vec.is_none() || o.id.is_none() {
                        return Err(bad("put needs id and vector"));
                    }
                    op.meta = meta;
                    op.vec = vec;
                }
                ffi::OP_UPSERT => {
                    if meta.as_ref().and_then(|m| m.key.as_ref()).is_none() {
                        return Err(bad("upsert needs metadata.key"));
                    }
                    op.id = 0;
                    op.meta = meta;
                    op.vec = vec;
                }
                ffi::OP_UPDATE => {
                    if meta.is_none() {
                        return Err(bad("update needs metadata"));
                    }
                    op.meta = meta;
                    op.flags |= keep;
                }
                ffi::OP_LINK => {
                    let to_ref = o.to_key.clone().map(|k| (o.to_key_namespace.clone(), k));
                    if o.to_id.is_none() && to_ref.is_none() {
                        return Err(bad("link needs to_id or to_key"));
                    }
                    op.to_id = o.to_id.unwrap_or(0);
                    if to_ref.is_some() {
                        op.flags |= ffi::FB_TO_REF;
                    }
                    op.to_key_ref = to_ref;
                    if o.require_target {
                        op.flags |= ffi::FB_REQUIRE_TARGET;
                    }
                }
                _ => {} // forget
            }
            if kind != ffi::OP_PUT && kind != ffi::OP_UPSERT && key_ref.is_some() {
                op.flags |= ffi::FB_REF;
                op.key_ref = key_ref;
            }
            batch.push(op);
        }
        let note = if key.is_empty() { "" } else { "{}" };
        let _g = n.write_lock.lock().unwrap();
        let p = n.db.write_batch(&batch, &ns2, &key, &fingerprint, note)?;
        let mut r = Reader::new(p.bytes());
        let replayed = r.u8().map_err(truncated)? != 0;
        r.str().map_err(truncated)?; // the stored note: "{}", which adds nothing to the body
        let count = r.u32().map_err(truncated)?;
        let mut out = Vec::with_capacity(16 + count as usize * 40);
        out.extend_from_slice(b"{\"results\":[");
        for i in 0..count {
            if i > 0 {
                out.push(b',');
            }
            out.extend_from_slice(b"{\"id\":");
            packed::json_int(&mut out, r.u64().map_err(truncated)?);
            out.extend_from_slice(b",\"version\":");
            packed::json_int(&mut out, r.u64().map_err(truncated)?);
            out.push(b'}');
        }
        out.extend_from_slice(b"]}");
        Ok((out, replayed))
    })
    .await?;
    let mut r = json_response(StatusCode::OK, out);
    if replayed {
        r.headers_mut().insert("idempotent-replayed", HeaderValue::from_static("true"));
    }
    Ok(r)
}

#[derive(Deserialize)]
struct LinkBody {
    to_id: u64,
    #[serde(default = "related_to")]
    rel_type: String,
    #[serde(default = "one")]
    weight: f64,
}

/// POST /{ns}/records/{id}/link: one edge, as embedded link(): the source
/// must be live (404), the target may be a forward reference.
/// (AgentMemory.observations.note links an observation to its evidence.)
pub async fn link(
    State(app): AppState, Path((ns, id)): Path<(String, String)>, method: Method, uri: Uri, headers: HeaderMap,
    body: Bytes,
) -> ApiResult {
    auth(&app, &headers, true)?;
    let (key, fingerprint) = idempotency(&method, &uri, &headers, &body)?;
    let from = record_id(&id)?;
    let req: LinkBody = parse(&body)?;
    if req.rel_type.chars().count() > 255 {
        return Err(ApiError::invalid(&["rel_type"], "String should have at most 255 characters", "string_too_long"));
    }
    let n = app.mgr.get(&ns, false).await?;
    let (body, version, replayed) = write(&app, move || {
        let mut body = json!({"from_id": from, "to_id": req.to_id, "linked": true});
        let _g = n.write_lock.lock().unwrap();
        // writes.prior first: a retry after the source was forgotten still
        // gets the first answer, not a 404.
        let prior = if key.is_empty() { None } else { n.db.idempotent_result(&key, &fingerprint)? };
        let (version, replayed, note) = match prior {
            Some((v, note)) => (v, true, note),
            None => {
                let live = match n.db.get(from, false)? {
                    Some(p) => !packed::read_meta(&mut Reader::new(p.bytes())).map_err(truncated)?.is_dead(),
                    None => false,
                };
                if !live {
                    return Err(ApiError::new(404, format!("Record {from} not found"), "not_found"));
                }
                // json.dumps(body), as feather-api stores it.
                let note = if key.is_empty() {
                    String::new()
                } else {
                    format!("{{\"from_id\": {from}, \"to_id\": {}, \"linked\": true}}", req.to_id)
                };
                let op = BatchOp {
                    kind: ffi::OP_LINK,
                    flags: 0, // require_target off: a forward reference is allowed
                    id: from,
                    key_ref: None,
                    to_key_ref: None,
                    expected: None,
                    meta: None,
                    vec: None,
                    modality: text(),
                    to_id: req.to_id,
                    rel_type: req.rel_type,
                    weight: req.weight as f32,
                };
                let p = n.db.write_batch(&[op], &ns, &key, &fingerprint, &note)?;
                let mut r = Reader::new(p.bytes());
                let replayed = r.u8().map_err(truncated)? != 0;
                let note = String::from_utf8_lossy(r.str().map_err(truncated)?).into_owned();
                let count = r.u32().map_err(truncated)?;
                let mut version = 0;
                for _ in 0..count {
                    r.u64().map_err(truncated)?;
                    version = r.u64().map_err(truncated)?;
                }
                (version, replayed, note)
            }
        };
        if replayed && !note.is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(&note) {
                body = v;
            }
        }
        if let Value::Object(m) = &mut body {
            m.insert("version".into(), version.into());
        }
        Ok((body, version, replayed))
    })
    .await?;
    let mut r = ok_json(&body);
    r.headers_mut().insert(header::ETAG, HeaderValue::from_str(&format!("\"{version}\"")).unwrap());
    if replayed {
        r.headers_mut().insert("idempotent-replayed", HeaderValue::from_static("true"));
    }
    Ok(r)
}

// ── Corrections and erasure ───────────────────────────────────────────
// What agents send outside a session's turns: Pocket.pin and keys.write
// (PUT), Pocket.forget / PocketSession.close / maintain() (DELETE), and every
// erasure (Pocket.erase, AgentMemory.erase_subject, FeatherStore.erase, the
// MCP forget tools). A namespace moved here 501'd on all of them.

/// (replayed, stored note, last version) of a write_batch result.
fn batch_outcome(p: &ffi::Packed) -> Result<(bool, String, u64), ApiError> {
    let mut r = Reader::new(p.bytes());
    let replayed = r.u8().map_err(truncated)? != 0;
    let note = String::from_utf8_lossy(r.str().map_err(truncated)?).into_owned();
    let mut version = 0;
    for _ in 0..r.u32().map_err(truncated)? {
        r.u64().map_err(truncated)?;
        version = r.u64().map_err(truncated)?;
    }
    Ok((replayed, note, version))
}

/// The body a replay answers with: the stored note, else the fresh body.
fn replay_body(fresh: Value, replayed: bool, note: &str) -> Value {
    if replayed && !note.is_empty() {
        if let Ok(v) = serde_json::from_str::<Value>(note) {
            return v;
        }
    }
    fresh
}

/// Python's repr() of a str, for messages that quote one.
fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// writes.if_match_version: `42`, `"42"` or `W/"42"`; `*` or nothing = no
/// condition.
fn if_match(headers: &HeaderMap) -> Result<Option<u64>, ApiError> {
    let Some(raw) = headers.get(header::IF_MATCH) else { return Ok(None) };
    let value: String = raw.as_bytes().iter().map(|&b| b as char).collect(); // latin-1, as Starlette
    let v = value.trim();
    if v.is_empty() || v == "*" {
        return Ok(None);
    }
    let v = v.strip_prefix("W/").unwrap_or(v).trim_matches('"');
    let bad = || ApiError::new(400, format!("If-Match must be a record version, got {}", py_repr(&value)),
                               "invalid_request");
    if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    v.parse::<u64>().map(Some).map_err(|_| bad())
}

/// writes.precondition_failed: a version conflict under If-Match is a 412
/// carrying the record's current version as its ETag.
fn precondition_failed(e: EngineError) -> ApiError {
    if e.kind != Kind::VersionConflict {
        return e.into();
    }
    let (id, expected, actual) = e.conflict.unwrap_or_default();
    let mut err = ApiError::new(412, e.msg, "precondition_failed")
        .with("id", id.into())
        .with("expected_version", expected.into())
        .with("actual_version", actual.into());
    err.headers.push((header::ETAG, format!("\"{actual}\"")));
    err
}

/// A record read back as the metadata that writes it again: every field,
/// the salience counters included, as get_metadata() hands WriteBatch.update.
fn meta_in_of(m: &packed::Meta) -> MetaIn {
    let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    MetaIn {
        timestamp: m.timestamp,
        importance: m.importance,
        type_: m.type_,
        source: s(m.source),
        content: s(m.content),
        tags_json: s(m.tags_json),
        namespace_id: s(m.namespace_id),
        entity_id: s(m.entity_id),
        key: (!m.key.is_empty()).then(|| s(m.key)),
        confidence: m.confidence,
        ttl: m.ttl,
        valid_at: m.valid_at,
        recall_count: Some(m.recall_count),
        last_recalled_at: Some(m.last_recalled_at),
        attributes: m.attributes.iter().map(|(k, v)| (s(k), s(v))).collect(),
        edges: m.edges.iter().map(|(t, r, w)| (*t, s(r), *w)).collect(),
    }
}

fn update_op(id: u64, meta: MetaIn, flags: u32, expected: Option<u64>) -> BatchOp {
    BatchOp {
        kind: ffi::OP_UPDATE,
        flags: flags | if expected.is_some() { ffi::FB_EXPECTED } else { 0 },
        id,
        key_ref: None,
        to_key_ref: None,
        expected,
        meta: Some(meta),
        vec: None,
        modality: text(),
        to_id: 0,
        rel_type: related_to(),
        weight: 1.0,
    }
}

#[derive(Deserialize)]
struct UpdateMetadataBody {
    metadata: MetadataIn,
}

/// PUT /{ns}/records/{id}: replace a record's metadata (update_metadata).
/// Edges, key and valid_at the request leaves out, the live record keeps.
/// `If-Match: <version>` makes it conditional (412 on a change).
pub async fn update_record(
    State(app): AppState, Path((ns, id)): Path<(String, String)>, method: Method, uri: Uri, headers: HeaderMap,
    body: Bytes,
) -> ApiResult {
    auth(&app, &headers, true)?;
    let (key, fingerprint) = idempotency(&method, &uri, &headers, &body)?;
    let id = record_id(&id)?;
    let req: UpdateMetadataBody = parse(&body)?;
    let m = &req.metadata;
    let keep = (if m.edges.is_none() { ffi::FB_KEEP_EDGES } else { 0 })
        | (if m.key.is_none() { ffi::FB_KEEP_KEY } else { 0 })
        | (if m.valid_at.is_none() { ffi::FB_KEEP_VALID_AT } else { 0 });
    let meta = req.metadata.into_meta(&ns)?;
    let n = app.mgr.get(&ns, false).await?;
    let expected = if_match(&headers)?;
    let (body, version, replayed) = write(&app, move || {
        let fresh = json!({"id": id, "updated": true});
        let note = if key.is_empty() { String::new() } else { format!("{{\"id\": {id}, \"updated\": true}}") };
        let _g = n.write_lock.lock().unwrap();
        let prior = if key.is_empty() { None } else { n.db.idempotent_result(&key, &fingerprint)? };
        let (replayed, stored, version) = match prior {
            Some((v, note)) => (true, note, v),
            None => {
                let p = n.db.write_batch(&[update_op(id, meta, keep, expected)], &ns, &key, &fingerprint, &note)
                    .map_err(precondition_failed)?;
                batch_outcome(&p)?
            }
        };
        let mut body = replay_body(fresh, replayed, &stored);
        if let Value::Object(m) = &mut body {
            m.insert("version".into(), version.into());
        }
        Ok((body, version, replayed))
    })
    .await?;
    let mut r = ok_json(&body);
    r.headers_mut().insert(header::ETAG, HeaderValue::from_str(&format!("\"{version}\"")).unwrap());
    if replayed {
        r.headers_mut().insert("idempotent-replayed", HeaderValue::from_static("true"));
    }
    Ok(r)
}

/// DELETE /{ns}/records/{id}: forget a record. `cascade` (default true)
/// also removes every edge pointing at it, in the same atomic write
/// (_prune_ops); RemoteDB.forget sends cascade=false, as embedded forget().
pub async fn delete_record(
    State(app): AppState, Path((ns, id)): Path<(String, String)>, RawQuery(raw): RawQuery, method: Method,
    uri: Uri, headers: HeaderMap, body: Bytes,
) -> ApiResult {
    auth(&app, &headers, true)?;
    let (key, fingerprint) = idempotency(&method, &uri, &headers, &body)?;
    let id = record_id(&id)?;
    let q = query_map(raw.as_deref());
    let cascade = if q.contains_key("cascade") { bool_param(&q, "cascade")? } else { true };
    let n = app.mgr.get(&ns, false).await?;
    let (body, replayed) = write(&app, move || {
        let _g = n.write_lock.lock().unwrap();
        // writes.prior before the existence check: a retry of a delete that
        // succeeded finds no record, and must get the first answer.
        if !key.is_empty() {
            if let Some((_, note)) = n.db.idempotent_result(&key, &fingerprint)? {
                return Ok((replay_body(json!({}), true, &note), true));
            }
        }
        let live = match n.db.get(id, false)? {
            Some(p) => !packed::read_meta(&mut Reader::new(p.bytes())).map_err(truncated)?.is_dead(),
            None => false,
        };
        if !live {
            return Err(ApiError::new(404, format!("Record {id} not found"), "not_found"));
        }
        let mut ops = vec![BatchOp {
            kind: ffi::OP_FORGET,
            flags: 0,
            id,
            key_ref: None,
            to_key_ref: None,
            expected: None,
            meta: None,
            vec: None,
            modality: text(),
            to_id: 0,
            rel_type: related_to(),
            weight: 1.0,
        }];
        let mut pruned = 0u64;
        if cascade {
            let p = n.db.incoming(id)?;
            let mut r = Reader::new(p.bytes());
            let mut sources = std::collections::BTreeSet::new();
            for _ in 0..r.u32().map_err(truncated)? {
                let src = r.u64().map_err(truncated)?;
                r.str().map_err(truncated)?;
                r.f32().map_err(truncated)?;
                if src != id {
                    sources.insert(src);
                }
            }
            for src in sources {
                let Some(p) = n.db.get(src, false)? else { continue };
                let m = packed::read_meta(&mut Reader::new(p.bytes())).map_err(truncated)?;
                let kept = m.edges.iter().filter(|e| e.0 != id).count();
                if kept != m.edges.len() {
                    pruned += (m.edges.len() - kept) as u64;
                    let mut meta = meta_in_of(&m);
                    meta.edges.retain(|e| e.0 != id);
                    ops.push(update_op(src, meta, 0, None));
                }
            }
        }
        let fresh = json!({"id": id, "deleted": true, "edges_pruned": pruned});
        let note = if key.is_empty() {
            String::new()
        } else {
            format!("{{\"id\": {id}, \"deleted\": true, \"edges_pruned\": {pruned}}}")
        };
        let p = n.db.write_batch(&ops, &ns, &key, &fingerprint, &note)?;
        let (replayed, stored, _) = batch_outcome(&p)?;
        Ok((replay_body(fresh, replayed, &stored), replayed))
    })
    .await?;
    let mut r = ok_json(&body);
    if replayed {
        r.headers_mut().insert("idempotent-replayed", HeaderValue::from_static("true"));
    }
    Ok(r)
}

/// models.py EraseRequest.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EraseBody {
    #[serde(default)]
    ids: Vec<u64>,
    #[serde(default)]
    namespace_id: String,
    #[serde(default)]
    prefix: bool,
    #[serde(default)]
    entity_id: String,
    cites: Option<Vec<String>>,
    feeds: Option<Vec<String>>,
    #[serde(default)]
    dry_run: bool,
}

/// POST /{ns}/erase: erase records so that nothing of them remains (those
/// selected, and every record derived from or citing them); the receipt.
/// The Python server also drops its as_of handles and embedding cache;
/// this server keeps neither.
pub async fn erase(State(app): AppState, Path(ns): Path<String>, headers: HeaderMap, body: Bytes) -> ApiResult {
    auth(&app, &headers, true)?;
    let req: EraseBody = parse(&body)?;
    if req.ids.is_empty() && req.namespace_id.is_empty() && req.entity_id.is_empty() {
        return Err(ApiError::new(422, "erase needs ids, a namespace_id or an entity_id", "invalid_request"));
    }
    let n = app.mgr.get(&ns, false).await?;
    let args = ffi::EraseArgs {
        ids: req.ids,
        namespace_id: req.namespace_id,
        prefix: req.prefix,
        entity_id: req.entity_id,
        cites: req.cites,
        feeds: req.feeds,
        dry_run: req.dry_run,
    };
    let out = write(&app, move || {
        let _g = n.write_lock.lock().unwrap();
        Ok(n.db.erase(&args)?.bytes().to_vec())
    })
    .await?;
    Ok(json_response(StatusCode::OK, out))
}

/// Any other /v1 route exists on the Python server only.
pub async fn not_served(method: Method, uri: Uri) -> ApiError {
    if uri.path().starts_with("/v1/") {
        return ApiError::new(
            501,
            format!("feather-server does not serve {method} {}; the Python Cloud API (feather-api) does", uri.path()),
            "not_implemented",
        );
    }
    ApiError::new(404, "Not Found", "not_found")
}

pub async fn method_not_allowed() -> ApiError {
    ApiError::new(405, "Method Not Allowed", "method_not_allowed")
}
