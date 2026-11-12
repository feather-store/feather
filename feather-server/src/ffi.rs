//! The engine through its C ABI (src/feather_core.cpp), wrapped so nothing
//! unsafe leaks out of this file. Every engine call is synchronous and reads
//! its error state from thread-locals on the calling thread, so an error is
//! collected right after the call, on the same thread.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FStr {
    p: *const c_char,
    n: usize,
}

impl FStr {
    pub const NONE: FStr = FStr { p: ptr::null(), n: 0 };
    pub fn of(s: &str) -> FStr {
        // A non-null pointer even for "", which C reads as "given, empty".
        FStr { p: if s.is_empty() { b"".as_ptr() as *const c_char } else { s.as_ptr() as *const c_char }, n: s.len() }
    }
    pub fn opt(s: Option<&str>) -> FStr {
        s.map(FStr::of).unwrap_or(FStr::NONE)
    }
}

#[repr(C)]
pub struct RawFilter {
    namespace_id: FStr,
    namespace_prefix: FStr,
    entity_id: FStr,
    source: FStr,
    source_prefix: FStr,
    types: *const u8,
    n_types: usize,
    attr_keys: *const FStr,
    attr_vals: *const FStr,
    n_attrs: usize,
    tags: *const FStr,
    n_tags: usize,
    has: u32,
    importance_gte: f32,
    confidence_gte: f32,
    timestamp_after: i64,
    timestamp_before: i64,
    valid_after: i64,
    valid_before: i64,
}

#[repr(C)]
pub struct RawScoring {
    pub half_life: f32,
    pub weight: f32,
    pub min: f32,
    pub relevance_floor: f32,
}

#[repr(C)]
struct RawMetaIn {
    timestamp: i64,
    importance: f32,
    type_: u8,
    source: FStr,
    content: FStr,
    tags_json: FStr,
    namespace_id: FStr,
    entity_id: FStr,
    key: FStr,
    confidence: f32,
    ttl: i64,
    valid_at: i64,
    has: u32,
    recall_count: u32,
    last_recalled_at: u64,
    attr_keys: *const FStr,
    attr_vals: *const FStr,
    n_attrs: usize,
    edge_targets: *const u64,
    edge_types: *const FStr,
    edge_weights: *const f32,
    n_edges: usize,
}

#[repr(C)]
struct RawHeatQuery {
    namespaces: *const FStr,
    n_namespaces: usize,
    now: f64,
    budget: f64,
    half_life_days: f64,
    min_heat: f64,
    usage_half_life: f64,
    unused_floor: f64,
    inherit_decay: f64,
    chars_per_token: i64,
    pinned_attr: FStr,
    due_attr: FStr,
    origin_attr: FStr,
    text_attr: FStr,
    default_origin: FStr,
    unknown_origin: FStr,
    trusted: *const FStr,
    n_trusted: usize,
    exclude_sources: *const FStr,
    n_exclude: usize,
    origins: *const FStr,
    n_origins: usize,
}

#[repr(C)]
struct RawBatchOp {
    kind: u8,
    flags: u32,
    id: u64,
    ref_ns: FStr,
    ref_key: FStr,
    to_ref_ns: FStr,
    to_ref_key: FStr,
    expected: u64,
    meta: *const RawMetaIn,
    vec: *const f32,
    vec_len: usize,
    modality: FStr,
    to_id: u64,
    rel_type: FStr,
    weight: f32,
}

extern "C" {
    fn feather_last_error() -> *const c_char;
    fn feather_last_error_kind() -> c_int;
    fn feather_last_conflict(id: *mut u64, expected: *mut u64, actual: *mut u64);
    fn feather_open(path: *const c_char, dim: usize) -> *mut c_void;
    fn feather_close_ex(db: *mut c_void, save: c_int) -> c_int;
    fn feather_save(db: *mut c_void) -> c_int;
    fn feather_free_buf(p: *mut u8);
    fn feather_search_bin(
        db: *mut c_void, q: *const f32, len: usize, k: usize, filter: *const RawFilter,
        scoring: *const RawScoring, modality: FStr, record_salience: c_int, with_cosine: c_int,
        include_metadata: c_int, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_keyword_search_bin(
        db: *mut c_void, query: FStr, k: usize, filter: *const RawFilter, record_salience: c_int,
        include_metadata: c_int, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_hybrid_search_bin(
        db: *mut c_void, q: *const f32, len: usize, query: FStr, k: usize, rrf_k: usize,
        filter: *const RawFilter, scoring: *const RawScoring, modality: FStr, fusion: FStr,
        include_metadata: c_int, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_get_bin(db: *mut c_void, id: u64, include_dead: c_int, out: *mut *mut u8, out_len: *mut usize) -> c_int;
    fn feather_get_many_bin(
        db: *mut c_void, ids: *const u64, n: usize, include_dead: c_int, out: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int;
    fn feather_get_vector(db: *mut c_void, id: u64, modality: FStr, out: *mut *mut f32, n: *mut usize) -> c_int;
    fn feather_modality_dim(db: *mut c_void, modality: FStr, out_dim: *mut usize) -> c_int;
    fn feather_write_error(db: *mut c_void) -> *mut c_char;
    fn feather_wal_size(db: *mut c_void) -> u64;
    fn feather_idempotent_result(
        db: *mut c_void, key: FStr, fingerprint: FStr, out_version: *mut u64, out_note: *mut *mut c_char,
        out_note_len: *mut usize,
    ) -> c_int;
    fn feather_dim(db: *mut c_void, modality: FStr, out: *mut usize) -> c_int;
    fn feather_lookup_key_bin(
        db: *mut c_void, key: FStr, ns: FStr, out_id: *mut u64, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_ids_for_many_bin(
        db: *mut c_void, nss: *const FStr, keys: *const FStr, n: usize, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_newest_bin(
        db: *mut c_void, ns: FStr, limit: usize, prefix: c_int, has_before: c_int, before_ts: i64,
        before_id: u64, by_valid: c_int, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_ids_bin(
        db: *mut c_void, which: c_int, a: FStr, b: FStr, prefix: c_int, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_metadata_namespaces_bin(db: *mut c_void, out: *mut *mut u8, out_len: *mut usize) -> c_int;
    fn feather_count_live(
        db: *mut c_void, ns: FStr, prefix: c_int, exclude: *const FStr, n_exclude: usize, out: *mut u64,
    ) -> c_int;
    fn feather_heat_select_bin(db: *mut c_void, q: *const RawHeatQuery, out: *mut *mut u8, out_len: *mut usize)
        -> c_int;
    fn feather_touch_many(db: *mut c_void, ids: *const u64, n: usize) -> c_int;
    fn feather_normalize_vectors(db: *mut c_void, modality: FStr, out: *mut u64) -> c_int;
    fn feather_engine_state_bin(db: *mut c_void, out: *mut *mut u8, out_len: *mut usize) -> c_int;
    fn feather_write_batch_bin(
        db: *mut c_void, ops: *const RawBatchOp, n: usize, default_ns: FStr, idem_key: FStr, fingerprint: FStr,
        note: FStr, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
    fn feather_put(
        db: *mut c_void, id: u64, vec: *const f32, len: usize, meta: *const RawMetaIn, modality: FStr,
        default_ns: FStr, idem_key: FStr, fingerprint: FStr, note: FStr, out_version: *mut u64,
        out_replayed: *mut c_int, out_note: *mut *mut c_char, out_note_len: *mut usize,
    ) -> c_int;
    fn feather_incoming_bin(db: *mut c_void, id: u64, out: *mut *mut u8, out_len: *mut usize) -> c_int;
    fn feather_erase_json(
        db: *mut c_void, ns: FStr, prefix: c_int, entity: FStr, ids: *const u64, n_ids: usize,
        has_cites: c_int, cites: *const FStr, n_cites: usize, has_feeds: c_int, feeds: *const FStr,
        n_feeds: usize, dry_run: c_int, out: *mut *mut u8, out_len: *mut usize,
    ) -> c_int;
}

/// DB.erase's arguments (feather-api EraseRequest). `cites` / `feeds`:
/// None keeps the engine's default edge kinds.
pub struct EraseArgs {
    pub ids: Vec<u64>,
    pub namespace_id: String,
    pub prefix: bool,
    pub entity_id: String,
    pub cites: Option<Vec<String>>,
    pub feeds: Option<Vec<String>>,
    pub dry_run: bool,
}

/// What kind of failure an engine call was (feather_core.cpp `ErrorKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Other,
    Invalid,
    Closed,
    WriteFailed,
    VersionConflict,
    DuplicateKey,
    IdempotencyReused,
}

#[derive(Debug, Clone)]
pub struct EngineError {
    pub kind: Kind,
    pub msg: String,
    /// (id, expected, actual) for a version conflict.
    pub conflict: Option<(u64, u64, u64)>,
}

fn last_error() -> EngineError {
    // SAFETY: both read thread-locals the failed call on this thread just set.
    let (kind, msg) = unsafe {
        let k = feather_last_error_kind();
        let p = feather_last_error();
        let m = if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() };
        (k, m)
    };
    let kind = match kind {
        2 => Kind::Invalid,
        3 => Kind::Closed,
        4 => Kind::WriteFailed,
        5 => Kind::VersionConflict,
        6 => Kind::DuplicateKey,
        7 => Kind::IdempotencyReused,
        _ => Kind::Other,
    };
    let conflict = (kind == Kind::VersionConflict).then(|| {
        let (mut a, mut b, mut c) = (0u64, 0u64, 0u64);
        // SAFETY: plain out-pointers to locals.
        unsafe { feather_last_conflict(&mut a, &mut b, &mut c) };
        (a, b, c)
    });
    EngineError { kind, msg, conflict }
}

fn check(rc: c_int) -> Result<c_int, EngineError> {
    if rc < 0 { Err(last_error()) } else { Ok(rc) }
}

/// A buffer the engine malloc'd; freed on drop.
pub struct Packed {
    p: *mut u8,
    n: usize,
}

impl Packed {
    pub fn bytes(&self) -> &[u8] {
        if self.n == 0 { &[] } else {
            // SAFETY: the engine wrote n bytes at p and nothing else holds it.
            unsafe { std::slice::from_raw_parts(self.p, self.n) }
        }
    }
}

impl Drop for Packed {
    fn drop(&mut self) {
        // SAFETY: p came from malloc in the engine (or is null).
        unsafe { feather_free_buf(self.p) }
    }
}

unsafe impl Send for Packed {}

fn take_string(p: *mut c_char, n: usize) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the engine malloc'd n bytes (+ NUL) at p.
    let s = unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, n)).into_owned() };
    unsafe { feather_free_buf(p as *mut u8) };
    s
}

/// A search filter, owned. `raw()` borrows it for one call.
#[derive(Default, Debug)]
pub struct Filter {
    pub namespace_id: Option<String>,
    pub namespace_prefix: Option<String>,
    pub entity_id: Option<String>,
    pub source: Option<String>,
    pub source_prefix: Option<String>,
    pub types: Vec<u8>,
    pub attributes: Vec<(String, String)>,
    pub tags: Vec<String>,
    pub importance_gte: Option<f32>,
    pub confidence_gte: Option<f32>,
    pub timestamp_after: Option<i64>,
    pub timestamp_before: Option<i64>,
    pub valid_after: Option<i64>,
    pub valid_before: Option<i64>,
}

impl Filter {
    pub fn is_empty(&self) -> bool {
        self.namespace_id.is_none() && self.namespace_prefix.is_none() && self.entity_id.is_none()
            && self.source.is_none() && self.source_prefix.is_none() && self.types.is_empty()
            && self.attributes.is_empty() && self.tags.is_empty() && self.importance_gte.is_none()
            && self.confidence_gte.is_none() && self.timestamp_after.is_none()
            && self.timestamp_before.is_none() && self.valid_after.is_none() && self.valid_before.is_none()
    }
}

/// The FStr arrays a RawFilter points into, kept alive for the call.
struct FilterRef {
    raw: RawFilter,
    _keys: Vec<FStr>,
    _vals: Vec<FStr>,
    _tags: Vec<FStr>,
}

fn filter_ref(f: &Filter) -> FilterRef {
    let keys: Vec<FStr> = f.attributes.iter().map(|(k, _)| FStr::of(k)).collect();
    let vals: Vec<FStr> = f.attributes.iter().map(|(_, v)| FStr::of(v)).collect();
    let tags: Vec<FStr> = f.tags.iter().map(|t| FStr::of(t)).collect();
    let mut has = 0u32;
    let mut bit = |o: bool, b: u32| if o { has |= b };
    bit(f.importance_gte.is_some(), 1);
    bit(f.confidence_gte.is_some(), 2);
    bit(f.timestamp_after.is_some(), 4);
    bit(f.timestamp_before.is_some(), 8);
    bit(f.valid_after.is_some(), 16);
    bit(f.valid_before.is_some(), 32);
    let raw = RawFilter {
        namespace_id: FStr::opt(f.namespace_id.as_deref()),
        namespace_prefix: FStr::opt(f.namespace_prefix.as_deref()),
        entity_id: FStr::opt(f.entity_id.as_deref()),
        source: FStr::opt(f.source.as_deref()),
        source_prefix: FStr::opt(f.source_prefix.as_deref()),
        types: f.types.as_ptr(),
        n_types: f.types.len(),
        attr_keys: keys.as_ptr(),
        attr_vals: vals.as_ptr(),
        n_attrs: keys.len(),
        tags: tags.as_ptr(),
        n_tags: tags.len(),
        has,
        importance_gte: f.importance_gte.unwrap_or(0.0),
        confidence_gte: f.confidence_gte.unwrap_or(0.0),
        timestamp_after: f.timestamp_after.unwrap_or(0),
        timestamp_before: f.timestamp_before.unwrap_or(0),
        valid_after: f.valid_after.unwrap_or(0),
        valid_before: f.valid_before.unwrap_or(0),
    };
    FilterRef { raw, _keys: keys, _vals: vals, _tags: tags }
}

/// The metadata a write gives (the API's MetadataIn after defaults).
#[derive(Debug, Default)]
pub struct MetaIn {
    pub timestamp: i64,
    pub importance: f32,
    pub type_: u8,
    pub source: String,
    pub content: String,
    pub tags_json: String,
    pub namespace_id: String,
    pub entity_id: String,
    pub key: Option<String>,
    pub confidence: f32,
    pub ttl: i64,
    pub valid_at: i64,
    pub recall_count: Option<u32>,
    pub last_recalled_at: Option<u64>,
    pub attributes: Vec<(String, String)>,
    pub edges: Vec<(u64, String, f32)>,
}

pub struct PutResult {
    pub version: u64,
    pub replayed: bool,
    pub note: String,
}

/// A RawMetaIn and the arrays it points into, kept alive for one call.
struct MetaRef {
    raw: RawMetaIn,
    _keys: Vec<FStr>,
    _vals: Vec<FStr>,
    _targets: Vec<u64>,
    _types: Vec<FStr>,
    _weights: Vec<f32>,
}

fn meta_ref(m: &MetaIn) -> MetaRef {
    let keys: Vec<FStr> = m.attributes.iter().map(|(k, _)| FStr::of(k)).collect();
    let vals: Vec<FStr> = m.attributes.iter().map(|(_, v)| FStr::of(v)).collect();
    let targets: Vec<u64> = m.edges.iter().map(|e| e.0).collect();
    let types: Vec<FStr> = m.edges.iter().map(|e| FStr::of(&e.1)).collect();
    let weights: Vec<f32> = m.edges.iter().map(|e| e.2).collect();
    let raw = RawMetaIn {
        timestamp: m.timestamp,
        importance: m.importance,
        type_: m.type_,
        source: FStr::of(&m.source),
        content: FStr::of(&m.content),
        tags_json: FStr::of(&m.tags_json),
        namespace_id: FStr::of(&m.namespace_id),
        entity_id: FStr::of(&m.entity_id),
        key: FStr::opt(m.key.as_deref()),
        confidence: m.confidence,
        ttl: m.ttl,
        valid_at: m.valid_at,
        has: (m.recall_count.is_some() as u32) | ((m.last_recalled_at.is_some() as u32) << 1),
        recall_count: m.recall_count.unwrap_or(0),
        last_recalled_at: m.last_recalled_at.unwrap_or(0),
        attr_keys: keys.as_ptr(),
        attr_vals: vals.as_ptr(),
        n_attrs: keys.len(),
        edge_targets: targets.as_ptr(),
        edge_types: types.as_ptr(),
        edge_weights: weights.as_ptr(),
        n_edges: targets.len(),
    };
    MetaRef { raw, _keys: keys, _vals: vals, _targets: targets, _types: types, _weights: weights }
}

/// DB.heat_select's arguments (feather-api HeatSelectRequest).
pub struct HeatQuery {
    pub namespaces: Vec<String>,
    pub now: f64,
    pub budget: f64,
    pub half_life_days: f64,
    pub min_heat: f64,
    pub usage_half_life: f64,
    pub unused_floor: f64,
    pub inherit_decay: f64,
    pub chars_per_token: i64,
    pub pinned_attr: String,
    pub due_attr: String,
    pub origin_attr: String,
    pub text_attr: String,
    pub default_origin: String,
    pub unknown_origin: String,
    pub trusted: Vec<String>,
    pub exclude_sources: Vec<String>,
    pub origins: Vec<String>,
}

/// One op of POST /batch (feather_core.cpp feather_batch_op).
pub struct BatchOp {
    pub kind: u8,
    pub flags: u32,
    pub id: u64,
    pub key_ref: Option<(String, String)>,
    pub to_key_ref: Option<(String, String)>,
    pub expected: Option<u64>,
    pub meta: Option<MetaIn>,
    pub vec: Option<Vec<f32>>,
    pub modality: String,
    pub to_id: u64,
    pub rel_type: String,
    pub weight: f32,
}

pub const OP_PUT: u8 = 1;
pub const OP_UPDATE: u8 = 2;
pub const OP_LINK: u8 = 3;
pub const OP_FORGET: u8 = 4;
pub const OP_UPSERT: u8 = 5;
pub const FB_REF: u32 = 1;
pub const FB_TO_REF: u32 = 2;
pub const FB_EXPECTED: u32 = 4;
pub const FB_REQUIRE_TARGET: u32 = 8;
pub const FB_KEEP_EDGES: u32 = 16;
pub const FB_KEEP_KEY: u32 = 32;
pub const FB_KEEP_VALID_AT: u32 = 64;

/// One open .feather file. The engine is thread-safe: every method takes
/// `&self` and may run on any thread at once.
pub struct Db {
    p: *mut c_void,
}

// SAFETY: feather::DB synchronises internally (shared/exclusive locks); the
// pointer is only freed by Drop, which needs ownership.
unsafe impl Send for Db {}
unsafe impl Sync for Db {}

impl Db {
    pub fn open(path: &str, dim: usize) -> Result<Db, EngineError> {
        let c = CString::new(path).map_err(|_| EngineError {
            kind: Kind::Invalid,
            msg: "path contains a NUL byte".into(),
            conflict: None,
        })?;
        // SAFETY: c outlives the call.
        let p = unsafe { feather_open(c.as_ptr(), dim) };
        if p.is_null() { Err(last_error()) } else { Ok(Db { p }) }
    }

    pub fn save(&self) -> Result<(), EngineError> {
        // SAFETY: p is a live handle for as long as self is.
        check(unsafe { feather_save(self.p) }).map(|_| ())
    }

    /// Why the handle refuses writes, if it does.
    pub fn write_error(&self) -> Option<String> {
        // SAFETY: as above; the returned string is ours to free.
        let p = unsafe { feather_write_error(self.p) };
        if p.is_null() {
            return None;
        }
        let s = unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() };
        unsafe { feather_free_buf(p as *mut u8) };
        Some(s)
    }

    pub fn wal_size(&self) -> u64 {
        unsafe { feather_wal_size(self.p) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self, q: &[f32], k: usize, filter: &Filter, scoring: Option<&RawScoring>, modality: &str,
        record_salience: bool, with_cosine: bool, include_metadata: bool,
    ) -> Result<Packed, EngineError> {
        let fr = (!filter.is_empty()).then(|| filter_ref(filter));
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        // SAFETY: every pointer is to data that lives across the call.
        check(unsafe {
            feather_search_bin(
                self.p, q.as_ptr(), q.len(), k, fr.as_ref().map_or(ptr::null(), |f| &f.raw),
                scoring.map_or(ptr::null(), |s| s), FStr::of(modality), record_salience as c_int,
                with_cosine as c_int, include_metadata as c_int, &mut out, &mut n,
            )
        })?;
        Ok(Packed { p: out, n })
    }

    pub fn keyword_search(
        &self, query: &str, k: usize, filter: &Filter, record_salience: bool, include_metadata: bool,
    ) -> Result<Packed, EngineError> {
        let fr = (!filter.is_empty()).then(|| filter_ref(filter));
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe {
            feather_keyword_search_bin(
                self.p, FStr::of(query), k, fr.as_ref().map_or(ptr::null(), |f| &f.raw),
                record_salience as c_int, include_metadata as c_int, &mut out, &mut n,
            )
        })?;
        Ok(Packed { p: out, n })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_search(
        &self, q: &[f32], query: &str, k: usize, rrf_k: usize, filter: &Filter,
        scoring: Option<&RawScoring>, modality: &str, fusion: &str, include_metadata: bool,
    ) -> Result<Packed, EngineError> {
        let fr = (!filter.is_empty()).then(|| filter_ref(filter));
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe {
            feather_hybrid_search_bin(
                self.p, q.as_ptr(), q.len(), FStr::of(query), k, rrf_k,
                fr.as_ref().map_or(ptr::null(), |f| &f.raw), scoring.map_or(ptr::null(), |s| s),
                FStr::of(modality), FStr::of(fusion), include_metadata as c_int, &mut out, &mut n,
            )
        })?;
        Ok(Packed { p: out, n })
    }

    pub fn get(&self, id: u64, include_dead: bool) -> Result<Option<Packed>, EngineError> {
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        let rc = check(unsafe { feather_get_bin(self.p, id, include_dead as c_int, &mut out, &mut n) })?;
        Ok((rc == 0).then_some(Packed { p: out, n }))
    }

    pub fn get_many(&self, ids: &[u64], include_dead: bool) -> Result<Packed, EngineError> {
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe {
            feather_get_many_bin(self.p, ids.as_ptr(), ids.len(), include_dead as c_int, &mut out, &mut n)
        })?;
        Ok(Packed { p: out, n })
    }

    pub fn get_vector(&self, id: u64, modality: &str) -> Result<Vec<f32>, EngineError> {
        let (mut out, mut n) = (ptr::null_mut::<f32>(), 0usize);
        check(unsafe { feather_get_vector(self.p, id, FStr::of(modality), &mut out, &mut n) })?;
        // SAFETY: the engine malloc'd n floats at out.
        let v = unsafe { std::slice::from_raw_parts(out, n).to_vec() };
        unsafe { feather_free_buf(out as *mut u8) };
        Ok(v)
    }

    /// The dim of `modality`'s index, or None while it has none yet.
    pub fn modality_dim(&self, modality: &str) -> Result<Option<usize>, EngineError> {
        let mut d = 0usize;
        let rc = check(unsafe { feather_modality_dim(self.p, FStr::of(modality), &mut d) })?;
        Ok((rc == 1).then_some(d))
    }

    /// The version and stored note of an earlier write under this key.
    pub fn idempotent_result(&self, key: &str, fingerprint: &str) -> Result<Option<(u64, String)>, EngineError> {
        let (mut v, mut note, mut n) = (0u64, ptr::null_mut(), 0usize);
        let rc = check(unsafe {
            feather_idempotent_result(self.p, FStr::of(key), FStr::of(fingerprint), &mut v, &mut note, &mut n)
        })?;
        Ok((rc == 1).then(|| (v, take_string(note, n))))
    }

    /// One PUT, atomic and versioned; `meta` None writes the vector only.
    #[allow(clippy::too_many_arguments)]
    pub fn put(
        &self, id: u64, vec: &[f32], meta: Option<&MetaIn>, modality: &str, default_ns: &str,
        idem_key: &str, fingerprint: &str, note: &str,
    ) -> Result<PutResult, EngineError> {
        let mr = meta.map(meta_ref);
        let (mut version, mut replayed, mut out_note, mut out_n) = (0u64, 0 as c_int, ptr::null_mut(), 0usize);
        check(unsafe {
            feather_put(
                self.p, id, vec.as_ptr(), vec.len(), mr.as_ref().map_or(ptr::null(), |r| &r.raw),
                FStr::of(modality), FStr::of(default_ns), FStr::of(idem_key), FStr::of(fingerprint),
                FStr::of(note), &mut version, &mut replayed, &mut out_note, &mut out_n,
            )
        })?;
        Ok(PutResult { version, replayed: replayed != 0, note: take_string(out_note, out_n) })
    }

    /// The dim of `modality` (the handle's default while it has no index).
    pub fn dim(&self, modality: &str) -> Result<usize, EngineError> {
        let mut d = 0usize;
        check(unsafe { feather_dim(self.p, FStr::of(modality), &mut d) })?;
        Ok(d)
    }

    /// The live record holding `key` in `ns`: its id and packed metadata.
    pub fn lookup_key(&self, key: &str, ns: &str) -> Result<Option<(u64, Packed)>, EngineError> {
        let (mut id, mut out, mut n) = (0u64, ptr::null_mut(), 0usize);
        let rc = check(unsafe { feather_lookup_key_bin(self.p, FStr::of(key), FStr::of(ns), &mut id, &mut out, &mut n) })?;
        Ok((rc == 0).then_some((id, Packed { p: out, n })))
    }

    /// [u32 n]{[u8 found][u64 id]}* for (namespace, key) refs.
    pub fn ids_for_many(&self, refs: &[(String, String)]) -> Result<Packed, EngineError> {
        let nss: Vec<FStr> = refs.iter().map(|r| FStr::of(&r.0)).collect();
        let keys: Vec<FStr> = refs.iter().map(|r| FStr::of(&r.1)).collect();
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe { feather_ids_for_many_bin(self.p, nss.as_ptr(), keys.as_ptr(), refs.len(), &mut out, &mut n) })?;
        Ok(Packed { p: out, n })
    }

    pub fn newest(
        &self, ns: &str, limit: usize, prefix: bool, before: Option<(i64, u64)>, by_valid: bool,
    ) -> Result<Packed, EngineError> {
        let (ts, id) = before.unwrap_or((0, 0));
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe {
            feather_newest_bin(self.p, FStr::of(ns), limit, prefix as c_int, before.is_some() as c_int, ts, id,
                               by_valid as c_int, &mut out, &mut n)
        })?;
        Ok(Packed { p: out, n })
    }

    /// GET /ids: see feather_ids_bin for `which`.
    pub fn ids(&self, which: i32, a: &str, b: &str, prefix: bool) -> Result<Packed, EngineError> {
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe { feather_ids_bin(self.p, which, FStr::of(a), FStr::of(b), prefix as c_int, &mut out, &mut n) })?;
        Ok(Packed { p: out, n })
    }

    pub fn metadata_namespaces(&self) -> Result<Packed, EngineError> {
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe { feather_metadata_namespaces_bin(self.p, &mut out, &mut n) })?;
        Ok(Packed { p: out, n })
    }

    pub fn count_live(&self, ns: &str, prefix: bool, exclude: &[String]) -> Result<u64, EngineError> {
        let ex: Vec<FStr> = exclude.iter().map(|e| FStr::of(e)).collect();
        let mut c = 0u64;
        check(unsafe { feather_count_live(self.p, FStr::of(ns), prefix as c_int, ex.as_ptr(), ex.len(), &mut c) })?;
        Ok(c)
    }

    pub fn heat_select(&self, q: &HeatQuery) -> Result<Packed, EngineError> {
        let f = |v: &[String]| -> Vec<FStr> { v.iter().map(|s| FStr::of(s)).collect() };
        let (nss, trusted, exclude, origins) = (f(&q.namespaces), f(&q.trusted), f(&q.exclude_sources), f(&q.origins));
        let raw = RawHeatQuery {
            namespaces: nss.as_ptr(),
            n_namespaces: nss.len(),
            now: q.now,
            budget: q.budget,
            half_life_days: q.half_life_days,
            min_heat: q.min_heat,
            usage_half_life: q.usage_half_life,
            unused_floor: q.unused_floor,
            inherit_decay: q.inherit_decay,
            chars_per_token: q.chars_per_token,
            pinned_attr: FStr::of(&q.pinned_attr),
            due_attr: FStr::of(&q.due_attr),
            origin_attr: FStr::of(&q.origin_attr),
            text_attr: FStr::of(&q.text_attr),
            default_origin: FStr::of(&q.default_origin),
            unknown_origin: FStr::of(&q.unknown_origin),
            trusted: trusted.as_ptr(),
            n_trusted: trusted.len(),
            exclude_sources: exclude.as_ptr(),
            n_exclude: exclude.len(),
            origins: origins.as_ptr(),
            n_origins: origins.len(),
        };
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe { feather_heat_select_bin(self.p, &raw, &mut out, &mut n) })?;
        Ok(Packed { p: out, n })
    }

    pub fn touch_many(&self, ids: &[u64]) -> Result<(), EngineError> {
        check(unsafe { feather_touch_many(self.p, ids.as_ptr(), ids.len()) }).map(|_| ())
    }

    pub fn normalize_vectors(&self, modality: &str) -> Result<u64, EngineError> {
        let mut c = 0u64;
        check(unsafe { feather_normalize_vectors(self.p, FStr::of(modality), &mut c) })?;
        Ok(c)
    }

    pub fn engine_state(&self) -> Result<Packed, EngineError> {
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe { feather_engine_state_bin(self.p, &mut out, &mut n) })?;
        Ok(Packed { p: out, n })
    }

    /// POST /batch, atomic; a stored result for the key answers first.
    pub fn write_batch(
        &self, ops: &[BatchOp], default_ns: &str, idem_key: &str, fingerprint: &str, note: &str,
    ) -> Result<Packed, EngineError> {
        let metas: Vec<Option<MetaRef>> = ops.iter().map(|o| o.meta.as_ref().map(meta_ref)).collect();
        let raws: Vec<RawBatchOp> = ops
            .iter()
            .zip(&metas)
            .map(|(o, m)| RawBatchOp {
                kind: o.kind,
                flags: o.flags,
                id: o.id,
                ref_ns: FStr::opt(o.key_ref.as_ref().map(|r| r.0.as_str())),
                ref_key: FStr::opt(o.key_ref.as_ref().map(|r| r.1.as_str())),
                to_ref_ns: FStr::opt(o.to_key_ref.as_ref().map(|r| r.0.as_str())),
                to_ref_key: FStr::opt(o.to_key_ref.as_ref().map(|r| r.1.as_str())),
                expected: o.expected.unwrap_or(0),
                meta: m.as_ref().map_or(ptr::null(), |r| &r.raw),
                vec: o.vec.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
                vec_len: o.vec.as_ref().map_or(0, |v| v.len()),
                modality: FStr::of(&o.modality),
                to_id: o.to_id,
                rel_type: FStr::of(&o.rel_type),
                weight: o.weight,
            })
            .collect();
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe {
            feather_write_batch_bin(
                self.p, raws.as_ptr(), raws.len(), FStr::of(default_ns), FStr::of(idem_key),
                FStr::of(fingerprint), FStr::of(note), &mut out, &mut n,
            )
        })?;
        Ok(Packed { p: out, n })
    }

    /// Edges into `id`: [u32 n]{[u64 source][str rel_type][f32 weight]}*.
    pub fn incoming(&self, id: u64) -> Result<Packed, EngineError> {
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe { feather_incoming_bin(self.p, id, &mut out, &mut n) })?;
        Ok(Packed { p: out, n })
    }

    /// erase(): the receipt, as the JSON the engine writes.
    pub fn erase(&self, a: &EraseArgs) -> Result<Packed, EngineError> {
        let f = |v: &[String]| -> Vec<FStr> { v.iter().map(|s| FStr::of(s)).collect() };
        let cites = f(a.cites.as_deref().unwrap_or(&[]));
        let feeds = f(a.feeds.as_deref().unwrap_or(&[]));
        let (mut out, mut n) = (ptr::null_mut(), 0usize);
        check(unsafe {
            feather_erase_json(
                self.p, FStr::of(&a.namespace_id), a.prefix as c_int, FStr::of(&a.entity_id), a.ids.as_ptr(),
                a.ids.len(), a.cites.is_some() as c_int, cites.as_ptr(), cites.len(), a.feeds.is_some() as c_int,
                feeds.as_ptr(), feeds.len(), a.dry_run as c_int, &mut out, &mut n,
            )
        })?;
        Ok(Packed { p: out, n })
    }

    /// Close now. A handle whose write failed is closed without a
    /// checkpoint, so its WAL is kept for the reopen to replay.
    pub fn close(mut self) -> Result<(), EngineError> {
        let p = std::mem::replace(&mut self.p, ptr::null_mut());
        close_ptr(p)
    }
}

fn close_ptr(p: *mut c_void) -> Result<(), EngineError> {
    if p.is_null() {
        return Ok(());
    }
    let poisoned = {
        let e = unsafe { feather_write_error(p) };
        let bad = !e.is_null();
        unsafe { feather_free_buf(e as *mut u8) };
        bad
    };
    check(unsafe { feather_close_ex(p, (!poisoned) as c_int) }).map(|_| ())
}

impl Drop for Db {
    fn drop(&mut self) {
        let p = std::mem::replace(&mut self.p, ptr::null_mut());
        if let Err(e) = close_ptr(p) {
            eprintln!("[feather-server] close failed: {}", e.msg);
        }
    }
}
