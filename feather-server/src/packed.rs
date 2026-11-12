//! Packed records from the engine (feather_core.cpp `Packer`) written out as
//! the Cloud API's JSON, field for field. Numbers are formatted as the
//! Python server formats them (orjson: ryu's shortest round-trip of the
//! float32 widened to f64; non-finite as null).

pub struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

#[derive(Debug)]
pub struct Truncated;

type R<T> = Result<T, Truncated>;

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Reader { b, at: 0 }
    }
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or(Truncated)?;
        let s = self.b.get(self.at..end).ok_or(Truncated)?;
        self.at = end;
        Ok(s)
    }
    pub fn u8(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn i64(&mut self) -> R<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> R<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn str(&mut self) -> R<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
}

/// One metadata record, borrowed from the packed buffer.
pub struct Meta<'a> {
    pub timestamp: i64,
    pub importance: f32,
    pub type_: u8,
    pub source: &'a [u8],
    pub content: &'a [u8],
    pub tags_json: &'a [u8],
    pub namespace_id: &'a [u8],
    pub entity_id: &'a [u8],
    pub attributes: Vec<(&'a [u8], &'a [u8])>,
    pub recall_count: u32,
    pub last_recalled_at: u64,
    pub edges: Vec<(u64, &'a [u8], f32)>,
    pub ttl: i64,
    pub confidence: f32,
    pub key: &'a [u8],
    pub version: u64,
    pub valid_at: i64,
}

pub fn read_meta<'a>(r: &mut Reader<'a>) -> R<Meta<'a>> {
    let timestamp = r.i64()?;
    let importance = r.f32()?;
    let type_ = r.u8()?;
    let source = r.str()?;
    let content = r.str()?;
    let tags_json = r.str()?;
    let namespace_id = r.str()?;
    let entity_id = r.str()?;
    let n_attrs = r.u32()? as usize;
    let mut attributes = Vec::with_capacity(n_attrs.min(1024));
    for _ in 0..n_attrs {
        attributes.push((r.str()?, r.str()?));
    }
    let recall_count = r.u32()?;
    let last_recalled_at = r.u64()?;
    let n_edges = r.u32()? as usize;
    let mut edges = Vec::with_capacity(n_edges.min(1024));
    for _ in 0..n_edges {
        edges.push((r.u64()?, r.str()?, r.f32()?));
    }
    Ok(Meta {
        timestamp,
        importance,
        type_,
        source,
        content,
        tags_json,
        namespace_id,
        entity_id,
        attributes,
        recall_count,
        last_recalled_at,
        edges,
        ttl: r.i64()?,
        confidence: r.f32()?,
        key: r.str()?,
        version: r.u64()?,
        valid_at: r.i64()?,
    })
}

impl Meta<'_> {
    pub fn is_dead(&self) -> bool {
        self.source == b"_forgotten"
            || self.attributes.iter().any(|(k, v)| *k == b"_deleted" && *v == b"true")
    }
}

// ── JSON output ───────────────────────────────────────────────────────

pub fn json_str(out: &mut Vec<u8>, s: &[u8]) {
    // Engine strings are UTF-8 by contract; anything else is replaced rather
    // than failing the whole response (the Python server raised).
    let s = String::from_utf8_lossy(s);
    serde_json::to_writer(&mut *out, s.as_ref()).expect("writing to a Vec cannot fail");
}

pub fn json_f64(out: &mut Vec<u8>, v: f64) {
    if v.is_finite() {
        out.extend_from_slice(ryu::Buffer::new().format_finite(v).as_bytes());
    } else {
        out.extend_from_slice(b"null");
    }
}

pub fn json_f32(out: &mut Vec<u8>, v: f32) {
    json_f64(out, v as f64)
}

pub fn json_int<I: itoa::Integer>(out: &mut Vec<u8>, v: I) {
    out.extend_from_slice(itoa::Buffer::new().format(v).as_bytes());
}

fn key(out: &mut Vec<u8>, k: &str, first: &mut bool) {
    if !*first {
        out.push(b',');
    }
    *first = false;
    out.push(b'"');
    out.extend_from_slice(k.as_bytes());
    out.extend_from_slice(b"\":");
}

fn attrs_obj(out: &mut Vec<u8>, m: &Meta) {
    out.push(b'{');
    for (i, (k, v)) in m.attributes.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        json_str(out, k);
        out.push(b':');
        json_str(out, v);
    }
    out.push(b'}');
}

fn links_arr(out: &mut Vec<u8>, m: &Meta) {
    out.push(b'[');
    for (i, e) in m.edges.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        json_int(out, e.0);
    }
    out.push(b']');
}

fn edges_arr(out: &mut Vec<u8>, m: &Meta) {
    out.push(b'[');
    for (i, (t, rel, w)) in m.edges.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"target_id\":");
        json_int(out, *t);
        out.extend_from_slice(b",\"rel_type\":");
        json_str(out, rel);
        out.extend_from_slice(b",\"weight\":");
        json_f32(out, *w);
        out.push(b'}');
    }
    out.push(b']');
}

/// `Metadata.to_dict()` (bindings/feather.cpp metadata_to_dict), the shape
/// of a search hit's and a get_many row's `metadata`.
pub fn meta_dict(out: &mut Vec<u8>, m: &Meta) {
    let mut f = true;
    out.push(b'{');
    key(out, "timestamp", &mut f);
    json_int(out, m.timestamp);
    key(out, "importance", &mut f);
    json_f32(out, m.importance);
    key(out, "type", &mut f);
    json_int(out, m.type_);
    key(out, "source", &mut f);
    json_str(out, m.source);
    key(out, "content", &mut f);
    json_str(out, m.content);
    key(out, "tags_json", &mut f);
    json_str(out, m.tags_json);
    key(out, "namespace_id", &mut f);
    json_str(out, m.namespace_id);
    key(out, "entity_id", &mut f);
    json_str(out, m.entity_id);
    key(out, "attributes", &mut f);
    attrs_obj(out, m);
    key(out, "recall_count", &mut f);
    json_int(out, m.recall_count);
    key(out, "last_recalled_at", &mut f);
    json_int(out, m.last_recalled_at);
    key(out, "links", &mut f);
    links_arr(out, m);
    key(out, "edges", &mut f);
    edges_arr(out, m);
    key(out, "ttl", &mut f);
    json_int(out, m.ttl);
    key(out, "confidence", &mut f);
    json_f32(out, m.confidence);
    key(out, "key", &mut f);
    json_str(out, m.key);
    key(out, "version", &mut f);
    json_int(out, m.version);
    key(out, "valid_at", &mut f);
    json_int(out, m.valid_at);
    out.push(b'}');
}

/// The `MetadataOut` model's fields in its order (GET /records/{id}),
/// without the braces, so the caller can append `id` / `metadata`.
pub fn meta_out_fields(out: &mut Vec<u8>, m: &Meta) {
    let mut f = true;
    key(out, "timestamp", &mut f);
    json_int(out, m.timestamp);
    key(out, "importance", &mut f);
    json_f32(out, m.importance);
    key(out, "type", &mut f);
    json_int(out, m.type_);
    key(out, "source", &mut f);
    json_str(out, m.source);
    key(out, "content", &mut f);
    json_str(out, m.content);
    key(out, "tags_json", &mut f);
    json_str(out, m.tags_json);
    key(out, "namespace_id", &mut f);
    json_str(out, m.namespace_id);
    key(out, "entity_id", &mut f);
    json_str(out, m.entity_id);
    key(out, "key", &mut f);
    json_str(out, m.key);
    key(out, "edges", &mut f);
    edges_arr(out, m);
    key(out, "recall_count", &mut f);
    json_int(out, m.recall_count);
    key(out, "last_recalled_at", &mut f);
    json_int(out, m.last_recalled_at);
    key(out, "attributes", &mut f);
    attrs_obj(out, m);
    key(out, "confidence", &mut f);
    json_f32(out, m.confidence);
    key(out, "ttl", &mut f);
    json_int(out, m.ttl);
    key(out, "valid_at", &mut f);
    json_int(out, m.valid_at);
    key(out, "links", &mut f);
    links_arr(out, m);
    key(out, "version", &mut f);
    json_int(out, m.version);
}

/// `{"results": [...], "count": n}` from packed search results.
pub fn results_json(b: &[u8]) -> R<Vec<u8>> {
    let mut r = Reader::new(b);
    let n = r.u32()?;
    let mut out = Vec::with_capacity(64 + b.len() + b.len() / 4);
    out.extend_from_slice(b"{\"results\":[");
    for i in 0..n {
        if i > 0 {
            out.push(b',');
        }
        let id = r.u64()?;
        let score = r.f32()?;
        let cosine = r.f32()?;
        let has_meta = r.u8()? != 0;
        out.extend_from_slice(b"{\"id\":");
        json_int(&mut out, id);
        out.extend_from_slice(b",\"score\":");
        json_f32(&mut out, score);
        out.extend_from_slice(b",\"cosine\":");
        json_f32(&mut out, cosine); // NaN (not computed) is null
        if has_meta {
            out.extend_from_slice(b",\"metadata\":");
            let m = read_meta(&mut r)?;
            meta_dict(&mut out, &m);
        }
        out.push(b'}');
    }
    out.extend_from_slice(b"],\"count\":");
    json_int(&mut out, n);
    out.push(b'}');
    Ok(out)
}
