//! Namespaces: one .feather file each under FEATHER_DATA_DIR, opened on first
//! use, checkpointed in the background, closed (with a checkpoint) on
//! shutdown. Mirrors feather-api/app/db_manager.py for what the hot routes
//! need. Not here: idle close, sharding, uploads, as_of handles.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::api::ApiError;
use crate::ffi::{Db, Kind};

/// The longest file name a namespace gets: `<name>.feather.wal.old`.
const LONGEST_SUFFIX: &str = ".feather.wal.old";
const NAME_MAX: usize = 255;

pub struct Namespace {
    pub db: Arc<Db>,
    /// Held by writes that read before they write (a PUT without metadata
    /// keeps the record's), as the Python server's manager.lock(namespace).
    pub write_lock: Arc<Mutex<()>>,
}

struct Entry {
    db: Arc<Db>,
    write_lock: Arc<Mutex<()>>,
    last_save: Instant,
}

pub struct Manager {
    data_dir: PathBuf,
    default_dim: usize,
    open: RwLock<HashMap<String, Entry>>,
    /// One per namespace being opened or recovered, so two requests never
    /// open the same file twice (the engine's lock is reentrant in-process).
    opening: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// (dev, inode) of each open namespace's `<path>.lock`.
    file_ids: Mutex<HashMap<String, (u64, u64)>>,
}

/// `validate_namespace` in db_manager.py: letters, digits, '-' and '_',
/// never rewritten.
pub fn validate_namespace(ns: &str) -> Result<(), ApiError> {
    if ns.is_empty() {
        return Err(ApiError::new(400, "namespace must be a non-empty string", "invalid_namespace"));
    }
    let mut bad: Vec<char> = ns.chars().filter(|c| !(c.is_alphanumeric() || *c == '-' || *c == '_')).collect();
    if !bad.is_empty() {
        bad.sort();
        bad.dedup();
        let bad: String = bad.into_iter().collect();
        return Err(ApiError::new(
            400,
            format!(
                "Invalid namespace '{ns}': only letters, digits, '-' and '_' are allowed (found '{bad}'). \
                 Names are not rewritten: an older server stored such a name under the name with those \
                 characters removed, and its data is there."
            ),
            "invalid_namespace",
        ));
    }
    if ns.len() + LONGEST_SUFFIX.len() > NAME_MAX {
        return Err(ApiError::new(
            400,
            format!(
                "Invalid namespace: {} bytes; at most {} fit in a file name with its WAL.",
                ns.len(),
                NAME_MAX - LONGEST_SUFFIX.len()
            ),
            "invalid_namespace",
        ));
    }
    Ok(())
}

impl Manager {
    pub fn new(data_dir: PathBuf, default_dim: usize) -> Self {
        Manager {
            data_dir,
            default_dim,
            open: RwLock::new(HashMap::new()),
            opening: Mutex::new(HashMap::new()),
            file_ids: Mutex::new(HashMap::new()),
        }
    }

    fn path(&self, ns: &str) -> PathBuf {
        self.data_dir.join(format!("{ns}.feather"))
    }

    /// Every namespace with a file in the data directory.
    pub fn list_namespaces(&self) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(&self.data_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let ns = name.strip_suffix(".feather")?;
                validate_namespace(ns).ok().map(|_| ns.to_string())
            })
            .collect();
        out.sort();
        out
    }

    pub fn open_count(&self) -> usize {
        self.open.read().unwrap().len()
    }

    fn healthy(&self, ns: &str) -> Option<Namespace> {
        let open = self.open.read().unwrap();
        let e = open.get(ns)?;
        // A handle whose write failed is replaced (reopened) before use, as
        // DBManager.get does: only a reopen replays what reached the WAL.
        if e.db.write_error().is_some() {
            return None;
        }
        Some(Namespace { db: e.db.clone(), write_lock: e.write_lock.clone() })
    }

    /// The namespace's handle, opening the file on first use. `create`
    /// false: a namespace without a file is a 404.
    pub async fn get(&self, ns: &str, create: bool) -> Result<Namespace, ApiError> {
        self.get_dim(ns, create, None).await
    }

    /// As `get`; a file this creates gets `dim` instead of FEATHER_DB_DIM
    /// (POST /v1/namespaces). An open namespace ignores it, as DBManager.get.
    pub async fn get_dim(&self, ns: &str, create: bool, dim: Option<usize>) -> Result<Namespace, ApiError> {
        if let Some(n) = self.healthy(ns) {
            return Ok(n);
        }
        validate_namespace(ns)?;
        let path = self.path(ns);
        // Answered before `opening` gets an entry, so requests for names that
        // do not exist cannot grow it: it holds only names with a file.
        if !create && !path.exists() && !self.open.read().unwrap().contains_key(ns) {
            return Err(ApiError::new(404, format!("Namespace '{ns}' not found"), "not_found"));
        }
        let gate = self.opening.lock().unwrap().entry(ns.to_string()).or_default().clone();
        let _turn = gate.lock().await;
        if let Some(n) = self.healthy(ns) {
            return Ok(n);
        }
        let stale = self.open.read().unwrap().get(ns).map(|e| e.db.clone());
        if stale.is_none() {
            if !create && !path.exists() {
                return Err(ApiError::new(404, format!("Namespace '{ns}' not found"), "not_found"));
            }
            self.claim_file(ns, &path)?;
        }
        let p = path.to_string_lossy().into_owned();
        let dim = dim.unwrap_or(self.default_dim);
        let opened = tokio::task::spawn_blocking(move || Db::open(&p, dim))
            .await
            .map_err(|e| ApiError::internal(format!("open panicked: {e}")))?;
        let db = match opened {
            Ok(db) => Arc::new(db),
            Err(e) => {
                if stale.is_none() {
                    self.file_ids.lock().unwrap().remove(ns);
                }
                if e.msg.contains("holds the write lock") {
                    return Err(ApiError::new(503, e.msg, "namespace_held").retry_after(1));
                }
                return Err(ApiError::internal(format!("could not open namespace '{ns}': {}", e.msg)));
            }
        };
        let n = {
            let mut open = self.open.write().unwrap();
            let lock = open.get(ns).map(|e| e.write_lock.clone()).unwrap_or_default();
            open.insert(
                ns.to_string(),
                Entry { db: db.clone(), write_lock: lock.clone(), last_save: Instant::now() },
            );
            Namespace { db, write_lock: lock }
        };
        if let Some(old) = stale {
            // The failed handle closes without a checkpoint once the last
            // request using it lets go (Db::drop checks write_error()).
            eprintln!("[feather-server] namespace '{ns}' reopened after a failed write");
            drop(old);
        }
        Ok(n)
    }

    /// Refuse a name that reaches the file of another open namespace (case
    /// or Unicode folding on macOS): two handles on one file each checkpoint
    /// their own view. Compared by the inode of `<path>.lock`, as
    /// DBManager._claim_file does.
    #[cfg(unix)]
    fn claim_file(&self, ns: &str, path: &std::path::Path) -> Result<(), ApiError> {
        use std::os::unix::fs::MetadataExt;
        let marker = PathBuf::from(format!("{}.lock", path.display()));
        let ident = match std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&marker)
            .and_then(|f| f.metadata())
        {
            Ok(m) => (m.dev(), m.ino()),
            Err(_) => return Ok(()), // unwritable dir: the open reports it
        };
        let mut ids = self.file_ids.lock().unwrap();
        if let Some((other, _)) = ids.iter().find(|(o, id)| **id == ident && o.as_str() != ns) {
            return Err(ApiError::new(
                409,
                format!(
                    "Namespace '{ns}' is the same file as namespace '{other}' on this filesystem \
                     (names that differ only in case or Unicode form share one file here). Use '{other}'."
                ),
                "namespace_alias",
            ));
        }
        ids.insert(ns.to_string(), ident);
        Ok(())
    }

    #[cfg(not(unix))]
    fn claim_file(&self, _ns: &str, _path: &std::path::Path) -> Result<(), ApiError> {
        Ok(())
    }

    /// One checkpoint pass: save a namespace whose WAL passed `wal_bytes` or
    /// has been dirty for `max_age` (feather-api's _checkpoint_pass).
    pub fn checkpoint_pass(&self, wal_bytes: u64, max_age: Duration) {
        let due: Vec<(String, Arc<Db>)> = {
            let mut open = self.open.write().unwrap();
            let now = Instant::now();
            open.iter_mut()
                .filter_map(|(ns, e)| {
                    if e.db.write_error().is_some() {
                        return None; // the next request reopens it
                    }
                    let wal = e.db.wal_size();
                    if wal == 0 || wal == u64::MAX {
                        e.last_save = now;
                        return None;
                    }
                    (wal >= wal_bytes || now.duration_since(e.last_save) >= max_age)
                        .then(|| (ns.clone(), e.db.clone()))
                })
                .collect()
        };
        for (ns, db) in due {
            match db.save() {
                Ok(()) => {
                    if let Some(e) = self.open.write().unwrap().get_mut(&ns) {
                        e.last_save = Instant::now();
                    }
                }
                Err(e) if e.kind == Kind::Closed => {}
                Err(e) => eprintln!("[feather-server] checkpoint of namespace '{ns}' failed: {}", e.msg),
            }
        }
    }

    /// POST /{ns}/save: checkpoint the namespace if it is open; a closed or
    /// unknown one has nothing to save (DBManager.save, which answers
    /// `saved: true` either way).
    pub fn save(&self, ns: &str) -> Result<(), crate::ffi::EngineError> {
        let db = self.open.read().unwrap().get(ns).map(|e| e.db.clone());
        if let Some(db) = db {
            db.save()?;
            if let Some(e) = self.open.write().unwrap().get_mut(ns) {
                e.last_save = Instant::now();
            }
        }
        Ok(())
    }

    /// Close every namespace, checkpointing each (shutdown).
    pub fn close_all(&self) {
        let all: Vec<(String, Entry)> = self.open.write().unwrap().drain().collect();
        for (ns, e) in all {
            match Arc::try_unwrap(e.db) {
                Ok(db) => {
                    if let Err(err) = db.close() {
                        eprintln!("[feather-server] close of namespace '{ns}' failed: {}", err.msg);
                    }
                }
                // Still referenced by a request that outlived shutdown: it
                // closes (and checkpoints) when that request drops it.
                Err(_) => eprintln!("[feather-server] namespace '{ns}' still in use at shutdown"),
            }
        }
    }
}
