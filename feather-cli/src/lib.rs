use std::ffi::{c_void, c_char, CStr};
use std::path::Path;

#[repr(C)]
pub struct DB(*mut c_void);  // ← `pub`

// Every call returns 0 on success and -1 on failure; the engine's message is
// then in feather_last_error(). No C++ exception crosses this boundary.
extern "C" {
    fn feather_last_error() -> *const c_char;
    fn feather_open(path: *const c_char, dim: usize) -> *mut c_void;
    fn feather_add(db: *mut c_void, id: u64, vec: *const f32, len: usize) -> i32;
    fn feather_add_with_meta(db: *mut c_void, id: u64, vec: *const f32, len: usize,
                              timestamp: i64, importance: f32, context_type: u8,
                              source: *const c_char, content: *const c_char, modality: *const c_char) -> i32;
    fn feather_link(db: *mut c_void, from_id: u64, to_id: u64) -> i32;
    fn feather_touch(db: *mut c_void, id: u64) -> i32;
    fn feather_search(db: *mut c_void, query: *const f32, len: usize, k: usize,
                      out_ids: *mut u64, out_dists: *mut f32, modality: *const c_char) -> i32;
    fn feather_search_with_filter(db: *mut c_void, query: *const f32, len: usize, k: usize,
                                   type_filter: u8, source_filter: *const c_char,
                                   out_ids: *mut u64, out_dists: *mut f32, modality: *const c_char) -> i32;
    fn feather_save(db: *mut c_void) -> i32;
    fn feather_close(db: *mut c_void);
}

/// The engine's message for the last failed call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(feather_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

impl DB {
    pub fn try_open(path: &Path, dim: usize) -> Result<Self, String> {
        let s = path.to_str().ok_or("path is not valid UTF-8")?;
        let c_path = std::ffi::CString::new(s).map_err(|e| e.to_string())?;
        let ptr = unsafe { feather_open(c_path.as_ptr(), dim) };
        if ptr.is_null() { Err(last_error()) } else { Ok(DB(ptr)) }
    }

    pub fn try_add(&self, id: u64, vec: &[f32]) -> Result<(), String> {
        check(unsafe { feather_add(self.0, id, vec.as_ptr(), vec.len()) })
    }

    pub fn try_add_with_meta(&self, id: u64, vec: &[f32], timestamp: i64, importance: f32, context_type: u8, 
                         source: Option<&str>, content: Option<&str>, modality: Option<&str>) -> Result<(), String> {
        let c_source = source.and_then(|s| std::ffi::CString::new(s).ok());
        let c_content = content.and_then(|s| std::ffi::CString::new(s).ok());
        let c_modality = modality.and_then(|s| std::ffi::CString::new(s).ok());
        
        check(unsafe {
            feather_add_with_meta(
                self.0, id, vec.as_ptr(), vec.len(),
                timestamp, importance, context_type,
                c_source.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                c_content.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                c_modality.as_ref().map_or(std::ptr::null(), |s| s.as_ptr())
            )
        })
    }

    pub fn try_link(&self, from_id: u64, to_id: u64) -> Result<(), String> {
        check(unsafe { feather_link(self.0, from_id, to_id) })
    }

    pub fn try_touch(&self, id: u64) -> Result<(), String> {
        check(unsafe { feather_touch(self.0, id) })
    }

    pub fn try_search(&self, query: &[f32], k: usize, modality: Option<&str>) -> Result<(Vec<u64>, Vec<f32>), String> {
        let mut ids = vec![0u64; k];
        let mut dists = vec![0f32; k];
        let c_modality = modality.and_then(|s| std::ffi::CString::new(s).ok());
        check(unsafe {
            feather_search(
                self.0, query.as_ptr(), query.len(), k, 
                ids.as_mut_ptr(), dists.as_mut_ptr(),
                c_modality.as_ref().map_or(std::ptr::null(), |s| s.as_ptr())
            )
        })?;
        Ok((ids, dists))
    }

    pub fn try_search_with_filter(&self, query: &[f32], k: usize, type_filter: Option<u8>, 
                               source_filter: Option<&str>, modality: Option<&str>) -> Result<(Vec<u64>, Vec<f32>), String> {
        let mut ids = vec![0u64; k];
        let mut dists = vec![0f32; k];
        let c_source = source_filter.and_then(|s| std::ffi::CString::new(s).ok());
        let c_modality = modality.and_then(|s| std::ffi::CString::new(s).ok());
        
        check(unsafe {
            feather_search_with_filter(
                self.0, query.as_ptr(), query.len(), k,
                type_filter.unwrap_or(255),
                c_source.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                ids.as_mut_ptr(), dists.as_mut_ptr(),
                c_modality.as_ref().map_or(std::ptr::null(), |s| s.as_ptr())
            )
        })?;
        Ok((ids, dists))
    }

    pub fn try_save(&self) -> Result<(), String> { check(unsafe { feather_save(self.0) }) }

    // ── Pre-0.22 signatures, kept so downstream crates still compile ─────
    // These used to abort the whole process (an uncaught C++ exception) on
    // any engine error. They now panic with the engine's message instead;
    // use the try_* forms above to handle the error.

    pub fn open(path: &Path, dim: usize) -> Option<Self> {
        Self::try_open(path, dim).ok()
    }

    pub fn add(&self, id: u64, vec: &[f32]) {
        self.try_add(id, vec).unwrap_or_else(|e| panic!("feather add: {e}"))
    }

    pub fn add_with_meta(&self, id: u64, vec: &[f32], timestamp: i64, importance: f32, context_type: u8,
                         source: Option<&str>, content: Option<&str>, modality: Option<&str>) {
        self.try_add_with_meta(id, vec, timestamp, importance, context_type, source, content, modality)
            .unwrap_or_else(|e| panic!("feather add: {e}"))
    }

    pub fn link(&self, from_id: u64, to_id: u64) {
        self.try_link(from_id, to_id).unwrap_or_else(|e| panic!("feather link: {e}"))
    }

    pub fn touch(&self, id: u64) {
        self.try_touch(id).unwrap_or_else(|e| panic!("feather touch: {e}"))
    }

    pub fn search(&self, query: &[f32], k: usize, modality: Option<&str>) -> (Vec<u64>, Vec<f32>) {
        self.try_search(query, k, modality).unwrap_or_else(|e| panic!("feather search: {e}"))
    }

    pub fn search_with_filter(&self, query: &[f32], k: usize, type_filter: Option<u8>,
                               source_filter: Option<&str>, modality: Option<&str>) -> (Vec<u64>, Vec<f32>) {
        self.try_search_with_filter(query, k, type_filter, source_filter, modality)
            .unwrap_or_else(|e| panic!("feather search: {e}"))
    }

    pub fn save(&self) {
        self.try_save().unwrap_or_else(|e| panic!("feather save: {e}"))
    }
}

impl Drop for DB {
    fn drop(&mut self) { unsafe { feather_close(self.0) } }
}
