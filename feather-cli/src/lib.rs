use std::ffi::{c_void, c_char};
use std::path::Path;

#[repr(C)]
pub struct DB(*mut c_void);  // ← `pub`

extern "C" {
    fn feather_open(path: *const c_char, dim: usize) -> *mut c_void;
    fn feather_add(db: *mut c_void, id: u64, vec: *const f32, len: usize);
    fn feather_add_with_meta(db: *mut c_void, id: u64, vec: *const f32, len: usize,
                              timestamp: i64, importance: f32, context_type: u8,
                              source: *const c_char, content: *const c_char);
    fn feather_search(db: *mut c_void, query: *const f32, len: usize, k: usize,
                      out_ids: *mut u64, out_dists: *mut f32);
    fn feather_search_with_filter(db: *mut c_void, query: *const f32, len: usize, k: usize,
                                   type_filter: u8, source_filter: *const c_char,
                                   out_ids: *mut u64, out_dists: *mut f32);
    fn feather_save(db: *mut c_void);
    fn feather_close(db: *mut c_void);
}

impl DB {
    pub fn open(path: &Path, dim: usize) -> Option<Self> {
        let c_path = std::ffi::CString::new(path.to_str()?).ok()?;
        let ptr = unsafe { feather_open(c_path.as_ptr(), dim) };
        if ptr.is_null() { None } else { Some(DB(ptr)) }
    }

    pub fn add(&self, id: u64, vec: &[f32]) {
        unsafe { feather_add(self.0, id, vec.as_ptr(), vec.len()) }
    }

    pub fn add_with_meta(&self, id: u64, vec: &[f32], timestamp: i64, importance: f32, context_type: u8, source: Option<&str>, content: Option<&str>) {
        let c_source = source.and_then(|s| std::ffi::CString::new(s).ok());
        let c_content = content.and_then(|s| std::ffi::CString::new(s).ok());
        
        unsafe {
            feather_add_with_meta(
                self.0, id, vec.as_ptr(), vec.len(),
                timestamp, importance, context_type,
                c_source.map_or(std::ptr::null(), |s| s.as_ptr()),
                c_content.map_or(std::ptr::null(), |s| s.as_ptr())
            )
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> (Vec<u64>, Vec<f32>) {
        let mut ids = vec![0u64; k];
        let mut dists = vec![0f32; k];
        unsafe {
            feather_search(self.0, query.as_ptr(), query.len(), k, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        (ids, dists)
    }

    pub fn search_with_filter(&self, query: &[f32], k: usize, type_filter: Option<u8>, source_filter: Option<&str>) -> (Vec<u64>, Vec<f32>) {
        let mut ids = vec![0u64; k];
        let mut dists = vec![0f32; k];
        let c_source = source_filter.and_then(|s| std::ffi::CString::new(s).ok());
        
        unsafe {
            feather_search_with_filter(
                self.0, query.as_ptr(), query.len(), k,
                type_filter.unwrap_or(255),
                c_source.map_or(std::ptr::null(), |s| s.as_ptr()),
                ids.as_mut_ptr(), dists.as_mut_ptr()
            )
        };
        (ids, dists)
    }

    pub fn save(&self) { unsafe { feather_save(self.0) } }
}

impl Drop for DB {
    fn drop(&mut self) { unsafe { feather_close(self.0) } }
}
