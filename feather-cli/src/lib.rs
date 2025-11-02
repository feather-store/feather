use std::ffi::{c_void, c_char};
use std::path::Path;

#[repr(C)]
pub struct DB(*mut c_void);  // ← `pub`

extern "C" {
    fn feather_open(path: *const c_char, dim: usize) -> *mut c_void;
    fn feather_add(db: *mut c_void, id: u64, vec: *const f32, len: usize);
    fn feather_search(db: *mut c_void, query: *const f32, len: usize, k: usize,
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

    pub fn search(&self, query: &[f32], k: usize) -> (Vec<u64>, Vec<f32>) {
        let mut ids = vec![0u64; k];
        let mut dists = vec![0f32; k];
        unsafe {
            feather_search(self.0, query.as_ptr(), query.len(), k, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        (ids, dists)
    }

    pub fn save(&self) { unsafe { feather_save(self.0) } }
}

impl Drop for DB {
    fn drop(&mut self) { unsafe { feather_close(self.0) } }
}
