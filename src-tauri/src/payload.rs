// payload.rs
//
// Defines the on-disk footer format that lets a single compiled executable
// carry extra binary blobs (the three llama-server CPU-tier variants + the
// .gguf model) appended after its own code, and extracts them at first run.
//
// Layout of the final shipped file:
//
//   [ original tauri binary bytes ............................. ]
//   [ engine-avx2 bytes ......................................... ]
//   [ engine-avx  bytes ......................................... ]
//   [ engine-noavx bytes ........................................ ]
//   [ model (.gguf) bytes ....................................... ]
//   [ FOOTER (fixed size, see Footer::SIZE) ..................... ]
//
// The footer sits at the very end so we can find it with a single seek to
// `file_len - Footer::SIZE`, without scanning the multi-gigabyte file.
//
// Each blob's (offset, length) is stored in the footer as absolute byte
// offsets into the file. Offsets are relative to byte 0 of the file (i.e.
// they include the leading app-binary bytes), so extraction is a plain
// seek+read on the running executable's own path.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"CLINEDG1"; // bump trailing digit if format changes

#[derive(Debug, Clone, Copy)]
pub struct BlobRange {
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Footer {
    pub engine_avx2: BlobRange,
    pub engine_avx: BlobRange,
    pub engine_noavx: BlobRange,
    pub model: BlobRange,
}

impl Footer {
    // magic(8) + 4 ranges * (offset:8 + len:8) + format_version(4) = 8 + 64 + 4 = 76
    pub const SIZE: u64 = 76;

    /// Reads and validates the footer from the file at `exe_path` (normally
    /// `std::env::current_exe()`). Returns None if the file is too small or
    /// the magic bytes don't match — meaning this isn't a packed executable
    /// (e.g. running under `cargo run` / `tauri dev`).
    pub fn read_from(exe_path: &Path) -> Result<Option<Footer>, String> {
        let mut file = File::open(exe_path).map_err(|e| format!("open exe: {e}"))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("stat exe: {e}"))?
            .len();

        if file_len < Self::SIZE {
            return Ok(None);
        }

        file.seek(SeekFrom::Start(file_len - Self::SIZE))
            .map_err(|e| format!("seek to footer: {e}"))?;

        let mut buf = vec![0u8; Self::SIZE as usize];
        file.read_exact(&mut buf).map_err(|e| format!("read footer: {e}"))?;

        if &buf[0..8] != MAGIC {
            return Ok(None);
        }

        let mut pos = 8usize;
        let mut next_u64 = || {
            let v = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            pos += 8;
            v
        };

        let engine_avx2 = BlobRange { offset: next_u64(), len: next_u64() };
        let engine_avx = BlobRange { offset: next_u64(), len: next_u64() };
        let engine_noavx = BlobRange { offset: next_u64(), len: next_u64() };
        let model = BlobRange { offset: next_u64(), len: next_u64() };

        let _format_version = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());

        // Sanity: every declared range must fit inside the file, before the footer itself.
        for r in [engine_avx2, engine_avx, engine_noavx, model] {
            if r.offset + r.len > file_len - Self::SIZE {
                return Err(format!(
                    "footer declares range [{}, {}) but file usable length is {}",
                    r.offset,
                    r.offset + r.len,
                    file_len - Self::SIZE
                ));
            }
        }

        Ok(Some(Footer { engine_avx2, engine_avx, engine_noavx, model }))
    }

    pub fn range_for(&self, engine_name: &str) -> BlobRange {
        match engine_name {
            "avx2" => self.engine_avx2,
            "avx" => self.engine_avx,
            _ => self.engine_noavx,
        }
    }
}

/// Copies `range` of bytes from `src_path` into a freshly created file at
/// `dest_path`, streaming in chunks so we never hold the whole (potentially
/// multi-gigabyte) blob in memory at once.
pub fn extract_range(src_path: &Path, range: BlobRange, dest_path: &Path) -> Result<(), String> {
    const CHUNK: usize = 8 * 1024 * 1024; // 8 MiB

    let mut src = File::open(src_path).map_err(|e| format!("open source: {e}"))?;
    src.seek(SeekFrom::Start(range.offset))
        .map_err(|e| format!("seek source: {e}"))?;

    // Write to a temp file first, then atomically rename into place, so a
    // crash/power-loss mid-extraction (very real concern on hospital
    // hardware) can't leave a half-written engine/model that looks "present".
    let tmp_path = dest_path.with_extension("partial");
    let mut dest = File::create(&tmp_path).map_err(|e| format!("create dest: {e}"))?;

    let mut remaining = range.len;
    let mut buf = vec![0u8; CHUNK];
    while remaining > 0 {
        let to_read = remaining.min(CHUNK as u64) as usize;
        src.read_exact(&mut buf[..to_read])
            .map_err(|e| format!("read chunk: {e}"))?;
        std::io::Write::write_all(&mut dest, &buf[..to_read])
            .map_err(|e| format!("write chunk: {e}"))?;
        remaining -= to_read as u64;
    }

    dest.sync_all().map_err(|e| format!("flush dest: {e}"))?;
    drop(dest);

    std::fs::rename(&tmp_path, dest_path).map_err(|e| format!("finalize dest: {e}"))?;
    println!("Extracted {} bytes to {:?}", range.len, dest_path);
    Ok(())
}