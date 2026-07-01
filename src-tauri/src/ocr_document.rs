// src-tauri/src/ocr_document.rs
//
// STUB. Placeholder for the printed/structured-document OCR pipeline
// (receipts, invoices, forms) — intended to be a PaddleOCR-based
// detection + recognition (+ optional PP-Structure table) pipeline,
// replacing DBNet+TrOCR for this mode. See chat history / HANDOFF.md for
// why TrOCR (handwriting or printed) isn't suitable for multi-column
// layouts: it's a single-line recognizer with no concept of table
// structure, and a plain top-to-bottom y-sort scrambles multi-column rows.
//
// This stub exists only so the frontend's mode toggle
// (OcrModal.tsx -> run_ocr_page_document) has something real to call
// while the actual pipeline is being built. It deliberately returns an
// error rather than empty/fake text, so a tester sees an honest "not
// implemented" message in the modal's error stage instead of silently
// getting a blank or misleading result.
//
// Tauri command React calls via:
//   await invoke("run_ocr_page_document", { imagePath: "/path/to/receipt.png" })

#[tauri::command]
pub fn run_ocr_page_document(image_path: String) -> Result<String, String> {
    // Touch the argument so it's not flagged unused once a real
    // implementation reads it; harmless for a stub.
    let _ = &image_path;

    Err(
        "Document/receipt OCR isn't implemented yet — only Handwriting mode works right now."
            .to_string(),
    )
}
