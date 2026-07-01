// src/components/OcrModal.tsx

import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";
import { tempDir } from "@tauri-apps/api/path";
import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RiFileImageLine, RiCloseLine, RiScanLine,
  RiCheckLine, RiFileCopyLine,
} from "react-icons/ri";
type Mode = "handwriting" | "document";
interface Props {
  isOpen: boolean;
  onClose: () => void;
  onTranscript: (text: string) => void;
  initialMode?: Mode;
}

type Stage = "idle" | "scanning" | "done" | "error";

export default function OcrModal({ isOpen, onClose, onTranscript, initialMode }: Props) {
  const [mode, setMode] = useState<Mode>(initialMode ?? "handwriting");
  const [stage, setStage] = useState<Stage>("idle");
  const [fileName, setFileName] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [dragOver, setDragOver] = useState(false);
  const [copied, setCopied] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) setMode(initialMode ?? "handwriting");
  }, [isOpen, initialMode]);

  const reset = useCallback(() => {
    setStage("idle");
    setFileName(null);
    setResult("");
    setErrorMsg("");
    setCopied(false);
  }, []);

  const handleClose = useCallback(() => {
    reset();
    onClose();
  }, [reset, onClose]);

  // Single method: always invoke run_ocr_page with a real file path.
  // Zero image data ever crosses the IPC bridge.
  const runOcrFromPath = useCallback(async (filePath: string, name: string) => {
    setFileName(name);
    setStage("scanning");
    try {
      const command = mode === "handwriting" ? "run_ocr_page_handwriting" : "run_ocr_page_document";
      /*const command =
  mode === "handwriting"
    ? "run_ocr"               // <-- temporary
    : "run_ocr_page_document";*/
      const text = await invoke<string>(command, { imagePath: filePath });
      setResult(text ?? "");
      setStage("done");
    } catch (e) {
      setErrorMsg(String(e));
      setStage("error");
    }
  }, [mode]);

  // Dialog pick — OS gives us a real path directly.
  const pickFile = useCallback(async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "bmp", "tiff", "webp"] }],
      });
      if (!selected || typeof selected !== "string") return;
      const name = selected.split(/[\\/]/).pop() ?? selected;
      await runOcrFromPath(selected, name);
    } catch {
      // Dialog plugin unavailable — fall back to the hidden file input.
      fileInputRef.current?.click();
    }
  }, [runOcrFromPath]);

  // Drag-drop — write bytes to a temp file so Rust can open it by path,
  // same as the dialog flow. Temp file is tiny overhead vs. serializing
  // the whole image over JSON.
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    const ext = file.name.split(".").pop() ?? "png";
    const tmp = await tempDir();
    const tmpPath = `${tmp}ocr_drop_${Date.now()}.${ext}`;

    const bytes = new Uint8Array(await file.arrayBuffer());
    await writeFile(tmpPath, bytes);
    await runOcrFromPath(tmpPath, file.name);
  }, [runOcrFromPath]);

  // Hidden input fallback — same write-to-temp approach.
  const handleFileInput = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split(".").pop() ?? "png";
    const tmp = await tempDir();
    const tmpPath = `${tmp}ocr_input_${Date.now()}.${ext}`;

    const bytes = new Uint8Array(await file.arrayBuffer());
    await writeFile(tmpPath, bytes);
    await runOcrFromPath(tmpPath, file.name);
    e.target.value = "";
  }, [runOcrFromPath]);

  const copyToClipboard = useCallback(async () => {
    await navigator.clipboard.writeText(result);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [result]);

  const useText = useCallback(() => {
    onTranscript(result);
    handleClose();
  }, [result, onTranscript, handleClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={handleClose}
            style={{
              position: "fixed", inset: 0,
              background: "rgba(0,0,0,0.55)",
              backdropFilter: "blur(4px)",
              zIndex: 100,
            }}
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            style={{
              position: "fixed", inset: 0, margin: "auto",
              width: "min(560px, 92vw)",
              height: "fit-content", maxHeight: "85vh",
              zIndex: 101,
              display: "flex", flexDirection: "column",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 16, overflow: "hidden",
              boxShadow: "0 24px 64px rgba(0,0,0,0.35)",
            }}
          >
            {/* Header */}
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "14px 18px",
              background: "var(--surface-2)",
              borderBottom: "1px solid var(--border)",
              flexShrink: 0,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <RiScanLine size={18} color="var(--accent)" />
                <span style={{
                  fontSize: 13, fontWeight: 700,
                  letterSpacing: "0.04em", textTransform: "uppercase",
                  color: "var(--text-dim)",
                }}>
                  {mode === "handwriting" ? "Upload handwritten document" : "Upload typed document"}
                </span>
              </div>
              <button
                onClick={handleClose}
                style={{
                  background: "none", border: "none",
                  cursor: "pointer", color: "var(--text-muted)",
                  display: "flex", alignItems: "center",
                  borderRadius: 6, padding: 4, transition: "color 0.15s",
                }}
                onMouseEnter={e => (e.currentTarget.style.color = "var(--text)")}
                onMouseLeave={e => (e.currentTarget.style.color = "var(--text-muted)")}
              >
                <RiCloseLine size={20} />
              </button>
            </div>

            {/* Body */}
            <div style={{ padding: 20, overflowY: "auto", flex: 1 }}>
              <AnimatePresence mode="wait">

                {stage === "idle" && (
                  <motion.div
                    key="idle"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    <motion.div
                      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                      onDragLeave={() => setDragOver(false)}
                      onDrop={handleDrop}
                      animate={{
                        borderColor: dragOver ? "var(--accent)" : "var(--border)",
                        background: dragOver ? "var(--accent-dim)" : "var(--surface-2)",
                      }}
                      onClick={pickFile}
                      style={{
                        border: "2px dashed var(--border)",
                        borderRadius: 12, padding: "40px 24px",
                        display: "flex", flexDirection: "column",
                        alignItems: "center", gap: 12,
                        cursor: "pointer",
                        transition: "border-color 0.2s, background 0.2s",
                      }}
                    >
                      <RiFileImageLine size={36} color="var(--accent)" style={{ opacity: 0.7 }} />
                      <div style={{ textAlign: "center" }}>
                        <p style={{ fontWeight: 600, fontSize: 14, color: "var(--text)" }}>
                          Drop a {mode==="handwriting"?"handwritten":"typed"} textual image
                        </p>
                        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                          PNG, JPG, BMP, TIFF, WebP — or{" "}
                          <span style={{ color: "var(--accent)", textDecoration: "underline" }}>
                            browse
                          </span>
                        </p>
                      </div>
                      <p style={{
                        fontSize: 10, color: "var(--text-muted)",
                        fontFamily: "var(--font-mono)",
                        letterSpacing: "0.06em", textTransform: "uppercase",
                        marginTop: 4,
                      }}>
                        fully offline · never leaves this device
                      </p>
                    </motion.div>

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      style={{ display: "none" }}
                      onChange={handleFileInput}
                    />
                  </motion.div>
                )}

                {stage === "scanning" && (
                  <motion.div
                    key="scanning"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    style={{
                      display: "flex", flexDirection: "column",
                      alignItems: "center", gap: 20, padding: "32px 0",
                    }}
                  >
                    <div style={{ position: "relative", width: 72, height: 72 }}>
                      <div style={{
                        width: 72, height: 72, borderRadius: 12,
                        background: "var(--accent-dim)",
                        border: "1px solid var(--accent)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                      }}>
                        <RiFileImageLine size={30} color="var(--accent)" />
                      </div>
                      <motion.div
                        animate={{ top: ["10%", "85%", "10%"] }}
                        transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                        style={{
                          position: "absolute", left: 4, right: 4,
                          height: 2, background: "var(--accent)",
                          borderRadius: 2, boxShadow: "0 0 8px var(--accent)",
                        }}
                      />
                    </div>

                    <div style={{ textAlign: "center" }}>
                      <p style={{ fontWeight: 600, color: "var(--text)", fontSize: 14 }}>
                        {mode === "handwriting" ? "Reading handwriting…" : "Reading document…"}
                      </p>
                      {fileName && (
                        <p style={{
                          fontSize: 11, color: "var(--text-muted)",
                          fontFamily: "var(--font-mono)", marginTop: 4,
                        }}>
                          {fileName}
                        </p>
                      )}
                    </div>

                    <div style={{
                      width: "100%", maxWidth: 280,
                      height: 3, background: "var(--border)",
                      borderRadius: 99, overflow: "hidden",
                    }}>
                      <motion.div
                        animate={{ x: ["-100%", "200%"] }}
                        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
                        style={{
                          width: "50%", height: "100%",
                          background: "var(--accent)", borderRadius: 99,
                        }}
                      />
                    </div>

                    <p style={{
                      fontSize: 10, color: "var(--text-muted)",
                      fontFamily: "var(--font-mono)",
                      letterSpacing: "0.08em", textTransform: "uppercase",
                    }}>
                      detecting lines · transcribing
                    </p>
                  </motion.div>
                )}

                {stage === "done" && (
                  <motion.div
                    key="done"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    style={{ display: "flex", flexDirection: "column", gap: 14 }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <RiCheckLine size={16} color="#22c98e" />
                      <span style={{ fontSize: 12, color: "#22c98e", fontFamily: "var(--font-mono)" }}>
                        Transcription complete
                      </span>
                      {fileName && (
                        <span style={{
                          fontSize: 11, color: "var(--text-muted)",
                          fontFamily: "var(--font-mono)", marginLeft: "auto",
                        }}>
                          {fileName}
                        </span>
                      )}
                    </div>

                    <textarea
                      value={result}
                      onChange={e => setResult(e.target.value)}
                      rows={10}
                      spellCheck
                      style={{ fontSize: 13, lineHeight: 1.7, resize: "vertical" }}
                    />

                    <p style={{ fontSize: 11, color: "var(--text-muted)" }}>
                      {mode === "handwriting"
                        ? "Review and correct before using — OCR on cursive handwriting isn't perfect."
                        : "Review and correct before using — complex layouts or tables may need manual fixes."}
                    </p>

                    <div style={{ display: "flex", gap: 8 }}>
                      <button className="btn btn-ghost" onClick={reset} style={{ flex: 0 }}>
                        Scan another
                      </button>
                      <button className="btn btn-ghost" onClick={copyToClipboard} style={{ flex: 0, gap: 6 }}>
                        {copied
                          ? <><RiCheckLine size={14} /> Copied</>
                          : <><RiFileCopyLine size={14} /> Copy</>
                        }
                      </button>
                      <button className="btn btn-primary btn-full" onClick={useText}>
                        Use this text
                      </button>
                    </div>
                  </motion.div>
                )}

                {stage === "error" && (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    style={{
                      display: "flex", flexDirection: "column",
                      alignItems: "center", gap: 16, padding: "32px 0",
                    }}
                  >
                    <p style={{ fontWeight: 600, color: "var(--danger)", fontSize: 14 }}>
                      Transcription failed
                    </p>
                    <p style={{
                      fontSize: 12, color: "var(--text-muted)",
                      fontFamily: "var(--font-mono)", textAlign: "center", maxWidth: 360,
                    }}>
                      {errorMsg}
                    </p>
                    <button className="btn btn-ghost" onClick={reset}>Try again</button>
                  </motion.div>
                )}

              </AnimatePresence>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}