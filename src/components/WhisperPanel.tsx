// src/components/WhisperPanel.tsx
import { invoke } from "@tauri-apps/api/core";
import { useRef, useState, useCallback } from "react";

const MAX_SECONDS = 120;

interface Props {
  onTranscript: (text: string) => void;
}

export default function WhisperPanel({ onTranscript }: Props) {
  const [status, setStatus] = useState("✅ Ready. Record or drop a file.");
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);

  const mediaRecRef     = useRef<MediaRecorder | null>(null);
  const chunksRef       = useRef<Blob[]>([]);
  const onTranscriptRef = useRef(onTranscript);
  onTranscriptRef.current = onTranscript;

  const transcribeBlob = useCallback(async (blob: Blob, label: string) => {
    setTranscribing(true);
    setStatus(`⏳ Transcribing ${label}…`);

    try {
      const arrayBuffer = await blob.arrayBuffer();
      const bytes = Array.from(new Uint8Array(arrayBuffer));

      const text = await invoke<string>("transcribe_audio", { bytes });

      if (text) {
        setStatus(`✅ Done — ${text.slice(0, 80)}${text.length > 80 ? "…" : ""}`);
        onTranscriptRef.current(text);
      } else {
        setStatus("⚠️ No speech detected.");
      }
    } catch (err) {
      console.error("transcribe_audio failed:", err);
      setStatus(`❌ ${(err as Error).message}`);
    } finally {
      setTranscribing(false);
    }
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")
        ? "audio/ogg;codecs=opus"
        : "audio/mp4";

      const mr = new MediaRecorder(stream, { mimeType });
      mediaRecRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data?.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: mr.mimeType });
        transcribeBlob(blob, "recording");
      };

      mr.start();
      setRecording(true);
      setStatus("🔴 Recording… click Stop when done.");

      setTimeout(() => {
        if (mediaRecRef.current === mr && mr.state === "recording") {
          mr.stop();
          setRecording(false);
        }
      }, MAX_SECONDS * 1000);
    } catch (err) {
      setStatus(`❌ Mic error: ${(err as Error).message}`);
    }
  }, [transcribeBlob]);

  const stopRecording = useCallback(() => {
    mediaRecRef.current?.stop();
    setRecording(false);
    setStatus("Processing audio…");
  }, []);

  const handleFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      transcribeBlob(file, file.name);
      e.target.value = "";
    },
    [transcribeBlob]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (file) transcribeBlob(file, file.name);
    },
    [transcribeBlob]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const busy = transcribing;

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">🎙 Audio Intake — Native Whisper</span>
      </div>

      <div
        className="card-body"
        style={{ display: "flex", flexDirection: "column", gap: 8, padding: "12px" }}
      >
        <button
          className={`btn btn-full ${recording ? "btn-danger" : "btn-primary"}`}
          disabled={busy && !recording}
          onClick={recording ? stopRecording : startRecording}
        >
          {recording ? "🛑 Stop Recording" : "🎙️ Start Recording"}
        </button>

        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          style={{
            border: "1.5px dashed var(--border-color, #555)",
            borderRadius: 6,
            padding: "10px 12px",
            textAlign: "center",
            fontSize: 11,
            color: "var(--text-muted)",
            cursor: busy ? "not-allowed" : "pointer",
            opacity: busy ? 0.5 : 1,
            transition: "opacity 0.2s",
          }}
        >
          Drop audio / video here
          <label
            style={{
              display: "block",
              marginTop: 4,
              color: "var(--text-dim)",
              cursor: busy ? "not-allowed" : "pointer",
            }}
          >
            <input
              type="file"
              accept="audio/*,video/*"
              style={{ display: "none" }}
              disabled={busy}
              onChange={handleFile}
            />
            or <span style={{ textDecoration: "underline" }}>browse a file</span>
          </label>
        </div>

        <div
          style={{
            fontSize: 11,
            color: "var(--text-muted)",
            lineHeight: 1.4,
            minHeight: 16,
            wordBreak: "break-word",
          }}
        >
          {status}
        </div>

        <p style={{ fontSize: 10, color: "var(--text-muted)", margin: 0 }}>
          Runs fully offline — audio never leaves this device.
        </p>
      </div>
    </div>
  );
}