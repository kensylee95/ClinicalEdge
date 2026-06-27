import { useState, useEffect } from "react";
import { runInference } from "../lib/llama";
import { insertRecord } from "../lib/db";

interface Props {
  transcript: string;
  serverOnline: boolean;
  onTpsUpdate: (tps: number) => void;
}

type Status = "idle" | "loading" | "success" | "error";

export default function TriagePanel({ transcript, serverOnline, onTpsUpdate }: Props) {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    if (transcript) setText(transcript);
  }, [transcript]);

  async function handleSubmit() {
    if (!text.trim() || !serverOnline) return;
    setStatus("loading");
    setError("");
    try {
      const { data, tps } = await runInference(text);
      if (!data) throw new Error("Schema verification failed — token sequence deviated from GBNF rules.");
      await insertRecord(data);
      if (tps != null) onTpsUpdate(tps);
      setStatus("success");
      setText("");
      setTimeout(() => setStatus("idle"), 3000);
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
    }
  }

  const canSubmit = serverOnline && text.trim().length > 0 && status !== "loading";

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">📋 Clinical Transcript</span>
        {status === "success" && (
          <span style={{ fontSize: "11px", color: "var(--accent)" }}>✓ Committed to ledger</span>
        )}
      </div>
      <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Whisper transcript appears here, or type manually…"
          rows={5}
        />
        {status === "loading" && (
          <div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "4px" }}>
              Running constrained inference via GBNF…
            </div>
            <div className="progress-wrap">
              <div className="progress-fill" style={{ width: "100%" }} />
            </div>
          </div>
        )}
        {status === "error" && (
          <div style={{ fontSize: "11px", color: "var(--danger)", padding: "8px 10px",
            background: "#e8445a11", borderRadius: "var(--radius)", border: "1px solid var(--danger)" }}>
            ❌ {error}
          </div>
        )}
        <button className="btn btn-primary btn-full" onClick={handleSubmit} disabled={!canSubmit}>
          {status === "loading" ? "Processing…" : "Submit to Triage Ledger"}
        </button>
        {!serverOnline && (
          <p style={{ fontSize: "11px", color: "var(--text-muted)", textAlign: "center" }}>
            llama-server offline — start the sidecar first.
          </p>
        )}
      </div>
    </div>
  );
}