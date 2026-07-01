import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { RiSunLine, RiMoonLine } from "react-icons/ri";
import WhisperPanel from "./components/WhisperPanel";
import TriagePanel from "./components/TriagePanel";
import LedgerPanel from "./components/LedgerPanel";
import TelemetryBar from "./components/TelemetryBar";
import SchemaDrawer from "./components/SchemaDrawer";
import { useLlamaHealth } from "./hooks/useLlamaHealth";
import { inferFields, parseUserJson } from "./lib/inferSchema";
import "./App.css";

export default function App() {
  const [transcript, setTranscript] = useState("");
  const [tpsHistory, setTpsHistory] = useState<number[]>([]);
  const [dark, setDark] = useState(true);
  const [schemaOpen, setSchemaOpen] = useState(false);
  const [rawJson, setRawJson] = useState<string | null>(null);
  const { online } = useLlamaHealth();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }, [dark]);

  // Derive field count for the header badge without re-running inferFields on every render
  const fieldCount = rawJson
    ? (() => { try { return inferFields(parseUserJson(rawJson)).length; } catch { return 0; } })()
    : 0;

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="logo-mark">CE</span>
          <div>
            <h1>Structra</h1>
            <p className="header-sub">— Unstructured input. Structured records. Zero cloud.</p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>

          {/* ── Schema trigger ── */}
          <motion.button
            onClick={() => setSchemaOpen(true)}
            whileTap={{ scale: 0.92 }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 7,
              padding: "5px 12px",
              border: "1px solid var(--border)",
              borderRadius: 99,
              background: rawJson ? "var(--accent-dim)" : "var(--surface-2)",
              cursor: "pointer",
              color: rawJson ? "var(--accent)" : "var(--text-dim)",
              fontSize: 12,
              fontFamily: "var(--font-mono)",
              borderColor: rawJson ? "var(--accent)" : "var(--border)",
              transition: "all 0.2s",
            }}
          >
            ⚙ {rawJson ? `Schema (${fieldCount} fields)` : "Schema"}
          </motion.button>

          {/* ── Theme toggle ── */}
          <motion.button
            onClick={() => setDark(d => !d)}
            whileTap={{ scale: 0.92 }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 7,
              padding: "5px 10px 5px 6px",
              border: "1px solid var(--border)",
              borderRadius: 99,
              background: "var(--surface-2)",
              cursor: "pointer",
              color: "var(--text-dim)",
              fontSize: 12,
              fontFamily: "var(--font-mono)",
            }}
          >
            <div style={{
              position: "relative",
              width: 40, height: 22,
              background: dark ? "var(--accent)" : "var(--border)",
              borderRadius: 99,
              transition: "background 0.3s",
              flexShrink: 0,
            }}>
              <motion.div
                animate={{ x: dark ? 20 : 2 }}
                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                style={{
                  position: "absolute",
                  top: 3, left: 0,
                  width: 16, height: 16,
                  borderRadius: "50%",
                  background: "#fff",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
                }}
              >
                <motion.div
                  animate={{ rotate: dark ? 0 : 180, opacity: 1 }}
                  transition={{ duration: 0.3 }}
                  style={{ display: "flex", alignItems: "center", justifyContent: "center" }}
                >
                  {dark
                    ? <RiMoonLine size={10} color="#0a0d12" />
                    : <RiSunLine  size={10} color="#1a6fc4" />
                  }
                </motion.div>
              </motion.div>
            </div>

            <motion.span
              key={dark ? "dark" : "light"}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2 }}
              style={{ color: "var(--text-dim)", userSelect: "none" }}
            >
              {dark ? "Dark" : "Light"}
            </motion.span>
          </motion.button>

          {/* ── Server status ── */}
          <div className={`status-pill ${online ? "online" : "offline"}`}>
            <span className="status-dot" />
            {online ? "llama-server connected" : "llama-server offline"}
          </div>
        </div>
      </header>

      <TelemetryBar tpsHistory={tpsHistory} />

      <main className="app-main">
        <section className="panel panel-left">
          <WhisperPanel onTranscript={setTranscript} />
          <TriagePanel
            transcript={transcript}
            serverOnline={online}
            rawJson={rawJson}
            onTpsUpdate={(tps) => setTpsHistory((h) => [...h.slice(-59), tps])}
          />
        </section>

        <section className="panel panel-right">
          <LedgerPanel />
        </section>
      </main>

      <SchemaDrawer
        open={schemaOpen}
        onClose={() => setSchemaOpen(false)}
        onApply={setRawJson}
      />
    </div>
  );
}