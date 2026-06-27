import { useState } from "react";
import WhisperPanel from "./components/WhisperPanel";
import TriagePanel from "./components/TrianglePanel";
import LedgerPanel from "./components/LedgerPanel";
import TelemetryBar from "./components/TelemetryBar";
import { useLlamaHealth } from "./hooks/useLlamaHealth";
import "./App.css";

export default function App() {
  const [transcript, setTranscript] = useState("");
  const [tpsHistory, setTpsHistory] = useState<number[]>([]);
  const { online } = useLlamaHealth();

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="logo-mark">CE</span>
          <div>
            <h1>ClinicalEdge</h1>
            <p className="header-sub">Sovereign Medical Intake Node</p>
          </div>
        </div>
        <div className={`status-pill ${online ? "online" : "offline"}`}>
          <span className="status-dot" />
          {online ? "llama-server connected" : "llama-server offline"}
        </div>
      </header>

      <TelemetryBar tpsHistory={tpsHistory} />

      <main className="app-main">
        <section className="panel panel-left">
          <WhisperPanel onTranscript={setTranscript} />
          <TriagePanel
            transcript={transcript}
            serverOnline={online}
            onTpsUpdate={(tps) => setTpsHistory((h) => [...h.slice(-59), tps])}
          />
        </section>

        <section className="panel panel-right">
          <LedgerPanel />
        </section>
      </main>
    </div>
  );
}