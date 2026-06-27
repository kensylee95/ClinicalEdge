import { useState, useEffect } from "react";
import { getRecords, type LedgerRow } from "../lib/db";

const PRIORITY_ORDER: Record<string, number> = { EMERGENCY: 0, URGENT: 1, ROUTINE: 2 };

export default function LedgerPanel() {
  const [records, setRecords] = useState<LedgerRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("ALL");

  async function refresh() {
    try {
      const rows = await getRecords();
      setRecords(rows);
    } catch (e) {
      console.error("Ledger read error:", e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, []);

  const filtered = filter === "ALL" ? records : records.filter(r => r.triage_priority === filter);
  const sorted = [...filtered].sort(
    (a, b) => (PRIORITY_ORDER[a.triage_priority] ?? 3) - (PRIORITY_ORDER[b.triage_priority] ?? 3)
  );

  return (
    <div className="card" style={{ flex: 1 }}>
      <div className="card-header">
        <span className="card-title">📊 Triage Ledger</span>
        <div style={{ display: "flex", gap: "6px" }}>
          {(["ALL", "EMERGENCY", "URGENT", "ROUTINE"] as const).map(f => (
            <button key={f} className="btn btn-ghost" onClick={() => setFilter(f)}
              style={{
                padding: "3px 8px", fontSize: "10px",
                background: filter === f ? "var(--accent-dim)" : undefined,
                borderColor: filter === f ? "var(--accent)" : undefined,
                color: filter === f ? "var(--accent)" : undefined,
              }}>
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="card-body" style={{ padding: "0 16px" }}>
        {loading ? (
          <div className="empty-state"><div className="empty-state-icon">⏳</div>Loading records…</div>
        ) : sorted.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🗂</div>
            No records yet. Submit a transcript to populate the ledger.
          </div>
        ) : (
          sorted.map(entry => <LedgerEntry key={entry.id} entry={entry} />)
        )}
      </div>
    </div>
  );
}

function LedgerEntry({ entry }: { entry: LedgerRow }) {
  let actions: string[] = [];
  try { actions = JSON.parse(entry.action_items); } catch { /* empty */ }

  const icon = entry.triage_priority === "EMERGENCY" ? "🚨" : entry.triage_priority === "URGENT" ? "⚠️" : "ℹ️";

  return (
    <div className="ledger-entry">
      <div className="ledger-entry-header">
        <span className="patient-name">{entry.patient_name}</span>
        <span className="patient-meta">&nbsp;· {entry.age} y/o</span>
        <span className={`priority-badge ${entry.triage_priority}`}>
          {icon} {entry.triage_priority}
        </span>
      </div>
      <div className="ledger-symptom">
        {entry.primary_symptom} &mdash; {entry.duration_days} day{entry.duration_days !== 1 ? "s" : ""}
      </div>
      {actions.length > 0 && (
        <div className="action-chips">
          {actions.map((a, i) => <span key={i} className="action-chip">{a}</span>)}
        </div>
      )}
      <div className="ledger-timestamp">{entry.timestamp}</div>
    </div>
  );
}