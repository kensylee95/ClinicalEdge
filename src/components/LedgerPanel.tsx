import { useState, useEffect } from "react";
import { getRecords, getFields, type LedgerRow } from "../lib/db";
import type { FlatField } from "../lib/inferSchema";

export default function LedgerPanel() {
  const [records, setRecords] = useState<LedgerRow[]>([]);
  const [fields, setFields] = useState<FlatField[]>([]);
  const [loading, setLoading] = useState(true);

  async function refresh() {
    try {
      const rows = await getRecords();
      setRecords(rows);
      try { setFields(getFields()); } catch { /* schema not init yet */ }
    } catch {
      // schema not initialised yet — silent, will retry
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="card" style={{ flex: 1 }}>
      <div className="card-header">
        <span className="card-title">📊 Records</span>
        <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {records.length} row{records.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="card-body" style={{ padding: 0, overflowX: "auto" }}>
        {loading ? (
          <div className="empty-state"><div className="empty-state-icon">⏳</div>Loading…</div>
        ) : records.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🗂</div>
            No records yet. Define a schema and submit some input.
          </div>
        ) : (
          <GenericTable records={records} fields={fields} />
        )}
      </div>
    </div>
  );
}

// ── Generic table ─────────────────────────────────────────────────
function GenericTable({ records, fields }: { records: LedgerRow[]; fields: FlatField[] }) {
  // Column order: id, timestamp, then schema fields in definition order,
  // falling back to whatever keys are on the first row if fields aren't loaded yet.
  const schemaCols = fields.length
    ? fields.map(f => f.key)
    : Object.keys(records[0]).filter(k => k !== "id" && k !== "timestamp");

  const cols = ["id", "timestamp", ...schemaCols];

  return (
    <table className="ledger-table">
      <thead>
        <tr>
          {cols.map(col => (
            <th key={col} className="ledger-th">
              {col.replace(/_/g, " ")}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {records.map(row => (
          <tr key={row.id} className="ledger-tr">
            {cols.map(col => (
              <td key={col} className="ledger-td">
                <CellValue value={row[col]} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Cell renderer ─────────────────────────────────────────────────
function CellValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span style={{ color: "var(--text-muted)" }}>—</span>;
  }

  // JSON-stringified arrays stored as TEXT
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.startsWith("[")) {
      try {
        const arr = JSON.parse(trimmed) as unknown[];
        return (
          <div className="action-chips">
            {arr.map((item, i) => (
              <span key={i} className="action-chip">{String(item)}</span>
            ))}
          </div>
        );
      } catch { /* fall through to plain string */ }
    }
    return <span>{value}</span>;
  }

  if (typeof value === "number") {
    return <span style={{ fontFamily: "var(--font-mono)", fontSize: "12px" }}>{value}</span>;
  }

  // booleans stored as 0/1
  if (typeof value === "number" && (value === 0 || value === 1)) {
    return <span style={{ color: value ? "var(--accent)" : "var(--text-muted)" }}>
      {value ? "true" : "false"}
    </span>;
  }

  return <span>{String(value)}</span>;
}

/* ── Scoped styles ───────────────────────────────────────────────── */
const tableStyles = `
  .ledger-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .ledger-th {
    padding: 8px 12px;
    text-align: left;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-dim);
    background: var(--surface-2);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    position: sticky;
    top: 0;
  }
  .ledger-tr { border-bottom: 1px solid var(--border); }
  .ledger-tr:last-child { border-bottom: none; }
  .ledger-tr:hover { background: var(--surface-2); }
  .ledger-td {
    padding: 8px 12px;
    color: var(--text);
    vertical-align: top;
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ledger-td:first-child {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    width: 40px;
  }
  .ledger-td:nth-child(2) {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
  }
`;

// Inject styles once
if (typeof document !== "undefined" && !document.getElementById("ledger-styles")) {
  const tag = document.createElement("style");
  tag.id = "ledger-styles";
  tag.textContent = tableStyles;
  document.head.appendChild(tag);
}