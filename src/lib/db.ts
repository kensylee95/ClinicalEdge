import Database from "@tauri-apps/plugin-sql";
import type { TriageRecord } from "./llama";

export interface LedgerRow {
  id: number;
  patient_name: string;
  age: number;
  primary_symptom: string;
  duration_days: number;
  triage_priority: "EMERGENCY" | "URGENT" | "ROUTINE";
  action_items: string;
  timestamp: string;
}

let _db: Database | null = null;

async function getDb(): Promise<Database> {
  if (_db) return _db;
  _db = await Database.load("sqlite:clinical_records.db");
  await _db.execute(`
    CREATE TABLE IF NOT EXISTS triage_ledger (
      id               INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_name     TEXT NOT NULL,
      age              INTEGER NOT NULL,
      primary_symptom  TEXT NOT NULL,
      duration_days    INTEGER NOT NULL,
      triage_priority  TEXT NOT NULL,
      action_items     TEXT NOT NULL,
      timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);
  return _db;
}

export async function insertRecord(parsed: TriageRecord): Promise<void> {
  const db = await getDb();
  await db.execute(
    `INSERT INTO triage_ledger
       (patient_name, age, primary_symptom, duration_days, triage_priority, action_items)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [
      String(parsed.patient_name ?? "Unknown"),
      Number(parsed.age ?? 0),
      String(parsed.primary_symptom ?? "Unspecified"),
      Number(parsed.duration_days ?? 0),
      String(parsed.triage_priority ?? "ROUTINE"),
      JSON.stringify(parsed.clinical_action_items ?? []),
    ]
  );
}

export async function getRecords(): Promise<LedgerRow[]> {
  const db = await getDb();
  return await db.select<LedgerRow[]>(
    `SELECT id, patient_name, age, primary_symptom, duration_days,
            triage_priority, action_items, timestamp
     FROM triage_ledger ORDER BY id DESC`
  );
}