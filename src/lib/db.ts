import Database from "@tauri-apps/plugin-sql";
import { inferFields, parseUserJson, type FlatField } from "./inferSchema";

// ── Active schema state ───────────────────────────────────────────
// Set once when the user applies their JSON schema.
// All subsequent insertions and reads use this.

let _db: Database | null = null;
let _tableName: string | null = null;
let _fields: FlatField[] | null = null;

async function getDb(): Promise<Database> {
  if (_db) return _db;
  _db = await Database.load("sqlite:records.db");
  return _db;
}

// ── Schema initialisation ─────────────────────────────────────────

/**
 * Call this when the user applies a JSON schema.
 * Derives the flat fields, creates the SQLite table if it doesn't exist,
 * and stores the field list for insertRecord / getRecords.
 *
 * Table name is derived from a slugified version of the JSON keys
 * (stable across reloads for the same schema shape).
 */
export async function initSchema(rawJson: string): Promise<void> {
  const obj = parseUserJson(rawJson);
  const fields = inferFields(obj);

  // Stable table name: hash of sorted field keys
  const keyHash = fields
    .map(f => f.key)
    .sort()
    .join("_")
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, "_")
    .slice(0, 48);
  const tableName = `rec_${keyHash}`;

  const db = await getDb();

  const columnDefs = fields
    .map(f => `  ${f.key} ${f.sqlType}`)
    .join(",\n");

  await db.execute(`
    CREATE TABLE IF NOT EXISTS ${tableName} (
      id        INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
${columnDefs}
    )
  `);

  _tableName = tableName;
  _fields    = fields;
}

// ── Insert ────────────────────────────────────────────────────────

/**
 * Insert a record. `parsed` is the flat JSON object the model produced
 * (already matching the user's schema). Nested arrays are JSON-stringified.
 */
export async function insertRecord(
  parsed: Record<string, unknown>
): Promise<void> {
  if (!_tableName || !_fields) {
    throw new Error("Schema not initialised — call initSchema() first.");
  }

  const db = await getDb();
  const cols = _fields.map(f => f.key).join(", ");
  const placeholders = _fields.map(() => "?").join(", ");

  const values = _fields.map(f => {
    const val = parsed[f.key] ?? null;
    // Arrays and any remaining objects → JSON string
    if (Array.isArray(val) || (typeof val === "object" && val !== null)) {
      return JSON.stringify(val);
    }
    // Booleans → 0/1 for SQLite
    if (typeof val === "boolean") return val ? 1 : 0;
    return val;
  });

  await db.execute(
    `INSERT INTO ${_tableName} (${cols}) VALUES (${placeholders})`,
    values
  );
}

// ── Read ──────────────────────────────────────────────────────────

export interface LedgerRow {
  id: number;
  timestamp: string;
  [key: string]: unknown;
}

export async function getRecords(): Promise<LedgerRow[]> {
  if (!_tableName) {
    throw new Error("Schema not initialised — call initSchema() first.");
  }
  const db = await getDb();
  return await db.select<LedgerRow[]>(
    `SELECT * FROM ${_tableName} ORDER BY id DESC`
  );
}

// ── Accessors ─────────────────────────────────────────────────────

export function getFields(): FlatField[] {
  if (!_fields) throw new Error("Schema not initialised.");
  return _fields;
}

export function getTableName(): string {
  if (!_tableName) throw new Error("Schema not initialised.");
  return _tableName;
}