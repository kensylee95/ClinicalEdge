import type { SchemaConfig } from "../components/SchemaDrawer";
import type { FieldType, ScalarType } from "../components/SchemaDrawer";

export interface FlatField {
  key: string;       // flattened key, e.g. "patient_name"
  type: FieldType;   // as SchemaDrawer FieldType
  sqlType: string;   // SQLite column type
}

/** Infer a scalar FieldType from a JSON value */
function inferScalar(value: unknown): ScalarType {
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return "number";
  if (typeof value === "string") {
    // ISO date heuristic: YYYY-MM-DD
    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return "date";
    return "string";
  }
  return "string"; // fallback
}

/** Infer array element scalar type from first element, fallback string */
function inferArrayScalar(arr: unknown[]): ScalarType {
  const first = arr.find(v => v !== null && v !== undefined);
  if (first === undefined) return "string";
  if (typeof first === "object") return "string"; // nested objects in arrays → stringify
  return inferScalar(first);
}

/** Flatten a JSON object into FlatField[], recursing into nested objects */
export function inferFields(
  obj: Record<string, unknown>,
  prefix = ""
): FlatField[] {
  const fields: FlatField[] = [];

  for (const [key, value] of Object.entries(obj)) {
    const flatKey = prefix ? `${prefix}_${key}` : key;

    if (Array.isArray(value)) {
      const inner = inferArrayScalar(value);
      fields.push({
        key: flatKey,
        type: `array<${inner}>`,
        sqlType: "TEXT", // stored as JSON string
      });
    } else if (value !== null && typeof value === "object") {
      // recurse into nested object
      fields.push(...inferFields(value as Record<string, unknown>, flatKey));
    } else {
      const scalar = inferScalar(value);
      const sqlType =
        scalar === "number"  ? "REAL" :
        scalar === "boolean" ? "INTEGER" : // SQLite has no bool
        "TEXT";
      fields.push({ key: flatKey, type: scalar, sqlType });
    }
  }

  return fields;
}

/** Build a SchemaConfig (for buildGrammar) from a parsed JSON object */
export function toSchemaConfig(obj: Record<string, unknown>): SchemaConfig {
  const fields = inferFields(obj);
  return {
    templateFile: null,
    fields: fields.map(f => ({ name: f.key, type: f.type })),
  };
}

/** Parse and validate user-pasted JSON, return the object or throw */
export function parseUserJson(raw: string): Record<string, unknown> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("Invalid JSON — please check your input.");
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("JSON must be an object { } at the top level.");
  }
  return parsed as Record<string, unknown>;
}