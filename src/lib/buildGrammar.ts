import type { SchemaConfig, FieldType, ScalarType } from "../components/SchemaDrawer";
import { toSchemaConfig, parseUserJson } from "./inferSchema";

// ── Entry points ──────────────────────────────────────────────────

/** Build a grammar from a user-pasted JSON string directly */
export function buildGrammarFromJson(rawJson: string): string {
  const obj = parseUserJson(rawJson);
  return buildGrammar(toSchemaConfig(obj));
}

/** Build a grammar from a SchemaConfig (existing path, still works) */
export function buildGrammar(schema: SchemaConfig): string {
  const fields = schema.fields.filter(f => f.name.trim().length > 0);

  if (fields.length === 0) {
    throw new Error("Schema has no named fields.");
  }

  const fieldRuleNames = fields.map((f, i) => ruleNameFor(f, i));

  const rootBody = fieldRuleNames
    .map((name, i) => (i === 0 ? name : `"," ws ${name}`))
    .join(" ws ");

  const lines: string[] = [];
  lines.push(`root ::= "{" ws ${rootBody} ws "}"`);
  lines.push("");

  fields.forEach((field, i) => {
    const ruleName    = fieldRuleNames[i];
    const nameLiteral = gbnfStringLiteral(`"${field.name.trim()}"`); // ← fix
    const valueRule   = valueRuleFor(field.type);
    lines.push(`${ruleName} ::= ${nameLiteral} ws ":" ws (${valueRule})`);
  });

  lines.push("");

  Object.entries(SHARED_RULES).forEach(([name, body]) => {
    lines.push(`${name} ::= ${body}`);
  });

  return lines.join("\n");
}

// ── Shared rules ──────────────────────────────────────────────────

const SHARED_RULES: Record<string, string> = {
  ws:      `[ \\t\\n]*`,
  string:  `"\\"" char* "\\""`,
  char:    `[^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" hex hex hex hex)`,
  hex:     `[0-9a-fA-F]`,
  number:  `"-"? int frac? exp?`,
  int:     `"0" | [1-9] [0-9]*`,
  frac:    `"." [0-9]+`,
  exp:     `[eE] [+-]? [0-9]+`,
  boolean: `"true" | "false"`,
  date:    `"\\"" digit digit digit digit "-" digit digit "-" digit digit "\\""`,
  digit:   `[0-9]`,
};

// ── Helpers ───────────────────────────────────────────────────────

function gbnfStringLiteral(raw: string): string {
  const escaped = raw.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  return `"${escaped}"`;
}

function ruleNameFor(field: { name: string }, index: number): string {
  const safe = field.name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  return `field-${index}-${safe || "f"}`;
}

function isArrayType(t: FieldType): t is `array<${ScalarType}>` {
  return t.startsWith("array<");
}

function valueRuleFor(t: FieldType): string {
  if (isArrayType(t)) {
    const inner = (t as string).slice(6, -1) as ScalarType;
    return `"[" ws (${inner} ("," ws ${inner})*)? ws "]"`;
  }
  return t;
}