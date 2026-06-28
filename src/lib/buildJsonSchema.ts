import { FieldType, ScalarType, SchemaConfig } from "../components/SchemaDrawer";

/**
 * Builds a JSON Schema object from a dynamic SchemaConfig produced by the
 * SchemaDrawer, for use with llama-server's `/v1/chat/completions`
 * `response_format: { type: "json_schema", schema }` option.
 *
 * This replaces the earlier hand-rolled GBNF approach (buildGrammar.ts).
 * JSON Schema is the natively-documented constraint mechanism for the
 * chat-completions endpoint; the server converts it to a grammar internally,
 * which avoids an entire class of hand-rolled-GBNF bugs (quote escaping,
 * lack of bounded repeat for dates, etc.) we hit going the grammar route.
 *
 * Notes:
 *   - Field names need no escaping here — JSON.stringify handles that when
 *     the schema object is serialized, unlike GBNF where we had to hand-roll
 *     literal escaping ourselves.
 *   - There's still no enum/literal scalar type in ScalarType, so a `string`
 *     field accepts any string — same limitation as the GBNF version, not
 *     introduced by this rewrite.
 *   - `additionalProperties: false` + `required` enforces the exact field
 *     set, mirroring what the GBNF root rule did structurally.
 *   - `date` uses a `pattern` regex rather than `format: "date"`, since
 *     `format` is advisory-only in JSON Schema and not reliably enforced by
 *     schema-to-grammar converters, whereas `pattern` typically compiles
 *     directly into the grammar.
 */

interface JsonSchemaNode {
  type: string;
  format?: string;
  pattern?: string;
  items?: JsonSchemaNode;
}

function isArrayType(t: FieldType): t is `array<${ScalarType}>` {
  return t.startsWith("array<");
}

function innerScalar(t: FieldType): ScalarType {
  return (t as string).slice(6, -1) as ScalarType;
}

function scalarSchema(t: ScalarType): JsonSchemaNode {
  switch (t) {
    case "string":
      return { type: "string" };
    case "number":
      return { type: "number" };
    case "boolean":
      return { type: "boolean" };
    case "date":
      // `format` alone is advisory in JSON Schema and not reliably enforced
      // by schema-to-grammar converters. `pattern` (a regex) is commonly
      // compiled directly into the grammar, so it's the stricter, actually
      // enforced choice — matches the rigor the old hand-written GBNF rule had.
      return { type: "string", pattern: "^\\d{4}-\\d{2}-\\d{2}$" };
  }
}

function fieldSchema(t: FieldType): JsonSchemaNode {
  if (isArrayType(t)) {
    return { type: "array", items: scalarSchema(innerScalar(t)) };
  }
  return scalarSchema(t);
}

export function buildJsonSchema(schema: SchemaConfig): object {
  const fields = schema.fields.filter((f) => f.name.trim().length > 0);

  if (fields.length === 0) {
    throw new Error("Schema has no named fields — cannot build a JSON schema from an empty schema.");
  }

  const properties: Record<string, JsonSchemaNode> = {};
  const required: string[] = [];

  fields.forEach((field) => {
    const name = field.name.trim();
    properties[name] = fieldSchema(field.type);
    required.push(name);
  });

  return {
    type: "object",
    properties,
    required,
    additionalProperties: false,
  };
}