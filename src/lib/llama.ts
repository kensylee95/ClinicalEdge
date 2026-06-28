import { buildGrammarFromJson } from "./buildGrammar";
import { parseUserJson, inferFields } from "./inferSchema";

const LLAMA_URL = "http://localhost:9191";

export type FieldValue = string | number | boolean | string[] | number[] | boolean[];
export type TriageRecord = Record<string, FieldValue>;

export interface InferenceResult {
  data: TriageRecord | null;
  tps: number | null;
}

export async function runInference(
  transcriptText: string,
  rawJson: string,         // ← user's pasted JSON schema, replaces SchemaConfig
): Promise<InferenceResult> {
  const grammar = buildGrammarFromJson(rawJson);

  // Derive field list for the system prompt from the same source of truth
  const obj    = parseUserJson(rawJson);
  const fields = inferFields(obj);
  const fieldList = fields
    .map(f => `${f.key} (${f.type})`)
    .join(", ");

  const systemContent =
    `You are a data extraction tool. Extract the following fields from the user's ` +
    `input and return them as a single flat JSON object with exactly these fields: ` +
    `${fieldList}. Use only information present in the input — do not invent values ` +
    `not supported by it. Output ONLY valid JSON. No conversational text, no markdown ` +
    `code blocks, no explanations.`;

  const prompt =
    `<|start_header_id|>system<|end_header_id|>\n\n${systemContent}<|eot_id|>` +
    `<|start_header_id|>user<|end_header_id|>\n\n${transcriptText}<|eot_id|>` +
    `<|start_header_id|>assistant<|end_header_id|>\n\n`;

  const payload = {
    prompt,
    temperature: 0.0,
    n_predict: 512,
    grammar,
  };

  const start = performance.now();
  const res = await fetch(`${LLAMA_URL}/completion`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) throw new Error(`llama-server returned ${res.status}`);

  const elapsed = (performance.now() - start) / 1000;
  const json = await res.json();

  const raw: string    = json.content ?? "";
  const tokens: number = json.tokens_predicted ?? 0;
  const tps = tokens > 0 && elapsed > 0 ? tokens / elapsed : null;

  let data: TriageRecord | null = null;
  let parseError: string | null = null;
  try {
    data = JSON.parse(raw);
  } catch {
    parseError =
      "Server returned non-JSON output — grammar likely wasn't applied " +
      "(check llama-server logs and confirm /completion accepted the 'grammar' field).";
  }

  if (parseError) throw new Error(parseError);

  return { data, tps };
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${LLAMA_URL}/health`, { signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch {
    return false;
  }
}