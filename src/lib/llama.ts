const LLAMA_URL = "http://localhost:9191";

export interface TriageRecord {
  patient_name: string;
  age: number;
  primary_symptom: string;
  duration_days: number;
  triage_priority: "EMERGENCY" | "URGENT" | "ROUTINE";
  clinical_action_items: string[];
}

export interface InferenceResult {
  data: TriageRecord | null;
  tps: number | null;
}

export async function runInference(transcriptText: string): Promise<InferenceResult> {
  const gbnfRes = await fetch("/medical_intake.gbnf");
  if (!gbnfRes.ok) throw new Error("medical_intake.gbnf not found in public/");
  const grammar = await gbnfRes.text();

  const payload = {
    messages: [
      { role: "system", content: "Extract clinical parameters matching the strict data schema precisely." },
      { role: "user", content: transcriptText },
    ],
    temperature: 0.0,
    grammar,
  };

  const start = performance.now();
  const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) throw new Error(`llama-server returned ${res.status}`);

  const elapsed = (performance.now() - start) / 1000;
  const json = await res.json();
  const raw: string = json.choices?.[0]?.message?.content ?? "";
  const tokens: number = json.usage?.completion_tokens ?? 0;
  const tps = tokens > 0 && elapsed > 0 ? tokens / elapsed : null;

  let data: TriageRecord | null = null;
  try { data = JSON.parse(raw); } catch { /* invalid JSON = schema failure */ }

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