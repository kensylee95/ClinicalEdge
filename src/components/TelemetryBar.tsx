import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

interface MemoryInfo { app: number; sys: number; }
interface Props { tpsHistory: number[]; }

export default function TelemetryBar({ tpsHistory }: Props) {
  const [ram, setRam] = useState<MemoryInfo>({ app: 0, sys: 0 });
  const [cpuTier, setCpuTier] = useState<string>("");

  useEffect(() => {
    // One-time CPU tier fetch
    invoke<string>("get_cpu_tier").then(setCpuTier).catch(() => {});

    let active = true;
    async function poll() {
      try {
        const result = await invoke<MemoryInfo>("get_memory_info");
        if (active) setRam(result);
      } catch {
        if (active) setRam({ app: 0, sys: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 0 });
      }
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => { active = false; clearInterval(id); };
  }, []);

  const avgTps = tpsHistory.length
    ? (tpsHistory.reduce((a, b) => a + b, 0) / tpsHistory.length).toFixed(1)
    : "—";
  const maxTps = tpsHistory.length ? Math.max(...tpsHistory) : 1;

  // Derive a short label from the binary name
  const engineLabel = cpuTier.includes("avx2")
    ? "AVX2" : cpuTier.includes("noavx")
    ? "no-AVX" : cpuTier.includes("avx")
    ? "AVX" : "…";

  return (
    <div className="telemetry-bar">
      <div className="telem-metric">
        <span className="telem-label">TPS</span>
        <span className="telem-value">{avgTps}</span>
        <div className="telem-sparkline">
          {tpsHistory.slice(-20).map((v, i) => (
            <div key={i} className="spark-bar"
              style={{ height: `${Math.max(2, (v / maxTps) * 18)}px` }} />
          ))}
        </div>
      </div>

      {ram.app > 0 && (
        <div className="telem-metric">
          <span className="telem-label">App RAM</span>
          <span className="telem-value">{ram.app.toFixed(0)} MB</span>
        </div>
      )}

      {ram.sys > 0 && (
        <div className="telem-metric">
          <span className="telem-label">Sys RAM</span>
          <span className="telem-value">{ram.sys.toFixed(2)} GB</span>
        </div>
      )}

      <div className="telem-metric" style={{ marginLeft: "auto" }}>
        <span className="telem-label">engine</span>
        <span className="telem-value">llama-server · {engineLabel} · :8080</span>
      </div>
    </div>
  );
}