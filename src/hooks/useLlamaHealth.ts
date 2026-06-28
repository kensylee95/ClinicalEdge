import { useState, useEffect, useRef } from "react";
import { checkHealth } from "../lib/llama";

export function useLlamaHealth(): { online: boolean } {
  const [online, setOnline] = useState(false);
  const consecutiveFailures = useRef(0);
  const consecutiveSuccesses = useRef(0);

  useEffect(() => {
    let active = true;
    async function poll() {
      const ok = await checkHealth();
      if (!active) return;
      if (ok) {
        consecutiveFailures.current = 0;
        consecutiveSuccesses.current += 1;
        if (consecutiveSuccesses.current >= 2) setOnline(true);
      } else {
        consecutiveSuccesses.current = 0;
        consecutiveFailures.current += 1;
        if (consecutiveFailures.current >= 1) setOnline(false);
      }
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => { active = false; clearInterval(id); };
  }, []);

  return { online };
}