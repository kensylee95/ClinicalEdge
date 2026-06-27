import { useState, useEffect } from "react";
import { checkHealth } from "../lib/llama";

export function useLlamaHealth(): { online: boolean } {
  const [online, setOnline] = useState(false);

  useEffect(() => {
    let active = true;
    async function poll() {
      const ok = await checkHealth();
      if (active) setOnline(ok);
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => { active = false; clearInterval(id); };
  }, []);

  return { online };
}