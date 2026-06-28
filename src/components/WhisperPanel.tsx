import { invoke } from "@tauri-apps/api/core";
import { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence, useAnimationFrame } from "framer-motion";
import { RiMicLine, RiStopCircleLine, RiUploadCloud2Line } from "react-icons/ri";

const MAX_SECONDS = 120;
const BAR_COUNT   = 48;

interface Props {
  onTranscript: (text: string) => void;
}

type Status = "idle" | "recording" | "transcribing" | "done";

const STATUS_META = {
  idle:         { label: "Press Enter or tap to speak", color: "var(--accent)",  dim: "var(--accent-dim)" },
  recording:    { label: "Listening — press Enter to stop", color: "#e8445a",   dim: "#e8445a18"         },
  transcribing: { label: "Transcribing…",                   color: "var(--accent)", dim: "var(--accent-dim)" },
  done:         { label: "Done",                            color: "#22c98e",    dim: "#22c98e18"         },
};

/* ── tiny synth sounds ─────────────────────────────────────────── */
function playTone(freq: number, type: OscillatorType, duration: number, fadeOut = true) {
  try {
    const ctx  = new AudioContext();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type      = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    if (fadeOut) gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
    osc.start();
    osc.stop(ctx.currentTime + duration);
    osc.onended = () => ctx.close();
  } catch {}
}

function playStartSound() {
  // ascending two-tone chime
  playTone(520, "sine", 0.18);
  setTimeout(() => playTone(780, "sine", 0.22), 120);
}

function playStopSound() {
  // descending two-tone
  playTone(680, "sine", 0.18);
  setTimeout(() => playTone(400, "sine", 0.28), 110);
}

export default function WhisperPanel({ onTranscript }: Props) {
  const [status, setStatus]           = useState<Status>("idle");
  const [barHeights, setBarHeights]   = useState<number[]>(Array(BAR_COUNT).fill(4));
  const [dragOver, setDragOver]       = useState(false);
  const [micScale, setMicScale]       = useState(1);

  const mediaRecRef     = useRef<MediaRecorder | null>(null);
  const chunksRef       = useRef<Blob[]>([]);
  const onTranscriptRef = useRef(onTranscript);
  onTranscriptRef.current = onTranscript;

  const analyserRef  = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const audioCtxRef  = useRef<AudioContext | null>(null);
  const statusRef    = useRef<Status>("idle");
  statusRef.current  = status;

  /* ── animation frame ── */
  useAnimationFrame((t) => {
    const s = statusRef.current;

    if (s === "recording" && analyserRef.current && dataArrayRef.current) {
      analyserRef.current.getByteFrequencyData(dataArrayRef.current as Uint8Array<ArrayBuffer>);
      const half = BAR_COUNT / 2;
      const step = Math.floor(dataArrayRef.current!.length / half);

      // overall volume → mic scale
      let sum = 0;
      for (let k = 0; k < dataArrayRef.current!.length; k++) sum += dataArrayRef.current![k];
      const avg = sum / dataArrayRef.current!.length / 255;
      setMicScale(1 + avg * 0.35);

      setBarHeights(
        Array.from({ length: BAR_COUNT }, (_, i) => {
          const mirror = i < half ? i : BAR_COUNT - 1 - i;
          const v = dataArrayRef.current![mirror * step] / 255;
          return 4 + v * 90;
        })
      );
    } else if (s === "transcribing") {
      setMicScale(1);
      setBarHeights(
        Array.from({ length: BAR_COUNT }, (_, i) => {
          const w1 = Math.sin(t / 180 + i * 0.38) * 0.5 + 0.5;
          const w2 = Math.sin(t / 260 + i * 0.22) * 0.5 + 0.5;
          return 5 + (w1 * 0.6 + w2 * 0.4) * 44;
        })
      );
    } else if (s === "done") {
      setMicScale(1);
      setBarHeights(
        Array.from({ length: BAR_COUNT }, (_, i) => {
          const wave = Math.sin(t / 1400 + i * 0.7) * 0.5 + 0.5;
          return 3 + wave * 6;
        })
      );
    } else {
      setMicScale(1);
      setBarHeights(
        Array.from({ length: BAR_COUNT }, (_, i) => {
          const wave = Math.sin(t / 1100 + i * 0.55) * 0.5 + 0.5;
          return 2 + wave * 5;
        })
      );
    }
  });

  /* ── transcribe ── */
  const transcribeBlob = useCallback(async (blob: Blob) => {
    setStatus("transcribing");
    try {
      const bytes = Array.from(new Uint8Array(await blob.arrayBuffer() as ArrayBuffer));
      const text  = await invoke<string>("transcribe_audio", { bytes });
      if (text) {
        onTranscriptRef.current(text);
        setStatus("done");
        setTimeout(() => setStatus("idle"), 2200);
      } else {
        setStatus("idle");
      }
    } catch {
      setStatus("idle");
    }
  }, []);

  /* ── recording control ── */
  const startRecording = useCallback(async () => {
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.78;
      analyserRef.current  = analyser;
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
      audioCtx.createMediaStreamSource(stream).connect(analyser);

      chunksRef.current = [];
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")
        ? "audio/ogg;codecs=opus"
        : "audio/mp4";

      const mr = new MediaRecorder(stream, { mimeType });
      mediaRecRef.current = mr;
      mr.ondataavailable = (e) => { if (e.data?.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        audioCtx.close();
        analyserRef.current  = null;
        dataArrayRef.current = null;
        transcribeBlob(new Blob(chunksRef.current, { type: mr.mimeType }));
      };

      playStartSound();
      mr.start();
      setStatus("recording");

      setTimeout(() => {
        if (mediaRecRef.current?.state === "recording") mr.stop();
      }, MAX_SECONDS * 1000);
    } catch {
      setStatus("idle");
    }
  }, [transcribeBlob]);

  const stopRecording = useCallback(() => {
    playStopSound();
    mediaRecRef.current?.stop();
  }, []);

  const toggleRecording = useCallback(() => {
    if (statusRef.current === "recording") stopRecording();
    else if (statusRef.current === "idle" || statusRef.current === "done") startRecording();
  }, [startRecording, stopRecording]);

  /* ── Enter key ── */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.repeat) {
        const s = statusRef.current;
        if (s === "recording") stopRecording();
        else if (s === "idle" || s === "done") startRecording();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [startRecording, stopRecording]);

  /* ── file handling ── */
  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    transcribeBlob(file);
    e.target.value = "";
  }, [transcribeBlob]);

  const meta  = STATUS_META[status];
  const isRec = status === "recording";
  const busy  = status === "transcribing";

  /* pulse rings — 3 rings staggered */
  const rings = [0, 1, 2];

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 16,
      overflow: "hidden",
      fontFamily: "var(--font-ui)",
    }}>

      {/* ── Stage ── */}
      <div style={{
        display: "flex",
        flexDirection: "column",    
        paddingBlock: "20px",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
        background: "var(--surface-2)",
        gap: 0,
      }}>

        {/* ambient bg glow */}
        <motion.div
          animate={{ opacity: isRec ? 0.15 : 0.06 }}
          transition={{ duration: 0.9 }}
          style={{
            position: "absolute", inset: 0,
            background: meta.color,
            filter: "blur(70px)",
            pointerEvents: "none",
          }}
        />

        {/* ── mic + rings ── */}
        <div style={{
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: 180,
          marginBottom: 8,
          flexShrink: 0,
        }}>

          {/* cyclic pulse rings */}
          {rings.map((r) => (
            <AnimatePresence key={r}>
              {(isRec || status === "transcribing") && (
                <motion.div
                  key={`ring-${r}-${status}`}
                  initial={{ scale: 0.55, opacity: 0 }}
                  animate={{ scale: [0.55, 1.55], opacity: [0.45, 0] }}
                  transition={{
                    duration: 2.2,
                    delay: r * 0.72,
                    repeat: Infinity,
                    ease: "easeOut",
                  }}
                  style={{
                    position: "absolute",
                    width:  64 * micScale,
                    height:  64 * micScale,
                    borderRadius: "50%",
                    border: `1.5px solid ${meta.color}`,
                    pointerEvents: "none",
                  }}
                />
              )}
            </AnimatePresence>
          ))}

          {/* mic button */}
          <motion.button
            animate={{
              scale: micScale,
              boxShadow: isRec
                ? `0 0 0 0px ${meta.color}00, 0 0 32px 6px ${meta.color}55, 0 0 80px 16px ${meta.color}22`
                : `0 0 0 0px ${meta.color}00, 0 0 14px 2px ${meta.color}33, 0 0 30px 4px ${meta.color}11`,
            }}
            transition={isRec
              ? { duration: 0.08, ease: "easeOut" }
              : { duration: 0.5 }
            }
            whileTap={!busy ? { scale: micScale * 0.91 } : {}}
            onClick={toggleRecording}
            disabled={busy}
            style={{
              position: "relative",
              zIndex: 3,
              width: 64, height: 64,
              borderRadius: "50%",
              background: isRec
                ? `radial-gradient(circle at 38% 36%, #ff6b7acc 0%, #e8445a88 55%, #e8445a22 100%)`
                : `radial-gradient(circle at 38% 36%, ${meta.color}cc 0%, ${meta.color}66 55%, ${meta.dim} 100%)`,
              border: `2px solid ${meta.color}99`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: busy ? "not-allowed" : "pointer",
              outline: "none",
              transition: "background 0.4s, border-color 0.4s",
            }}
          >
            <AnimatePresence mode="wait">
              {isRec ? (
                <motion.div
                  key="stop"
                  initial={{ scale: 0, opacity: 0, rotate: -30 }}
                  animate={{ scale: 1, opacity: 1, rotate: 0 }}
                  exit={{ scale: 0, opacity: 0, rotate: 30 }}
                  transition={{ duration: 0.22, ease: "backOut" }}
                >
                  <RiStopCircleLine size={28} color="#fff" />
                </motion.div>
              ) : (
                <motion.div
                  key="mic"
                  initial={{ scale: 0, opacity: 0, rotate: 30 }}
                  animate={{ scale: 1, opacity: busy ? 0.4 : 1, rotate: 0 }}
                  exit={{ scale: 0, opacity: 0, rotate: -30 }}
                  transition={{ duration: 0.22, ease: "backOut" }}
                >
                  <RiMicLine size={28} color="#fff" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>
        </div>

        {/* waveform bars */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 2,
          height: 48,
          zIndex: 1,
          flexShrink: 0,
        }}>
          {barHeights.map((h, i) => {
            const centre = Math.abs(i - BAR_COUNT / 2) / (BAR_COUNT / 2);
            const boost  = 1 - centre * 0.28;
            return (
              <motion.div
                key={i}
                animate={{
                  height: Math.max(2, h * boost),
                  opacity: status === "idle" ? 0.25 : 0.8,
                }}
                transition={{ duration: 0.04, ease: "easeOut" }}
                style={{
                  width: 2.5,
                  borderRadius: 2,
                  background: meta.color,
                  flexShrink: 0,
                }}
              />
            );
          })}
        </div>

        {/* status label */}
        <AnimatePresence mode="wait">
          <motion.span
            key={status}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.22 }}
            style={{
              marginTop: 10,
              fontSize: 10,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: meta.color,
              fontFamily: "var(--font-mono)",
              zIndex: 2,
              flexShrink: 0,
            }}
          >
            {meta.label}
          </motion.span>
        </AnimatePresence>
      </div>

      {/* ── Drop zone ── */}
      <motion.div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files?.[0];
          if (file) transcribeBlob(file);
        }}
        animate={{
          borderColor: dragOver ? meta.color : "var(--border)",
          background:  dragOver ? meta.dim   : "transparent",
        }}
        style={{
          margin: "12px 14px",
          border: "1px dashed var(--border)",
          borderRadius: 10,
          padding: "10px 14px",
          display: "flex",
          alignItems: "center",
          gap: 10,
          cursor: busy ? "not-allowed" : "pointer",
          opacity: busy ? 0.4 : 1,
          transition: "opacity 0.2s",
        }}
      >
        <RiUploadCloud2Line size={16} color="var(--text-muted)" style={{ flexShrink: 0 }} />
        <label style={{
          fontSize: 12,
          color: "var(--text-muted)",
          cursor: busy ? "not-allowed" : "pointer",
          flex: 1,
        }}>
          <input
            type="file"
            accept="audio/*,video/*"
            style={{ display: "none" }}
            disabled={busy}
            onChange={handleFile}
          />
          Drop audio or video — or{" "}
          <span style={{ color: "var(--accent)", textDecoration: "underline" }}>browse</span>
        </label>
      </motion.div>

      {/* ── Footer ── */}
      <p style={{
        fontSize: 10,
        color: "var(--text-muted)",
        textAlign: "center",
        padding: "0 14px 12px",
        fontFamily: "var(--font-mono)",
        letterSpacing: "0.04em",
      }}>
        fully offline · audio never leaves this device
      </p>
    </div>
  );
}