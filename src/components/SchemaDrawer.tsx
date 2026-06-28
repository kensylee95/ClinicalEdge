import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RiText,
  RiHashtag,
  RiToggleLine,
  RiCalendarLine,
  RiBracketsLine,
  RiCodeBoxLine,
  RiFileUploadLine,
  RiFileLine,
  RiAddLine,
  RiCloseLine,
  RiCheckLine,
  RiLayoutGridLine,
} from "react-icons/ri";

// ── Types ────────────────────────────────────────────────────────
export type ScalarType = "string" | "number" | "boolean" | "date";
export type FieldType  = ScalarType | `array<${ScalarType}>`;

export interface CustomField {
  id: number;
  name: string;
  type: FieldType;
}

export interface SchemaConfig {
  templateFile: File | null;
  fields: Omit<CustomField, "id">[];
}

interface Props {
  open: boolean;
  onClose: () => void;
   onApply: (rawJson: string) => void;
}

// ── Constants ────────────────────────────────────────────────────
const SCALAR_TYPES: { type: ScalarType; Icon: React.ElementType; label: string }[] = [
  { type: "string",  Icon: RiText,         label: "Text"    },
  { type: "number",  Icon: RiHashtag,      label: "Number"  },
  { type: "boolean", Icon: RiToggleLine,   label: "Yes / No"},
  { type: "date",    Icon: RiCalendarLine, label: "Date"    },
];

const TYPE_COLORS: Record<ScalarType, string> = {
  string:  "#1a6fc4",
  number:  "#b87000",
  boolean: "#0f8a5f",
  date:    "#7c3aed",
};

const ACCEPTED_MIME =
  ".pdf,.docx,.txt,.json,.csv,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain,application/json,text/csv";

// ── Helpers ───────────────────────────────────────────────────────
function isArray(t: FieldType): t is `array<${ScalarType}>` {
  return t.startsWith("array<");
}

function arrayInner(t: FieldType): ScalarType {
  return (t as string).slice(6, -1) as ScalarType;
}

function colorOf(t: FieldType): string {
  return isArray(t) ? TYPE_COLORS[arrayInner(t)] : TYPE_COLORS[t as ScalarType];
}

// ── DropZone ─────────────────────────────────────────────────────
function DropZone({ file, onFile, onRemove }: { file: File | null; onFile: (f: File) => void; onRemove: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0]; if (f) onFile(f);
  }, [onFile]);

  if (file) return (
    <div className="dz-pill">
      <RiFileLine size={16} style={{ color: "var(--accent)", flexShrink: 0 }} />
      <span className="dz-pill-name">{file.name}</span>
      <span className="dz-pill-size">{(file.size / 1024).toFixed(1)} KB</span>
      <button className="dz-pill-remove" onClick={onRemove} aria-label="Remove">
        <RiCloseLine size={14} />
      </button>
    </div>
  );

  return (
    <div
      className={`dz-zone${dragging ? " dz-zone--over" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      role="button" tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept={ACCEPTED_MIME} style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
      <RiFileUploadLine size={28} style={{ marginBottom: 8, opacity: 0.4 }} />
      <div className="dz-text">Drop a template or <strong>browse</strong></div>
      <div className="dz-hint">.pdf · .docx · .txt · .json · .csv</div>
    </div>
  );
}

// ── Type selector for a field ────────────────────────────────────
function TypeSelector({ value, onChange }: { value: FieldType; onChange: (t: FieldType) => void }) {
  const asArray = isArray(value);
  const inner   = asArray ? arrayInner(value) : (value as ScalarType);

  const toggleArray = () => {
    onChange(asArray ? inner : `array<${inner}>`);
  };

  const setScalar = (s: ScalarType) => {
    onChange(asArray ? `array<${s}>` : s);
  };

  return (
    <div className="type-sel">
      {/* Scalar pills */}
      {SCALAR_TYPES.map(({ type, Icon, label }) => {
        const active = inner === type;
        return (
          <button
            key={type}
            className={`sd-type-pill${active ? " sd-type-pill--active" : ""}`}
            style={active ? { borderColor: TYPE_COLORS[type], color: TYPE_COLORS[type], background: TYPE_COLORS[type] + "18" } : {}}
            onClick={() => setScalar(type)}
            title={label}
          >
            <Icon size={13} />
          </button>
        );
      })}

      {/* Divider */}
      <div className="type-sel-div" />

      {/* Array toggle */}
      <button
        className={`sd-type-pill sd-array-pill${asArray ? " sd-type-pill--active" : ""}`}
        style={asArray ? { borderColor: colorOf(value), color: colorOf(value), background: colorOf(value) + "18" } : {}}
        onClick={toggleArray}
        title={asArray ? `array of ${inner}` : "Make this a list (array)"}
      >
        <RiBracketsLine size={13} />
      </button>
    </div>
  );
}

// ── JSON value representation ────────────────────────────────────
function jsonValue(t: FieldType): { text: string; color: string } {
  if (isArray(t)) {
    const inner = arrayInner(t);
    return { text: `[ <${inner}>, … ]`, color: TYPE_COLORS[inner] };
  }
  const examples: Record<ScalarType, string> = {
    string:  '"…"',
    number:  "0",
    boolean: "true",
    date:    '"2024-01-01"',
  };
  return { text: examples[t as ScalarType], color: TYPE_COLORS[t as ScalarType] };
}

// ── JSON Preview ─────────────────────────────────────────────────
function JsonPreview({ fields }: { fields: CustomField[] }) {
  const named = fields.filter(f => f.name.trim());

  if (!named.length) return (
    <div className="json-empty">
      <RiCodeBoxLine size={36} style={{ marginBottom: 10, opacity: 0.2 }} />
      <div>Fields you add will appear here</div>
    </div>
  );

  return (
    <pre className="json-pre">
      <span className="j-brace">{"{"}</span>{"\n"}
      {named.map((f, i) => {
        const { text, color } = jsonValue(f.type);
        return (
          <span key={f.id}>
            {"  "}
            <span className="j-key">"{f.name.trim()}"</span>
            <span className="j-colon">: </span>
            <span style={{ color }}>{text}</span>
            {i < named.length - 1 && <span className="j-comma">,</span>}
            {"\n"}
          </span>
        );
      })}
      <span className="j-brace">{"}"}</span>
    </pre>
  );
}

// ── Main ─────────────────────────────────────────────────────────
export default function SchemaDrawer({ open, onClose, onApply }: Props) {
  const [templateFile, setTemplateFile] = useState<File | null>(null);
  const [fields, setFields] = useState<CustomField[]>([]);

  const addField    = () => setFields(prev => [...prev, { id: Date.now(), name: "", type: "string" }]);
  const updateField = (id: number, patch: Partial<CustomField>) =>
    setFields(prev => prev.map(f => f.id === id ? { ...f, ...patch } : f));
  const removeField = (id: number) => setFields(prev => prev.filter(f => f.id !== id));

const handleApply = () => {
  // Build the JSON string from the fields the user defined,
  // using a representative value per type so inferFields gets the shape right.
  const example: Record<string, unknown> = {};
  fields
    .filter(f => f.name.trim())
    .forEach(f => {
      const name = f.name.trim();
      switch (f.type) {
        case "number":           example[name] = 0;          break;
        case "boolean":          example[name] = true;        break;
        case "date":             example[name] = "2024-01-01"; break;
        case "array<string>":    example[name] = [""];        break;
        case "array<number>":    example[name] = [0];         break;
        case "array<boolean>":   example[name] = [true];      break;
        case "array<date>":      example[name] = ["2024-01-01"]; break;
        default:                 example[name] = "";          break; // string
      }
    });

  onApply(JSON.stringify(example));
  onClose();
};
  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.div className="sd-overlay"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }} onClick={onClose} />
        )}
      </AnimatePresence>

      <motion.aside className="sd-drawer"
        initial={{ x: "100%" }}
        animate={{ x: open ? "0%" : "100%" }}
        transition={{ type: "spring", stiffness: 340, damping: 34 }}
        aria-hidden={!open} aria-label="Schema builder"
      >
        {/* Header */}
        <div className="sd-header">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <RiLayoutGridLine size={18} style={{ color: "var(--accent)" }} />
            <div>
              <div className="sd-header-title">Schema Builder</div>
              <div className="sd-header-sub">Flat JSON — define every output field and its type</div>
            </div>
          </div>
          <button className="sd-close" onClick={onClose} aria-label="Close">
            <RiCloseLine size={16} />
          </button>
        </div>

        {/* Columns */}
        <div className="sd-columns">

          {/* LEFT */}
          <div className="sd-left">

            <div className="sd-section">
              <div className="sd-label">
                <RiFileUploadLine size={13} /> Template file
                <span className="sd-label-hint">optional</span>
              </div>
              <DropZone file={templateFile} onFile={setTemplateFile} onRemove={() => setTemplateFile(null)} />
            </div>

            <div className="sd-divider" />

            <div className="sd-section" style={{ flex: 1 }}>
              <div className="sd-label">
                <RiLayoutGridLine size={13} /> Output fields
                <span className="sd-field-badge">{fields.filter(f => f.name.trim()).length}</span>
              </div>

              {/* Legend */}
              <div className="sd-legend">
                {SCALAR_TYPES.map(({ type, Icon, label }) => (
                  <span key={type} className="sd-legend-item" style={{ color: TYPE_COLORS[type] }}>
                    <Icon size={11} /> {label}
                  </span>
                ))}
                <span className="sd-legend-item" style={{ color: "var(--text-muted)" }}>
                  <RiBracketsLine size={11} /> = list
                </span>
              </div>

              <div className="sd-fields">
                <AnimatePresence initial={false}>
                  {fields.length === 0 && (
                    <div className="sd-no-fields">No fields yet — hit <strong>Add field</strong> below</div>
                  )}
                  {fields.map((f, idx) => (
                    <motion.div key={f.id} className="sd-field-row"
                      initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: 16 }} transition={{ duration: 0.14 }}
                    >
                      <span className="sd-field-num">{idx + 1}</span>
                      <input className="sd-field-input" type="text" placeholder="field_name"
                        value={f.name} onChange={e => updateField(f.id, { name: e.target.value })}
                        spellCheck={false} />
                      <TypeSelector
                        value={f.type}
                        onChange={type => updateField(f.id, { type })}
                      />
                      <button className="sd-remove" onClick={() => removeField(f.id)} aria-label="Remove">
                        <RiCloseLine size={13} />
                      </button>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>

              <button className="sd-add-btn" onClick={addField}>
                <RiAddLine size={14} /> Add field
              </button>
            </div>
          </div>

          {/* RIGHT */}
          <div className="sd-right">
            <div className="sd-label" style={{ marginBottom: 14 }}>
              <RiCodeBoxLine size={13} /> Live preview
            </div>
            <JsonPreview fields={fields} />
          </div>
        </div>

        {/* Footer */}
        <div className="sd-footer">
          <button className="sd-btn-ghost" onClick={onClose}>Cancel</button>
          <button className="sd-btn-primary" onClick={handleApply}>
            <RiCheckLine size={14} /> Apply schema
          </button>
        </div>
      </motion.aside>

      <style>{`
        .sd-overlay { position: fixed; inset: 0; background: rgba(10,13,18,0.5); z-index: 200; }

        .sd-drawer {
          position: fixed; top: 0; right: 0; bottom: 0;
          width: 60vw; max-width: 960px; min-width: 640px;
          background: var(--surface); border-left: 1px solid var(--border);
          box-shadow: -8px 0 32px rgba(0,0,0,0.12);
          display: flex; flex-direction: column; z-index: 201; overflow: hidden;
        }

        .sd-header {
          display: flex; align-items: center; justify-content: space-between;
          padding: 16px 24px; background: var(--surface-2);
          border-bottom: 1px solid var(--border); flex-shrink: 0;
        }
        .sd-header-title { font-size: 15px; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }
        .sd-header-sub   { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
        .sd-close {
          width: 30px; height: 30px; border: 1px solid var(--border);
          border-radius: var(--radius); background: var(--surface); cursor: pointer;
          color: var(--text-muted); display: flex; align-items: center; justify-content: center;
          transition: background 0.15s, color 0.15s;
        }
        .sd-close:hover { background: var(--surface-2); color: var(--text); }

        .sd-columns { display: grid; grid-template-columns: 1fr 1fr; flex: 1; overflow: hidden; }
        .sd-left {
          display: flex; flex-direction: column; overflow-y: auto;
          padding: 20px; border-right: 1px solid var(--border);
        }
        .sd-right { display: flex; flex-direction: column; padding: 20px; background: var(--surface-2); overflow-y: auto; }

        .sd-section { margin-bottom: 20px; }
        .sd-label {
          display: flex; align-items: center; gap: 6px;
          font-size: 11px; font-weight: 700; letter-spacing: 0.07em;
          text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px;
        }
        .sd-label-hint {
          margin-left: auto; font-size: 10px; font-weight: 400; text-transform: none;
          letter-spacing: 0; color: var(--text-muted); background: var(--surface-2);
          border: 1px solid var(--border); padding: 1px 6px; border-radius: 99px;
        }
        .sd-field-badge {
          margin-left: auto; min-width: 20px; height: 18px; padding: 0 6px;
          background: var(--accent); color: #fff; border-radius: 99px;
          font-size: 10px; font-weight: 700; letter-spacing: 0; text-transform: none;
          display: flex; align-items: center; justify-content: center;
        }
        .sd-divider { height: 1px; background: var(--border); margin-bottom: 20px; }

        /* Legend */
        .sd-legend {
          display: flex; gap: 10px; flex-wrap: wrap;
          margin-bottom: 10px; padding: 6px 10px;
          background: var(--surface-2); border: 1px solid var(--border);
          border-radius: var(--radius);
        }
        .sd-legend-item {
          display: flex; align-items: center; gap: 4px;
          font-size: 10px; font-family: var(--font-mono);
        }

        /* Drop zone */
        .dz-zone {
          border: 1.5px dashed var(--border); border-radius: var(--radius-lg);
          padding: 22px 20px; text-align: center; cursor: pointer;
          background: var(--surface-2); color: var(--text-muted);
          display: flex; flex-direction: column; align-items: center;
          transition: border-color 0.15s, background 0.15s;
        }
        .dz-zone:hover, .dz-zone--over { border-color: var(--accent); background: var(--accent-dim); color: var(--accent); }
        .dz-text { font-size: 13px; }
        .dz-text strong { color: var(--accent); }
        .dz-hint { font-size: 10px; margin-top: 4px; font-family: var(--font-mono); opacity: 0.7; }
        .dz-pill {
          display: flex; align-items: center; gap: 8px; padding: 10px 12px;
          background: var(--accent-dim); border: 1px solid var(--accent); border-radius: var(--radius);
        }
        .dz-pill-name { flex: 1; font-family: var(--font-mono); font-size: 11px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .dz-pill-size { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); white-space: nowrap; }
        .dz-pill-remove { background: none; border: none; cursor: pointer; color: var(--text-muted); display: flex; align-items: center; padding: 2px; }
        .dz-pill-remove:hover { color: var(--danger); }

        /* Fields */
        .sd-fields { display: flex; flex-direction: column; gap: 6px; min-height: 40px; }
        .sd-no-fields { font-size: 12px; color: var(--text-muted); padding: 16px 0; text-align: center; }
        .sd-field-row {
          display: flex; align-items: center; gap: 8px; padding: 8px 10px;
          background: var(--surface-2); border: 1px solid var(--border);
          border-radius: var(--radius); transition: border-color 0.15s;
        }
        .sd-field-row:focus-within { border-color: var(--accent); }
        .sd-field-num { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); min-width: 14px; text-align: center; flex-shrink: 0; }
        .sd-field-input {
          flex: 1; border: none; background: transparent; color: var(--text);
          font-size: 12px; font-family: var(--font-mono); outline: none; min-width: 0;
        }
        .sd-field-input::placeholder { color: var(--text-muted); }

        /* Type selector */
        .type-sel { display: flex; align-items: center; gap: 3px; flex-shrink: 0; }
        .type-sel-div { width: 1px; height: 16px; background: var(--border); margin: 0 2px; }
        .sd-type-pill {
          width: 26px; height: 24px; border: 1px solid var(--border); border-radius: 4px;
          background: var(--surface); cursor: pointer; color: var(--text-muted);
          display: flex; align-items: center; justify-content: center;
          transition: border-color 0.12s, background 0.12s, color 0.12s;
        }
        .sd-type-pill:hover { border-color: var(--text-dim); color: var(--text); }
        .sd-type-pill--active { font-weight: 700; }
        .sd-array-pill { font-size: 11px; }

        .sd-remove {
          width: 24px; height: 24px; flex-shrink: 0; border: 1px solid transparent;
          border-radius: var(--radius); background: none; cursor: pointer;
          color: var(--text-muted); display: flex; align-items: center; justify-content: center;
          transition: border-color 0.12s, color 0.12s;
        }
        .sd-remove:hover { border-color: var(--danger); color: var(--danger); }

        .sd-add-btn {
          margin-top: 8px; width: 100%; padding: 8px; border: 1px dashed var(--border);
          border-radius: var(--radius); background: none; cursor: pointer; gap: 6px;
          color: var(--text-muted); font-size: 12px; font-family: var(--font-ui);
          display: flex; align-items: center; justify-content: center;
          transition: border-color 0.15s, color 0.15s;
        }
        .sd-add-btn:hover { border-color: var(--accent); color: var(--accent); }

        /* JSON preview */
        .json-empty {
          flex: 1; display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          font-size: 12px; color: var(--text-muted); text-align: center; padding: 40px 0;
        }
        .json-pre {
          font-family: var(--font-mono); font-size: 13px; line-height: 1.9;
          color: var(--text-dim); white-space: pre; overflow-x: auto; margin: 0; padding: 0;
        }
        .j-brace { color: var(--text); font-weight: 700; }
        .j-key   { color: var(--accent); }
        .j-colon { color: var(--text-muted); }
        .j-comma { color: var(--text-muted); }

        /* Footer */
        .sd-footer {
          display: flex; justify-content: flex-end; gap: 8px; padding: 14px 24px;
          border-top: 1px solid var(--border); background: var(--surface-2); flex-shrink: 0;
        }
        .sd-btn-ghost {
          padding: 9px 18px; border: 1px solid var(--border); border-radius: var(--radius);
          background: var(--surface); color: var(--text); font-size: 13px; font-weight: 600;
          cursor: pointer; font-family: var(--font-ui); transition: background 0.15s;
        }
        .sd-btn-ghost:hover { background: var(--surface-2); }
        .sd-btn-primary {
          display: flex; align-items: center; gap: 6px; padding: 9px 22px;
          border: none; border-radius: var(--radius); background: var(--accent); color: #fff;
          font-size: 13px; font-weight: 600; cursor: pointer; font-family: var(--font-ui);
          transition: opacity 0.15s;
        }
        .sd-btn-primary:hover { opacity: 0.88; }
      `}</style>
    </>
  );
}