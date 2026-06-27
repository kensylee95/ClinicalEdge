#!/usr/bin/env bash
# scripts/pack.sh
#
# Produces the single shippable executable by appending an entire
# llama-server engine directory (the server binary AND every shared
# library it depends on — e.g. ggml.dll / ggml-base.dll / ggml-cpu.dll on
# Windows, or the equivalent .so files on Linux) plus the model file,
# after the plain Tauri binary, followed by a variable-length manifest +
# fixed-size footer (see src-tauri/src/payload.rs for the exact format
# this script and the Rust reader must agree on).
#
# WHY A WHOLE DIRECTORY, NOT JUST ONE ENGINE BINARY:
# llama-server does not run standalone. The official llama.cpp release
# packages ship the server executable alongside several shared libraries
# it loads at startup, and removing the AVX2/AVX/noavx split also means
# there is now exactly ONE recommended CPU build per OS, not three
# CPU-tier variants. So this script embeds everything in ONE engine
# directory you provide, as-is, preserving filenames so the engine's own
# library lookup at runtime is unaffected by the packing step.
#
# Usage:
#   scripts/pack.sh <built-binary> <engine-dir> <model-file> <output-file>
#
#   <built-binary>  path to the plain cargo-built executable
#                   (src-tauri/target/release/clinical-edge[.exe])
#   <engine-dir>    directory containing the FULL llama-server deployment
#                   for this OS — the server exe plus every DLL/SO it
#                   needs alongside it (i.e. the unzipped contents of the
#                   official llama.cpp release zip for this platform).
#                   Every regular file directly inside this directory is
#                   embedded; subdirectories are not walked.
#   <model-file>    path to the .gguf model
#   <output-file>   path to write the final packed single executable
#
# Example (Linux build, using the official llama.cpp Ubuntu CPU release):
#   unzip llama-bXXXX-bin-ubuntu-x64.zip -d ./engine-linux
#   scripts/pack.sh \
#     src-tauri/target/release/clinical-edge \
#     ./engine-linux \
#     ./models/my_clinical_model.gguf \
#     ./dist/ClinicalEdge-linux
#
# Example (Windows build, using the official llama.cpp Windows CPU release):
#   unzip llama-bXXXX-bin-win-cpu-x64.zip -d ./engine-windows
#   scripts/pack.sh \
#     src-tauri/target/release/clinical-edge.exe \
#     ./engine-windows \
#     ./models/my_clinical_model.gguf \
#     ./dist/ClinicalEdge-windows.exe

set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <built-binary> <engine-dir> <model-file> <output-file>" >&2
  exit 1
fi

BIN_PATH="$1"
ENGINE_DIR="$2"
MODEL_PATH="$3"
OUT_PATH="$4"

if [ ! -f "$BIN_PATH" ]; then
  echo "ERROR: built binary not found: $BIN_PATH" >&2
  exit 1
fi
if [ ! -d "$ENGINE_DIR" ]; then
  echo "ERROR: engine directory not found: $ENGINE_DIR" >&2
  exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model file not found: $MODEL_PATH" >&2
  exit 1
fi

# Collect every regular file directly inside ENGINE_DIR (no subdirectories).
shopt -s nullglob
ENGINE_FILE_LIST=()
for f in "$ENGINE_DIR"/*; do
  [ -f "$f" ] && ENGINE_FILE_LIST+=("$f")
done
shopt -u nullglob

if [ "${#ENGINE_FILE_LIST[@]}" -eq 0 ]; then
  echo "ERROR: no files found directly inside $ENGINE_DIR" >&2
  exit 1
fi

echo "Engine directory contains ${#ENGINE_FILE_LIST[@]} file(s):"
for f in "${ENGINE_FILE_LIST[@]}"; do
  echo "  $(basename "$f")"
done

mkdir -p "$(dirname "$OUT_PATH")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"

echo "Building packed executable -> $OUT_PATH"

# All the actual byte-packing, offset tracking, and manifest/footer
# construction happens in this Python step. Earlier shell-only attempts at
# tracking running byte offsets across multiple function calls hit a real
# bug (command-substitution subshells silently discarding variable
# mutation), and hand-rolling variable-length binary structures in pure
# bash is fragile in the same way — Python's struct module is a much more
# reliable place to do this correctly once, rather than re-debugging shell
# semantics. The script just gathers file lists/paths from bash and hands
# them to this script.
"$PYTHON_BIN" - "$OUT_PATH" "$BIN_PATH" "$MODEL_PATH" "${ENGINE_FILE_LIST[@]}" <<'PYEOF'
import struct
import sys
import os

out_path = sys.argv[1]
bin_path = sys.argv[2]
model_path = sys.argv[3]
engine_file_paths = sys.argv[4:]

MAGIC = b"CLINMAN1"
CHUNK = 8 * 1024 * 1024  # 8 MiB, avoid loading multi-GB files fully into memory

def copy_file_streaming(src_path, out_f):
    """Append src_path's bytes to out_f, return the number of bytes written."""
    total = 0
    with open(src_path, "rb") as src_f:
        while True:
            chunk = src_f.read(CHUNK)
            if not chunk:
                break
            out_f.write(chunk)
            total += len(chunk)
    return total

entries = []  # (name, offset, length, is_model)

with open(out_path, "wb") as out_f:
    # 1. app binary, unmodified
    copy_file_streaming(bin_path, out_f)
    offset = out_f.tell()

    # 2. every engine file, preserving its original filename
    for path in engine_file_paths:
        name = os.path.basename(path)
        length = copy_file_streaming(path, out_f)
        entries.append((name, offset, length, 0))
        offset += length

    # 3. the model file
    model_name = os.path.basename(model_path)
    model_length = copy_file_streaming(model_path, out_f)
    entries.append((model_name, offset, model_length, 1))
    offset += model_length

    # 4. the manifest itself
    #    entry_count: u32
    #    per entry: name_len:u16, name bytes (utf-8), offset:u64, len:u64, is_model:u8
    manifest_start = offset
    manifest_buf = bytearray()
    manifest_buf += struct.pack("<I", len(entries))
    for (name, ent_off, ent_len, is_model) in entries:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise SystemExit(f"filename too long for manifest: {name}")
        manifest_buf += struct.pack("<H", len(name_bytes))
        manifest_buf += name_bytes
        manifest_buf += struct.pack("<Q", ent_off)
        manifest_buf += struct.pack("<Q", ent_len)
        manifest_buf += struct.pack("<B", is_model)
    out_f.write(manifest_buf)
    manifest_len = len(manifest_buf)

    # 5. fixed-size footer: magic(8) + manifest_offset:u64(8) + manifest_len:u32(4) = 20 bytes
    out_f.write(MAGIC)
    out_f.write(struct.pack("<Q", manifest_start))
    out_f.write(struct.pack("<I", manifest_len))

print(f"Wrote {len(entries)} embedded file(s):")
for (name, ent_off, ent_len, is_model) in entries:
    kind = "model" if is_model else "engine"
    print(f"  [{kind}] {name}: offset={ent_off} len={ent_len}")

final_size = os.path.getsize(out_path)
print(f"\nDone. {out_path} is {final_size} bytes.")
PYEOF

chmod +x "$OUT_PATH" 2>/dev/null || true