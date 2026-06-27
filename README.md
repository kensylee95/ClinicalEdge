# ClinicalEdge — Sovereign Medical Intake Node

ClinicalEdge is a desktop app for offline medical intake. It records and transcribes a conversation, then uses a locally-running AI model to help structure that conversation into useful data — all without sending anything over the internet. Audio, transcripts, and patient information never leave the device it runs on.

It's built with [Tauri](https://tauri.app), which means the interface is a normal web app (React) but the heavy lifting — audio processing, AI inference, the database — runs as native Rust code on the user's machine.

## What it does

- **Records audio** directly from the microphone, or accepts an uploaded audio/video file.
- **Transcribes speech to text** entirely on-device, using a small local copy of OpenAI's Whisper model (no cloud API, no internet required).
- **Runs a local language model** (via [llama.cpp](https://github.com/ggml-org/llama.cpp)) to process that transcript — for example, turning a rambling conversation into structured notes.
- **Stores everything locally** in a SQLite database on the user's machine.

Because every step happens on the device itself, this is designed for environments where sending patient conversations to a third-party server isn't acceptable — clinics with poor or no internet access, or settings with strict data-residency requirements.

## How it's built, at a glance

```
Microphone / file
      │
      ▼
React UI (the window you see)
      │  records audio, sends raw bytes to Rust
      ▼
Rust backend (runs natively, not in a browser)
      │
      ├── Symphonia + libopus  →  decode whatever audio format came in
      ├── Rubato               →  resample it to 16kHz mono
      ├── whisper-rs           →  transcribe it to text (Whisper tiny.en model)
      └── llama-server         →  a small local AI model, running as its own
                                   background process, for any further
                                   text processing
      │
      ▼
SQLite database (on disk, never uploaded anywhere)
```

The React frontend never talks to the internet for any of this — it calls into the Rust backend through Tauri's built-in command system, and the Rust backend does all the actual work using libraries that run entirely on the local CPU.

### Why a local AI model instead of an API?

Calling an AI service over the internet (like the ones behind ChatGPT or similar tools) is simpler to build, but it means sending the conversation's content to someone else's server. ClinicalEdge instead bundles a small, efficient model that runs directly on the user's computer. It's not as capable as the largest cloud models, but it never has to leave the building.

## Project layout

```
tauri-app/
├── src/                    # React frontend (what the user sees and clicks)
│   └── components/
│       └── WhisperPanel.tsx  # Recording / transcription UI
├── src-tauri/               # Rust backend (everything else)
│   ├── src/
│   │   ├── main.rs          # App startup, launches the local AI model
│   │   ├── audio.rs         # Audio decoding + resampling
│   │   ├── whisper.rs       # Speech-to-text transcription
│   │   └── payload.rs       # Packs the AI model + engine into the installer
│   ├── models/               # The Whisper speech-to-text model file
│   ├── binaries/              # The local AI engine + its supporting files
│   └── capabilities/          # Tauri's permission rules (what the UI is allowed to do)
└── tauri.conf.json           # App configuration
```

## Setting up a development environment

You'll need:

- **[Rust](https://www.rust-lang.org/tools/install)** — the language the backend is written in.
- **[Node.js](https://nodejs.org/) and [pnpm](https://pnpm.io/)** — for the frontend.
- **A C++ build toolchain** — required because some dependencies (the speech and audio libraries) compile native code at build time.
  - **Windows:** Visual Studio Build Tools, specifically the "Desktop development with C++" workload. You'll also need [CMake](https://cmake.org/) and [LLVM/clang](https://releases.llvm.org/) (used to generate some bindings; install LLVM's official Windows build, not a MinGW-bundled one — see note below).
  - **macOS:** Xcode Command Line Tools (`xcode-select --install`).
  - **Linux:** `build-essential` (or your distro's equivalent), plus `webkit2gtk` development packages for Tauri itself.
- **Git**, for fetching some dependencies during the build.

> **Windows note:** building this project requires Microsoft's MSVC compiler to be active in your terminal session. The easiest way is to open **"Developer PowerShell for VS 2022"** (search for it in the Start Menu) instead of a regular terminal — it sets up the compiler automatically. A plain PowerShell or Command Prompt window will not have the compiler available unless you run Visual Studio's setup script in it first.
>
> Also avoid running this project from a folder path containing spaces (e.g. a path under a Windows username with a space in it, like `C:\Users\Jane Doe\...`) — some build tools used here don't handle spaces in paths correctly. A path like `C:\dev\clinical-edge` is safer.

### First-time setup

```bash
pnpm install
```

This installs the frontend's JavaScript dependencies. The Rust dependencies install automatically the first time you build.

### Running it locally

```bash
cargo tauri dev
```

This starts the app in development mode: the frontend hot-reloads as you edit it, and Rust changes trigger an automatic rebuild. The **first build will take a while** — it's compiling the speech-recognition and audio libraries from source, which only happens once. Later builds are much faster.

### Building a release version

```bash
cargo tauri build
```

Produces an installable application for your current platform.

## Models and binaries

This project depends on a few large files that aren't part of the source code itself:

- A **Whisper speech-to-text model** (`src-tauri/models/`) — downloaded separately, not committed to git.
- A **local AI engine** (`src-tauri/binaries/llama-server`) plus its supporting files — also downloaded separately.

These are excluded from version control (see `.gitignore`) because they're large binary files that change independently of the source code. Ask a maintainer for the current download links, or check the project's internal setup notes for where to fetch them.

## A note on the audio format

Recordings are captured by the browser's built-in recording API, which produces different audio formats depending on the operating system — generally an Opus-encoded stream. The Rust backend handles decoding that automatically. If you're troubleshooting a "no audio track" or "unsupported codec" error, the relevant code is in `src-tauri/src/audio.rs`.

## Troubleshooting

**The first build fails with a compiler error mentioning `cl.exe` or "no such file" (Windows):** the MSVC compiler isn't active in your terminal. Open "Developer PowerShell for VS 2022" and try again from there.

**A build fails with "not enough space on disk":** Rust build artifacts (especially with the speech and AI libraries here) can use a surprising amount of disk space across repeated builds. Running `cargo clean` inside `src-tauri/` will remove old build output and free up space.

**Transcription fails after recording:** open the app's developer tools (right-click → Inspect, or press F12) and check the Console tab for the actual error — the status message shown in the app itself is intentionally brief.