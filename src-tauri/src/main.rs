// src-tauri/src/main.rs
//
// Loads both ONNX engines once at app startup (not per-call): TrOCR
// (recognition, ocr.rs) and DBNet (detection, ocr_page.rs). Resolves
// bundled model files via Tauri's path resolver (installer-safe -- works
// regardless of where the app gets installed, unlike a hardcoded dev path),
// stores both engines in managed state, and registers both Tauri commands.
//
// Also manages the whisper transcription engine and llama-server sidecar.
//
// ocr_page.rs / run_ocr_page now uses the ONNX-based DBNet detector
// (Felix92/onnxtr-db-mobilenet-v3-large) via `ort`, replacing the earlier
// EAST/OpenCV-detection-model approach. NOTE: ocr_page.rs's postprocessing
// still uses the `opencv` crate internally for contour-finding, morphology,
// and minAreaRect -- those specific CV primitives, not the detection model
// itself, are why `opencv` remains a dependency.
//
// NOTE on ort version: written against ort 2.0.0-rc.12, which has NO
// separate Environment type to construct -- Session::builder() is
// self-contained per session.
//
// NOTE on Tauri version: written against Tauri 2.x. `app.path()` returns a
// `PathResolver`; resources are resolved via
// `.resolve(path, BaseDirectory::Resource)`.

//#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod ocr;
mod ocr_document;
mod ocr_page;
mod preprocessing;
mod whisper;

use ocr::OcrEngine;
use ocr_page::DetectorEngine;
use std::path::PathBuf;
use std::sync::Mutex;
use sysinfo::{Pid, System};
use tauri::path::BaseDirectory;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use whisper::WhisperState;
use whisper_rs::{WhisperContext, WhisperContextParameters};

struct LlamaProcess(Mutex<Option<tauri_plugin_shell::process::CommandChild>>);

fn resource_root(app: &tauri::App) -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("assets/model");
        println!("Dev mode: using local assets at {:?}", dev_path);
        return Ok(dev_path);
    }

    app.path()
        .resource_dir()
        .map_err(|e| format!("Cannot resolve resource dir: {e}"))
}

fn llm_model_path(resource_root: &PathBuf) -> PathBuf {
    if cfg!(debug_assertions) {
        resource_root.join("my_clinical_model.gguf")
    } else {
        resource_root.join("models").join("my_clinical_model.gguf")
    }
}

fn whisper_model_path(resource_root: &PathBuf) -> PathBuf {
    if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("ggml-tiny.en.bin")
    } else {
        resource_root.join("models").join("ggml-tiny.en.bin")
    }
}

#[tauri::command]
fn get_memory_info() -> serde_json::Value {
    let mut sys = System::new_all();
    sys.refresh_all();

    let pid = std::process::id();
    let app_ram_mb = sys
        .process(Pid::from_u32(pid))
        .map(|p| p.memory() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    let sys_used_gb = sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

    serde_json::json!({ "app": app_ram_mb, "sys": sys_used_gb })
}

fn main() {
    let ok = ort::init_from(r"C:\Users\HP EliteBook 840 G7\AppData\Roaming\Python\Python312\site-packages\onnxruntime\capi\onnxruntime.dll")
    .expect("failed to init ort with system onnxruntime.dll")
    .commit();
    println!("DEBUG ort init_from commit() returned: {}", ok);
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .manage(LlamaProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            get_memory_info,
            audio::decode_audio,
            whisper::transcribe_audio,
            ocr::run_ocr,
            ocr_page::run_ocr_page_handwriting,
            ocr_document::run_ocr_page_document,
        ])
        .setup(|app| {
            let root = resource_root(app).unwrap_or_else(|e| {
                eprintln!("Setup error: {e}");
                std::process::exit(1);
            });
            eprintln!("Resource root: {:?}", root);

            // ── LLM model check ──────────────────────────────────────────
            let model_path = llm_model_path(&root);
            eprintln!("LLM model path: {:?}", model_path);
            eprintln!("LLM model exists: {}", model_path.exists());

            if !model_path.exists() {
                eprintln!(
                    "Setup error: LLM model not found at {:?}. Check tauri.conf.json's \
                     bundle.resources mapping for my_clinical_model.gguf, and confirm \
                     assets/model/my_clinical_model.gguf exists before building.",
                    model_path
                );
                std::process::exit(1);
            }

            // ── Whisper model — load once, keep in memory ─────────────────
            let whisper_path = whisper_model_path(&root);
            eprintln!("Whisper model path: {:?}", whisper_path);
            eprintln!("Whisper model exists: {}", whisper_path.exists());

            let whisper_ctx = WhisperContext::new_with_params(
                whisper_path.to_str().unwrap(),
                WhisperContextParameters::default(),
            )
            .expect("failed to load whisper model");

            app.manage(WhisperState(Mutex::new(whisper_ctx)));

            // ── ONNX OCR engines — load once, keep in memory ─────────────
            let path = app.path();

            // Stage 2 (TrOCR recognition) resources -- required.
            let encoder_path = path
                .resolve(
                    "models/onnx/encoder_model_quantized.onnx",
                    BaseDirectory::Resource,
                )
                .map_err(|e| format!("could not resolve encoder_model_quantized.onnx: {e}"))?;
            let decoder_path = path
                .resolve(
                    "models/onnx/decoder_model_quantized.onnx",
                    BaseDirectory::Resource,
                )
                .map_err(|e| format!("could not resolve decoder_model_quantized.onnx: {e}"))?;
            let tokenizer_path = path
                .resolve("models/tokenizer.json", BaseDirectory::Resource)
                .map_err(|e| format!("could not resolve tokenizer.json: {e}"))?;

            // Stage 1 (DBNet line detector) resource.
            let detector_path = path
                .resolve(
                    "models/onnx/onnxtr_db_mobilenet_v3_large.onnx",
                    BaseDirectory::Resource,
                )
                .map_err(|e| format!("could not resolve db_mobilenet_v3_large.onnx: {e}"))?;

            // num_threads: match your target hardware's core count.
            // 4 intra-op threads suits a 4-core / 8-thread i5-8th-gen target;
            // tune after profiling on the real device.
            let num_threads: usize = 4;

            let ocr_engine = OcrEngine::load(&encoder_path, &decoder_path, &tokenizer_path)
                .map_err(|e| format!("load OcrEngine: {e}"))?;

            let detector = DetectorEngine::load(&detector_path, num_threads)
                .map_err(|e| format!("load DetectorEngine: {e}"))?;

            app.manage(Mutex::new(ocr_engine));
            app.manage(Mutex::new(detector));

            // ── llama-server sidecar ──────────────────────────────────────
            let (mut rx, child) = app
                .shell()
                .sidecar("llama-server")
                .map_err(|e| {
                    eprintln!("Sidecar find error: {e:?}");
                    format!("{e}")
                })?
                .args([
                    "-m",
                    model_path.to_str().unwrap(),
                    "--port",
                    "9191",
                    "--host",
                    "127.0.0.1",
                    "-c",
                    "1024",
                    "-t",
                    "4",
                    "--n-gpu-layers",
                    "0",
                    "--flash-attn",
                    "off",
                ])
                .spawn()
                .map_err(|e| {
                    eprintln!("Sidecar spawn error: {e:?}");
                    format!("{e}")
                })?;

            tauri::async_runtime::spawn(async move {
                use tauri_plugin_shell::process::CommandEvent;
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            eprintln!("[llama] {}", String::from_utf8_lossy(&line))
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("[llama:err] {}", String::from_utf8_lossy(&line))
                        }
                        CommandEvent::Error(e) => eprintln!("[llama:error] {e}"),
                        CommandEvent::Terminated(s) => {
                            eprintln!("[llama] terminated: {s:?}");
                            break;
                        }
                        _ => {}
                    }
                }
            });

            let state = app.state::<LlamaProcess>();
            *state.0.lock().unwrap() = Some(child);

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<LlamaProcess>();
                let mut guard = state.0.lock().unwrap();
                if let Some(child) = guard.take() {
                    eprintln!("Killing llama-server sidecar on window close");
                    let _ = child.kill();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
                let state = app_handle.state::<LlamaProcess>();
                let mut guard = state.0.lock().unwrap();
                if let Some(child) = guard.take() {
                    eprintln!("Killing llama-server sidecar on exit");
                    let _ = child.kill();
                }
            }
        });
}
