// src-tauri/src/main.rs
//#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod whisper;

use std::path::PathBuf;
use std::sync::Mutex;
use sysinfo::{Pid, System};
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
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .manage(LlamaProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            get_memory_info,
            audio::decode_audio,
            whisper::transcribe_audio,
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
            ).expect("failed to load whisper model");

            app.manage(WhisperState(Mutex::new(whisper_ctx)));

            // ── llama-server sidecar ──────────────────────────────────────
            let (mut rx, child) = app
                .shell()
                .sidecar("llama-server")
                .map_err(|e| { eprintln!("Sidecar find error: {e:?}"); format!("{e}") })?
                .args([
                    "-m", model_path.to_str().unwrap(),
                    "--port", "9191",
                    "--host", "127.0.0.1",
                    "-c", "1024",
                    "-t", "4",
                    "--n-gpu-layers", "0",
                    "--flash-attn",
                ])
                .spawn()
                .map_err(|e| { eprintln!("Sidecar spawn error: {e:?}"); format!("{e}") })?;

            tauri::async_runtime::spawn(async move {
                use tauri_plugin_shell::process::CommandEvent;
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line)  => eprintln!("[llama] {}", String::from_utf8_lossy(&line)),
                        CommandEvent::Stderr(line)  => eprintln!("[llama:err] {}", String::from_utf8_lossy(&line)),
                        CommandEvent::Error(e)      => eprintln!("[llama:error] {e}"),
                        CommandEvent::Terminated(s) => { eprintln!("[llama] terminated: {s:?}"); break; }
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