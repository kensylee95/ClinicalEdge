// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod payload;
mod audio;      // ADDED
mod whisper;    // ADDED

use std::path::PathBuf;
use std::sync::Mutex;
use sysinfo::{Pid, System};
use tauri::Manager;
use tauri_plugin_shell::ShellExt;

struct LlamaProcess(Mutex<Option<tauri_plugin_shell::process::CommandChild>>);

#[cfg(target_arch = "x86_64")]
fn pick_engine_tier() -> &'static str {
    if is_x86_feature_detected!("avx2") {
        "avx2"
    } else if is_x86_feature_detected!("avx") {
        "avx"
    } else {
        "noavx"
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn pick_engine_tier() -> &'static str {
    "avx2"
}

fn engine_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

fn ensure_engine_installed(app: &tauri::App) -> Result<PathBuf, String> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Cannot resolve app data dir: {e}"))?;

    std::fs::create_dir_all(&data_dir).map_err(|e| format!("Cannot create data dir: {e}"))?;

    if cfg!(debug_assertions) {
        let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("assets/model");
        println!("Dev mode: Using local assets at {:?}", dev_path);
        return Ok(dev_path);
    }

    let engine_dest = data_dir.join(engine_filename());
    let model_dest = data_dir.join("my_clinical_model.gguf");

    if engine_dest.exists() && model_dest.exists() {
        return Ok(data_dir);
    }

    let exe_path = std::env::current_exe().map_err(|e| format!("Cannot resolve current exe: {e}"))?;

    let footer = payload::Footer::read_from(&exe_path)
        .map_err(|e| format!("Cannot read embedded payload footer: {e}"))?
        .ok_or_else(|| {
            "This build has no embedded engine/model payload. Run scripts/pack.sh for production.".to_string()
        })?;

    Ok(data_dir)
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

#[tauri::command]
fn get_cpu_tier() -> &'static str {
    pick_engine_tier()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .manage(LlamaProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            get_memory_info,
            get_cpu_tier,
            audio::decode_audio,        // ADDED
            whisper::transcribe_audio,  // ADDED
        ])
        .setup(|app| {
            let data_dir = ensure_engine_installed(app).unwrap_or_else(|e| {
                eprintln!("Setup error: {e}");
                std::process::exit(1);
            });

            let model_path = data_dir.join("my_clinical_model.gguf");
            eprintln!("Model path: {:?}", model_path);
            eprintln!("Model exists: {}", model_path.exists());

            let (mut rx, child) = app
                .shell()
                .sidecar("llama-server")
                .map_err(|e| { eprintln!("Sidecar find error: {e:?}"); format!("{e}") })?
                .args([
                    "-m", model_path.to_str().unwrap(),
                    "--port", "9191",
                    "--host", "127.0.0.1",
                    "-c", "2048",
                    "-t", "4",
                    "--n-gpu-layers", "0",
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