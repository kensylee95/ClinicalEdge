// src-tauri/src/whisper.rs
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use tauri::{AppHandle, Manager};
use crate::audio::decode_to_16k_mono;

#[tauri::command]
pub async fn transcribe_audio(app: AppHandle, bytes: Vec<u8>) -> Result<String, String> {
    let pcm = decode_to_16k_mono(bytes)?;

    let model_path = app
        .path()
        .resolve("models/ggml-tiny.en.bin", tauri::path::BaseDirectory::Resource)
        .map_err(|e| e.to_string())?;

    let model_path_str = model_path
        .to_str()
        .ok_or("Model path is not valid UTF-8")?
        .to_string();

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let ctx = WhisperContext::new_with_params(
            &model_path_str,
            WhisperContextParameters::default(),
        ).map_err(|e| e.to_string())?;

        let mut state = ctx.create_state().map_err(|e| e.to_string())?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(4);
        params.set_language(Some("auto"));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.full(params, &pcm).map_err(|e| e.to_string())?;

        let n = state.full_n_segments();
        let mut text = String::new();
        for i in 0..n {
           let segment = state.get_segment(i).ok_or("segment index out of range")?;
            text.push_str(segment.to_str().map_err(|e| e.to_string())?.trim());
            text.push(' ');
        }

        Ok(text.trim().to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}