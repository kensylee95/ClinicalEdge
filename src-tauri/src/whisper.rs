// src-tauri/src/whisper.rs
use crate::audio::decode_to_16k_mono;
use std::sync::Mutex;
use tauri::{AppHandle, Manager};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext };

pub struct WhisperState(pub Mutex<WhisperContext>);

#[tauri::command]
pub async fn transcribe_audio(app: AppHandle, bytes: Vec<u8>) -> Result<String, String> {
    let pcm = decode_to_16k_mono(bytes)?;

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let state = app.state::<WhisperState>();
        let ctx = state.0.lock().unwrap();

        let mut wstate = ctx.create_state().map_err(|e| e.to_string())?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(4);
        params.set_language(Some("en"));
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        wstate.full(params, &pcm).map_err(|e| e.to_string())?;

        let n = wstate.full_n_segments();
        let mut text = String::new();
        for i in 0..n {
            let segment = wstate.get_segment(i).ok_or("segment index out of range")?;
            text.push_str(segment.to_str().map_err(|e| e.to_string())?.trim());
            text.push(' ');
        }

        Ok(text.trim().to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}
