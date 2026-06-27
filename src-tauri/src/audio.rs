// src-tauri/src/audio.rs
use symphonia::core::{
    audio::SampleBuffer, codecs::{CodecRegistry, DecoderOptions},
    formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};
use symphonia::default::register_enabled_codecs;
use symphonia_adapter_libopus::OpusDecoder;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use once_cell::sync::Lazy;

/// Custom codec registry: Symphonia's built-in defaults (FLAC, Vorbis, PCM, etc.)
/// plus Opus via libopus, since Symphonia has no native Opus decoder.
static CODECS: Lazy<CodecRegistry> = Lazy::new(|| {
    let mut registry = CodecRegistry::new();
    register_enabled_codecs(&mut registry);
    registry.register_all::<OpusDecoder>();
    registry
});

/// Decodes any audio/video byte stream to mono f32 samples at 16kHz.
/// Used internally by both `decode_audio` (IPC) and `transcribe_audio`.
pub fn decode_to_16k_mono(bytes: Vec<u8>) -> Result<Vec<f32>, String> {
    let cursor = std::io::Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| e.to_string())?;

    let mut format = probed.format;
    let track = format.default_track().ok_or("No audio track")?;
    let native_sr = track.codec_params.sample_rate.ok_or("Unknown sample rate")? as f64;
    let mut decoder = CODECS
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let track_id = track.id;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id { continue; }

        let decoded = decoder.decode(&packet).map_err(|e| e.to_string())?;
        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);

        let chans = spec.channels.count();
        let s = buf.samples();
        for frame in s.chunks(chans) {
            samples.push(frame.iter().sum::<f32>() / chans as f32);
        }
    }

    if (native_sr - 16000.0).abs() < 1.0 {
        return Ok(samples);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        16000.0 / native_sr,
        2.0,
        params,
        samples.len(),
        1,
    ).map_err(|e| e.to_string())?;

    let out = resampler
        .process(&[&samples], None)
        .map_err(|e| e.to_string())?;

    Ok(out.into_iter().next().unwrap_or_default())
}

/// Thin IPC wrapper — kept in case the frontend still calls this directly
/// (e.g. for waveform preview). Transcription no longer needs this round-trip.
#[tauri::command]
pub async fn decode_audio(bytes: Vec<u8>) -> Result<Vec<f32>, String> {
    decode_to_16k_mono(bytes)
}