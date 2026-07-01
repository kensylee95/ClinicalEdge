use ndarray::{Array2, ArrayD};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tauri::State;
use tokenizers::Tokenizer;

use crate::preprocessing::preprocess_image;

const MAX_NEW_TOKENS: usize = 64;
const BOS_TOKEN_ID: i64 = 2;
const EOS_TOKEN_ID: i64 = 2;

pub struct OcrEngine {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
}

impl OcrEngine {
    pub fn load(
        encoder_path: &Path,
        decoder_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, String> {
        // Match Python's ort.InferenceSession() defaults:
        //   - Level1: basic graph optimizations including op fusion
        //   - no explicit thread count: let ORT decide, same as Python default
        //
        // Note: the ort crate version must match the onnxruntime Python package
        // version for identical quantized model outputs.
        // ort 2.0.0-rc.12 bundles ORT 1.24.4 — ensure Python uses the same:
        //   pip install onnxruntime==1.24.4
        let encoder = Session::builder()
            .map_err(|e| format!("session builder (encoder): {e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| format!("set opt level (encoder): {e}"))?
            .commit_from_file(encoder_path)
            .map_err(|e| format!("load encoder: {e}"))?;

        let decoder = Session::builder()
            .map_err(|e| format!("session builder (decoder): {e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| format!("set opt level (decoder): {e}"))?
            .commit_from_file(decoder_path)
            .map_err(|e| format!("load decoder: {e}"))?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("load tokenizer: {e}"))?;

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
        })
    }
}

fn run_encoder(engine: &mut OcrEngine, img: &image::DynamicImage) -> Result<ArrayD<f32>, String> {
    let pixel_values = preprocess_image(img);
    let input_tensor = Tensor::from_array(pixel_values.into_dyn())
        .map_err(|e| format!("encoder input tensor: {e}"))?;

    let outputs = engine
        .encoder
        .run(ort::inputs!["pixel_values" => input_tensor])
        .map_err(|e| format!("encoder run: {e}"))?;

    let (_, hidden_view) = outputs.iter().next().ok_or("encoder produced no outputs")?;

    let hidden = hidden_view
        .try_extract_array::<f32>()
        .map_err(|e| format!("encoder output extract: {e}"))?
        .to_owned()
        .into_dyn();
    Ok(hidden)
}

fn run_decode_loop(
    engine: &mut OcrEngine,
    encoder_hidden: &ArrayD<f32>,
) -> Result<Vec<i64>, String> {
    let mut tokens: Vec<i64> = vec![BOS_TOKEN_ID];

    for _ in 0..MAX_NEW_TOKENS {
        let seq_len = tokens.len();
        let token_arr = Array2::from_shape_vec((1, seq_len), tokens.clone())
            .map_err(|e| format!("token array shape: {e}"))?;

        let token_tensor = Tensor::from_array(token_arr.into_dyn())
            .map_err(|e| format!("decoder token tensor: {e}"))?;
        let hidden_tensor = Tensor::from_array(encoder_hidden.clone())
            .map_err(|e| format!("decoder hidden tensor: {e}"))?;

        let outputs = engine
            .decoder
            .run(ort::inputs![
                "input_ids"             => token_tensor,
                "encoder_hidden_states" => hidden_tensor,
            ])
            .map_err(|e| format!("decoder run: {e}"))?;

        let (_, logits_value) = outputs.iter().next().ok_or("decoder produced no outputs")?;

        // .to_owned() forces a contiguous layout before indexing,
        // matching numpy's logits[0, -1] regardless of ORT's memory layout.
        let logits = logits_value
            .try_extract_array::<f32>()
            .map_err(|e| format!("decoder logits extract: {e}"))?
            .to_owned();

        let vocab_size = logits.shape()[2];
        let last_step = seq_len - 1;

        let next_token = (0..vocab_size)
            .map(|v| (v, logits[[0, last_step, v]]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(idx, _)| idx as i64)
            .ok_or("argmax on empty logits")?;

        tokens.push(next_token);

        if next_token == EOS_TOKEN_ID {
            break;
        }
    }

    Ok(tokens)
}

fn detokenize(engine: &OcrEngine, tokens: &[i64]) -> Result<String, String> {
    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    engine
        .tokenizer
        .decode(&ids, true) // skip_special_tokens = true, matches Python exactly
        .map_err(|e| format!("detokenize: {e}"))
}

pub fn recognize_line(engine: &mut OcrEngine, img: &image::DynamicImage) -> Result<String, String> {
    let encoder_hidden = run_encoder(engine, img)?;
    let tokens = run_decode_loop(engine, &encoder_hidden)?;
    detokenize(engine, &tokens)
}

#[tauri::command]
pub fn run_ocr(image_path: String, engine: State<Mutex<OcrEngine>>) -> Result<String, String> {
    let path = Path::new(&image_path);
    if !path.exists() {
        return Err(format!("image not found at {image_path}"));
    }

    let img = image::open(path).map_err(|e| format!("open image: {e}"))?;
    let mut engine = engine
        .lock()
        .map_err(|e| format!("engine lock poisoned: {e}"))?;
    recognize_line(&mut engine, &img)
}
