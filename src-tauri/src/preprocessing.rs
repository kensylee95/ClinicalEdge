//! PIL-compatible image preprocessing for OCR inference.
//!
//! This module is intentionally isolated from the inference engine so that
//! verified preprocessing logic is never accidentally modified when working
//! on model/session changes.
//!
//! Verified byte-exact against real Pillow 12.1.1 output for upscale,
//! downscale, and the production 384x384 target size. Do not modify unless
//! the Python pipeline's preprocessing changes.

use ndarray::Array4;

pub const IMG_SIZE: u32 = 384;
use std::fs::File;
use std::io::Write;

// ---------------------------------------------------------------------------
// PIL-exact bicubic resize
//
// `image`'s FilterType::CatmullRom is NOT pixel-identical to PIL's default
// BICUBIC. PIL uses a separable two-pass convolution with:
//   - Keys cubic kernel, a = -0.5
//   - filterscale support-widening on downscale
//   - round/clamp to byte range BETWEEN the horizontal and vertical passes
//     (not just at the final output)
//
// All three details are implemented here and verified numerically.
// ---------------------------------------------------------------------------

const CUBIC_SUPPORT: f64 = 2.0;

/// Keys cubic convolution kernel, a = -0.5 (PIL's "bicubic").
fn cubic_kernel(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct AxisCoeffs {
    bounds: Vec<(usize, usize)>,
    weights: Vec<Vec<f64>>,
}

/// Per-output-pixel source bounds + normalized weights.
/// Mirrors PIL's `precompute_coeffs` in Resample.c, including
/// filterscale support-widening when downscaling.
fn precompute_coeffs(in_size: usize, out_size: usize) -> AxisCoeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = CUBIC_SUPPORT * filterscale;

    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = Vec::with_capacity(out_size);

    for out_x in 0..out_size {
        let center = (out_x as f64 + 0.5) * scale;

        let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as i64).min(in_size as i64) as usize;

        let mut ws: Vec<f64> = (xmin..xmax)
            .map(|x| cubic_kernel((x as f64 - center + 0.5) / filterscale))
            .collect();

        let total: f64 = ws.iter().sum();
        if total != 0.0 {
            ws.iter_mut().for_each(|w| *w /= total);
        }

        bounds.push((xmin, xmax));
        weights.push(ws);
    }

    AxisCoeffs { bounds, weights }
}

/// Clamp to [0, 255] and round-half-up in place (PIL's `clip8` macro).
fn clip_round_byte(buf: &mut [f64]) {
    for v in buf.iter_mut() {
        *v = (v.clamp(0.0, 255.0) + 0.5).floor();
    }
}

/// Resize along the width axis: (in_h, in_w, 3) → (in_h, out_w, 3).
fn resize_width(src: &[f64], in_w: usize, in_h: usize, out_w: usize) -> Vec<f64> {
    let coeffs = precompute_coeffs(in_w, out_w);
    let mut out = vec![0.0f64; in_h * out_w * 3];

    for y in 0..in_h {
        let row_offset = y * in_w * 3;
        for ox in 0..out_w {
            let (xmin, xmax) = coeffs.bounds[ox];
            let ws = &coeffs.weights[ox];
            let mut acc = [0.0f64; 3];
            for (i, x) in (xmin..xmax).enumerate() {
                let w = ws[i];
                let px = row_offset + x * 3;
                acc[0] += src[px] * w;
                acc[1] += src[px + 1] * w;
                acc[2] += src[px + 2] * w;
            }
            let out_px = (y * out_w + ox) * 3;
            out[out_px] = acc[0];
            out[out_px + 1] = acc[1];
            out[out_px + 2] = acc[2];
        }
    }

    out
}

/// Resize along the height axis: (in_h, img_w, 3) → (out_h, img_w, 3).
fn resize_height(src: &[f64], img_w: usize, in_h: usize, out_h: usize) -> Vec<f64> {
    let coeffs = precompute_coeffs(in_h, out_h);
    let mut out = vec![0.0f64; out_h * img_w * 3];

    for x in 0..img_w {
        for oy in 0..out_h {
            let (ymin, ymax) = coeffs.bounds[oy];
            let ws = &coeffs.weights[oy];
            let mut acc = [0.0f64; 3];
            for (i, y) in (ymin..ymax).enumerate() {
                let wt = ws[i];
                let px = (y * img_w + x) * 3;
                acc[0] += src[px] * wt;
                acc[1] += src[px + 1] * wt;
                acc[2] += src[px + 2] * wt;
            }
            let out_px = (oy * img_w + x) * 3;
            out[out_px] = acc[0];
            out[out_px + 1] = acc[1];
            out[out_px + 2] = acc[2];
        }
    }

    out
}

/// PIL-exact bicubic resize of an RGB8 image to `out_w` × `out_h`.
/// Matches `Image.resize((out_w, out_h))` with PIL's default BICUBIC filter.
fn pil_bicubic_resize_rgb(img: &image::RgbImage, out_w: u32, out_h: u32) -> Vec<f64> {
    let in_w = img.width() as usize;
    let in_h = img.height() as usize;
    let out_w = out_w as usize;
    let out_h = out_h as usize;

    // Flatten RGB image to flat HWC f64 buffer.
    let mut src = vec![0.0f64; in_h * in_w * 3];
    for y in 0..in_h {
        for x in 0..in_w {
            let p = img.get_pixel(x as u32, y as u32);
            let idx = (y * in_w + x) * 3;
            src[idx] = p[0] as f64;
            src[idx + 1] = p[1] as f64;
            src[idx + 2] = p[2] as f64;
        }
    }

    // Horizontal pass → clamp to byte range → vertical pass → clamp.
    // The between-pass clamp is required to match PIL's Resample.c exactly.
    let mut horiz = resize_width(&src, in_w, in_h, out_w);
    clip_round_byte(&mut horiz);

    let mut result = resize_height(&horiz, out_w, in_h, out_h);
    clip_round_byte(&mut result);

    result
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Preprocess a `DynamicImage` into a model-ready `Array4<f32>`.
///
/// Mirrors the Python pipeline exactly:
/// ```python
/// img = Image.open(path).convert("RGB").resize((384, 384))
/// arr = np.array(img).astype(np.float32) / 255.0
/// arr = (arr - 0.5) / 0.5
/// arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]
/// ```
pub fn preprocess_image(img: &image::DynamicImage) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let resized = pil_bicubic_resize_rgb(&rgb, IMG_SIZE, IMG_SIZE);

    let sz = IMG_SIZE as usize;
    let mut arr = Array4::<f32>::zeros((1, 3, sz, sz));

    for y in 0..sz {
        for x in 0..sz {
            let px = (y * sz + x) * 3;
            for c in 0..3 {
                let v = resized[px + c] as f32 / 255.0;
                arr[[0, c, y, x]] = (v - 0.5) / 0.5;
            }
        }
    }
    arr
}
