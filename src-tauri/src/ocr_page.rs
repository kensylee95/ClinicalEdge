// src-tauri/src/ocr_page.rs
//
// Full-page OCR. Pairs a DBNet text-detection model (finds where lines of
// text are on a page) with the existing TrOCR pipeline in ocr.rs (reads
// what a single line says).
//
// REWRITTEN to drop the `opencv` crate entirely. The previous version used
// opencv for contour-finding, morphological opening, minAreaRect, and
// perspective warp -- but opencv-rust needs a real OpenCV install (vcpkg on
// Windows) plus a working C/C++ build toolchain (Ninja or NMake), which
// turned into a genuine multi-hour setup wall (vcpkg bootstrap, disk space,
// no pkg-config/cmake generator configured). Given the goal is a low-end
// device that "just works," that native dependency wasn't worth keeping.
//
// What changed:
//   - Contour-finding: now uses the `imageproc` crate's
//     find_contours_with_threshold (confirmed real API, pure Rust, already
//     widely used -- 7M+ downloads). Replaces cv2.findContours.
//   - Morphological opening, threshold, minAreaRect, fillPoly+mean scoring,
//     and perspective warp: hand-implemented directly over raw pixel
//     buffers below. These are well-defined, bounded algorithms (not
//     guessed against an unfamiliar crate) -- each is commented with what
//     OpenCV operation it stands in for, so the mapping to the original
//     verified OnnxTR postprocessing spec (still accurate, see ocr.rs-
//     adjacent HANDOFF.md) is traceable.
//   - `opencv` dependency removed from Cargo.toml entirely. No native
//     C/C++ toolchain or vcpkg needed anymore -- this file is pure Rust.
//   - NEW: a post-detection, pre-recognition "line reconstruction" pass
//     (`merge_fragmented_lines`) using `rstar` (spatial index) +
//     `petgraph`'s `UnionFind` (connected components) to merge DBNet
//     fragments that are really one handwritten line split apart by pen
//     lifts / cursive gaps / slant. This runs between `detect_lines` and
//     sorting/cropping in `run_ocr_page_inner`. See HANDOFF.md for the
//     rationale and tuning notes.
//   - NEW (debug): two opt-in instrumentation hooks for diagnosing missed
//     detections (see "DEBUG INSTRUMENTATION" comments below):
//       * OCR_DEBUG_PROBA=1   dumps the raw DBNet probability map as a
//         grayscale heatmap (ocr_debug_proba.png) before any thresholding,
//         so you can see whether the model activated on a given stroke at
//         all.
//       * OCR_DEBUG_STAGES=1  prints a contour/box survivor count after
//         each filtering stage in postprocess_to_boxes (threshold -> morph
//         open -> raw contours -> outer-only -> size filter -> score
//         filter -> unclip), so you can see exactly which stage drops a
//         given region.
//
// Detection model source:
//   https://huggingface.co/Felix92/onnxtr-db-mobilenet-v3-large
//
// Confirmed real config for db_mobilenet_v3_large (unchanged from before):
//   input_shape = (3, 1024, 1024), mean = (0.798, 0.785, 0.772),
//   std = (0.264, 0.2749, 0.287)
//   DBNet model defaults: bin_thresh = 0.3, box_thresh = 0.1
//   GeneralDetectionPostProcessor: unclip_ratio = 1.5
//   Preprocessing: preserve_aspect_ratio = True, symmetric_pad = True
//     (letterbox: resize so the longer side fits 1024, pad the shorter
//     side symmetrically on both sides to reach 1024x1024)
//
// Tauri command React calls via:
//   await invoke("run_ocr_page_handwriting", { imagePath: "/path/to/full_page.png" })

use image::{GrayImage, Luma, RgbImage};
use imageproc::contours::{find_contours_with_threshold, BorderType};
use ndarray::{Array4, ArrayD};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use geo::{Area, BoundingRect as GeoBoundingRect, ConvexHull, EuclideanLength};
use geo::{Coord, LineString, Polygon};
use petgraph::unionfind::UnionFind;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tauri::State;

use crate::ocr::{recognize_line, OcrEngine};

// ---- Confirmed real config, from onnxtr's differentiable_binarization.py ----
const DET_INPUT_SIZE: u32 = 1024;
const DET_MEAN: [f32; 3] = [0.798, 0.785, 0.772];
const DET_STD: [f32; 3] = [0.264, 0.2749, 0.287];
const BIN_THRESH: f32 = 0.3;
const BOX_THRESH: f32 = 0.1;
const UNCLIP_RATIO: f64 = 1.5;
const MIN_CONTOUR_SIZE: i32 = 6; // was 2 -- too small to filter real noise speckle

// ---- Line-merging tuning constants. Starting points -- adjust against
// real handwriting samples. See `dump_debug_overlay` below for a way to
// iterate on these visually instead of guessing from OCR output alone. ----
const MERGE_MAX_ANGLE_DIFF_RAD: f32 = 0.12; // ~7 degrees
const MERGE_MAX_VGAP_RATIO: f32 = 0.5; // vertical gap vs avg line height
const MERGE_MAX_HGAP_RATIO: f32 = 1.2; // horizontal gap vs avg line height

/// One detected text line, in ORIGINAL PAGE pixel coordinates (already
/// remapped back from the 1024x1024 letterboxed detector frame).
pub struct DetectedLine {
    pub quad: [(f32, f32); 4],
    #[allow(dead_code)]
    pub score: f32,
}

impl DetectedLine {
    fn approx_y(&self) -> f32 {
        self.quad.iter().map(|p| p.1).sum::<f32>() / 4.0
    }
}

struct LetterboxInfo {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    orig_width: f32,
    orig_height: f32,
}

/// A minimal rotated-rectangle type, replacing opencv::core::RotatedRect.
/// `points()` returns the 4 corners in the same clockwise-from-bottom-left
/// convention used throughout this file's geometry.
struct RotatedRect {
    points: [(f32, f32); 4],
}

/// Holds the loaded DBNet ONNX session.
pub struct DetectorEngine {
    session: Session,
}

impl DetectorEngine {
    pub fn load(model_path: &Path, num_threads: usize) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("session builder (detector): {e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| format!("set opt level (detector): {e}"))?
            .with_intra_threads(num_threads)
            .map_err(|e| format!("set threads (detector): {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("load detector onnx: {e}"))?;

        if session.inputs().is_empty() {
            return Err("detector model has no inputs".to_string());
        }
        if session.outputs().is_empty() {
            return Err("detector model has no outputs".to_string());
        }

        Ok(Self { session })
    }
}

/// Letterbox-resize: scale the image so its longer side fits DET_INPUT_SIZE,
/// then pad the shorter dimension symmetrically to reach a square
/// DET_INPUT_SIZE x DET_INPUT_SIZE frame. Matches OnnxTR's confirmed
/// default (preserve_aspect_ratio=True, symmetric_pad=True).
fn letterbox_resize(img: &image::DynamicImage) -> (image::DynamicImage, LetterboxInfo) {
    use image::imageops::FilterType;

    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
    let scale = (DET_INPUT_SIZE as f32 / orig_w).min(DET_INPUT_SIZE as f32 / orig_h);
    let new_w = (orig_w * scale).round() as u32;
    let new_h = (orig_h * scale).round() as u32;

    let resized = img.resize_exact(new_w.max(1), new_h.max(1), FilterType::Triangle);

    let pad_x = ((DET_INPUT_SIZE - new_w) as f32) / 2.0;
    let pad_y = ((DET_INPUT_SIZE - new_h) as f32) / 2.0;

    let mut canvas = image::DynamicImage::new_rgb8(DET_INPUT_SIZE, DET_INPUT_SIZE);
    image::imageops::overlay(
        &mut canvas,
        &resized,
        pad_x.round() as i64,
        pad_y.round() as i64,
    );

    (
        canvas,
        LetterboxInfo {
            scale,
            pad_x,
            pad_y,
            orig_width: orig_w,
            orig_height: orig_h,
        },
    )
}

/// Normalize per DET_MEAN/DET_STD and arrange into NCHW float32 tensor.
fn preprocess_for_detector(img: &image::DynamicImage) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let mut arr = Array4::<f32>::zeros((1, 3, DET_INPUT_SIZE as usize, DET_INPUT_SIZE as usize));
    for y in 0..DET_INPUT_SIZE {
        for x in 0..DET_INPUT_SIZE {
            let pixel = rgb.get_pixel(x, y);
            for c in 0..3 {
                let v = pixel[c] as f32 / 255.0;
                arr[[0, c as usize, y as usize, x as usize]] =
                    (v - DET_MEAN[c as usize]) / DET_STD[c as usize];
            }
        }
    }
    arr
}

fn run_detector(engine: &mut DetectorEngine, input: Array4<f32>) -> Result<ArrayD<f32>, String> {
    let input_name = engine
        .session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or("detector session has no inputs")?;

    let input_tensor =
        Tensor::from_array(input.into_dyn()).map_err(|e| format!("detector input tensor: {e}"))?;

    let outputs = engine
        .session
        .run(ort::inputs![input_name => input_tensor])
        .map_err(|e| format!("detector run: {e}"))?;

    let (_, proba_value) = outputs
        .iter()
        .next()
        .ok_or("detector produced no outputs")?;
    let proba = proba_value
        .try_extract_array::<f32>()
        .map_err(|e| format!("detector output extract: {e}"))?
        .to_owned()
        .into_dyn();

    Ok(proba)
}

/// Squeeze the raw model output down to a flat (height, width, Vec<f32>)
/// probability plane, dropping any size-1 batch/channel dimensions.
/// Replaces the earlier opencv::core::Mat wrapping step.
fn proba_map_to_plane(proba: &ArrayD<f32>) -> Result<(usize, usize, Vec<f32>), String> {
    let kept_dims: Vec<usize> = proba.shape().iter().filter(|&&d| d != 1).cloned().collect();
    if kept_dims.len() != 2 {
        return Err(format!(
            "expected a 2D probability map after dropping size-1 dims, got shape {:?}",
            proba.shape()
        ));
    }
    let (h, w) = (kept_dims[0], kept_dims[1]);
    let squeezed = proba
        .clone()
        .into_shape_with_order(kept_dims)
        .map_err(|e| format!("squeeze proba map: {e}"))?;

    let mut plane = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            plane[y * w + x] = squeezed[[y, x]];
        }
    }
    Ok((h, w, plane))
}

/// DEBUG INSTRUMENTATION: dump the raw (pre-threshold) DBNet probability
/// plane as a grayscale heatmap. Gated behind OCR_DEBUG_PROBA so it never
/// runs in production. This answers the question "did the model activate
/// over this stroke at all?" independent of any postprocessing filter --
/// if a region is dark here, the bug is upstream (model input / detector
/// itself); if it's bright here but the region never produces a box, the
/// bug is in postprocess_to_boxes's filtering chain.
fn dump_proba_heatmap(h: usize, w: usize, plane: &[f32], out_path: &str) {
    let mut heat = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let v = (plane[y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            heat.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }
    if let Err(e) = heat.save(out_path) {
        eprintln!("[ocr_page] failed to save proba heatmap to {out_path}: {e}");
    } else {
        eprintln!("[ocr_page] wrote proba heatmap to {out_path}");
    }
}

/// Threshold a probability plane into a binary GrayImage. Replaces
/// cv2.threshold(..., THRESH_BINARY). Foreground pixels (>= thresh) become
/// 255, background becomes 0.
fn threshold_plane(h: usize, w: usize, plane: &[f32], thresh: f32) -> GrayImage {
    let mut img = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let v = if plane[y * w + x] >= thresh {
                255u8
            } else {
                0u8
            };
            img.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }
    img
}

/// Count foreground (255) pixels in a binary GrayImage. Debug helper only.
fn count_foreground(img: &GrayImage) -> u32 {
    img.pixels().filter(|p| p[0] > 0).count() as u32
}

/// Morphological opening (erosion then dilation) with a 3x3 square
/// structuring element, matching OpenCV's MORPH_OPEN with a
/// np.ones((3,3)) kernel exactly (8-connected neighborhood, all 9 cells
/// including center must be foreground for erosion to keep a pixel).
/// Denoising pass before contour-finding. NOTE: despite the original
/// comment here claiming this matches cv2.MORPH_OPEN (erode, then
/// dilate) with a 3x3 kernel, it does not -- it's erode -> dilate ->
/// erode. Tried correcting it to a textbook opening and it caused a
/// 0-to-32-quad swing on real handwriting samples (see HANDOFF.md):
/// standard opening alone isn't enough to clean the speckle noise that
/// survives thresholding on these probability maps (paper texture,
/// anti-aliasing around thin cursive strokes), and that noise either got
/// silently dropped downstream (masking the problem) or surfaced as
/// dozens of spurious tiny contours. Reverted to the original sequence,
/// which is closer to "open, then erode again" -- non-standard, but
/// empirically necessary for this detector's actual noise profile.
/// MIN_CONTOUR_SIZE below is also a backstop against any noise that
/// survives this step.
///
/// NOTE for debugging missed-thin-stroke cases: this sequence (erode ->
/// dilate -> erode) has a *net erosive* bias on anything that isn't
/// already a blob with margin -- thin/faint strokes that only narrowly
/// survive the initial threshold can be erased by the first erode and
/// never recover, even though a single textbook erode->dilate "opening"
/// would have preserved them. If OCR_DEBUG_STAGES shows foreground pixel
/// count for a known region dropping to near-zero between "post-threshold"
/// and "post-morph-open", this is the stage doing it.
fn morphological_open(img: &GrayImage) -> GrayImage {
    erode_3x3(&dilate_3x3(&erode_3x3(img)))
}

fn erode_3x3(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut all_fg = true;
            'check: for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    let fg = nx >= 0
                        && ny >= 0
                        && nx < w as i32
                        && ny < h as i32
                        && img.get_pixel(nx as u32, ny as u32)[0] > 0;
                    if !fg {
                        all_fg = false;
                        break 'check;
                    }
                }
            }
            out.put_pixel(x as u32, y as u32, Luma([if all_fg { 255 } else { 0 }]));
        }
    }
    out
}

fn dilate_3x3(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut any_fg = false;
            'check: for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0
                        && ny >= 0
                        && nx < w as i32
                        && ny < h as i32
                        && img.get_pixel(nx as u32, ny as u32)[0] > 0
                    {
                        any_fg = true;
                        break 'check;
                    }
                }
            }
            out.put_pixel(x as u32, y as u32, Luma([if any_fg { 255 } else { 0 }]));
        }
    }
    out
}

/// Build a closed geo::Polygon from a (possibly open) point ring. geo
/// expects the exterior ring's first and last points to match.
fn points_to_polygon(points: &[(f32, f32)]) -> Polygon<f64> {
    let mut coords: Vec<Coord<f64>> = points
        .iter()
        .map(|&(x, y)| Coord { x: x as f64, y: y as f64 })
        .collect();
    if let (Some(first), Some(last)) = (coords.first().cloned(), coords.last().cloned()) {
        if first.x != last.x || first.y != last.y {
            coords.push(first);
        }
    }
    Polygon::new(LineString::new(coords), vec![])
}

/// Axis-aligned bounding box of a set of points -- via geo's BoundingRect.
fn bounding_rect(points: &[(f32, f32)]) -> (f32, f32, f32, f32) {
    let poly = points_to_polygon(points);
    match GeoBoundingRect::bounding_rect(&poly) {
        Some(rect) => (
            rect.min().x as f32,
            rect.min().y as f32,
            (rect.max().x - rect.min().x) as f32,
            (rect.max().y - rect.min().y) as f32,
        ),
        None => (0.0, 0.0, 0.0, 0.0),
    }
}

/// Convex hull via geo's ConvexHull trait (Andrew's monotone chain
/// internally) -- replaces the convex-hull step implicit in
/// cv2.minAreaRect. Returns points in order, hull-closing point dropped.
fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        let mut pts: Vec<(f32, f32)> = points.to_vec();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
        pts.dedup();
        return pts;
    }
    let poly = points_to_polygon(points);
    let hull = poly.convex_hull();
    let mut hull_points: Vec<(f32, f32)> = hull
        .exterior()
        .coords()
        .map(|c| (c.x as f32, c.y as f32))
        .collect();
    // geo's hull is closed (first == last); drop the duplicate closing point
    // since the rest of this file works with open point rings.
    if hull_points.len() > 1 && hull_points.first() == hull_points.last() {
        hull_points.pop();
    }
    hull_points
}

/// Polygon area -- via geo's Area trait (shoelace internally). Replaces
/// cv2.contourArea.
fn contour_area(points: &[(f32, f32)]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    points_to_polygon(points).unsigned_area()
}

/// Closed-polygon perimeter -- via geo's EuclideanLength on the exterior
/// ring. Replaces cv2.arcLength(..., closed=True).
fn arc_length(points: &[(f32, f32)]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    points_to_polygon(points).exterior().euclidean_length()
}


/// Minimum-area rotated bounding rectangle of a point set, via rotating
/// calipers over the convex hull -- replaces cv2.minAreaRect. This is the
/// standard textbook algorithm (the same one OpenCV implements
/// internally): the minimum-area rectangle enclosing a convex polygon
/// always has one side flush with one of the hull's edges, so checking
/// every edge's orientation and taking the smallest-area fit is exact, not
/// an approximation.
fn min_area_rect(points: &[(f32, f32)]) -> RotatedRect {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        // Degenerate (collinear or single point) -- fall back to the
        // axis-aligned bounding box so callers always get 4 distinct
        // corners rather than a panic.
        let (x, y, w, h) = bounding_rect(points);
        return RotatedRect {
            points: [(x, y), (x, y + h), (x + w, y + h), (x + w, y)],
        };
    }

    let mut best_area = f32::INFINITY;
    let mut best_rect = [(0.0, 0.0); 4];

    let n = hull.len();
    for i in 0..n {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % n];
        let edge_angle = (p2.1 - p1.1).atan2(p2.0 - p1.0);
        let (cos_a, sin_a) = (edge_angle.cos(), edge_angle.sin());

        // Rotate every hull point into this edge's frame, find the
        // axis-aligned extent there, then rotate the resulting rect back.
        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for &(px, py) in &hull {
            let u = px * cos_a + py * sin_a;
            let v = -px * sin_a + py * cos_a;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }

        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            let corners_uv = [
                (min_u, min_v),
                (min_u, max_v),
                (max_u, max_v),
                (max_u, min_v),
            ];
            best_rect = corners_uv.map(|(u, v)| (u * cos_a - v * sin_a, u * sin_a + v * cos_a));
        }
    }

    RotatedRect { points: best_rect }
}

/// (convex_hull moved above, now backed by geo::ConvexHull)

/// (contour_area and arc_length moved above, now backed by geo::Area and
/// geo::EuclideanLength)

/// box_score: mean of the probability plane's values inside the polygon
/// region. Replaces cv2.fillPoly + masked mean (the
/// assume_straight_pages=False branch of DetectionPostProcessor.box_score).
/// Uses a standard scanline point-in-polygon fill, which is exactly what
/// fillPoly does internally.
fn box_score(h: usize, w: usize, plane: &[f32], points: &[(f32, f32)]) -> f32 {
    let ymin = points
        .iter()
        .map(|p| p.1)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as usize;
    let ymax = points
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(h as f32 - 1.0)
        .max(0.0) as usize;

    let mut sum = 0.0f64;
    let mut count = 0u32;

    for y in ymin..=ymax.min(h.saturating_sub(1)) {
        // Find x-intersections of this scanline with each polygon edge.
        let mut xs: Vec<f32> = Vec::new();
        let n = points.len();
        for i in 0..n {
            let (x1, y1) = points[i];
            let (x2, y2) = points[(i + 1) % n];
            let (y1, y2, x1, x2) = (y1, y2, x1, x2);
            let yf = y as f32 + 0.5;
            if (y1 <= yf && y2 > yf) || (y2 <= yf && y1 > yf) {
                let t = (yf - y1) / (y2 - y1);
                xs.push(x1 + t * (x2 - x1));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for pair in xs.chunks(2) {
            if pair.len() < 2 {
                continue;
            }
            let xstart = pair[0].max(0.0).round() as usize;
            let xend = pair[1].min(w as f32 - 1.0).round() as usize;
            for x in xstart..=xend.min(w.saturating_sub(1)) {
                sum += plane[y * w + x] as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Unclip (expand) a polygon outward by a distance derived from its
/// area/perimeter ratio and unclip_ratio. Previously this used a
/// centroid-radial approximation (each point pushed outward from the
/// polygon centroid) -- flagged in earlier versions of this file as the
/// one approximated, not-verified-exact step in the pipeline. Now uses
/// `geo-buffer`'s real polygon offset (a proper Minkowski-sum-style
/// buffer, the same family of operation pyclipper/Clipper perform in the
/// original OnnxTR Python reference), which is the actually-correct
/// operation DBNet's postprocessing spec calls for.
fn unclip_polygon(points: &[(f32, f32)], unclip_ratio: f64) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        return Vec::new();
    }
    let area = contour_area(points);
    let length = arc_length(points);
    if length == 0.0 {
        return Vec::new();
    }
    let distance = area * unclip_ratio / length;

    let poly = points_to_polygon(points);
    let buffered: geo::MultiPolygon<f64> = geo_buffer::buffer_polygon(&poly, distance);

    let largest = buffered
        .into_iter()
        .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap());

    match largest {
        Some(p) => p
            .exterior()
            .coords()
            .map(|c| (c.x as f32, c.y as f32))
            .collect(),
        None => {
            if std::env::var("OCR_DEBUG_UNCLIP").is_ok() {
                eprintln!(
                    "[unclip] geo_buffer returned empty for {} points, area={:.2}, length={:.2}, distance={:.2} -- falling back to centroid-radial approximation",
                    points.len(), area, length, distance
                );
            }
            unclip_polygon_fallback(points, distance)
        }
    }
}

/// Centroid-radial expansion fallback -- the original approximation this
/// file used before the geo_buffer swap. Used only if geo_buffer's
/// straight-skeleton offset fails (returns no polygons) for a given
/// input, e.g. on a very thin/near-degenerate detected box, so a box
/// is never silently dropped just because the more "correct" offset
/// algorithm chokes on it.
fn unclip_polygon_fallback(points: &[(f32, f32)], distance: f64) -> Vec<(f32, f32)> {
    let cx: f64 = points.iter().map(|p| p.0 as f64).sum::<f64>() / points.len() as f64;
    let cy: f64 = points.iter().map(|p| p.1 as f64).sum::<f64>() / points.len() as f64;

    points
        .iter()
        .map(|&(px, py)| {
            let dx = px as f64 - cx;
            let dy = py as f64 - cy;
            let norm = (dx * dx + dy * dy).sqrt().max(1e-6);
            (
                (px as f64 + (dx / norm) * distance) as f32,
                (py as f64 + (dy / norm) * distance) as f32,
            )
        })
        .collect()
}

/// Full postprocessing chain: threshold -> morphological opening ->
/// find_contours (imageproc) -> filter tiny contours -> box_score -> filter
/// by box_thresh -> unclip -> min_area_rect. Same algorithm as the prior
/// OpenCV version, now built entirely on imageproc + hand-rolled geometry.
///
/// DEBUG INSTRUMENTATION: when OCR_DEBUG_STAGES is set, prints a survivor
/// count after each filtering stage so you can see exactly which stage
/// drops a given region's contour(s). Compare this against
/// OCR_DEBUG_PROBA's heatmap: if a region is bright in the heatmap but its
/// contour count goes to zero at "post-morph-open", the erode-heavy
/// morphological_open is the culprit; if it survives morph-open but dies
/// at "post-size-filter", MIN_CONTOUR_SIZE is too aggressive for that
/// region's stroke width; if it survives size-filter but dies at
/// "post-score-filter", BOX_THRESH is cutting it.
fn postprocess_to_boxes(h: usize, w: usize, plane: &[f32]) -> Vec<([(f32, f32); 4], f32)> {
    let debug_stages = std::env::var("OCR_DEBUG_STAGES").is_ok();

    let binary = threshold_plane(h, w, plane, BIN_THRESH);
    if debug_stages {
        eprintln!(
            "[stages] post-threshold: {} foreground px (thresh={})",
            count_foreground(&binary),
            BIN_THRESH
        );
    }

    let opened = morphological_open(&binary);
    if debug_stages {
        eprintln!(
            "[stages] post-morph-open: {} foreground px",
            count_foreground(&opened)
        );
    }

    // find_contours_with_threshold treats pixels strictly > threshold as
    // foreground; our binary image is already 0/255, so threshold=0 is
    // the correct call here (matches cv2.RETR_EXTERNAL + CHAIN_APPROX_SIMPLE
    // closely enough for this use -- imageproc returns both Outer and Hole
    // borders, so Outer-only filtering below reproduces RETR_EXTERNAL).
    let contours = find_contours_with_threshold::<i32>(&opened, 0);
    if debug_stages {
        let outer_count = contours
            .iter()
            .filter(|c| c.border_type == BorderType::Outer)
            .count();
        eprintln!(
            "[stages] post-find-contours: {} total ({} outer)",
            contours.len(),
            outer_count
        );
    }

    let mut boxes = Vec::new();
    let mut dropped_by_size = 0u32;
    let mut dropped_by_score = 0u32;
    let mut dropped_by_unclip = 0u32;
    let mut survived = 0u32;

    for contour in contours
        .iter()
        .filter(|c| c.border_type == BorderType::Outer)
    {
        let points: Vec<(f32, f32)> = contour
            .points
            .iter()
            .map(|p| (p.x as f32, p.y as f32))
            .collect();
        if points.len() < 3 {
            continue;
        }

        let (rx, ry, rw, rh) = bounding_rect(&points);
        if (rw as i32) < MIN_CONTOUR_SIZE || (rh as i32) < MIN_CONTOUR_SIZE {
            dropped_by_size += 1;
            if debug_stages {
                eprintln!(
                    "[stages]   dropped by MIN_CONTOUR_SIZE: bbox=({:.0},{:.0}) {}x{} (limit {})",
                    rx, ry, rw, rh, MIN_CONTOUR_SIZE
                );
            }
            continue;
        }

        let rotated = min_area_rect(&points);
        let score = box_score(h, w, plane, &rotated.points);
        if score < BOX_THRESH {
            dropped_by_score += 1;
            if debug_stages {
                eprintln!(
                    "[stages]   dropped by BOX_THRESH: bbox=({:.0},{:.0}) {}x{} score={:.4} (limit {})",
                    rx, ry, rw, rh, score, BOX_THRESH
                );
            }
            continue;
        }

        let unclipped = unclip_polygon(&points, UNCLIP_RATIO);
        if unclipped.is_empty() {
            dropped_by_unclip += 1;
            if debug_stages {
                eprintln!(
                    "[stages]   dropped by empty unclip: bbox=({:.0},{:.0}) {}x{}",
                    rx, ry, rw, rh
                );
            }
            continue;
        }
        let expanded = min_area_rect(&unclipped);

        survived += 1;
        boxes.push((expanded.points, score));
    }

    if debug_stages {
        eprintln!(
            "[stages] post-size-filter dropped={}, post-score-filter dropped={}, post-unclip dropped={}, survived={}",
            dropped_by_size, dropped_by_score, dropped_by_unclip, survived
        );
    }

    boxes
}

/// Map a point from the letterboxed 1024x1024 detector frame back into the
/// original page image's pixel coordinates.
fn remap_point(p: (f32, f32), info: &LetterboxInfo) -> (f32, f32) {
    let orig_x = (p.0 - info.pad_x) / info.scale;
    let orig_y = (p.1 - info.pad_y) / info.scale;
    (
        orig_x.clamp(0.0, info.orig_width - 1.0),
        orig_y.clamp(0.0, info.orig_height - 1.0),
    )
}

fn detect_lines(
    detector: &mut DetectorEngine,
    page_img: &image::DynamicImage,
) -> Result<Vec<DetectedLine>, String> {
    let (letterboxed, info) = letterbox_resize(page_img);
    let input = preprocess_for_detector(&letterboxed);
    let proba = run_detector(detector, input)?;
    let (h, w, plane) = proba_map_to_plane(&proba)?;

    // DEBUG INSTRUMENTATION: see comment on dump_proba_heatmap above.
    if std::env::var("OCR_DEBUG_PROBA").is_ok() {
        dump_proba_heatmap(h, w, &plane, "ocr_debug_proba.png");
    }

    let raw_boxes = postprocess_to_boxes(h, w, &plane);

    let lines = raw_boxes
        .into_iter()
        .map(|(points, score)| {
            let quad = points.map(|p| remap_point(p, &info));
            DetectedLine { quad, score }
        })
        .collect();

    Ok(lines)
}

// ============================================================================
// Line reconstruction: merge DBNet fragments that are really one handwritten
// line split apart by pen lifts, cursive word gaps, or slant. Runs between
// detect_lines() and sorting/cropping. Uses `rstar` purely as a spatial
// index to avoid O(n^2) neighbor checks, and `petgraph`'s UnionFind for
// connected-components grouping once candidate pairs are confirmed by the
// angle/vgap/hgap geometry checks below.
// ============================================================================

/// Lightweight spatial-index entry for one detected line fragment. Indexed
/// by position in the original Vec<DetectedLine> rather than wrapping the
/// struct itself, since DetectedLine doesn't need to be Clone elsewhere.
#[derive(Clone)]
struct IndexedLine {
    idx: usize,
    bbox: [f32; 4], // [xmin, ymin, xmax, ymax]
    centroid: (f32, f32),
    angle: f32,
}

impl RTreeObject for IndexedLine {
    type Envelope = AABB<[f32; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.bbox[0], self.bbox[1]], [self.bbox[2], self.bbox[3]])
    }
}

impl PointDistance for IndexedLine {
    fn distance_2(&self, point: &[f32; 2]) -> f32 {
        let dx = self.centroid.0 - point[0];
        let dy = self.centroid.1 - point[1];
        dx * dx + dy * dy
    }
}

fn quad_bbox(quad: &[(f32, f32); 4]) -> [f32; 4] {
    let (x, y, w, h) = bounding_rect(quad);
    [x, y, x + w, y + h]
}

fn quad_centroid(quad: &[(f32, f32); 4]) -> (f32, f32) {
    let cx = quad.iter().map(|p| p.0).sum::<f32>() / 4.0;
    let cy = quad.iter().map(|p| p.1).sum::<f32>() / 4.0;
    (cx, cy)
}

/// Approximate text-line angle from the quad's longer edge (0..1 vs 1..2).
fn quad_angle(quad: &[(f32, f32); 4]) -> f32 {
    let edge01 = ((quad[1].0 - quad[0].0).powi(2) + (quad[1].1 - quad[0].1).powi(2)).sqrt();
    let edge12 = ((quad[2].0 - quad[1].0).powi(2) + (quad[2].1 - quad[1].1).powi(2)).sqrt();
    if edge01 >= edge12 {
        (quad[1].1 - quad[0].1).atan2(quad[1].0 - quad[0].0)
    } else {
        (quad[2].1 - quad[1].1).atan2(quad[2].0 - quad[1].0)
    }
}

/// Smallest angle between two line orientations, folded into [0, PI/2] so
/// e.g. a near-horizontal line and one rotated ~179 degrees from it (same
/// line, opposite winding) still reads as "aligned."
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b).abs() % std::f32::consts::PI;
    if d > std::f32::consts::FRAC_PI_2 {
        d = std::f32::consts::PI - d;
    }
    d
}

/// Merge DBNet fragments that are really one handwritten line. Returns one
/// DetectedLine per merged group, each with a quad re-derived from the
/// union of fragment points and re-unclipped so the merged line still gets
/// the same crop padding a single fresh detection would.
fn merge_fragmented_lines(lines: Vec<DetectedLine>) -> Vec<DetectedLine> {
    if lines.len() <= 1 {
        return lines;
    }

    let indexed: Vec<IndexedLine> = lines
        .iter()
        .enumerate()
        .map(|(idx, l)| IndexedLine {
            idx,
            bbox: quad_bbox(&l.quad),
            centroid: quad_centroid(&l.quad),
            angle: quad_angle(&l.quad),
        })
        .collect();

    let tree = RTree::bulk_load(indexed.clone());
    let mut uf = UnionFind::new(lines.len());

    let avg_height: f32 = {
        let total: f32 = lines.iter().map(|l| bounding_rect(&l.quad).3).sum();
        (total / lines.len() as f32).max(1.0)
    };

    for a in &indexed {
        // Generous search radius; the angle/vgap/hgap checks below do the
        // actual filtering, this just bounds the candidate set.
        let radius = avg_height * (MERGE_MAX_HGAP_RATIO + 1.0);
        let candidates =
            tree.locate_within_distance([a.centroid.0, a.centroid.1], radius * radius);

        for b in candidates {
            if a.idx == b.idx {
                continue;
            }

            let ang_diff = angle_diff(a.angle, b.angle);
            let vgap = (a.centroid.1 - b.centroid.1).abs();
            let hgap = if a.bbox[0] < b.bbox[0] {
                b.bbox[0] - a.bbox[2]
            } else {
                a.bbox[0] - b.bbox[2]
            };

            if std::env::var("OCR_DEBUG_MERGE").is_ok() {
                eprintln!(
                    "[merge] pair ({}, {}): angle_diff={:.4} (limit {:.4}), vgap={:.2} (limit {:.2}), hgap={:.2} (limit {:.2})",
                    a.idx,
                    b.idx,
                    ang_diff,
                    MERGE_MAX_ANGLE_DIFF_RAD,
                    vgap,
                    avg_height * MERGE_MAX_VGAP_RATIO,
                    hgap,
                    avg_height * MERGE_MAX_HGAP_RATIO,
                );
            }

            if ang_diff > MERGE_MAX_ANGLE_DIFF_RAD {
                continue;
            }

            if vgap > avg_height * MERGE_MAX_VGAP_RATIO {
                continue;
            }

            if hgap > avg_height * MERGE_MAX_HGAP_RATIO {
                continue;
            }

            uf.union(a.idx, b.idx);
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..lines.len() {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    groups
        .into_values()
        .map(|members| {
            if members.len() == 1 {
                let l = &lines[members[0]];
                return DetectedLine {
                    quad: l.quad,
                    score: l.score,
                };
            }

            let mut all_points: Vec<(f32, f32)> = Vec::new();
            let mut score_sum = 0.0f32;
            for &m in &members {
                all_points.extend_from_slice(&lines[m].quad);
                score_sum += lines[m].score;
            }

            let merged_rect = min_area_rect(&all_points);
            let unclipped = unclip_polygon(&merged_rect.points, UNCLIP_RATIO);
            let final_quad = if unclipped.is_empty() {
                merged_rect.points
            } else {
                min_area_rect(&unclipped).points
            };

            DetectedLine {
                quad: final_quad,
                score: score_sum / members.len() as f32,
            }
        })
        .collect()
}

// ============================================================================
// Column detection: a single y-sort assumes one column. On multi-column
// pages, lines from different columns interleave by vertical position and
// the resulting reading order is scrambled. This pass groups lines into
// columns by finding gaps in the horizontal distribution of line centers,
// then orders columns left-to-right and lines top-to-bottom within each.
// Runs after merge_fragmented_lines, replacing the old flat sort_by(approx_y).
// ============================================================================

const COLUMN_HISTOGRAM_BINS: usize = 100;
// A gap must be at least this fraction of the average line height to count
// as a real column gutter rather than ordinary word/letter spacing.
const COLUMN_MIN_GAP_RATIO: f32 = 1.5;
// A gap region must span at least this many consecutive empty bins to be
// trusted (avoids single-bin noise from histogram quantization).
const COLUMN_MIN_GAP_BINS: usize = 2;

/// Split lines into left-to-right column groups using a horizontal
/// histogram of line centroids. Falls back to a single column (the
/// original whole-page behavior) if no sufficiently wide gap is found.
fn split_into_columns(lines: Vec<DetectedLine>) -> Vec<Vec<DetectedLine>> {
    if lines.len() <= 1 {
        return vec![lines];
    }

    let avg_height: f32 = {
        let total: f32 = lines.iter().map(|l| bounding_rect(&l.quad).3).sum();
        (total / lines.len() as f32).max(1.0)
    };

    let centers: Vec<f32> = lines.iter().map(|l| quad_centroid(&l.quad).0).collect();
    let xmin = centers.iter().cloned().fold(f32::INFINITY, f32::min);
    let xmax = centers.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (xmax - xmin).max(1.0);
    let bin_width = span / COLUMN_HISTOGRAM_BINS as f32;

    let min_gap_width = avg_height * COLUMN_MIN_GAP_RATIO;
    let min_gap_bins = COLUMN_MIN_GAP_BINS.max((min_gap_width / bin_width).ceil() as usize);

    let mut counts = vec![0u32; COLUMN_HISTOGRAM_BINS];
    for &c in &centers {
        let mut bin = (((c - xmin) / bin_width) as usize).min(COLUMN_HISTOGRAM_BINS - 1);
        if bin >= COLUMN_HISTOGRAM_BINS {
            bin = COLUMN_HISTOGRAM_BINS - 1;
        }
        counts[bin] += 1;
    }

    // Find runs of empty bins long enough to count as column gutters, and
    // record the x boundary at the midpoint of each run.
    let mut boundaries: Vec<f32> = Vec::new();
    let mut run_start: Option<usize> = None;
    for i in 0..COLUMN_HISTOGRAM_BINS {
        if counts[i] == 0 {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start.take() {
            let run_len = i - start;
            if run_len >= min_gap_bins {
                let mid_bin = start as f32 + run_len as f32 / 2.0;
                boundaries.push(xmin + mid_bin * bin_width);
            }
        }
    }
    // (A trailing empty run with no following non-empty bin isn't a gutter
    // between columns -- it's just trailing whitespace -- so it's
    // intentionally not closed out here.)

    if boundaries.is_empty() {
        return vec![lines];
    }

    let mut columns: Vec<Vec<DetectedLine>> = (0..=boundaries.len()).map(|_| Vec::new()).collect();
    for line in lines {
        let cx = quad_centroid(&line.quad).0;
        let col = boundaries.iter().filter(|&&b| cx > b).count();
        columns[col].push(line);
    }

    columns.retain(|c| !c.is_empty());
    columns
}

/// Order lines for OCR: split into columns left-to-right, sort each column
/// top-to-bottom, concatenate. Single-column pages behave exactly as the
/// old flat sort did (split_into_columns returns one group in that case).
fn order_lines_for_reading(lines: Vec<DetectedLine>) -> Vec<DetectedLine> {
    let mut columns = split_into_columns(lines);
    for col in &mut columns {
        col.sort_by(|a, b| a.approx_y().partial_cmp(&b.approx_y()).unwrap());
    }
    columns.into_iter().flatten().collect()
}

/// Debug helper: draws detected-line quads as colored outlines on a copy of
/// the page image and saves it to disk, so merge-threshold tuning can be
/// done visually instead of by inference on OCR output quality alone. Not
/// wired into the Tauri command -- call manually from a test or a temporary
/// debug branch, e.g.:
///   dump_debug_overlay(&page_img.to_rgb8(), &lines, "/tmp/debug_overlay.png");
fn dump_debug_overlay(page_img: &RgbImage, lines: &[DetectedLine], out_path: &str) {
    use image::Rgb;

    let mut canvas = page_img.clone();
    let (w, h) = canvas.dimensions();

    let colors = [
        Rgb([255, 0, 0]),
        Rgb([0, 200, 0]),
        Rgb([0, 100, 255]),
        Rgb([255, 165, 0]),
        Rgb([200, 0, 200]),
    ];

    for (i, line) in lines.iter().enumerate() {
        let color = colors[i % colors.len()];
        for edge in 0..4 {
            let (x0, y0) = line.quad[edge];
            let (x1, y1) = line.quad[(edge + 1) % 4];
            draw_line_segment(&mut canvas, x0, y0, x1, y1, color, w, h);
        }
    }

    if let Err(e) = canvas.save(out_path) {
        eprintln!("dump_debug_overlay: failed to save {out_path}: {e}");
    }
}

/// Simple Bresenham-style line draw for the debug overlay above. Not used
/// anywhere in the production pipeline.
fn draw_line_segment(
    img: &mut RgbImage,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: image::Rgb<u8>,
    w: u32,
    h: u32,
) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs())).ceil().max(1.0) as i32;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (x0 + (x1 - x0) * t).round() as i32;
        let y = (y0 + (y1 - y0) * t).round() as i32;
        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

/// Perspective transform: solve for the 8 coefficients of a projective
/// homography mapping `src` quad -> `dst` quad, replacing
/// cv2.getPerspectiveTransform. Standard direct linear solve of the
/// 8x8 system derived from the 4 point correspondences.
fn get_perspective_transform(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> [[f64; 3]; 3] {
    // Build the 8x8 linear system A*h = b for homography coefficients
    // [a,b,c,d,e,f,g,h] with the 3x3 matrix normalized so the bottom-right
    // entry is 1.
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];

    for i in 0..4 {
        let (sx, sy) = (src[i].0 as f64, src[i].1 as f64);
        let (dx, dy) = (dst[i].0 as f64, dst[i].1 as f64);

        a[2 * i] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy];
        b[2 * i] = dx;
        a[2 * i + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy];
        b[2 * i + 1] = dy;
    }

    let h = solve_8x8(a, b);

    [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
}

/// Gaussian elimination with partial pivoting for the 8x8 homography
/// system. Small, fixed-size, well-conditioned for 4 non-degenerate point
/// correspondences -- no need for a general linear-algebra crate here.
fn solve_8x8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> [f64; 8] {
    for col in 0..8 {
        let mut pivot_row = col;
        for row in (col + 1)..8 {
            if a[row][col].abs() > a[pivot_row][col].abs() {
                pivot_row = row;
            }
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot = a[col][col];
        if pivot.abs() < 1e-12 {
            continue; // degenerate; leave as-is rather than divide by ~0
        }
        for row in (col + 1)..8 {
            let factor = a[row][col] / pivot;
            for k in col..8 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0f64; 8];
    for row in (0..8).rev() {
        let mut sum = b[row];
        for col in (row + 1)..8 {
            sum -= a[row][col] * x[col];
        }
        x[row] = if a[row][row].abs() > 1e-12 {
            sum / a[row][row]
        } else {
            0.0
        };
    }
    x
}

/// Perspective-warp the source image's quadrilateral region into a
/// straightened output crop -- replaces cv2.warpPerspective. Uses inverse
/// mapping with nearest-neighbor sampling: for each destination pixel, map
/// back into source space via the inverse homography and sample. Simpler
/// than bilinear but sufficient for this use (TrOCR resizes the crop again
/// afterward, so a small amount of nearest-neighbor softness here is not
/// the limiting factor on output quality).
fn warp_perspective(
    src_img: &RgbImage,
    src_quad: &[(f32, f32); 4],
    out_w: u32,
    out_h: u32,
) -> RgbImage {
    let dst_quad = [
        (0.0, (out_h - 1) as f32),
        (0.0, 0.0),
        ((out_w - 1) as f32, 0.0),
        ((out_w - 1) as f32, (out_h - 1) as f32),
    ];

    // Forward transform maps src -> dst; invert it so we can map each
    // dst pixel back to a src sample location.
    let forward = get_perspective_transform(src_quad, &dst_quad);
    let inverse = invert_3x3(&forward);

    let (src_w, src_h) = src_img.dimensions();
    let mut out = RgbImage::new(out_w, out_h);

    for dy in 0..out_h {
        for dx in 0..out_w {
            let (x, y) = (dx as f64, dy as f64);
            let denom = inverse[2][0] * x + inverse[2][1] * y + inverse[2][2];
            if denom.abs() < 1e-12 {
                continue;
            }
            let sx = (inverse[0][0] * x + inverse[0][1] * y + inverse[0][2]) / denom;
            let sy = (inverse[1][0] * x + inverse[1][1] * y + inverse[1][2]) / denom;

            if sx >= 0.0 && sy >= 0.0 && (sx as u32) < src_w && (sy as u32) < src_h {
                let pixel = *src_img.get_pixel(sx as u32, sy as u32);
                out.put_pixel(dx, dy, pixel);
            }
        }
    }

    out
}

/// 3x3 matrix inverse via the adjugate/cofactor method -- replaces the
/// implicit inverse OpenCV computes internally for warpPerspective with
/// WARP_INVERSE_MAP, made explicit here since we solve the forward
/// transform and need its inverse ourselves.
fn invert_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };

    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

// Crop sizing bounds. Height is held fixed (it's the meaningful per-line
// normalization -- x-height/cap-height should be consistent across crops
// for TrOCR); width scales with the line's actual content length instead
// of being forced into one fixed box, which previously stretched short
// lines and compressed long ones into the same 600x150 frame, distorting
// character proportions before TrOCR ever saw them.
const CROP_OUT_HEIGHT: u32 = 150;
const CROP_MIN_WIDTH: u32 = 100;
const CROP_MAX_WIDTH: u32 = 2000;

/// Quad edge length between two corners -- used to measure the detected
/// line's actual width and height (post-unclip, post-merge) rather than
/// assuming a fixed box.
fn edge_len(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
}

/// Derive (out_w, out_h) for a crop from the quad's actual aspect ratio.
/// quad winding here matches the rest of this file: 0=bottom-left,
/// 1=top-left, 2=top-right, 3=bottom-right (see warp_perspective's
/// dst_quad for the same convention), so width is edge 1->2 (or 0->3) and
/// height is edge 0->1 (or 3->2).
fn crop_dims_for_quad(quad: &[(f32, f32); 4]) -> (u32, u32) {
    let width = (edge_len(quad[1], quad[2]) + edge_len(quad[0], quad[3])) / 2.0;
    let height = (edge_len(quad[0], quad[1]) + edge_len(quad[3], quad[2])) / 2.0;
    let aspect = if height > 0.5 { width / height } else { 4.0 };

    let out_h = CROP_OUT_HEIGHT;
    let out_w = ((out_h as f32) * aspect)
        .round()
        .clamp(CROP_MIN_WIDTH as f32, CROP_MAX_WIDTH as f32) as u32;

    (out_w, out_h)
}

fn crop_line(page_img: &RgbImage, line: &DetectedLine) -> image::DynamicImage {
    let (out_w, out_h) = crop_dims_for_quad(&line.quad);
    let cropped = warp_perspective(page_img, &line.quad, out_w, out_h);
    image::DynamicImage::ImageRgb8(cropped)
}

/// The Tauri command for full-page OCR. Detects lines with DBNet, merges
/// fragments back into whole lines, crops and recognizes each with the
/// existing TrOCR pipeline, reassembles top-to-bottom into one joined
/// string.
///
/// Register Mutex<OcrEngine> and Mutex<DetectorEngine> in app state (not
/// bare structs) -- see the note on run_ocr in ocr.rs for why.
#[tauri::command]
/// Shared inner pipeline — called by both entry points once they have a
/// DynamicImage. Detection, merging, line sorting, cropping, and
/// recognition all live here once rather than being duplicated.
fn run_ocr_page_inner(
    page_img: image::DynamicImage,
    engine: State<Mutex<OcrEngine>>,
    detector: State<Mutex<DetectorEngine>>,
) -> Result<String, String> {
    let mut detector = detector
        .lock()
        .map_err(|e| format!("detector lock poisoned: {e}"))?;
    let raw_lines = detect_lines(&mut detector, &page_img)?;
    eprintln!("[ocr_page] DBNet raw quads detected: {}", raw_lines.len());

    if std::env::var("OCR_DEBUG_OVERLAY").is_ok() {
        dump_debug_overlay(&page_img.to_rgb8(), &raw_lines, "ocr_debug_raw.png");
    }

    let lines = merge_fragmented_lines(raw_lines);
    eprintln!("[ocr_page] quads after fragment merge: {}", lines.len());

    if lines.is_empty() {
        return Err("no text lines detected on this page".to_string());
    }

    if std::env::var("OCR_DEBUG_OVERLAY").is_ok() {
        dump_debug_overlay(&page_img.to_rgb8(), &lines, "ocr_debug_merged.png");
    }

    let lines = order_lines_for_reading(lines);
    eprintln!(
        "[ocr_page] quads after column split into reading order: {}",
        lines.len()
    );

    let page_rgb = page_img.to_rgb8();
    let mut engine = engine
        .lock()
        .map_err(|e| format!("engine lock poisoned: {e}"))?;
    let mut page_text = String::new();

    for line in &lines {
        let cropped_img = crop_line(&page_rgb, line);
        let line_text = recognize_line(&mut engine, &cropped_img)?;
        if !page_text.is_empty() {
            page_text.push('\n');
        }
        page_text.push_str(&line_text);
    }

    Ok(page_text)
}

/// The single Tauri command both dialog-pick and drag-drop now call, when
/// the frontend's mode toggle is set to "handwriting". Always receives a
/// real file path — zero image data over the IPC bridge. For drag-drop the
/// frontend writes a temp file first; this command is identical either
/// way, which is the whole point.
///
/// See ocr_document.rs's run_ocr_page_document for the printed/structured-
/// document counterpart (PaddleOCR-based), registered as a separate Tauri
/// command rather than a branch here so each pipeline's state/engine types
/// stay independent.
#[tauri::command]
pub fn run_ocr_page_handwriting(
    image_path: String,
    engine: State<Mutex<OcrEngine>>,
    detector: State<Mutex<DetectorEngine>>,
) -> Result<String, String> {
    let page_img =
        image::open(&image_path).map_err(|e| format!("open image at {image_path}: {e}"))?;
    run_ocr_page_inner(page_img, engine, detector)
}