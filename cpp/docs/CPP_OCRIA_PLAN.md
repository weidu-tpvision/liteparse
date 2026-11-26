# C++ Implementation Plan: English/Latin `ocria-nano` OCR (Linux `x86_64` & `aarch64`)

**Target Hosts:** General Linux (`x86_64` and `aarch64`, GCC/Clang with C++17, CMake 3.20+)  
**Models Used:** `det.onnx` (348 KB) + `rec_latin.onnx` (944 KB) + `rec_latin.charset.json` + `det.meta.json` (**~1.3 MB total**)  
**Runtime:** ONNX Runtime 1.31.1 (`libonnxruntime.so` for `linux-x64` or `linux-arm64`)  
**Design Goal:** Zero-OpenCV, zero-PDFium, dependency-free C++17 library + CLI with exact parity against LiteParse's Rust `ocria-nano` engine, portable across 64-bit and 32-bit Linux targets.

---

## 1. Scope & Non-Goals

### In Scope (English / Latin Only)
- Load raw RGB/grayscale images (PNG/JPEG via header-only `stb_image.h` in CLI; raw `uint8_t*` buffer in library API).
- Run **`det.onnx`** text-line detector with exact preprocessing and postprocessing (4-connected components, `min_area_rect`, `SHRINK=0.3` unclip, `order_quad`, `merge_collinear`, optional `split_gutters`).
- Perspective-crop each detected quad (bicubic warp, rotate vertical boxes `h/w >= 1.5`).
- Run **`rec_latin.onnx`** batched CTC recognizer (grayscale `H=32`, width bucketing to multiples of `128`, `+1.0` white right-padding, in-graph argmax `ids`/`conf` → greedy CTC decode).
- Output line text, confidence, axis-aligned bbox `[x1, y1, x2, y2]`, and 4-point quad, plus per-stage latency breakdown (`det_ms`, `crop_ms`, `rec_ms`, `total_ms`).
- Automated parity verification against LiteParse's Python `ocria-nano` output on `integration_tests_data/receipt.png` and rendered pages of `receipt_scan.pdf`.

### Out of Scope for Phase 1
- CJK / RTL / World recognizers and `router.onnx` (can be added later as optional modules if upside-down 180° flip detection is needed).
- `tablenet.onnx` (LayoutNet) and PDFium integration.
- Cross-compilation toolchains (supported via `-DORT_INCLUDE_DIR` / `-DORT_LIB_PATH`, not bundled).

---

## 2. Project Layout (under `tests/cpp_ocria/`)

```text
tests/cpp_ocria/
├── CMakeLists.txt
├── third_party/                  # Header-only deps (no system packages required)
│   ├── onnxruntime/              # Official ORT 1.31.1 C/C++ headers (onnxruntime_c_api.h, onnxruntime_cxx_api.h)
│   └── stb_image.h               # Image decoding for CLI only
├── include/
│   └── ocria_latin.hpp           # Public C++17 API (OcriaConfig, OcrLine, OcriaLatinEngine)
├── src/
│   ├── img_ops.hpp / .cpp        # Dependency-free port of crates/liteparse/src/ocr/ocria/img.rs
│   ├── ort_session.hpp / .cpp    # Thin ORT C++ wrapper (env, telemetry off, thread pool, run("x"))
│   ├── detector.hpp / .cpp       # Exact port of det.rs (preprocess, CC4, unclip, collinear merge)
│   ├── recognizer.hpp / .cpp     # Exact port of rec.rs (charset parser, stride probe, batch CTC)
│   ├── ocria_latin.cpp           # Pipeline orchestrator (det -> crop -> rec -> clean/junk filter)
│   └── main.cpp                  # CLI binary: ./ocria_cli <model_dir> <image_path> [--long-side N] [--threads N] [--json]
└── verify_parity.py              # Compares C++ CLI output vs Python LiteParse ocria-nano
```

**Why zero-OpenCV?**
1. DGX Spark has no `opencv4` system package installed.
2. `crates/liteparse/src/ocr/ocria/img.rs` is only ~689 lines of self-contained math (`resize_area`, `resize_linear`, `warp_perspective_cubic`, `components4`, `min_area_rect`, `otsu`). Porting it directly guarantees **numerical parity** with LiteParse and keeps the final binary tiny (~200 KB + `libonnxruntime.so`) for future TV deployment.

---

## 3. Exact ONNX Tensor Contract

Both models take a single NCHW `float32` input named **`"x"`** with symmetric normalization `x = u8 / 127.5f - 1.0f`:

| Model | Input `"x"` | Output(s) | Notes |
|---|---|---|---|
| `det.onnx` | `[1, 3, H, W]` (`f32`, RGB) | `"prob"`: `[1, 1, H/s, W/s]` (`f32`) | `H, W` multiples of 32, `max(H,W) <= det_long_side` (default `1600`, never upscale). Threshold from `det.meta.json` (`0.55`). |
| `rec_latin.onnx` | `[B, 1, 32, W_pad]` (`f32`, gray) | `"ids"`: `[B, T]` (`int64_t`)<br>`"conf"`: `[B, T]` (`float`) | `W_pad` rounded up to multiple of `128` (`<= 1280`), right-padded with `+1.0f`. Stride probed once at init via `[1, 1, 32, 64]` (`stride = 64 / T`). |

**Charset loading (`rec_latin.charset.json`)**:
- Simple JSON array of UTF-8 strings (e.g. `[" ", "!", ...]`, 4.9 KB). Parse with a ~40-line minimal JSON string-array parser (no need for `nlohmann/json`). Token `0` is CTC blank; token `c > 0` maps to `charset[c - 1]`.

---

## 4. Public C++ API Sketch (`include/ocria_latin.hpp`)

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ocria {

struct OcriaConfig {
    int threads = 0;            // 0 = hardware_concurrency
    int det_long_side = 1600;   // configurable (e.g., 960/1280 for faster inference)
    float det_thresh = -1.0f;   // <0 = load from det.meta.json (0.55), else override
    float det_min_score = 0.6f;
    float det_merge = 0.6f;     // collinear merge gap in line-heights (0 = off)
    bool split_gutters = true;
    bool filter_junk = true;
};

struct OcrLine {
    std::string text;
    float confidence;
    float bbox[4];              // [x1, y1, x2, y2] in original image pixels
    float quad[4][2];           // [TL, TR, BR, BL]
};

struct OcrTimings {
    double det_ms = 0.0;
    double crop_ms = 0.0;
    double rec_ms = 0.0;
    double total_ms = 0.0;
};

class OcriaLatinEngine {
public:
    explicit OcriaLatinEngine(const std::string& model_dir, const OcriaConfig& cfg = {});
    ~OcriaLatinEngine();

    // Input: row-major uint8 buffer, channels = 1 (gray) or 3 (RGB).
    std::vector<OcrLine> run(const uint8_t* data, int width, int height, int channels,
                             OcrTimings* timings = nullptr);
};

} // namespace ocria
```

---

## 5. Step-by-Step Implementation Plan

### Milestone 1: CMake Skeleton & ONNX Runtime Verification
1. Fetch official ONNX Runtime **v1.31.1** C/C++ headers (`onnxruntime_c_api.h`, `onnxruntime_cxx_api.h`, `onnxruntime_cxx_inline.h`) into `tests/cpp_ocria/third_party/onnxruntime/` and `stb_image.h` into `tests/cpp_ocria/third_party/`.
2. Write `CMakeLists.txt` (`C++17`, `-O3`, link `tests/models/ocria-nano/runtime/libonnxruntime.so` with `BUILD_RPATH` so no `LD_LIBRARY_PATH` hack is needed).
3. Implement `ort_session.{hpp,cpp}`:
   - Set `ORT_DISABLE_TELEMETRY=1` before `Ort::Env` creation (matches `crates/liteparse/src/ocr/ocria/mod.rs:154-158` to prevent 1DS teardown crashes on Linux).
   - Disable telemetry (`env.DisableTelemetryEvents()`), configure intra-op thread count and `GraphOptimizationLevel::ORT_ENABLE_ALL`.
4. **Verify:** Build a minimal smoke binary that loads `det.onnx` + `rec_latin.onnx` and runs the `[1, 1, 32, 64]` stride probe successfully.

### Milestone 2: Dependency-Free Image & Geometry Primitives (`img_ops.{hpp,cpp}`)
Port directly from `crates/liteparse/src/ocr/ocria/img.rs`:
1. `Img` struct (`w`, `h`, `ch`, `std::vector<uint8_t> data`), `to_gray()` (exact BT.601 weights matching `img.rs`), `rot90(k)`.
2. `resize_area()` (box-filter downscaling for `det` and shrinking crops) and `resize_linear()` (bilinear upscaling for small crops).
3. `perspective_transform(src_quad, dst_quad)` (8×8 Gaussian elimination) + `warp_perspective_cubic()` (Catmull-Rom / OpenCV cubic kernel `a = -0.75`).
4. `components4(mask, w, h)` (iterative BFS/union-find 4-connected components).
5. `min_area_rect(pts)` (Andrew's monotone chain convex hull + rotating calipers) + `box_points(rect)`.
6. `otsu()` threshold (needed for optional `split_gutters`).
7. **Verify:** Small C++ unit test checking `resize_area`, `min_area_rect`, `components4`, and `warp_perspective_cubic` on synthetic shapes.

### Milestone 3: Detector (`detector.{hpp,cpp}`) & Recognizer (`recognizer.{hpp,cpp}`)
1. **`Detector`**:
   - Resize RGB image to `(h, w)` multiples of 32 (`s = min(1.0, long_side / max(H, W))`), normalize to `[-1, 1]`, run `det.onnx` → `"prob"`.
   - Threshold `prob > thresh`, run `components4`, filter `< 2` px or mean prob `< 0.6`.
   - Build cell-corner hull scaled by `stride = h / ph`, fit `min_area_rect`, unclip with `SHRINK = 0.3` (`line_h = kernel_h / 0.4`, `d = 0.3 * line_h`).
   - Rescale to original image coordinates, `order_quad` (`[TL, TR, BR, BL]`), `merge_collinear` (`gap = 0.6`), stable sort by `(round(min_y / 8), min_x)`.
2. **`LatinRecognizer`**:
   - Parse `rec_latin.charset.json`, probe `stride` via `[1, 1, 32, 64]`.
   - For each quad: perspective-crop (`crop()`), rotate 90° CCW if `h / w >= 1.5`, convert to gray, resize to `H = 32`, clamp width `[8, 1280]`.
   - Sort indices by width, chunk into batches of `<= 32`, pad width to multiple of `128` with `+1.0f`, run `rec_latin.onnx` → `"ids"` (`int64_t`) & `"conf"` (`float`).
   - Greedy CTC collapse over `t = min(line_w / stride, T)` + mean confidence + `is_junk()` filter.

### Milestone 4: CLI, Parity Check & Spark Benchmarks
1. Implement `main.cpp` CLI accepting `<model_dir> <image_path>` with `--long-side`, `--threads`, `--json`, and `--repeat N` (for warm latency measurement).
2. Write `verify_parity.py` to:
   - Run C++ CLI on `integration_tests_data/receipt.png` and a 250-DPI PNG rendered from `receipt_scan.pdf`.
   - Compare extracted lines and bounding boxes against `LiteParse(ocr_backend="ocria-nano")`.
3. Measure warm latency & peak RSS (`/proc/self/status` `VmHWM`) on DGX Spark at `det_long_side = 1600`, `1280`, and `960` with `threads = 1, 2, 4`.

---

## 6. Cross-Architecture & 32-Bit Portability Notes

Keeping the C++17 implementation portable across `x86_64`, `aarch64`, and 32-bit Linux targets:
1. **ORT tensor shapes**: Always use `int64_t` (`std::array<int64_t, 4>`) for ONNX Runtime shape vectors—never `size_t` or `long`.
2. **Pixel buffer indexing**: Use `size_t` for flat array indices and guard `width * height * channels` against overflow.
3. **Bounded recognizer batching**: Keep a configurable `max_run_pixels` cap (default unbounded; e.g. `4 * 32 * 1280` on memory-constrained devices) to bound peak activation RAM.
4. **No architecture-specific intrinsics**: Keep `img_ops.cpp` in clean portable C++17 so GCC/Clang auto-vectorizes `-O3` loops across `x86_64` (SSE/AVX) and `aarch64` (NEON).
5. **Runtime swap**: Point CMake's `-DORT_LIB_PATH` and `-DORT_INCLUDE_DIR` at any target's `libonnxruntime.so` and headers.
