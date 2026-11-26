# `ocria-nano` OCR Model Guide (Python & TV/Embedded Reference)

This document summarizes how LiteParse's `ocria-nano` OCR pipeline works end-to-end: Python usage, ONNX model specifications, exact preprocessing, and exact postprocessing—derived directly from `crates/liteparse/src/ocr/ocria/`.

---

## 1. Using `ocria-nano` from Python (`liteparse`)

```python
from liteparse import LiteParse, models

# Download once (~60-75 MB: ONNX models + ONNX Runtime, SHA-256 verified)
# Default cache: ~/.cache/liteparse/models/ocria-nano (override: LITEPARSE_MODEL_DIR)
models.pull("ocria-nano")

# Parse a scanned PDF or image (ocria-nano defaults to 250 DPI)
parser = LiteParse(
    ocr_enabled=True,
    ocr_backend="ocria-nano",
    output_format="markdown",   # "text" | "json" | "markdown"
)
result = parser.parse("integration_tests_data/receipt_scan.pdf")
print(result.text)
```

- `ocr_language` is ignored by `ocria-nano` (the router detects scripts automatically).
- For air-gapped devices/TVs, download once on a host and install offline with `models.pull("ocria-nano", from_dir="/path/to/files")`.

---

## 2. Model Spec & Architecture (`v1.0.0`)

Published at `https://huggingface.co/llamaindex/OCRia-nano-1/resolve/v1.0.0`.

| Model | Size | Purpose | Input (`f32`, NCHW) | Outputs |
|---|---:|---|---|---|
| `det.onnx` | 348 KB | Line segmentation detector | `[1, 3, H, W]` (`H,W` mult of 32, max side 1600, RGB) | `prob`: `[1, 1, H/s, W/s]` (`f32`) |
| `router.onnx` | 159 KB | Script router + 180° flip | `[B, 1, 32, 512]` (grayscale, `B<=64`) | `group`: `[B, 4]` (`latin,cjk,rtl,world`), `flip`: `[B, 2]` |
| `rec_latin.onnx` | 944 KB | Latin/Cyrillic/Greek CTC | `[B, 1, 32, W_pad]` (`W_pad` mult of 128, `<=1280`) | `ids`: `[B, T]` (`i64`), `conf`: `[B, T]` (`f32`) |
| `rec_cjk.onnx` | 5.08 MB | CJK CTC recognizer | `[B, 1, 32, W_pad]` | `ids`: `[B, T]` (`i64`), `conf`: `[B, T]` (`f32`) |
| `rec_rtl.onnx` | 2.84 MB | Arabic/Hebrew CTC (`visual_rtl`) | `[B, 1, 32, W_pad]` | `ids`: `[B, T]` (`i64`), `conf`: `[B, T]` (`f32`) |
| `rec_world.onnx` | 3.91 MB | ~17 other scripts CTC | `[B, 1, 32, W_pad]` | `ids`: `[B, T]` (`i64`), `conf`: `[B, T]` (`f32`) |
| `tablenet.onnx` | 17.18 MB | Optional LayoutNet/TableNet | 66-d unit features + optional `[1, 1, 1024, 1024]` ink canvas | Table/block node & pair logits |

> **Why it's fast on CPU/TVs:** `rec_*.onnx` performs argmax + confidence scoring **inside the ONNX graph**, emitting `[B, T]` `ids` and `conf` instead of `[B, T, Vocab]` logits.

---

## 3. Preprocessing (Exact Formulas)

All three model families normalize `uint8` pixels identically to **`[-1.0, 1.0]`**:
```python
x = pixel_u8 / 127.5 - 1.0   # 0 (black) -> -1.0, 255 (white) -> +1.0
```

1. **Detector (`det.onnx`)**:
   - RGB input (replicate grayscale to 3 channels if needed).
   - `s = min(1.0, 1600 / max(H, W))` (never upscales).
   - `h = max(32, round(H * s / 32) * 32)`, `w = max(32, round(W * s / 32) * 32)`.
   - Downscale with **area interpolation**, transpose HWC → NCHW `[1, 3, h, w]`, normalize.
2. **Line Crop (`page::crop`)**:
   - Perspective-warp detected quad `[TL, TR, BR, BL]` with bicubic interpolation to `(w, h)`.
   - If `h / w >= 1.5`, rotate 90° CCW so the line is horizontal.
3. **Router (`router.onnx`)**:
   - Grayscale, resize to height `32` (`min_w = 8`; area if shrinking, bilinear if enlarging).
   - **Tile horizontally** (`x[:, xx] = img[:, xx % w]`) to fixed width `512`.
   - Normalize, batch up to 64 → `[B, 1, 32, 512]`.
4. **Recognizer (`rec_*.onnx`)**:
   - Grayscale, resize to height `32`, clamp width to `[8, 1280]`.
   - Sort by width, batch up to `32`, right-pad width to a multiple of `128` using **`+1.0` (white)**.
   - Normalize active pixels → `[B, 1, 32, W_pad]`.

---

## 4. Postprocessing

1. **Detector (`prob` → quads)**:
   - Binary mask `prob > thresh` (default `0.5` or `det.meta.json`).
   - 4-connected components; discard `< 2` pixels or mean prob `< 0.6`.
   - Fit `min_area_rect`, expand (unclip) with `SHRINK = 0.3`: `line_h = kernel_h / 0.4`, `d = 0.3 * line_h`, `w += 2*d`, `h += 2*d`.
   - Scale back to original image pixels, order corners `[TL, TR, BR, BL]`.
   - Merge collinear boxes (`gap <= 0.6 * line_h`), split column gutters (`split_gutters`), and optionally run a faint-token pass (`thresh = 0.1`, kept if recognized at `conf >= 0.9`).
2. **Router (`group`, `flip`)**:
   - Width-weighted softmax across lines. Default page script is `latin` unless another script has `> 65%` width share; individual lines override only if `p_max > 0.9` and `w / h >= 4.0`.
   - If `>= 3` lines and width-weighted `P(flip) > 0.7`, rotate crops 180°.
3. **Recognizer (`ids`, `conf` → text)**:
   - Active steps `t = min(line_w // stride, T)` (`stride` = 4 or 8, probed via a `[1, 1, 32, 64]` dummy run).
   - Greedy CTC collapse: skip `0` (blank) and consecutive duplicates; map `c -> charset[c - 1]`; line confidence = mean of emitted token confidences.
   - If `visual_rtl` (`rec_rtl`), reorder visual → logical using Unicode Bidi (`base_dir="R"`).
   - Fold Latin lookalikes (`clean_latin`) and drop junk (`is_junk`).

---

## 5. Companion Test Script

See [test_ocria_pipeline.py](file:///home/weidu/WorkAtTPVision/test/liteparse/tests/test_ocria_pipeline.py) for runnable unit tests of the exact preprocessing/postprocessing math and an optional live `ocria-nano` end-to-end check.

---

## 6. Comparison: `ocria-nano` vs. PP-OCRv5 & PP-OCRv6 (TV Deployment)

| Aspect | `ocria-nano` v1.0.0 | PP-OCRv5 Mobile | PP-OCRv6 (`tiny` / `small`) |
|---|---|---|---|
| **Size** | **~1.3 MB** (Latin-only) / **~13.3 MB** (all 4 scripts, 8.1M params) | ~15–21 MB (det + cls + 1 rec) | **`tiny`**: **~1.5M params**; **`small`**: **~7.7M params** (50 langs unified) |
| **Architecture** | CNN det + script/flip router + 4 CTC recs (**gray `H=32`**, in-graph argmax) | PP-LCNetV3/MobileNetV3 + SVTR-LCNet (**RGB `H=48`**) | Unified **`PPLCNetV4`** (reparam MetaFormer) + **`RepLKFPN`** + **`EncoderWithLightSVTR`** |
| **Best Domain** | Document/PDF scans (dense text, gutters, faint table digits) | Documents + natural scene text / TV UI / subtitles | Documents + scene text / TV UI; `tiny` targets edge/TV C++ |
| **C++ / ORT** | Native ONNX Runtime (`libonnxruntime.so`) | Export to ONNX (ORT / MNN / NCNN) | Export to ONNX (ORT / OpenVINO / Paddle Lite) |
| **Model License** | **Modified OpenRAIL-M**: OK for internal R&D; **prohibits commercial use for >$2M revenue entities** without a LlamaIndex commercial license (`tests/models/ocria-nano/LICENSE` Attachment A §2) | **Apache-2.0** | **Apache-2.0** |

**Local model location:** `tests/models/ocria-nano/` (use `LITEPARSE_MODEL_DIR=$PWD/tests/models` when running `lit` or Python scripts).
