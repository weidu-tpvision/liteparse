# LiteParse `ocria-nano` Standalone C++17 Engine (English / Latin)

Dependency-free C++17 port of LiteParse's `ocria-nano` OCR pipeline (`det.onnx` + `rec_latin.onnx`, ~1.3 MB weights) running on ONNX Runtime — **zero OpenCV, zero PDFium, zero Rust**.

## Quick Start (Linux `x86_64` & `aarch64`)

1. **Pull `ocria-nano` models locally** (into `tests/models/`):
   ```bash
   LITEPARSE_MODEL_DIR=$PWD/tests/models lit models pull ocria-nano
   ```

2. **Fetch header-only dependencies** (ONNX Runtime 1.31.1 headers + `stb_image.h`):
   ```bash
   python3 cpp/scripts/fetch_headers.py
   ```
   *(If you don't have `lit` / `libonnxruntime.so` installed yet, pass `--with-runtime` to also download `libonnxruntime.so` for your current Linux architecture, `x86_64` or `aarch64`.)*

3. **Build with CMake**:
   ```bash
   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
   cmake --build cpp/build -j
   ```
   *(To use a custom ONNX Runtime SDK or cross-compilation toolchain, pass `-DORT_INCLUDE_DIR=/path/to/include -DORT_LIB_PATH=/path/to/libonnxruntime.so`.)*

4. **Run CLI**:
   ```bash
   ./cpp/build/ocria_cli tests/models/ocria-nano integration_tests_data/receipt.png --threads 4
   ```

5. **Verify exact parity against Python LiteParse**:
   ```bash
   tests/.venv/bin/python cpp/tests/verify_parity.py
   ```

## Docs
- [docs/CPP_OCRIA_PLAN.md](docs/CPP_OCRIA_PLAN.md) — Architecture, tensor contract, and portability notes.
- [docs/OCR_MODEL_GUIDE.md](docs/OCR_MODEL_GUIDE.md) — `ocria-nano` specs, pre/postprocessing formulas, and PP-OCRv5/v6 comparison.
