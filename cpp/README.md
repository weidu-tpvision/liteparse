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

3. **Build with CMake** (native `x86_64` or `aarch64`):
   ```bash
   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
   cmake --build cpp/build -j
   ```

   **Cross-compiling** (`x86_64`, `arm64`, or `arm32` / `armv7-a` with NEON hard-float):
   ```bash
   # Linux x86_64
   cmake -S cpp -B cpp/build-x86_64 \
     -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-linux-x86_64.cmake \
     -DORT_LIB_PATH=/path/to/x86_64/libonnxruntime.so
   cmake --build cpp/build-x86_64 -j

   # Linux arm64 (aarch64)
   cmake -S cpp -B cpp/build-arm64 \
     -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-linux-arm64.cmake \
     -DORT_LIB_PATH=/path/to/arm64/libonnxruntime.so
   cmake --build cpp/build-arm64 -j

   # Linux arm32 (armv7-a, NEON hard-float)
   cmake -S cpp -B cpp/build-arm32 \
     -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-linux-arm32.cmake \
     -DORT_LIB_PATH=/path/to/arm32/libonnxruntime.so
   cmake --build cpp/build-arm32 -j
   ```
   *(Custom flags like `-DCMAKE_CXX_FLAGS="-march=..."` override the default arch flags; cross-compiled binaries use `$ORIGIN` RPATH so placing `libonnxruntime.so.1` alongside `ocria_cli` works on target.)*

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
