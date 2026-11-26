#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ocria {

struct OcriaConfig {
    int threads = 0;            // 0 = std::thread::hardware_concurrency()
    int det_long_side = 1600;   // max long side for det.onnx (multiples of 32)
    float det_thresh = -1.0f;   // < 0 reads "thresh" from det.meta.json (fallback 0.5)
    float det_min_score = 0.6f; // minimum mean prob per connected component
    float det_merge = 0.6f;     // collinear box merge gap in line-heights (0 = off)
    bool filter_junk = true;    // drop symbol-only / tiny low-confidence noise
    size_t max_run_pixels = 0;  // 0 = unbounded (set e.g. 4*32*1280 for low-RAM TVs)
};

struct OcrLine {
    std::string text;
    float confidence = 0.0f;
    float bbox[4] = {0, 0, 0, 0};          // [x1, y1, x2, y2] in original image pixels
    float quad[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}}; // [TL, TR, BR, BL]
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

    OcriaLatinEngine(const OcriaLatinEngine&) = delete;
    OcriaLatinEngine& operator=(const OcriaLatinEngine&) = delete;
    OcriaLatinEngine(OcriaLatinEngine&&) noexcept;
    OcriaLatinEngine& operator=(OcriaLatinEngine&&) noexcept;

    /// Run English/Latin OCR on an 8-bit row-major image (channels = 1 or 3).
    std::vector<OcrLine> run(const uint8_t* data, int width, int height, int channels,
                             OcrTimings* timings = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ocria
