#include "ocria_latin.hpp"
#include "detector.hpp"
#include "img_ops.hpp"
#include "recognizer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ocria {

namespace {

float read_det_meta_thresh(const std::string& model_dir, float fallback) {
    std::ifstream ifs(model_dir + "/det.meta.json");
    if (!ifs) return fallback;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string s = ss.str();
    auto pos = s.find("\"thresh\"");
    if (pos == std::string::npos) return fallback;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return fallback;
    try {
        return std::stof(s.substr(pos + 1));
    } catch (...) {
        return fallback;
    }
}

std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

bool is_junk_latin(const std::string& text, float conf) {
    std::string t = trim_ws(text);
    if (t.empty()) return true;
    size_t alnum = 0;
    for (unsigned char c : t) {
        if (std::isalnum(c) || c >= 0x80) ++alnum;
    }
    if (alnum == 0) return true;
    if (t.size() <= 2 && conf < 0.6f) return true;
    return false;
}

} // namespace

struct OcriaLatinEngine::Impl {
    OcriaConfig cfg;
    Detector det;
    LatinRecognizer rec;

    Impl(const std::string& model_dir, const OcriaConfig& c)
        : cfg(c),
          det(model_dir + "/det.onnx",
              c.threads,
              c.det_long_side,
              c.det_thresh >= 0.0f ? c.det_thresh : read_det_meta_thresh(model_dir, 0.5f),
              c.det_min_score,
              c.det_merge),
          rec(model_dir + "/rec_latin.onnx",
              model_dir + "/rec_latin.charset.json",
              c.threads,
              c.max_run_pixels) {}
};

OcriaLatinEngine::OcriaLatinEngine(const std::string& model_dir, const OcriaConfig& cfg)
    : impl_(std::make_unique<Impl>(model_dir, cfg)) {}

OcriaLatinEngine::~OcriaLatinEngine() = default;
OcriaLatinEngine::OcriaLatinEngine(OcriaLatinEngine&&) noexcept = default;
OcriaLatinEngine& OcriaLatinEngine::operator=(OcriaLatinEngine&&) noexcept = default;

std::vector<OcrLine> OcriaLatinEngine::run(const uint8_t* data, int width, int height, int channels,
                                           OcrTimings* timings) const {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    if (!data || width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        throw std::invalid_argument("invalid image buffer (expected 1 or 3 channels)");
    }

    size_t w = static_cast<size_t>(width);
    size_t h = static_cast<size_t>(height);
    Img img(w, h, 3);
    if (channels == 1) {
        for (size_t i = 0; i < w * h; ++i) {
            uint8_t v = data[i];
            img.data[i * 3 + 0] = v;
            img.data[i * 3 + 1] = v;
            img.data[i * 3 + 2] = v;
        }
    } else {
        img.data.assign(data, data + w * h * 3);
    }

    auto boxes = impl_->det.detect(img);
    auto t1 = Clock::now();

    std::vector<Img> crops;
    crops.reserve(boxes.size());
    for (const auto& q : boxes) {
        crops.push_back(Detector::crop(img, q));
    }
    auto t2 = Clock::now();

    auto reads = impl_->rec.read(crops);
    auto t3 = Clock::now();

    std::vector<OcrLine> out;
    out.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& text = reads[i].first;
        float conf = reads[i].second;
        if (trim_ws(text).empty()) continue;
        if (impl_->cfg.filter_junk && is_junk_latin(text, conf)) continue;

        OcrLine line;
        line.text = text;
        line.confidence = conf;
        float x1 = std::numeric_limits<float>::infinity();
        float y1 = std::numeric_limits<float>::infinity();
        float x2 = -std::numeric_limits<float>::infinity();
        float y2 = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < 4; ++k) {
            line.quad[k][0] = boxes[i][k][0];
            line.quad[k][1] = boxes[i][k][1];
            x1 = std::min(x1, boxes[i][k][0]);
            y1 = std::min(y1, boxes[i][k][1]);
            x2 = std::max(x2, boxes[i][k][0]);
            y2 = std::max(y2, boxes[i][k][1]);
        }
        line.bbox[0] = std::trunc(x1);
        line.bbox[1] = std::trunc(y1);
        line.bbox[2] = std::trunc(x2);
        line.bbox[3] = std::trunc(y2);
        out.push_back(std::move(line));
    }

    auto t4 = Clock::now();
    if (timings) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        timings->det_ms = ms(t0, t1);
        timings->crop_ms = ms(t1, t2);
        timings->rec_ms = ms(t2, t3);
        timings->total_ms = ms(t0, t4);
    }
    return out;
}

} // namespace ocria
