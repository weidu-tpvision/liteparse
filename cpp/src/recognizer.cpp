#include "recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ocria {

namespace {

constexpr size_t REC_H = 32;
constexpr size_t REC_MIN_W = 8;
constexpr size_t REC_MAX_W = 1280;
constexpr size_t REC_BUCKET_W = 128;
constexpr size_t REC_BATCH = 32;

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

uint32_t parse_hex4(const std::string& s, size_t pos) {
    if (pos + 4 > s.size()) throw std::runtime_error("truncated \\uXXXX escape in charset");
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) {
        char c = s[pos + i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
        else throw std::runtime_error("invalid hex digit in charset");
    }
    return v;
}

/// Minimal RFC-8259 JSON string-array parser for `rec_latin.charset.json`.
std::vector<std::string> load_charset_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("cannot open charset: " + path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string s = ss.str();

    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size() && s[i] != '[') ++i;
    if (i >= s.size()) throw std::runtime_error("invalid charset JSON (missing '[')");
    ++i;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n' || s[i] == ',')) ++i;
        if (i >= s.size() || s[i] == ']') break;
        if (s[i] != '"') throw std::runtime_error("expected string in charset JSON");
        ++i;
        std::string item;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                if (i >= s.size()) break;
                switch (s[i]) {
                    case '"': item.push_back('"'); ++i; break;
                    case '\\': item.push_back('\\'); ++i; break;
                    case '/': item.push_back('/'); ++i; break;
                    case 'b': item.push_back('\b'); ++i; break;
                    case 'f': item.push_back('\f'); ++i; break;
                    case 'n': item.push_back('\n'); ++i; break;
                    case 'r': item.push_back('\r'); ++i; break;
                    case 't': item.push_back('\t'); ++i; break;
                    case 'u': {
                        ++i;
                        uint32_t cp = parse_hex4(s, i);
                        i += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                            uint32_t lo = parse_hex4(s, i + 2);
                            i += 6;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                        }
                        append_utf8(item, cp);
                        break;
                    }
                    default: item.push_back(s[i]); ++i; break;
                }
            } else {
                item.push_back(s[i++]);
            }
        }
        if (i < s.size() && s[i] == '"') ++i;
        out.push_back(std::move(item));
    }
    return out;
}

Img prep_line(const Img& crop) {
    Img g = crop.to_gray();
    double raw_w = static_cast<double>(g.w) * static_cast<double>(REC_H) /
                   static_cast<double>(std::max<size_t>(1, g.h));
    size_t nw = std::clamp<size_t>(static_cast<size_t>(std::nearbyint(raw_w)), REC_MIN_W, REC_MAX_W);
    return (g.h > REC_H) ? resize_area(g, nw, REC_H) : resize_linear(g, nw, REC_H);
}

} // namespace

LatinRecognizer::LatinRecognizer(const std::string& model_path,
                                 const std::string& charset_path,
                                 int threads,
                                 size_t max_run_pixels)
    : session_(model_path, threads),
      chars_(load_charset_json(charset_path)),
      max_run_pixels_(max_run_pixels) {
    // Probe timestep stride via one [1, 1, 32, 64] run (matches rec.rs:50-58).
    std::vector<float> probe(REC_H * 64, 1.0f);
    TensorI64 ids;
    TensorF32 conf;
    session_.run_rec({1, 1, static_cast<int64_t>(REC_H), 64}, probe, ids, conf);
    size_t steps = ids.shape.size() >= 2 ? static_cast<size_t>(ids.shape[1]) : 16;
    stride_ = 64 / std::max<size_t>(1, steps);
}

std::vector<std::pair<std::string, float>> LatinRecognizer::read(const std::vector<Img>& crops) const {
    if (crops.empty()) return {};
    std::vector<Img> lines;
    lines.reserve(crops.size());
    for (const auto& c : crops) lines.push_back(prep_line(c));

    std::vector<size_t> order(lines.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return lines[a].w < lines[b].w; });

    std::vector<std::pair<std::string, float>> out(lines.size(), {"", 0.0f});
    for (size_t start = 0; start < order.size(); start += REC_BATCH) {
        size_t end = std::min(start + REC_BATCH, order.size());
        size_t maxw = 1;
        for (size_t k = start; k < end; ++k) maxw = std::max(maxw, lines[order[k]].w);
        size_t w_pad = ((maxw + REC_BUCKET_W - 1) / REC_BUCKET_W) * REC_BUCKET_W;

        size_t chunk = (max_run_pixels_ > 0)
            ? std::max<size_t>(1, max_run_pixels_ / (REC_H * w_pad))
            : (end - start);
        for (size_t sub = start; sub < end; sub += chunk) {
            size_t sub_n = std::min(chunk, end - sub);
            read_run(lines, order.data() + sub, sub_n, w_pad, out);
        }
    }
    return out;
}

void LatinRecognizer::read_run(const std::vector<Img>& lines,
                               const size_t* idx,
                               size_t n_idx,
                               size_t w_pad,
                               std::vector<std::pair<std::string, float>>& out) const {
    std::vector<float> x(n_idx * REC_H * w_pad, 1.0f);
    for (size_t j = 0; j < n_idx; ++j) {
        const Img& l = lines[idx[j]];
        for (size_t y = 0; y < REC_H; ++y) {
            float* dst = x.data() + (j * REC_H + y) * w_pad;
            const uint8_t* src = l.data.data() + y * l.w;
            for (size_t xx = 0; xx < l.w; ++xx) {
                dst[xx] = static_cast<float>(static_cast<double>(src[xx]) / 127.5 - 1.0);
            }
        }
    }

    TensorI64 ids;
    TensorF32 conf;
    session_.run_rec({static_cast<int64_t>(n_idx), 1, static_cast<int64_t>(REC_H),
                      static_cast<int64_t>(w_pad)},
                     x, ids, conf);
    size_t t_all = static_cast<size_t>(ids.shape[1]);

    for (size_t j = 0; j < n_idx; ++j) {
        size_t orig_i = idx[j];
        size_t t = std::min(lines[orig_i].w / stride_, t_all);
        std::string text;
        double conf_sum = 0.0;
        size_t conf_cnt = 0;
        int64_t prev = 0;
        for (size_t k = 0; k < t; ++k) {
            int64_t c = ids.data[j * t_all + k];
            if (c != 0 && c != prev) {
                size_t ci = static_cast<size_t>(c - 1);
                if (ci < chars_.size()) text += chars_[ci];
                conf_sum += static_cast<double>(conf.data[j * t_all + k]);
                ++conf_cnt;
            }
            prev = c;
        }
        float mean_conf = conf_cnt > 0 ? static_cast<float>(conf_sum / static_cast<double>(conf_cnt)) : 0.0f;
        out[orig_i] = {std::move(text), mean_conf};
    }
}

} // namespace ocria
