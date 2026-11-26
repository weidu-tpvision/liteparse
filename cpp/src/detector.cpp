#include "detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ocria {

namespace {

constexpr float SHRINK = 0.3f;

inline float norm2(float x, float y) {
    return std::sqrt(x * x + y * y);
}

inline float dist2(const std::array<float, 2>& a, const std::array<float, 2>& b) {
    return norm2(a[0] - b[0], a[1] - b[1]);
}

inline float min_y(const Quad& q) {
    float m = std::numeric_limits<float>::infinity();
    for (const auto& p : q) m = std::min(m, p[1]);
    return m;
}

inline float min_x(const Quad& q) {
    float m = std::numeric_limits<float>::infinity();
    for (const auto& p : q) m = std::min(m, p[0]);
    return m;
}

Quad order_quad(const Quad& q) {
    float cx = (q[0][0] + q[1][0] + q[2][0] + q[3][0]) * 0.25f;
    float cy = (q[0][1] + q[1][1] + q[2][1] + q[3][1]) * 0.25f;
    float e1x = q[1][0] - q[0][0], e1y = q[1][1] - q[0][1];
    float e2x = q[2][0] - q[1][0], e2y = q[2][1] - q[1][1];
    float ux = (norm2(e1x, e1y) >= norm2(e2x, e2y)) ? e1x : e2x;
    float uy = (norm2(e1x, e1y) >= norm2(e2x, e2y)) ? e1y : e2y;
    float n = norm2(ux, uy) + 1e-6f;
    ux /= n;
    uy /= n;
    if (ux < 0.0f) {
        ux = -ux;
        uy = -uy;
    }
    float vx = -uy, vy = ux;
    float a[4], b[4];
    for (int i = 0; i < 4; ++i) {
        a[i] = (q[i][0] - cx) * ux + (q[i][1] - cy) * uy;
        b[i] = (q[i][0] - cx) * vx + (q[i][1] - cy) * vy;
    }
    auto pick = [&](auto fn, bool find_max) {
        int best = 0;
        for (int i = 1; i < 4; ++i) {
            if ((find_max && fn(i) > fn(best)) || (!find_max && fn(i) < fn(best))) best = i;
        }
        return best;
    };
    int tl = pick([&](int i) { return a[i] + b[i]; }, false);
    int tr = pick([&](int i) { return a[i] - b[i]; }, true);
    int br = pick([&](int i) { return a[i] + b[i]; }, true);
    int bl = pick([&](int i) { return a[i] - b[i]; }, false);
    return {q[tl], q[tr], q[br], q[bl]};
}

std::vector<Quad> merge_collinear(const std::vector<Quad>& boxes, float gap) {
    constexpr float MAX_TILT = 0.35f;
    constexpr float MAX_PERP = 0.35f;
    constexpr float MIN_H_RATIO = 0.6f;
    constexpr float MAX_DANGLE = 0.07f;
    size_t n = boxes.size();
    if (n < 2) return boxes;

    std::vector<std::array<float, 2>> c(n), u(n), v(n);
    std::vector<float> len(n), hgt(n);
    std::vector<uint8_t> flat(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const auto& q = boxes[i];
        c[i] = {(q[0][0] + q[1][0] + q[2][0] + q[3][0]) * 0.25f,
                (q[0][1] + q[1][1] + q[2][1] + q[3][1]) * 0.25f};
        float dx = q[1][0] - q[0][0], dy = q[1][1] - q[0][1];
        float l = std::max(norm2(dx, dy), 1e-3f);
        u[i] = {dx / l, dy / l};
        v[i] = {-u[i][1], u[i][0]};
        len[i] = l;
        hgt[i] = std::max(norm2(q[3][0] - q[0][0], q[3][1] - q[0][1]), 1e-3f);
        flat[i] = (std::abs(u[i][1]) <= MAX_TILT && len[i] >= 0.5f * hgt[i]) ? 1 : 0;
    }

    float cos_min = 1.0f - MAX_DANGLE * MAX_DANGLE * 0.5f;
    std::vector<int> right_link(n, -1);
    std::vector<std::pair<float, int>> left_best(n, {std::numeric_limits<float>::infinity(), -1});

    for (size_t a = 0; a < n; ++a) {
        if (!flat[a]) continue;
        std::pair<float, int> best = {std::numeric_limits<float>::infinity(), -1};
        for (size_t b = 0; b < n; ++b) {
            if (a == b || !flat[b]) continue;
            float dx = c[b][0] - c[a][0];
            float dy = c[b][1] - c[a][1];
            float along = dx * u[a][0] + dy * u[a][1];
            float perp = std::abs(dx * v[a][0] + dy * v[a][1]);
            float hm = (hgt[a] + hgt[b]) * 0.5f;
            float g = along - (len[a] + len[b]) * 0.5f;
            bool ok = along > 0.0f &&
                      std::min(hgt[a], hgt[b]) / std::max(hgt[a], hgt[b]) >= MIN_H_RATIO &&
                      std::abs(u[a][0] * u[b][0] + u[a][1] * u[b][1]) >= cos_min &&
                      perp <= MAX_PERP * hm &&
                      g <= gap * hm &&
                      g >= -0.5f * std::min(len[a], len[b]);
            if (ok) {
                if (g < best.first) best = {g, static_cast<int>(b)};
                if (g < left_best[b].first) left_best[b] = {g, static_cast<int>(a)};
            }
        }
        right_link[a] = best.second;
    }

    // Keep only mutual nearest links
    std::vector<int> nxt(n, -1), prv(n, -1);
    for (size_t a = 0; a < n; ++a) {
        int b = right_link[a];
        if (b >= 0 && left_best[b].second == static_cast<int>(a)) {
            nxt[a] = b;
            prv[b] = static_cast<int>(a);
        }
    }

    std::vector<uint8_t> used(n, 0);
    std::vector<Quad> out;
    for (size_t i = 0; i < n; ++i) {
        if (used[i] || prv[i] >= 0) continue;
        std::vector<size_t> chain;
        int cur = static_cast<int>(i);
        while (cur >= 0 && !used[cur]) {
            used[cur] = 1;
            chain.push_back(static_cast<size_t>(cur));
            cur = nxt[cur];
        }
        if (chain.size() == 1) {
            out.push_back(boxes[chain[0]]);
            continue;
        }
        size_t longest = chain[0];
        for (size_t idx : chain) {
            if (len[idx] > len[longest]) longest = idx;
        }
        auto ax = u[longest];
        auto ay = v[longest];
        float u0 = std::numeric_limits<float>::infinity();
        float u1 = -std::numeric_limits<float>::infinity();
        float v0 = std::numeric_limits<float>::infinity();
        float v1 = -std::numeric_limits<float>::infinity();
        for (size_t idx : chain) {
            for (const auto& pt : boxes[idx]) {
                float pu = pt[0] * ax[0] + pt[1] * ax[1];
                float pv = pt[0] * ay[0] + pt[1] * ay[1];
                u0 = std::min(u0, pu);
                u1 = std::max(u1, pu);
                v0 = std::min(v0, pv);
                v1 = std::max(v1, pv);
            }
        }
        auto mk = [&](float pu, float pv) -> std::array<float, 2> {
            return {pu * ax[0] + pv * ay[0], pu * ax[1] + pv * ay[1]};
        };
        out.push_back({mk(u0, v0), mk(u1, v0), mk(u1, v1), mk(u0, v1)});
    }
    for (size_t i = 0; i < n; ++i) {
        if (!used[i]) out.push_back(boxes[i]);
    }
    return out;
}

} // namespace

Detector::Detector(const std::string& model_path,
                   int threads,
                   int long_side,
                   float thresh,
                   float min_score,
                   float merge_gap)
    : session_(model_path, threads),
      long_side_(long_side),
      thresh_(thresh),
      min_score_(min_score),
      merge_gap_(merge_gap) {}

std::vector<Quad> Detector::detect(const Img& img) const {
    double big_h = static_cast<double>(img.h);
    double big_w = static_cast<double>(img.w);
    double s = std::min(1.0, static_cast<double>(long_side_) / std::max(big_h, big_w));
    size_t h = std::max<size_t>(32, static_cast<size_t>(std::nearbyint(big_h * s / 32.0)) * 32);
    size_t w = std::max<size_t>(32, static_cast<size_t>(std::nearbyint(big_w * s / 32.0)) * 32);
    double sx = static_cast<double>(w) / big_w;
    double sy = static_cast<double>(h) / big_h;

    Img small = resize_area(img, w, h);
    std::vector<float> x(3 * h * w);
    const size_t hw = h * w;
    for (size_t c = 0; c < 3; ++c) {
        float* dst = x.data() + c * hw;
        for (size_t i = 0; i < hw; ++i) {
            dst[i] = static_cast<float>(small.data[i * 3 + c]) / 127.5f - 1.0f;
        }
    }

    TensorF32 prob = session_.run_f32(
        {1, 3, static_cast<int64_t>(h), static_cast<int64_t>(w)}, x, "prob");
    size_t ph = static_cast<size_t>(prob.shape[2]);
    size_t pw = static_cast<size_t>(prob.shape[3]);
    float stride = static_cast<float>(h / ph);

    std::vector<uint8_t> mask(prob.data.size());
    for (size_t i = 0; i < prob.data.size(); ++i) {
        mask[i] = (prob.data[i] > thresh_) ? 1 : 0;
    }

    std::vector<Quad> boxes;
    for (const auto& comp : components4(mask, pw, ph)) {
        if (comp.pixels.size() < 2) continue;
        double sum = 0.0;
        for (const auto& px : comp.pixels) {
            sum += prob.data[px.second * pw + px.first];
        }
        float mean = static_cast<float>(sum / static_cast<double>(comp.pixels.size()));
        if (mean < min_score_) continue;

        size_t rows = comp.y1 - comp.y0 + 1;
        std::vector<std::pair<size_t, size_t>> span(rows, {std::numeric_limits<size_t>::max(), 0});
        for (const auto& px : comp.pixels) {
            auto& r = span[px.second - comp.y0];
            r.first = std::min(r.first, px.first);
            r.second = std::max(r.second, px.first);
        }
        std::vector<std::pair<float, float>> pts;
        pts.reserve(rows * 4);
        for (size_t i = 0; i < rows; ++i) {
            if (span[i].first == std::numeric_limits<size_t>::max()) continue;
            float y = static_cast<float>(comp.y0 + i);
            float x0 = static_cast<float>(span[i].first);
            float x1 = static_cast<float>(span[i].second + 1);
            pts.emplace_back(x0 * stride, y * stride);
            pts.emplace_back(x0 * stride, (y + 1.0f) * stride);
            pts.emplace_back(x1 * stride, y * stride);
            pts.emplace_back(x1 * stride, (y + 1.0f) * stride);
        }
        RotRect r = min_area_rect(pts);
        float a_mod = std::fmod(r.angle + 90.0f, 180.0f);
        if (a_mod < 0.0f) a_mod += 180.0f;
        bool w_is_horizontal = std::abs(a_mod - 90.0f) < 45.0f;
        float kernel_h = w_is_horizontal ? r.h : r.w;
        float line_h = kernel_h / (1.0f - 2.0f * SHRINK);
        float d = SHRINK * line_h;
        r.w += 2.0f * d;
        r.h += 2.0f * d;

        Quad quad = box_points(r);
        for (auto& pt : quad) {
            pt[0] = std::clamp(static_cast<float>(static_cast<double>(pt[0]) / sx), 0.0f, static_cast<float>(img.w));
            pt[1] = std::clamp(static_cast<float>(static_cast<double>(pt[1]) / sy), 0.0f, static_cast<float>(img.h));
        }
        boxes.push_back(order_quad(quad));
    }

    if (merge_gap_ > 0.0f) {
        boxes = merge_collinear(boxes, merge_gap_);
    }

    std::stable_sort(boxes.begin(), boxes.end(), [](const Quad& a, const Quad& b) {
        double ya = std::nearbyint(static_cast<double>(min_y(a)) / 8.0);
        double yb = std::nearbyint(static_cast<double>(min_y(b)) / 8.0);
        if (ya != yb) return ya < yb;
        return min_x(a) < min_x(b);
    });
    return boxes;
}

Img Detector::crop(const Img& img, const Quad& q) {
    size_t w = std::max<size_t>(2, static_cast<size_t>(std::max(dist2(q[0], q[1]), dist2(q[2], q[3]))));
    size_t h = std::max<size_t>(2, static_cast<size_t>(std::max(dist2(q[0], q[3]), dist2(q[1], q[2]))));
    Quad dst = {{
        {0.0f, 0.0f},
        {static_cast<float>(w), 0.0f},
        {static_cast<float>(w), static_cast<float>(h)},
        {0.0f, static_cast<float>(h)},
    }};
    auto m = perspective_transform(q, dst);
    Img c = warp_perspective_cubic(img, m, w, h);
    if (static_cast<double>(h) / static_cast<double>(w) >= 1.5) {
        return c.rot90(1);
    }
    return c;
}

} // namespace ocria
