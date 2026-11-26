#include "img_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ocria {

namespace {

constexpr int32_t RESIZE_BITS = 11;
constexpr int32_t RESIZE_ONE = 1 << RESIZE_BITS;
constexpr int64_t INTER_BITS = 5;
constexpr int64_t INTER_TAB = 1 << INTER_BITS;
constexpr int32_t REMAP_SCALE = 1 << 15;

inline int64_t cv_round(double v) {
    // Round half to even (FE_TONEAREST default in IEEE 754)
    return static_cast<int64_t>(std::nearbyint(v));
}

inline int32_t sat_short(float v) {
    int64_t r = cv_round(static_cast<double>(v));
    if (r < std::numeric_limits<int16_t>::min()) return std::numeric_limits<int16_t>::min();
    if (r > std::numeric_limits<int16_t>::max()) return std::numeric_limits<int16_t>::max();
    return static_cast<int32_t>(r);
}

struct LinearTap {
    size_t sx;
    int32_t a0;
    int32_t a1;
};

std::vector<LinearTap> linear_taps(size_t src, size_t dst, bool area) {
    double inv_scale = static_cast<double>(dst) / static_cast<double>(src);
    double scale = 1.0 / inv_scale;
    std::vector<LinearTap> out;
    out.reserve(dst);
    for (size_t d = 0; d < dst; ++d) {
        int64_t sx = 0;
        float f = 0.0f;
        if (area) {
            sx = static_cast<int64_t>(std::floor(static_cast<double>(d) * scale));
            float raw = static_cast<float>(static_cast<double>(d + 1) -
                                           static_cast<double>(sx + 1) * inv_scale);
            f = raw <= 0.0f ? 0.0f : raw - std::floor(raw);
        } else {
            float fx = static_cast<float>((static_cast<double>(d) + 0.5) * scale - 0.5);
            float fsx = std::floor(fx);
            sx = static_cast<int64_t>(fsx);
            f = fx - fsx;
        }
        if (sx < 0) {
            sx = 0;
            f = 0.0f;
        }
        if (sx >= static_cast<int64_t>(src) - 1) {
            sx = static_cast<int64_t>(src) - 1;
            f = 0.0f;
        }
        int32_t a0 = sat_short((1.0f - f) * static_cast<float>(RESIZE_ONE));
        int32_t a1 = sat_short(f * static_cast<float>(RESIZE_ONE));
        out.push_back({static_cast<size_t>(sx), a0, a1});
    }
    return out;
}

Img resize_linear_taps(const Img& img, size_t w, size_t h, bool area) {
    auto xt = linear_taps(img.w, w, area);
    auto yt = linear_taps(img.h, h, area);
    size_t ch = img.ch;
    std::vector<std::vector<int32_t>> rows(img.h, std::vector<int32_t>(w * ch, 0));
    for (size_t y = 0; y < img.h; ++y) {
        auto& r = rows[y];
        for (size_t dx = 0; dx < w; ++dx) {
            const auto& t = xt[dx];
            for (size_t c = 0; c < ch; ++c) {
                int32_t s0 = static_cast<int32_t>(img.at(t.sx, y, c));
                if (t.sx + 1 < img.w) {
                    r[dx * ch + c] = s0 * t.a0 + static_cast<int32_t>(img.at(t.sx + 1, y, c)) * t.a1;
                } else {
                    r[dx * ch + c] = s0 * RESIZE_ONE;
                }
            }
        }
    }
    Img out(w, h, ch);
    for (size_t dy = 0; dy < h; ++dy) {
        const auto& t = yt[dy];
        const auto& r0 = rows[t.sx];
        const auto& r1 = rows[std::min(t.sx + 1, img.h - 1)];
        uint8_t* o = out.data.data() + dy * w * ch;
        for (size_t i = 0; i < w * ch; ++i) {
            int64_t v = (static_cast<int64_t>(r0[i]) * t.a0 +
                         static_cast<int64_t>(r1[i]) * t.a1 + (1 << 21)) >> 22;
            o[i] = static_cast<uint8_t>(std::clamp<int64_t>(v, 0, 255));
        }
    }
    return out;
}

std::vector<std::vector<std::pair<size_t, float>>> area_tab(size_t src, size_t dst, double scale) {
    std::vector<std::vector<std::pair<size_t, float>>> out(dst);
    for (size_t d = 0; d < dst; ++d) {
        double fsx1 = static_cast<double>(d) * scale;
        double fsx2 = fsx1 + scale;
        double cell = std::min(scale, static_cast<double>(src) - fsx1);
        int64_t sx2 = std::min(static_cast<int64_t>(std::floor(fsx2)),
                               static_cast<int64_t>(src) - 1);
        int64_t sx1 = std::min(static_cast<int64_t>(std::ceil(fsx1)), sx2);
        auto& t = out[d];
        if (static_cast<double>(sx1) - fsx1 > 1e-3) {
            t.push_back({static_cast<size_t>(sx1 - 1),
                         static_cast<float>((static_cast<double>(sx1) - fsx1) / cell)});
        }
        for (int64_t sx = sx1; sx < sx2; ++sx) {
            t.push_back({static_cast<size_t>(sx), static_cast<float>(1.0 / cell)});
        }
        if (fsx2 - static_cast<double>(sx2) > 1e-3) {
            t.push_back({static_cast<size_t>(sx2),
                         static_cast<float>(std::min({fsx2 - static_cast<double>(sx2), 1.0, cell}) / cell)});
        }
    }
    return out;
}

std::array<double, 9> invert3(const std::array<double, 9>& m) {
    double det = m[0] * (m[4] * m[8] - m[5] * m[7])
               - m[1] * (m[3] * m[8] - m[5] * m[6])
               + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (det == 0.0) return {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double d = 1.0 / det;
    return {
        (m[4] * m[8] - m[5] * m[7]) * d,
        (m[2] * m[7] - m[1] * m[8]) * d,
        (m[1] * m[5] - m[2] * m[4]) * d,
        (m[5] * m[6] - m[3] * m[8]) * d,
        (m[0] * m[8] - m[2] * m[6]) * d,
        (m[2] * m[3] - m[0] * m[5]) * d,
        (m[3] * m[7] - m[4] * m[6]) * d,
        (m[1] * m[6] - m[0] * m[7]) * d,
        (m[0] * m[4] - m[1] * m[3]) * d,
    };
}

std::array<float, 4> cubic_coeffs(float x) {
    constexpr float A = -0.75f;
    float c0 = ((A * (x + 1.0f) - 5.0f * A) * (x + 1.0f) + 8.0f * A) * (x + 1.0f) - 4.0f * A;
    float c1 = ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
    float c2 = ((A + 2.0f) * (1.0f - x) - (A + 3.0f)) * (1.0f - x) * (1.0f - x) + 1.0f;
    return {c0, c1, c2, 1.0f - c0 - c1 - c2};
}

const std::vector<std::array<int32_t, 16>>& bicubic_tab() {
    static const std::vector<std::array<int32_t, 16>> tab = []() {
        std::vector<std::array<float, 4>> one_d(INTER_TAB);
        for (int64_t i = 0; i < INTER_TAB; ++i) {
            one_d[i] = cubic_coeffs(static_cast<float>(i) / static_cast<float>(INTER_TAB));
        }
        std::vector<std::array<int32_t, 16>> out;
        out.reserve(INTER_TAB * INTER_TAB);
        for (int64_t iy = 0; iy < INTER_TAB; ++iy) {
            for (int64_t ix = 0; ix < INTER_TAB; ++ix) {
                const auto& vy = one_d[iy];
                const auto& vx = one_d[ix];
                std::array<int32_t, 16> t{};
                int32_t isum = 0;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        t[i * 4 + j] = sat_short(vy[i] * vx[j] * static_cast<float>(REMAP_SCALE));
                        isum += t[i * 4 + j];
                    }
                }
                if (isum != REMAP_SCALE) {
                    int32_t diff = isum - REMAP_SCALE;
                    int mk0 = 2, mk1 = 2, big0 = 2, big1 = 2;
                    for (int k1 = 2; k1 < 4; ++k1) {
                        for (int k2 = 2; k2 < 4; ++k2) {
                            if (t[k1 * 4 + k2] < t[mk0 * 4 + mk1]) {
                                mk0 = k1; mk1 = k2;
                            } else if (t[k1 * 4 + k2] > t[big0 * 4 + big1]) {
                                big0 = k1; big1 = k2;
                            }
                        }
                    }
                    if (diff < 0) t[big0 * 4 + big1] -= diff;
                    else t[mk0 * 4 + mk1] -= diff;
                }
                out.push_back(t);
            }
        }
        return out;
    }();
    return tab;
}

double cross2(std::pair<double, double> o,
              std::pair<double, double> a,
              std::pair<double, double> b) {
    return (a.first - o.first) * (b.second - o.second) -
           (a.second - o.second) * (b.first - o.first);
}

} // namespace

Img Img::to_gray() const {
    if (ch == 1) return *this;
    Img out(w, h, 1);
    const size_t n = w * h;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* p = data.data() + i * ch;
        uint32_t v = static_cast<uint32_t>(p[0]) * 4899u +
                     static_cast<uint32_t>(p[1]) * 9617u +
                     static_cast<uint32_t>(p[2]) * 1868u + (1u << 13);
        out.data[i] = static_cast<uint8_t>(v >> 14);
    }
    return out;
}

Img Img::rot90(uint32_t k) const {
    switch (k % 4) {
        case 0:
            return *this;
        case 1: {
            Img out(h, w, ch);
            for (size_t i = 0; i < out.h; ++i) {
                for (size_t j = 0; j < out.w; ++j) {
                    size_t sx = w - 1 - i;
                    size_t sy = j;
                    for (size_t c = 0; c < ch; ++c) {
                        out.data[(i * out.w + j) * ch + c] = at(sx, sy, c);
                    }
                }
            }
            return out;
        }
        case 2: {
            Img out(w, h, ch);
            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    for (size_t c = 0; c < ch; ++c) {
                        out.data[(y * w + x) * ch + c] = at(w - 1 - x, h - 1 - y, c);
                    }
                }
            }
            return out;
        }
        default:
            return rot90(1).rot90(2);
    }
}

Img resize_linear(const Img& img, size_t w, size_t h) {
    if (w == img.w && h == img.h) return img;
    return resize_linear_taps(img, w, h, false);
}

Img resize_area(const Img& img, size_t w, size_t h) {
    if (w == img.w && h == img.h) return img;
    double scale_x = static_cast<double>(img.w) / static_cast<double>(w);
    double scale_y = static_cast<double>(img.h) / static_cast<double>(h);
    if (scale_x < 1.0 || scale_y < 1.0) {
        return resize_linear_taps(img, w, h, true);
    }
    size_t ch = img.ch;
    double isx = std::round(scale_x);
    double isy = std::round(scale_y);
    if (std::abs(scale_x - isx) < std::numeric_limits<double>::epsilon() &&
        std::abs(scale_y - isy) < std::numeric_limits<double>::epsilon()) {
        size_t kx = static_cast<size_t>(isx);
        size_t ky = static_cast<size_t>(isy);
        uint32_t area = static_cast<uint32_t>(kx * ky);
        Img out(w, h, ch);
        for (size_t dy = 0; dy < h; ++dy) {
            for (size_t dx = 0; dx < w; ++dx) {
                for (size_t c = 0; c < ch; ++c) {
                    uint32_t s = 0;
                    for (size_t yy = 0; yy < ky; ++yy) {
                        for (size_t xx = 0; xx < kx; ++xx) {
                            s += img.at(dx * kx + xx, dy * ky + yy, c);
                        }
                    }
                    uint32_t v = (area == 4)
                        ? ((s + 2) >> 2)
                        : static_cast<uint32_t>(cv_round(static_cast<double>(s) *
                              static_cast<double>(1.0f / static_cast<float>(area))));
                    out.data[(dy * w + dx) * ch + c] = static_cast<uint8_t>(std::min<uint32_t>(v, 255));
                }
            }
        }
        return out;
    }

    struct HTap { size_t d, s; float a; };
    auto xtab_raw = area_tab(img.w, w, scale_x);
    std::vector<HTap> xt;
    for (size_t dx = 0; dx < w; ++dx) {
        for (const auto& kv : xtab_raw[dx]) {
            xt.push_back({dx * ch, kv.first * ch, kv.second});
        }
    }
    auto yt = area_tab(img.h, h, scale_y);
    Img out(w, h, ch);
    size_t row_len = img.w * ch;

    auto hsum = [&](size_t sy, std::vector<float>& buf) {
        std::fill(buf.begin(), buf.end(), 0.0f);
        const uint8_t* row = img.data.data() + sy * row_len;
        for (const auto& t : xt) {
            for (size_t c = 0; c < ch; ++c) {
                buf[t.d + c] += static_cast<float>(row[t.s + c]) * t.a;
            }
        }
    };

    std::vector<float> buf(w * ch, 0.0f);
    size_t buf_row = std::numeric_limits<size_t>::max();
    std::vector<float> sum(w * ch, 0.0f);

    for (size_t dy = 0; dy < h; ++dy) {
        std::fill(sum.begin(), sum.end(), 0.0f);
        for (const auto& yt_item : yt[dy]) {
            size_t sy = yt_item.first;
            float beta = yt_item.second;
            if (buf_row != sy) {
                hsum(sy, buf);
                buf_row = sy;
            }
            for (size_t i = 0; i < w * ch; ++i) {
                sum[i] += buf[i] * beta;
            }
        }
        uint8_t* o = out.data.data() + dy * w * ch;
        for (size_t i = 0; i < w * ch; ++i) {
            o[i] = static_cast<uint8_t>(std::clamp<int64_t>(cv_round(sum[i]), 0, 255));
        }
    }
    return out;
}

std::array<double, 9> perspective_transform(const Quad& src, const Quad& dst) {
    double a[8][9] = {};
    for (int i = 0; i < 4; ++i) {
        double x = src[i][0], y = src[i][1];
        double u = dst[i][0], v = dst[i][1];
        a[i][0] = x; a[i][1] = y; a[i][2] = 1.0;
        a[i][6] = -x * u; a[i][7] = -y * u; a[i][8] = u;
        a[i + 4][3] = x; a[i + 4][4] = y; a[i + 4][5] = 1.0;
        a[i + 4][6] = -x * v; a[i + 4][7] = -y * v; a[i + 4][8] = v;
    }
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        for (int r = col + 1; r < 8; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
        }
        if (piv != col) {
            for (int k = 0; k < 9; ++k) std::swap(a[col][k], a[piv][k]);
        }
        double d = a[col][col];
        if (std::abs(d) < 1e-300) continue;
        for (int r = 0; r < 8; ++r) {
            if (r == col) continue;
            double f = a[r][col] / d;
            if (f != 0.0) {
                for (int k = col; k < 9; ++k) a[r][k] -= f * a[col][k];
            }
        }
    }
    std::array<double, 9> m{};
    for (int i = 0; i < 8; ++i) {
        m[i] = (a[i][i] != 0.0) ? (a[i][8] / a[i][i]) : 0.0;
    }
    m[8] = 1.0;
    return m;
}

Img warp_perspective_cubic(const Img& img, const std::array<double, 9>& m, size_t w, size_t h) {
    auto inv = invert3(m);
    const auto& tab = bicubic_tab();
    size_t ch = img.ch;
    Img out(w, h, ch);
    int64_t sw = static_cast<int64_t>(img.w);
    int64_t sh = static_cast<int64_t>(img.h);

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            double xf = static_cast<double>(x);
            double yf = static_cast<double>(y);
            double x0 = inv[0] * xf + inv[1] * yf + inv[2];
            double y0 = inv[3] * xf + inv[4] * yf + inv[5];
            double ww = inv[6] * xf + inv[7] * yf + inv[8];
            ww = (ww != 0.0) ? (static_cast<double>(INTER_TAB) / ww) : 0.0;
            double fx = std::clamp(x0 * ww,
                                   static_cast<double>(std::numeric_limits<int32_t>::min()),
                                   static_cast<double>(std::numeric_limits<int32_t>::max()));
            double fy = std::clamp(y0 * ww,
                                   static_cast<double>(std::numeric_limits<int32_t>::min()),
                                   static_cast<double>(std::numeric_limits<int32_t>::max()));
            int64_t xi = cv_round(fx);
            int64_t yi = cv_round(fy);
            int64_t sx = xi >> INTER_BITS;
            int64_t sy = yi >> INTER_BITS;
            size_t a = static_cast<size_t>((yi & (INTER_TAB - 1)) * INTER_TAB + (xi & (INTER_TAB - 1)));
            const auto& wt = tab[a];
            bool inside = (sx >= 1 && sy >= 1 && sx + 2 < sw && sy + 2 < sh);
            size_t cols[4];
            for (int j = 0; j < 4; ++j) {
                cols[j] = inside ? static_cast<size_t>(sx - 1 + j)
                                 : static_cast<size_t>(std::clamp<int64_t>(sx - 1 + j, 0, sw - 1));
            }
            int64_t sum[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4; ++i) {
                size_t yy = inside ? static_cast<size_t>(sy - 1 + i)
                                   : static_cast<size_t>(std::clamp<int64_t>(sy - 1 + i, 0, sh - 1));
                const uint8_t* row = img.data.data() + yy * img.w * ch;
                for (int j = 0; j < 4; ++j) {
                    int64_t wv = wt[i * 4 + j];
                    const uint8_t* px = row + cols[j] * ch;
                    for (size_t c = 0; c < ch; ++c) {
                        sum[c] += static_cast<int64_t>(px[c]) * wv;
                    }
                }
            }
            uint8_t* o = out.data.data() + (y * w + x) * ch;
            for (size_t c = 0; c < ch; ++c) {
                o[c] = static_cast<uint8_t>(std::clamp<int64_t>((sum[c] + (1 << 14)) >> 15, 0, 255));
            }
        }
    }
    return out;
}

std::vector<Component> components4(const std::vector<uint8_t>& mask, size_t w, size_t h) {
    std::vector<uint8_t> seen(w * h, 0);
    std::vector<Component> out;
    std::vector<size_t> stack;
    for (size_t start = 0; start < w * h; ++start) {
        if (!mask[start] || seen[start]) continue;
        seen[start] = 1;
        stack.push_back(start);
        Component comp;
        comp.x0 = std::numeric_limits<size_t>::max();
        comp.y0 = std::numeric_limits<size_t>::max();
        while (!stack.empty()) {
            size_t p = stack.back();
            stack.pop_back();
            size_t x = p % w;
            size_t y = p / w;
            comp.x0 = std::min(comp.x0, x);
            comp.y0 = std::min(comp.y0, y);
            comp.x1 = std::max(comp.x1, x);
            comp.y1 = std::max(comp.y1, y);
            comp.pixels.emplace_back(x, y);
            auto visit = [&](size_t q) {
                if (mask[q] && !seen[q]) {
                    seen[q] = 1;
                    stack.push_back(q);
                }
            };
            if (x > 0) visit(p - 1);
            if (x + 1 < w) visit(p + 1);
            if (y > 0) visit(p - w);
            if (y + 1 < h) visit(p + w);
        }
        out.push_back(std::move(comp));
    }
    return out;
}

RotRect min_area_rect(const std::vector<std::pair<float, float>>& pts) {
    if (pts.empty()) return {};
    std::vector<std::pair<double, double>> p;
    p.reserve(pts.size());
    for (const auto& pt : pts) p.emplace_back(pt.first, pt.second);
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    if (p.size() == 1) {
        return {static_cast<float>(p[0].first), static_cast<float>(p[0].second), 0.0f, 0.0f, 0.0f};
    }
    std::vector<std::pair<double, double>> hull;
    hull.reserve(p.size() * 2);
    for (const auto& q : p) {
        while (hull.size() >= 2 && cross2(hull[hull.size() - 2], hull.back(), q) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(q);
    }
    size_t lower = hull.size() + 1;
    for (size_t i = p.size() - 1; i > 0; --i) {
        const auto& q = p[i - 1];
        while (hull.size() >= lower && cross2(hull[hull.size() - 2], hull.back(), q) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(q);
    }
    hull.pop_back();

    bool has_best = false;
    double best_area = std::numeric_limits<double>::infinity();
    RotRect best{};
    constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;

    for (size_t i = 0; i < hull.size(); ++i) {
        auto a = hull[i];
        auto b = hull[(i + 1) % hull.size()];
        double dx = b.first - a.first;
        double dy = b.second - a.second;
        double len = std::hypot(dx, dy);
        if (len == 0.0) continue;
        double ux = dx / len, uy = dy / len;
        double vx = -uy, vy = ux;
        double u0 = std::numeric_limits<double>::infinity();
        double u1 = -std::numeric_limits<double>::infinity();
        double v0 = std::numeric_limits<double>::infinity();
        double v1 = -std::numeric_limits<double>::infinity();
        for (const auto& q : hull) {
            double pu = q.first * ux + q.second * uy;
            double pv = q.first * vx + q.second * vy;
            u0 = std::min(u0, pu);
            u1 = std::max(u1, pu);
            v0 = std::min(v0, pv);
            v1 = std::max(v1, pv);
        }
        double area = (u1 - u0) * (v1 - v0);
        if (!has_best || area < best_area) {
            has_best = true;
            best_area = area;
            double cu = (u0 + u1) * 0.5;
            double cv = (v0 + v1) * 0.5;
            best = {
                static_cast<float>(cu * ux + cv * vx),
                static_cast<float>(cu * uy + cv * vy),
                static_cast<float>(u1 - u0),
                static_cast<float>(v1 - v0),
                static_cast<float>(std::atan2(uy, ux) * RAD2DEG),
            };
        }
    }
    return has_best ? best
                    : RotRect{static_cast<float>(p[0].first), static_cast<float>(p[0].second), 0, 0, 0};
}

Quad box_points(const RotRect& r) {
    constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
    double a = static_cast<double>(r.angle) * DEG2RAD;
    float b = static_cast<float>(std::cos(a) * 0.5);
    float c = static_cast<float>(std::sin(a) * 0.5);
    std::array<float, 2> p0 = {r.cx - c * r.h - b * r.w, r.cy + b * r.h - c * r.w};
    std::array<float, 2> p1 = {r.cx + c * r.h - b * r.w, r.cy - b * r.h - c * r.w};
    std::array<float, 2> p2 = {2.0f * r.cx - p0[0], 2.0f * r.cy - p0[1]};
    std::array<float, 2> p3 = {2.0f * r.cx - p1[0], 2.0f * r.cy - p1[1]};
    return {p0, p1, p2, p3};
}

} // namespace ocria
