#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ocria {

using Quad = std::array<std::array<float, 2>, 4>;

/// 8-bit interleaved image (ch = 1 grayscale or 3 RGB).
struct Img {
    size_t w = 0;
    size_t h = 0;
    size_t ch = 0;
    std::vector<uint8_t> data;

    Img() = default;
    Img(size_t w_, size_t h_, size_t ch_) : w(w_), h(h_), ch(ch_), data(w_ * h_ * ch_, 0) {}

    inline uint8_t at(size_t x, size_t y, size_t c) const {
        return data[(y * w + x) * ch + c];
    }

    /// 14-bit fixed-point BT.601 RGB -> Gray (matches OpenCV COLOR_RGB2GRAY & img.rs).
    Img to_gray() const;

    /// k quarter turns counter-clockwise (matches np.rot90).
    Img rot90(uint32_t k) const;
};

struct Component {
    size_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    std::vector<std::pair<size_t, size_t>> pixels;
};

struct RotRect {
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float angle = 0.0f; // degrees
};

Img resize_linear(const Img& img, size_t w, size_t h);
Img resize_area(const Img& img, size_t w, size_t h);

std::array<double, 9> perspective_transform(const Quad& src, const Quad& dst);
Img warp_perspective_cubic(const Img& img, const std::array<double, 9>& m, size_t w, size_t h);

std::vector<Component> components4(const std::vector<uint8_t>& mask, size_t w, size_t h);
RotRect min_area_rect(const std::vector<std::pair<float, float>>& pts);
Quad box_points(const RotRect& r);

} // namespace ocria
