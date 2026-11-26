#pragma once

#include "img_ops.hpp"
#include "ort_session.hpp"

#include <string>
#include <vector>

namespace ocria {

class Detector {
public:
    Detector(const std::string& model_path,
             int threads,
             int long_side,
             float thresh,
             float min_score,
             float merge_gap);

    /// Detect text-line quads [TL, TR, BR, BL] in `img` pixels, sorted in reading-ish order.
    std::vector<Quad> detect(const Img& img) const;

    /// Perspective-crop a (possibly rotated) quad; rotates vertical boxes (h/w >= 1.5) 90 deg CCW.
    static Img crop(const Img& img, const Quad& q);

private:
    OrtModelSession session_;
    int long_side_;
    float thresh_;
    float min_score_;
    float merge_gap_;
};

} // namespace ocria
