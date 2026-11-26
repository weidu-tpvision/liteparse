#pragma once

#include "img_ops.hpp"
#include "ort_session.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ocria {

class LatinRecognizer {
public:
    LatinRecognizer(const std::string& model_path,
                    const std::string& charset_path,
                    int threads,
                    size_t max_run_pixels = 0);

    /// Returns (text, mean_confidence) for each line crop in input order.
    std::vector<std::pair<std::string, float>> read(const std::vector<Img>& crops) const;

    size_t stride() const { return stride_; }
    size_t vocab_size() const { return chars_.size(); }

private:
    void read_run(const std::vector<Img>& lines,
                  const size_t* idx,
                  size_t n_idx,
                  size_t w_pad,
                  std::vector<std::pair<std::string, float>>& out) const;

    OrtModelSession session_;
    std::vector<std::string> chars_;
    size_t stride_ = 4;
    size_t max_run_pixels_ = 0;
};

} // namespace ocria
