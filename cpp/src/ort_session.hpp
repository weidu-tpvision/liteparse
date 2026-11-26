#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ocria {

struct TensorF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct TensorI64 {
    std::vector<int64_t> shape;
    std::vector<int64_t> data;
};

/// Thin RAII wrapper over one ONNX Runtime session whose single NCHW f32 input is named "x".
class OrtModelSession {
public:
    OrtModelSession(const std::string& model_path, int threads);
    ~OrtModelSession();

    OrtModelSession(const OrtModelSession&) = delete;
    OrtModelSession& operator=(const OrtModelSession&) = delete;

    TensorF32 run_f32(const std::array<int64_t, 4>& shape,
                      std::vector<float>& input,
                      const char* output_name) const;

    void run_rec(const std::array<int64_t, 4>& shape,
                 std::vector<float>& input,
                 TensorI64& ids_out,
                 TensorF32& conf_out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ocria
