#include "ort_session.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace ocria {

namespace {

Ort::Env& shared_env() {
    static Ort::Env env = []() {
        // Prevent 1DS telemetry background thread teardown race on Linux ORT builds.
        if (std::getenv("ORT_DISABLE_TELEMETRY") == nullptr) {
            setenv("ORT_DISABLE_TELEMETRY", "1", 0);
        }
        Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "ocria_latin_cpp");
        e.DisableTelemetryEvents();
        return e;
    }();
    return env;
}

} // namespace

struct OrtModelSession::Impl {
    Ort::Session session;
    Ort::MemoryInfo mem_info;

    Impl(const std::string& model_path, int threads)
        : session(nullptr),
          mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        Ort::SessionOptions opts;
        int t = threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency());
        if (t < 1) t = 1;
        opts.SetIntraOpNumThreads(t);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
        session = Ort::Session(shared_env(), model_path.c_str(), opts);
    }
};

OrtModelSession::OrtModelSession(const std::string& model_path, int threads)
    : impl_(std::make_unique<Impl>(model_path, threads)) {}

OrtModelSession::~OrtModelSession() = default;

TensorF32 OrtModelSession::run_f32(const std::array<int64_t, 4>& shape,
                                   std::vector<float>& input,
                                   const char* output_name) const {
    const char* in_names[] = {"x"};
    const char* out_names[] = {output_name};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        impl_->mem_info, input.data(), input.size(), shape.data(), shape.size());

    auto outputs = impl_->session.Run(
        Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    TensorF32 out;
    out.shape = info.GetShape();
    size_t count = info.GetElementCount();
    const float* ptr = outputs[0].GetTensorData<float>();
    out.data.assign(ptr, ptr + count);
    return out;
}

void OrtModelSession::run_rec(const std::array<int64_t, 4>& shape,
                              std::vector<float>& input,
                              TensorI64& ids_out,
                              TensorF32& conf_out) const {
    const char* in_names[] = {"x"};
    const char* out_names[] = {"ids", "conf"};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        impl_->mem_info, input.data(), input.size(), shape.data(), shape.size());

    auto outputs = impl_->session.Run(
        Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 2);

    auto id_info = outputs[0].GetTensorTypeAndShapeInfo();
    ids_out.shape = id_info.GetShape();
    size_t id_count = id_info.GetElementCount();
    const int64_t* id_ptr = outputs[0].GetTensorData<int64_t>();
    ids_out.data.assign(id_ptr, id_ptr + id_count);

    auto cf_info = outputs[1].GetTensorTypeAndShapeInfo();
    conf_out.shape = cf_info.GetShape();
    size_t cf_count = cf_info.GetElementCount();
    const float* cf_ptr = outputs[1].GetTensorData<float>();
    conf_out.data.assign(cf_ptr, cf_ptr + cf_count);
}

} // namespace ocria
